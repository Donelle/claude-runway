"""
A fake Qdrant client for the codebase-indexer's ownership guard
(libs/qdrant_code_index_marker.py). Not a test module (no TestCase), imported
by the ones that exercise the guard.

`scroll` evaluates the filter the guard sends against the payloads the
collection is given, the way Qdrant would, so each case is decided by what
the collection holds rather than by a canned answer. Every other method the
guard does not call is absent: a caller that reaches past a refusal and tries
to write or delete raises AttributeError instead of passing silently.
"""

from types import SimpleNamespace

import memory_bank_lib as mb
from qdrant_client import models


def no_retry(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _get(payload: dict, dotted: str):
    value = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _matches(payload: dict, condition) -> bool:
    if isinstance(condition, models.FieldCondition):
        return _get(payload, condition.key) == condition.match.value
    if isinstance(condition, models.IsEmptyCondition):
        return _get(payload, condition.is_empty.key) in (None, "", [])
    raise AssertionError(f"unexpected condition {condition!r}")


def _passes(payload: dict, flt: models.Filter) -> bool:
    if flt.must and not all(_matches(payload, c) for c in flt.must):
        return False
    if flt.must_not and any(_matches(payload, c) for c in flt.must_not):
        return False
    if flt.should and not any(_matches(payload, c) for c in flt.should):
        return False
    return True


class FakeOwnershipClient:
    """Just enough of QdrantClient for check_code_index_ownership."""

    def __init__(self, exists=True, metadata=None, payloads=(), fail_on=None):
        self.exists = exists
        self.metadata = metadata
        self.payloads = list(payloads)
        self.fail_on = fail_on
        self.updates: list = []
        self.calls: list = []

    def _maybe_fail(self, name):
        self.calls.append(name)
        if self.fail_on == name:
            raise RuntimeError(f"{name} failed")

    def collection_exists(self, collection):
        self._maybe_fail("collection_exists")
        return self.exists

    def get_collection(self, collection):
        self._maybe_fail("get_collection")
        return SimpleNamespace(config=SimpleNamespace(metadata=self.metadata))

    def scroll(self, collection_name, scroll_filter, limit, with_payload, with_vectors):
        self._maybe_fail("scroll")
        hits = [p for p in self.payloads if _passes(p, scroll_filter)][:limit]
        return hits, None

    def update_collection(self, collection_name, metadata):
        self._maybe_fail("update_collection")
        self.updates.append(metadata)
        # Qdrant merges metadata key by key; a later get_collection sees it.
        self.metadata = {**(self.metadata or {}), **metadata}


def code_chunk(path="a.py"):
    return {"document": "x", "metadata": {"file_path": path, "type": "code"}}


def memory_point():
    return {"document": "m", "metadata": {"source": mb.MEMORY_BANK_SOURCE, "repo": "p"}}


def compact_point():
    return {"document": "c", "metadata": {"source": mb.COMPACT_SOURCE}}


def unrelated_point():
    return {"document": "u", "metadata": {"kind": "something"}}


def accept_ownership(*modules):
    """Starts patchers making each module's ownership guard
    (`check_code_index_ownership`) and its later re-checks
    (`current_ownership_error`, and `metadata_ownership_error` where the
    module calls it directly) accept every collection, and returns them for
    the caller to stop.

    For test modules about what happens AFTER the ownership guard: their
    MagicMock clients carry no real collection metadata, so the real guard
    (which fails closed) would refuse every run. The guard itself is tested
    in test_qdrant_code_index_marker.py and, through the tools that call it,
    in test_codebase_indexer_mcp_server_memory_bank_guards.py and
    test_ingest_to_qdrant.py.
    """
    from unittest import mock

    from qdrant_code_index_marker import OwnershipCheck

    patchers = []
    for module in modules:
        patchers.append(mock.patch.object(module, "check_code_index_ownership", return_value=OwnershipCheck()))
        for name in ("current_ownership_error", "metadata_ownership_error"):
            if hasattr(module, name):
                patchers.append(mock.patch.object(module, name, return_value=None))
    for patcher in patchers:
        patcher.start()
    return patchers
