#!/usr/bin/env python3
"""Tests for tools/qdrant_mcp_server.py.

Stdlib-only (unittest, no pytest) and no network -- needs neither a live model
nor a real database (a fake Qdrant client and a fake embedding provider stand
in for both):

    .venv/bin/python -m unittest discover -s tests

Most of this file is the conversation-compact tests (compact_store /
compact_find / compact_prune and their project/collection resolution helpers),
moved here unchanged apart from the module they load when those tools moved
from the local-compress server to this one. The qdrant-find / qdrant-store
tests and the new compact_store compression step are at the bottom.
"""

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "tools"))


async def _identity_compression(information):
    """Stands in for _compress_for_storage in every test that is not about
    compression itself, so none of them reach for a live LM Studio."""
    return information, "stored uncompressed: test"


def _load_qdrant_mcp_server(stub_compression=True):
    """
    Loads a FRESH copy of the module per call (its own MCPServer instance, its
    own registered tools) so tests that register an extra ad-hoc tool, or
    replace a module attribute, can't leak that into other tests. With
    `stub_compression` (the default) compact_store's compression step is
    replaced by a no-op; the compression tests load the module with it off.
    """
    spec = importlib.util.spec_from_file_location(
        "qdrant_mcp_server", REPO_ROOT / "tools" / "qdrant_mcp_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if stub_compression:
        mod._compress_for_storage = _identity_compression
    return mod


class SanitizeProjectCollectionName(unittest.TestCase):
    """_sanitize_project maps a project string to a stable, valid Qdrant
    collection-name segment. It deliberately appends NO hash suffix (issue #45's
    hash was dropped so the name stays readable and one project string always
    maps to the same plain collection name), so punctuation variants collapse
    together.
    """

    def test_punctuation_variants_collapse_to_the_same_name(self):
        mod = _load_qdrant_mcp_server()
        variants = ["my.project", "my project", "my@project", "my-project"]
        sanitized = {mod._sanitize_project(v) for v in variants}
        self.assertEqual(sanitized, {"my-project"})

    def test_deterministic_across_calls(self):
        # compact_store and compact_find each call _sanitize_project
        # independently and must agree on the same collection name for the
        # same project, or /my-resume would never find what /my-compact
        # just stored.
        mod = _load_qdrant_mcp_server()
        first = mod._sanitize_project("Acme.Support.TicketsApi")
        second = mod._sanitize_project("Acme.Support.TicketsApi")
        self.assertEqual(first, second)

    def test_output_contains_only_valid_qdrant_collection_characters(self):
        mod = _load_qdrant_mcp_server()
        for project in ["my.project", "my project", "my@project", "plain-name", "Mixed_Case-123"]:
            result = mod._sanitize_project(project)
            self.assertRegex(result, r"^[a-zA-Z0-9_-]+$")

    def test_plain_alphanumeric_name_is_unchanged_with_no_suffix(self):
        mod = _load_qdrant_mcp_server()
        self.assertEqual(mod._sanitize_project("myproject"), "myproject")
        self.assertEqual(mod._sanitize_project("claude-runway"), "claude-runway")

    def test_casing_variants_collapse_to_the_same_collection(self):
        # Regression coverage for issue #37: casing variants of the exact
        # same project string must sanitize IDENTICALLY, or a casing drift
        # between /my-compact and /my-resume points at a nonexistent
        # collection.
        mod = _load_qdrant_mcp_server()
        variants = ["MyProject", "myproject", "MYPROJECT", "mYpRoJeCt"]
        sanitized = {mod._sanitize_project(v) for v in variants}
        self.assertEqual(len(sanitized), 1)


class _FakeEmbeddingProvider:
    """Stands in for qdrant_embeddings.FastEmbedProvider
    -- compact_store/compact_find only ever call get_vector_name(),
    embed_documents(), and embed_query() on it, never anything else."""

    VECTOR_NAME = "fake-vector"
    VECTOR_SIZE = 3

    def __init__(self, model_name):
        self.model_name = model_name

    def get_vector_name(self):
        return self.VECTOR_NAME

    def get_vector_size(self):
        return self.VECTOR_SIZE

    async def embed_documents(self, documents):
        return [[0.1, 0.2, 0.3] for _ in documents]

    async def embed_query(self, query):
        return [0.1, 0.2, 0.3]


class _FakeQueryResult:
    def __init__(self, points):
        self.points = points


def _make_fake_qdrant_client_class():
    """
    Returns a FRESH _FakeQdrantClient class per call, closing over its own
    private `shared_collections` dict. compact_store/compact_find each
    construct their own `QdrantClient(...)` instance internally (there's no
    way to inject a pre-built instance), so a plain per-instance dict would
    make every call see an empty store -- exactly like a real Qdrant client
    reconnecting to a real, persistent server, ALL instances built from the
    SAME class here need to share one backing store. Returning a new class
    (not a shared module-level singleton) is what keeps different test
    methods from leaking state into each other.

    Real qdrant_models.PointStruct/Filter/FieldCondition/MatchValue objects
    are used directly by callers (plain pydantic models, no network) -- only
    the client connection itself needs faking here.
    """
    shared_collections = {}  # name -> {point_id: PointStruct}

    class _FakeQdrantClient:
        def __init__(self, url=None, api_key=None):
            self.url = url
            self.api_key = api_key
            self.collections = shared_collections

        def collection_exists(self, collection_name):
            return collection_name in self.collections

        def create_collection(self, collection_name, vectors_config):
            self.collections.setdefault(collection_name, {})

        def upsert(self, collection_name, points):
            col = self.collections.setdefault(collection_name, {})
            for p in points:
                # Real Qdrant upsert semantics: a point sharing an existing
                # id overwrites it in place rather than adding a second one
                # -- this is exactly the behavior issue #36's fix relies on.
                col[p.id] = p

        def _matching(self, collection_name, qdrant_filter):
            points = list(self.collections.get(collection_name, {}).values())
            if qdrant_filter is not None:
                for cond in qdrant_filter.must:
                    # Real qdrant_models.MatchValue has a scalar `.value`;
                    # MatchAny (used by compact_find's per-collection
                    # exact-casing filter, round 5) has a list `.any`
                    # instead -- support both, since real Filter objects
                    # are used directly by this fake's callers.
                    if hasattr(cond.match, "any"):
                        points = [p for p in points if (p.payload or {}).get(cond.key) in cond.match.any]
                    else:
                        points = [p for p in points if (p.payload or {}).get(cond.key) == cond.match.value]
            return points

        def scroll(self, collection_name, scroll_filter=None, limit=100, offset=None, with_payload=True):
            points = self._matching(collection_name, scroll_filter)
            # Single-page fake (tests here never exceed one page) -- (batch, next_offset).
            return points, None

        def query_points(self, collection_name, query, using, query_filter=None, limit=10, with_payload=True):
            points = self._matching(collection_name, query_filter)
            return _FakeQueryResult(points[:limit])

        def delete(self, collection_name, points_selector):
            col = self.collections.get(collection_name, {})
            # PointIdsList has a `.points` attribute that is the list of ids.
            for pid in points_selector.points:
                col.pop(pid, None)

        def get_collections(self):
            # Stands in for qdrant_client's real CollectionsResponse -- only
            # the `.collections[i].name` shape is used by
            # _sibling_compact_collections, so that's all this fakes.
            names = list(self.collections.keys())
            return _FakeCollectionsResponse(names)

    return _FakeQdrantClient


class _FakeCollectionDescription:
    def __init__(self, name):
        self.name = name


class _FakeCollectionsResponse:
    def __init__(self, names):
        self.collections = [_FakeCollectionDescription(n) for n in names]


def _run(coro):
    import asyncio
    return asyncio.run(coro)


class CompactPointIdIsDeterministic(unittest.TestCase):
    """Regression coverage for issue #36's core fix: the point id must be a
    pure function of (project, label, date) so a retried compact_store call
    lands on the SAME id -- otherwise there's nothing to dedupe against."""

    def test_same_inputs_produce_same_id(self):
        mod = _load_qdrant_mcp_server()
        first = mod._compact_point_id("proj", "label", "2026-08-27")
        second = mod._compact_point_id("proj", "label", "2026-08-27")
        self.assertEqual(first, second)

    def test_different_inputs_produce_different_ids(self):
        mod = _load_qdrant_mcp_server()
        base = mod._compact_point_id("proj", "label", "2026-08-27")
        variants = [
            mod._compact_point_id("other-proj", "label", "2026-08-27"),
            mod._compact_point_id("proj", "other-label", "2026-08-27"),
            mod._compact_point_id("proj", "label", "2026-08-28"),
        ]
        for v in variants:
            self.assertNotEqual(base, v)
        # Field-boundary collision check: "a" + "b|c" must not equal
        # "a|b" + "c" -- the null-byte separator (not "|") is what
        # guarantees this.
        a = mod._compact_point_id("a", "b|c", "2026-08-27")
        b = mod._compact_point_id("a|b", "c", "2026-08-27")
        self.assertNotEqual(a, b)


class CompactStoreIsIdempotent(unittest.TestCase):
    """Regression coverage for issue #36: a retried compact_store call with
    identical project/label/date must overwrite the earlier point (same id)
    instead of inserting a second, indistinguishable one."""

    def _patched_module(self):
        mod = _load_qdrant_mcp_server()
        client_cls = _make_fake_qdrant_client_class()
        mod.QdrantClient = client_cls
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod, client_cls

    def test_retry_overwrites_instead_of_duplicating(self):
        mod, client_cls = self._patched_module()

        _run(mod.compact_store(
            information="first attempt content",
            project="my-project",
            label="my-label",
            date="2026-08-27",
        ))
        # Same three identity fields, simulating an external retry --
        # /my-compact's own SKILL.md only calls this once per run, so this
        # models a transport-level retry from outside it, not a code bug in
        # the skill itself.
        _run(mod.compact_store(
            information="retried content (should overwrite, not duplicate)",
            project="my-project",
            label="my-label",
            date="2026-08-27",
        ))

        # A fresh instance of the same class sees the same shared store --
        # this mirrors how compact_store's own internal QdrantClient()
        # constructions all talk to the same real server in production.
        col = next(iter(client_cls().collections.values()))
        # Exactly one point exists -- the retry overwrote it, it didn't add
        # a second one.
        self.assertEqual(len(col), 1)
        [point] = list(col.values())
        self.assertEqual(point.payload["information"], "retried content (should overwrite, not duplicate)")

    def test_payload_carries_compact_source_marker(self):
        # Issue #282: the indexer's reset/delete guard excludes on
        # metadata.source, so compact points must carry the shared constant.
        mod, client_cls = self._patched_module()
        _run(mod.compact_store(information="x", project="my-project", label="l", date="2026-08-27"))
        col = next(iter(client_cls().collections.values()))
        [point] = list(col.values())
        self.assertEqual(point.payload["metadata"]["source"], mod.COMPACT_SOURCE)
        # Writer and guard must agree: the exclusion filter names this value.
        import memory_bank_lib
        excluded = {c.match.value for c in memory_bank_lib.memory_bank_exclusion_filter().must_not}
        self.assertIn(point.payload["metadata"]["source"], excluded)

    def test_different_label_does_not_collide(self):
        mod, client_cls = self._patched_module()

        _run(mod.compact_store(information="a", project="my-project", label="label-a", date="2026-08-27"))
        _run(mod.compact_store(information="b", project="my-project", label="label-b", date="2026-08-27"))

        col = next(iter(client_cls().collections.values()))
        self.assertEqual(len(col), 2)

    def test_returned_message_includes_the_deterministic_id(self):
        mod, _client_cls = self._patched_module()
        expected_id = mod._compact_point_id("my-project", "my-label", "2026-08-27")
        result = _run(mod.compact_store(
            information="content", project="my-project", label="my-label", date="2026-08-27",
        ))
        self.assertIn(expected_id, result)


class CompactStoreServerSideLabelDerivation(unittest.TestCase):
    """Issue #72: compact_store derives a label server-side when the caller
    omits it or passes blank, reducing label-quality variance across models."""

    def _patched_module(self):
        mod = _load_qdrant_mcp_server()
        client_cls = _make_fake_qdrant_client_class()
        mod.QdrantClient = client_cls
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod, client_cls

    def _stored_label(self, client_cls):
        col = next(iter(client_cls().collections.values()))
        [point] = list(col.values())
        return point.payload["label"]

    STRUCTURED_INFO = (
        "PROJECT: my-project\nDATE: 2026-09-09\nLABEL:\n\n"
        "## What we were working on\n"
        "Migrating the cache layer to Redis. Other context follows.\n\n"
        "## Key decisions made\n- Chose Redis over Memcached."
    )

    def test_blank_label_is_derived_from_working_on_section(self):
        mod, client_cls = self._patched_module()
        _run(mod.compact_store(
            information=self.STRUCTURED_INFO,
            project="my-project",
            label="",
            date="2026-09-09",
        ))
        label = self._stored_label(client_cls)
        # Must contain the key phrase from "What we were working on"
        self.assertIn("Migrating", label)
        # Must NOT be blank
        self.assertNotEqual(label.strip(), "")

    def test_omitted_label_defaults_to_blank_and_is_derived(self):
        """label has a default of "" so callers can omit it entirely."""
        mod, client_cls = self._patched_module()
        _run(mod.compact_store(
            information=self.STRUCTURED_INFO,
            project="my-project",
            date="2026-09-09",
        ))
        label = self._stored_label(client_cls)
        self.assertIn("Migrating", label)

    def test_explicit_label_is_not_overridden(self):
        """When the caller passes an explicit label it must be used verbatim."""
        mod, client_cls = self._patched_module()
        _run(mod.compact_store(
            information=self.STRUCTURED_INFO,
            project="my-project",
            label="explicit caller label",
            date="2026-09-09",
        ))
        self.assertEqual(self._stored_label(client_cls), "explicit caller label")

    def test_unstructured_info_with_blank_label_uses_unlabeled_fallback(self):
        """When derive_compact_label returns '' (no parseable content), the
        server falls back to '(unlabeled)' so the /my-resume picker always
        shows something."""
        mod, client_cls = self._patched_module()
        _run(mod.compact_store(
            information="",
            project="my-project",
            label="",
            date="2026-09-09",
        ))
        self.assertEqual(self._stored_label(client_cls), "(unlabeled)")

    def test_derived_label_appears_in_returned_message(self):
        """The confirmation string must show the DERIVED label, not blank."""
        mod, _client_cls = self._patched_module()
        result = _run(mod.compact_store(
            information=self.STRUCTURED_INFO,
            project="my-project",
            label="",
            date="2026-09-09",
        ))
        self.assertIn("Migrating", result)


class CompactFindSurfacesPointId(unittest.TestCase):
    """Regression coverage for issue #36's second gap: compact_find's
    output must include enough of each point's own id for /my-resume's
    selection UI to disambiguate entries that otherwise render identically
    (same date and label)."""

    def _patched_module(self):
        mod = _load_qdrant_mcp_server()
        mod.QdrantClient = _make_fake_qdrant_client_class()
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod

    def test_scroll_path_includes_id_suffix_per_entry(self):
        mod = self._patched_module()
        _run(mod.compact_store(information="alpha", project="proj", label="dup-label", date="2026-08-20"))
        _run(mod.compact_store(information="beta", project="proj", label="dup-label", date="2026-08-21"))

        result = _run(mod.compact_find(project="proj"))
        self.assertEqual(result.count("(id: "), 2)
        # The two ids must actually differ -- otherwise this wouldn't
        # disambiguate anything.
        id_a = mod._compact_point_id("proj", "dup-label", "2026-08-20").replace("-", "")[:8]
        id_b = mod._compact_point_id("proj", "dup-label", "2026-08-21").replace("-", "")[:8]
        self.assertNotEqual(id_a, id_b)
        self.assertIn(f"(id: {id_a})", result)
        self.assertIn(f"(id: {id_b})", result)

    def test_query_path_includes_id_suffix_per_entry(self):
        mod = self._patched_module()
        _run(mod.compact_store(information="alpha content", project="proj", label="l1", date="2026-08-20"))

        result = _run(mod.compact_find(project="proj", query="alpha"))
        self.assertIn("(id: ", result)


class CompactFindIsCaseInsensitiveToProject(unittest.TestCase):
    """Regression coverage for issue #37: compact_find must find a compact
    stored under a differently-cased project string -- casing is the one
    kind of project-name drift the fix closes directly (rather than via
    the sibling-candidate fallback), since /my-compact and /my-resume can
    legitimately see the working directory name with different casing
    (different shell, different mount, etc.)."""

    def _patched_module(self):
        mod = _load_qdrant_mcp_server()
        mod.QdrantClient = _make_fake_qdrant_client_class()
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod

    def test_find_locates_entry_stored_under_different_casing(self):
        mod = self._patched_module()
        _run(mod.compact_store(information="alpha", project="MyProject", label="l1", date="2026-08-20"))

        result = _run(mod.compact_find(project="myproject"))
        self.assertIn("Found 1 compact(s)", result)
        self.assertIn("alpha", result)

    def test_find_with_query_also_locates_entry_stored_under_different_casing(self):
        mod = self._patched_module()
        _run(mod.compact_store(information="alpha content", project="MyProject", label="l1", date="2026-08-20"))

        result = _run(mod.compact_find(project="MYPROJECT", query="alpha"))
        self.assertIn("alpha content", result)


class CompactFindSuggestsSiblingProjects(unittest.TestCase):
    """Regression coverage for issue #37's sibling-candidate fallback: when
    a project genuinely has nothing (not just a casing mismatch, which is
    handled separately -- a truly different project name, e.g. after a
    working-directory rename), compact_find should surface OTHER projects
    that do have saved compacts instead of just failing flatly."""

    def _patched_module(self):
        mod = _load_qdrant_mcp_server()
        mod.QdrantClient = _make_fake_qdrant_client_class()
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod

    def test_collection_not_found_lists_sibling_candidates(self):
        mod = self._patched_module()
        # Seed a sibling project's collection so it exists when the
        # renamed/mismatched project's own lookup below finds nothing.
        _run(mod.compact_store(information="old content", project="old-project-name", label="last session", date="2026-08-20"))

        result = _run(mod.compact_find(project="new-project-name"))
        self.assertIn("No compacts collection found", result)
        self.assertIn("Did you mean one of these other projects", result)
        self.assertIn("old-project-name", result)
        self.assertIn("last session", result)

    def test_zero_results_within_existing_collection_lists_sibling_candidates(self):
        mod = self._patched_module()
        # Two distinct projects, each with their own collection.
        _run(mod.compact_store(information="old content", project="old-project-name", label="last session", date="2026-08-20"))
        _run(mod.compact_store(information="other content", project="another-project", label="other session", date="2026-08-19"))

        # Query "another-project"'s own (real, existing) collection
        # explicitly, but with a project value that matches nothing in it
        # -- forcing the zero-results path rather than collection-not-found.
        another_col = f"{mod.COMPACT_COLLECTION}-{mod._sanitize_project('another-project')}"
        result = _run(mod.compact_find(project="totally-different-project", collection=another_col))
        self.assertIn("No compacts found", result)
        self.assertIn("Did you mean one of these other projects", result)
        # "another-project" (the collection actually queried) must NOT be
        # offered as a candidate for itself -- only the OTHER sibling.
        self.assertIn("old-project-name", result)

    def test_no_siblings_means_no_candidate_section(self):
        mod = self._patched_module()
        result = _run(mod.compact_find(project="never-compacted-project"))
        self.assertIn("No compacts collection found", result)
        self.assertNotIn("Did you mean", result)


class CollectionsForProjectFindsEveryCasingVariant(unittest.TestCase):
    """Regression coverage for TWO real bugs flagged in PR review on #123:

    1. (Round 1's own bug) Unconditionally case-folding _sanitize_project's
       output would silently orphan every ALREADY-EXISTING project name
       containing an uppercase letter, since it derives a DIFFERENT
       collection name for the exact same literal project string that was
       always used before.
    2. (Round 1's fix was still incomplete) Picking exactly ONE collection
       per call (a "legacy vs. canonical" choice) still fragmented history
       across a genuine casing drift: a pre-existing "MyProject" collection
       plus a later call passing "myproject" landed on a brand new
       canonical collection instead of the existing one, after which
       lowercase and uppercase reads showed two disjoint histories.

    _collections_for_project fixes both by scanning EVERY existing
    '<COMPACT_COLLECTION>-*' collection and matching by each one's own
    stored `project` payload value, case-insensitively, rather than
    re-deriving a name algorithmically."""

    def _patched_module(self):
        mod = _load_qdrant_mcp_server()
        mod.QdrantClient = _make_fake_qdrant_client_class()
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod

    def test_finds_preexisting_collection_under_original_literal_casing(self):
        mod = self._patched_module()
        _run(mod.compact_store(information="content", project="Acme.Support.TicketsApi", label="l1", date="2026-08-27"))

        matches = mod._collections_for_project(mod.QdrantClient(), "Acme.Support.TicketsApi")
        self.assertEqual(len(matches), 1)

    def test_finds_preexisting_collection_under_a_DIFFERENT_casing(self):
        mod = self._patched_module()
        _run(mod.compact_store(information="content", project="MyProject", label="l1", date="2026-08-27"))

        # A later call passing a totally different casing must still find
        # the SAME collection -- this is exactly the round-2 fragmentation
        # bug: a name-based resolver misses this, a payload-based scan
        # doesn't.
        matches = mod._collections_for_project(mod.QdrantClient(), "myproject")
        self.assertEqual(len(matches), 1)

    def test_no_matches_for_a_brand_new_project(self):
        mod = self._patched_module()
        matches = mod._collections_for_project(mod.QdrantClient(), "brand-new-project")
        self.assertEqual(matches, [])

    def test_finds_a_collection_under_a_name_the_canonical_algorithm_would_never_derive(self):
        # THE actual round-2 regression case: a collection stored under a
        # name that has nothing to do with _sanitize_project's current
        # algorithm at all (simulating history from before issue #37
        # existed in any form) must still be discovered, purely by reading
        # its own stored `project` payload -- proving this fix doesn't
        # depend on the collection name being algorithmically related to
        # the query in any way, unlike the round-1 fix it replaced.
        mod = self._patched_module()
        client_cls = mod.QdrantClient
        legacy_name = f"{mod.COMPACT_COLLECTION}-some-arbitrary-legacy-name-12345678"
        point = mod.qdrant_models.PointStruct(
            id=str(mod.uuid.uuid4()),
            vector={"fake-vector": [0.1, 0.2, 0.3]},
            payload={"project": "MyProject", "label": "old", "date": "2026-08-01", "information": "legacy content"},
        )
        client_cls().upsert(collection_name=legacy_name, points=[point])

        # Found under the original literal casing...
        self.assertIn(legacy_name, mod._collections_for_project(client_cls(), "MyProject"))
        # ...AND under a completely different casing -- this second
        # assertion is what round 1's name-based resolver could never
        # satisfy, since there's no algorithm linking "myproject" to
        # "some-arbitrary-legacy-name-12345678".
        self.assertIn(legacy_name, mod._collections_for_project(client_cls(), "myproject"))

    def test_compact_store_converges_a_differently_named_legacy_collection(self):
        mod = self._patched_module()
        client_cls = mod.QdrantClient
        legacy_name = f"{mod.COMPACT_COLLECTION}-some-arbitrary-legacy-name-12345678"
        point = mod.qdrant_models.PointStruct(
            id=str(mod.uuid.uuid4()), vector={"fake-vector": [0.1, 0.2, 0.3]},
            payload={"project": "MyProject", "label": "old", "date": "2026-08-01", "information": "legacy content"},
        )
        client_cls().upsert(collection_name=legacy_name, points=[point])

        # A NEW /my-compact call, in a different casing, must land in the
        # EXISTING legacy-named collection rather than creating a fresh
        # canonical one alongside it.
        _run(mod.compact_store(information="new content", project="myproject", label="l2", date="2026-08-27"))

        self.assertEqual(len(client_cls().collections[legacy_name]), 2)
        canonical_col = f"{mod.COMPACT_COLLECTION}-{mod._sanitize_project('myproject')}"
        self.assertNotIn(canonical_col, client_cls().collections)

    def test_compact_find_aggregates_a_differently_named_legacy_collection(self):
        mod = self._patched_module()
        client_cls = mod.QdrantClient
        legacy_name = f"{mod.COMPACT_COLLECTION}-some-arbitrary-legacy-name-12345678"
        legacy_point = mod.qdrant_models.PointStruct(
            id=str(mod.uuid.uuid4()), vector={"fake-vector": [0.1, 0.2, 0.3]},
            payload={"project": "MyProject", "label": "old", "date": "2026-08-01", "information": "legacy content"},
        )
        client_cls().upsert(collection_name=legacy_name, points=[legacy_point])
        _run(mod.compact_store(information="new content", project="myproject", label="l2", date="2026-08-27"))

        result = _run(mod.compact_find(project="MyProject"))
        self.assertIn("legacy content", result)
        self.assertIn("new content", result)

    def test_compact_store_reuses_existing_collection_regardless_of_casing(self):
        mod = self._patched_module()
        client_cls = mod.QdrantClient

        _run(mod.compact_store(information="first", project="MyProject", label="l1", date="2026-08-20"))
        # A later call under a DIFFERENT casing must land in the SAME
        # collection, not fragment into a second one.
        _run(mod.compact_store(information="second", project="myproject", label="l2", date="2026-08-27"))

        # Exactly one collection exists for this project no matter how many
        # different casings were used to store into it.
        compact_cols = [name for name in client_cls().collections if name.startswith(mod.COMPACT_COLLECTION)]
        self.assertEqual(len(compact_cols), 1)
        [col_name] = compact_cols
        self.assertEqual(len(client_cls().collections[col_name]), 2)

    def test_compact_find_sees_entries_stored_under_every_casing(self):
        mod = self._patched_module()

        _run(mod.compact_store(information="first", project="MyProject", label="l1", date="2026-08-20"))
        _run(mod.compact_store(information="second", project="myproject", label="l2", date="2026-08-27"))

        # Both /my-compact calls' entries must be visible regardless of
        # which casing /my-resume happens to query with.
        result_upper = _run(mod.compact_find(project="MyProject"))
        self.assertIn("first", result_upper)
        self.assertIn("second", result_upper)

        result_lower = _run(mod.compact_find(project="myproject"))
        self.assertIn("first", result_lower)
        self.assertIn("second", result_lower)

    def test_explicit_collection_param_bypasses_case_insensitive_resolution(self):
        mod = self._patched_module()
        _run(mod.compact_store(
            information="in shared collection", project="MyProject", label="l1",
            date="2026-08-20", collection="some-shared-collection",
        ))

        # An explicit collection override keeps its EXACT-match project
        # filter -- a different-cased query against the same explicit
        # collection must NOT match (this path is caller-controlled, not
        # part of the case-insensitive default scheme).
        result = _run(mod.compact_find(project="myproject", collection="some-shared-collection"))
        self.assertIn("No compacts found", result)

        result_exact = _run(mod.compact_find(project="MyProject", collection="some-shared-collection"))
        self.assertIn("in shared collection", result_exact)


class MostRecentSiblingEntryPaginatesFully(unittest.TestCase):
    """Regression coverage for a real pagination bug flagged in PR review
    on #123: _most_recent_sibling_entry used to read only a single
    100-point scroll page. scroll's page order is by point id, not by
    date, so a collection with more than 100 points could have its actual
    most-recently-dated entry sitting on a LATER page -- a single-page
    read would silently pick an older entry as "most recent" instead."""

    class _TwoPageClient:
        """Minimal fake exposing exactly the paginated scroll() shape
        _most_recent_sibling_entry depends on: two pages, with the
        genuinely most-recent-dated point placed on the SECOND page."""

        def __init__(self, pages):
            self._pages = pages  # {offset_key: (batch, next_offset)}

        def scroll(self, collection_name, scroll_filter=None, limit=100, offset=None, with_payload=True):
            return self._pages[offset]

    def test_picks_up_most_recent_entry_from_a_later_page(self):
        mod = _load_qdrant_mcp_server()

        class _Point:
            def __init__(self, payload):
                self.payload = payload

        page_1 = ([_Point({"date": "2026-08-01", "label": "old", "project": "proj"})], "page-2")
        page_2 = ([_Point({"date": "2026-08-27", "label": "newest", "project": "proj"})], None)
        client = self._TwoPageClient({None: page_1, "page-2": page_2})

        result = mod._most_recent_sibling_entry(client, "some-collection")
        self.assertIsNotNone(result)
        date, label, project = result
        self.assertEqual(date, "2026-08-27")
        self.assertEqual(label, "newest")


class CompactFindDoesNotLeakOtherProjectsFromAnUnprovenCollection(unittest.TestCase):
    """Regression coverage for a real cross-project leakage bug flagged in
    PR review on #123 (round 3): _collections_for_project only proves a
    candidate collection contains AT LEAST ONE matching point -- finding
    one match proves containment, not exclusivity, so a collection that
    ACTUALLY holds multiple different projects' entries (an explicitly
    shared collection someone created via the `collection` override, or
    in principle a hash collision) can still get included in the
    aggregated result. Dropping the per-entry Qdrant filter for that path
    (round 2's own change) would then return every OTHER project's
    entries in that same collection too, not just the one that matched.
    compact_find must still isolate by project at the per-entry level for
    any collection it didn't derive by construction."""

    def _patched_module(self):
        mod = _load_qdrant_mcp_server()
        mod.QdrantClient = _make_fake_qdrant_client_class()
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod

    def test_only_the_matching_projects_own_entries_are_returned(self):
        mod = self._patched_module()
        client_cls = mod.QdrantClient
        # A collection that LOOKS like a sibling (matches the
        # '<COMPACT_COLLECTION>-*' prefix _collections_for_project scans)
        # but actually holds TWO different projects' entries -- exactly
        # what an explicitly shared `collection` override could produce.
        shared_name = f"{mod.COMPACT_COLLECTION}-a-shared-collection-abcdef01"
        target_point = mod.qdrant_models.PointStruct(
            id=str(mod.uuid.uuid4()), vector={"fake-vector": [0.1, 0.2, 0.3]},
            payload={"project": "target-project", "label": "l1", "date": "2026-08-27", "information": "target content"},
        )
        other_point = mod.qdrant_models.PointStruct(
            id=str(mod.uuid.uuid4()), vector={"fake-vector": [0.1, 0.2, 0.3]},
            payload={"project": "other-project", "label": "l2", "date": "2026-08-20", "information": "other content"},
        )
        # _collections_for_project's _collection_contains_project check
        # will find target_point's matching payload and correctly include
        # this collection -- the whole point of THIS test is that the
        # OTHER point (a genuinely different project sharing the same
        # collection) must still be excluded from the RESULT despite the
        # collection itself being a legitimate match.
        client_cls().upsert(collection_name=shared_name, points=[other_point, target_point])

        result = _run(mod.compact_find(project="target-project"))
        self.assertIn("target content", result)
        self.assertNotIn("other content", result)

    def test_query_path_also_excludes_other_projects_entries(self):
        mod = self._patched_module()
        client_cls = mod.QdrantClient
        shared_name = f"{mod.COMPACT_COLLECTION}-a-shared-collection-abcdef01"
        target_point = mod.qdrant_models.PointStruct(
            id=str(mod.uuid.uuid4()), vector={"fake-vector": [0.1, 0.2, 0.3]},
            payload={"project": "target-project", "label": "l1", "date": "2026-08-27", "information": "target content alpha"},
        )
        other_point = mod.qdrant_models.PointStruct(
            id=str(mod.uuid.uuid4()), vector={"fake-vector": [0.1, 0.2, 0.3]},
            payload={"project": "other-project", "label": "l2", "date": "2026-08-20", "information": "other content alpha"},
        )
        client_cls().upsert(collection_name=shared_name, points=[other_point, target_point])

        result = _run(mod.compact_find(project="target-project", query="alpha"))
        self.assertIn("target content alpha", result)
        self.assertNotIn("other content alpha", result)


class AuthoritativeResolutionPropagatesFailuresInsteadOfHidingThem(unittest.TestCase):
    """Regression coverage for a real error-handling bug flagged in PR
    review on #123 (round 3): _collections_for_project (authoritative
    resolution for compact_store/compact_find) used to swallow a failed
    get_collections() call and proceed as if no other collections
    existed -- which could make compact_store create a second,
    fragmenting collection right next to a real match it just couldn't
    see, or make compact_find falsely report a project as
    never-compacted. It must now propagate such failures instead. The
    ONE place that still degrades gracefully on a lookup failure is
    _candidate_hint, a pure best-effort DISPLAY hint layered on top of an
    already-decided result."""

    class _BrokenGetCollectionsClient:
        def collection_exists(self, collection_name):
            return False

        def get_collections(self):
            raise RuntimeError("simulated Qdrant outage")

    def test_collections_for_project_propagates_get_collections_failure(self):
        mod = _load_qdrant_mcp_server()
        with self.assertRaises(RuntimeError):
            mod._collections_for_project(self._BrokenGetCollectionsClient(), "some-project")

    def test_candidate_hint_degrades_gracefully_on_the_same_failure(self):
        mod = _load_qdrant_mcp_server()
        # _candidate_hint's own _sibling_compact_collections call hits the
        # SAME get_collections() failure, but this path is a best-effort
        # hint, not authoritative resolution -- it must return "" rather
        # than raise.
        result = mod._candidate_hint(self._BrokenGetCollectionsClient(), [])
        self.assertEqual(result, "")


class CollectionMatchingScansEveryPointNotJustTheNewest(unittest.TestCase):
    """Regression coverage for a real omission bug flagged in PR review on
    #123 (round 4): _collections_for_project used to decide whether a
    sibling collection belonged to a project by checking only that
    collection's MOST RECENT point. In a genuinely shared collection where
    a DIFFERENT project's entry happens to have a LATER date, this project's
    own (older) real history was silently excluded from the match entirely
    -- compact_find would omit it, and compact_store could create a
    second, fragmenting canonical collection right alongside a shared one
    that actually already held this project's data."""

    def _patched_module(self):
        mod = _load_qdrant_mcp_server()
        mod.QdrantClient = _make_fake_qdrant_client_class()
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod

    def test_older_matching_point_is_still_found_behind_a_newer_other_project_point(self):
        mod = self._patched_module()
        client_cls = mod.QdrantClient
        shared_name = f"{mod.COMPACT_COLLECTION}-a-shared-collection-abcdef01"
        # target-project's entry is OLDER...
        target_point = mod.qdrant_models.PointStruct(
            id=str(mod.uuid.uuid4()), vector={"fake-vector": [0.1, 0.2, 0.3]},
            payload={"project": "target-project", "label": "l1", "date": "2026-08-01", "information": "target content"},
        )
        # ...but other-project's entry is NEWER, so a "most recent point"
        # check would see only this one and wrongly conclude the
        # collection doesn't belong to target-project at all.
        other_point = mod.qdrant_models.PointStruct(
            id=str(mod.uuid.uuid4()), vector={"fake-vector": [0.1, 0.2, 0.3]},
            payload={"project": "other-project", "label": "l2", "date": "2026-08-27", "information": "other content"},
        )
        client_cls().upsert(collection_name=shared_name, points=[target_point, other_point])

        matches = mod._collections_for_project(client_cls(), "target-project")
        self.assertIn(shared_name, matches)

    def test_compact_store_finds_the_shared_collection_instead_of_fragmenting(self):
        mod = self._patched_module()
        client_cls = mod.QdrantClient
        shared_name = f"{mod.COMPACT_COLLECTION}-a-shared-collection-abcdef01"
        target_point = mod.qdrant_models.PointStruct(
            id=str(mod.uuid.uuid4()), vector={"fake-vector": [0.1, 0.2, 0.3]},
            payload={"project": "target-project", "label": "l1", "date": "2026-08-01", "information": "target content"},
        )
        other_point = mod.qdrant_models.PointStruct(
            id=str(mod.uuid.uuid4()), vector={"fake-vector": [0.1, 0.2, 0.3]},
            payload={"project": "other-project", "label": "l2", "date": "2026-08-27", "information": "other content"},
        )
        client_cls().upsert(collection_name=shared_name, points=[target_point, other_point])

        _run(mod.compact_store(information="new content", project="target-project", label="l3", date="2026-08-28"))

        # The new entry must have landed in the EXISTING shared collection,
        # not a freshly created (fragmenting) canonical one.
        self.assertEqual(len(client_cls().collections[shared_name]), 3)
        canonical_col = f"{mod.COMPACT_COLLECTION}-{mod._sanitize_project('target-project')}"
        self.assertNotIn(canonical_col, client_cls().collections)


class CompactFindQueryPathFiltersBeforeQdrantTruncates(unittest.TestCase):
    """Regression coverage for a truncation-before-filter bug flagged in PR
    review on #123 (rounds 4 AND 5): the semantic query path used to ask
    Qdrant for exactly `limit` results per collection and only THEN apply
    the case-insensitive per-entry project filter client-side. A shared
    collection holding more than `limit` entries that rank ahead of a
    project's own entries could fill the whole truncated page with OTHER
    projects' points, which the filter then removed entirely -- silently
    reporting no matches even though the target project's history exists
    past the truncation point.

    Over-fetching a larger number of results only moves the threshold, since
    ANY fixed cap can be exceeded. The fix builds an EXACT, Qdrant-SIDE
    MatchAny filter (from every literal casing variant stored in that
    collection -- see _collection_project_values), so Qdrant only ever ranks
    and truncates ALREADY-isolated points. This test inserts far more
    other-project points than any plausible over-fetch buffer to prove that."""

    def _patched_module(self):
        mod = _load_qdrant_mcp_server()
        mod.QdrantClient = _make_fake_qdrant_client_class()
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod

    def test_target_projects_entry_still_found_behind_many_higher_ranked_others(self):
        mod = self._patched_module()
        client_cls = mod.QdrantClient
        shared_name = f"{mod.COMPACT_COLLECTION}-a-shared-collection-abcdef01"
        # Insert far more other-project points than any fixed over-fetch
        # cap would cover -- the fake's query_points returns points in
        # insertion order, standing in for "rank", so these go in FIRST.
        points = []
        for i in range(250):
            points.append(mod.qdrant_models.PointStruct(
                id=str(mod.uuid.uuid4()), vector={"fake-vector": [0.1, 0.2, 0.3]},
                payload={"project": "other-project", "label": f"o{i}", "date": "2026-08-01", "information": f"other content {i}"},
            ))
        target_point = mod.qdrant_models.PointStruct(
            id=str(mod.uuid.uuid4()), vector={"fake-vector": [0.1, 0.2, 0.3]},
            payload={"project": "target-project", "label": "l1", "date": "2026-08-27", "information": "target content alpha"},
        )
        points.append(target_point)
        client_cls().upsert(collection_name=shared_name, points=points)

        result = _run(mod.compact_find(project="target-project", query="alpha"))
        self.assertIn("target content alpha", result)


class CandidateHintSkipsPastEmptySiblingsToFindRealOnes(unittest.TestCase):
    """Regression coverage for a real bug flagged in PR review on #123 (round
    6): _sibling_compact_collections used to cap the RAW list of sibling
    collection names to 3 BEFORE _candidate_hint ever got a chance to probe
    and skip the ones that turn out empty. compact_store creates a collection
    before it has any points, so an empty sibling is a realistic state -- if
    the first 3 siblings (in whatever order get_collections() happens to
    return them) were all empty, a 4th sibling with real saved compacts was
    never even inspected, and the recovery path wrongly reported zero
    candidates. The fix moves the cap into _candidate_hint itself, applied to
    hints actually collected (i.e. AFTER filtering), not to the raw name
    list."""

    def _patched_module(self):
        mod = _load_qdrant_mcp_server()
        mod.QdrantClient = _make_fake_qdrant_client_class()
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod

    def test_a_real_sibling_behind_three_empty_ones_is_still_surfaced(self):
        mod = self._patched_module()
        client = mod.QdrantClient()
        # Three genuinely empty sibling collections, inserted FIRST so the
        # fake's insertion-ordered get_collections() returns them ahead of
        # the real one below -- reproducing "the first 3 siblings happen to
        # be empty" exactly.
        for i in range(3):
            client.collections[f"{mod.COMPACT_COLLECTION}-empty-sibling-{i}"] = {}
        # A fourth sibling that DOES have a real saved compact.
        _run(mod.compact_store(information="real content", project="real-sibling-project", label="l1", date="2026-08-27"))

        result = _run(mod.compact_find(project="never-compacted-project"))
        self.assertIn("Did you mean one of these other projects", result)
        self.assertIn("real-sibling-project", result)


class CompactFindFiltersTheCanonicalCollectionToo(unittest.TestCase):
    """Regression coverage for a real truncation-before-filter bug flagged in
    PR review on #123 (round 6): the semantic query path's exact MatchAny
    filter (round 5's fix) was applied to every resolved collection EXCEPT
    the canonical one, which was treated as "trusted by construction" and
    queried with no filter at all. But `collection` is a caller-controlled
    override on compact_store's WRITE side too (see compact_store's own
    docstring) -- nothing actually prevents another project's points from
    being written directly into a project's exact canonical collection name.
    If enough of those foreign points rank ahead of the real ones, the
    (previously unfiltered) canonical-collection query could return an
    all-foreign page, which the post-fetch per-entry filter then strips to
    nothing -- silently hiding real matches that still exist further down
    that same collection. The fix applies the same exact MatchAny filter to
    the canonical collection too."""

    def _patched_module(self):
        mod = _load_qdrant_mcp_server()
        mod.QdrantClient = _make_fake_qdrant_client_class()
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod

    def test_target_projects_entry_in_the_canonical_collection_still_found_behind_many_foreign_points(self):
        mod = self._patched_module()
        canonical_col = f"{mod.COMPACT_COLLECTION}-{mod._sanitize_project('target-project')}"
        # Write another project's points directly into target-project's own
        # canonical collection name, via the caller-controlled `collection`
        # override -- the exact vector round 6 flagged as still open.
        for i in range(250):
            _run(mod.compact_store(
                information=f"foreign content {i}", project="other-project",
                label=f"o{i}", date="2026-08-01", collection=canonical_col,
            ))
        # The real project's own entry, stored normally -- lands in the
        # same canonical collection by construction, since it now exists.
        _run(mod.compact_store(information="target content alpha", project="target-project", label="l1", date="2026-08-27"))

        result = _run(mod.compact_find(project="target-project", query="alpha"))
        self.assertIn("target content alpha", result)


class CompactPruneTests(unittest.TestCase):
    """Tests for compact_prune (issue #70): a retention/cleanup tool for
    stored conversation compacts, with dry_run-by-default safety.

    Uses the same _FakeQdrantClient/_FakeEmbeddingProvider pattern as
    compact_store/compact_find tests -- no real Qdrant or embedding model
    needed.
    """

    def _patched_module(self):
        mod = _load_qdrant_mcp_server()
        mod.QdrantClient = _make_fake_qdrant_client_class()
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod

    # ── argument validation ──────────────────────────────────────────────

    def test_error_when_both_params_given(self):
        mod = self._patched_module()
        result = _run(mod.compact_prune(project="proj", keep_last_n=2, older_than_days=7))
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("exactly one", result)

    def test_error_when_neither_param_given(self):
        mod = self._patched_module()
        result = _run(mod.compact_prune(project="proj"))
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("exactly one", result)

    def test_error_on_negative_keep_last_n(self):
        mod = self._patched_module()
        result = _run(mod.compact_prune(project="proj", keep_last_n=-1))
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("keep_last_n", result)

    def test_error_on_negative_older_than_days(self):
        mod = self._patched_module()
        result = _run(mod.compact_prune(project="proj", older_than_days=-1))
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("older_than_days", result)

    # ── collection-not-found ─────────────────────────────────────────────

    def test_no_collection_returns_helpful_message(self):
        mod = self._patched_module()
        result = _run(mod.compact_prune(project="never-compacted", keep_last_n=1))
        self.assertIn("No compacts collection found", result)

    # ── dry_run (default True) ───────────────────────────────────────────

    def test_dry_run_reports_candidates_without_deleting(self):
        mod = self._patched_module()
        _run(mod.compact_store(information="old", project="proj", label="l1", date="2026-07-01"))
        _run(mod.compact_store(information="new", project="proj", label="l2", date="2026-08-01"))

        result = _run(mod.compact_prune(project="proj", keep_last_n=1))
        # DRY RUN prefix expected
        self.assertIn("[DRY RUN]", result)
        self.assertIn("dry_run=False", result)
        # The older entry is listed as a deletion candidate
        self.assertIn("l1", result)
        # Nothing was actually deleted -- both points still exist
        col_name = next(k for k in mod.QdrantClient().collections if k.startswith(mod.COMPACT_COLLECTION))
        self.assertEqual(len(mod.QdrantClient().collections[col_name]), 2)

    # ── keep_last_n ──────────────────────────────────────────────────────

    def test_keep_last_n_deletes_older_entries_and_keeps_newest(self):
        mod = self._patched_module()
        _run(mod.compact_store(information="old content", project="proj", label="l1", date="2026-07-01"))
        _run(mod.compact_store(information="mid content", project="proj", label="l2", date="2026-07-15"))
        _run(mod.compact_store(information="new content", project="proj", label="l3", date="2026-08-01"))

        result = _run(mod.compact_prune(project="proj", keep_last_n=1, dry_run=False))
        self.assertNotIn("[DRY RUN]", result)
        self.assertIn("Deleted 2 compact(s)", result)

        # Only the newest entry survives
        col_name = next(k for k in mod.QdrantClient().collections if k.startswith(mod.COMPACT_COLLECTION))
        remaining = list(mod.QdrantClient().collections[col_name].values())
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].payload["date"], "2026-08-01")

    def test_keep_last_n_zero_deletes_all(self):
        mod = self._patched_module()
        _run(mod.compact_store(information="a", project="proj", label="l1", date="2026-07-01"))
        _run(mod.compact_store(information="b", project="proj", label="l2", date="2026-08-01"))

        result = _run(mod.compact_prune(project="proj", keep_last_n=0, dry_run=False))
        self.assertIn("Deleted 2 compact(s)", result)

        col_name = next(k for k in mod.QdrantClient().collections if k.startswith(mod.COMPACT_COLLECTION))
        self.assertEqual(len(mod.QdrantClient().collections[col_name]), 0)

    def test_keep_last_n_larger_than_count_deletes_nothing(self):
        mod = self._patched_module()
        _run(mod.compact_store(information="a", project="proj", label="l1", date="2026-07-01"))

        result = _run(mod.compact_prune(project="proj", keep_last_n=100, dry_run=False))
        self.assertIn("Nothing to prune", result)

    # ── older_than_days ──────────────────────────────────────────────────

    def test_older_than_days_deletes_entries_past_cutoff(self):
        import datetime as _dt
        mod = self._patched_module()
        today = _dt.date.today().isoformat()
        old_date = (_dt.date.today() - _dt.timedelta(days=10)).isoformat()
        _run(mod.compact_store(information="today's compact", project="proj", label="today", date=today))
        _run(mod.compact_store(information="old compact", project="proj", label="old", date=old_date))

        result = _run(mod.compact_prune(project="proj", older_than_days=5, dry_run=False))
        self.assertIn("Deleted 1 compact(s)", result)

        col_name = next(k for k in mod.QdrantClient().collections if k.startswith(mod.COMPACT_COLLECTION))
        remaining = list(mod.QdrantClient().collections[col_name].values())
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].payload["date"], today)

    def test_older_than_days_nothing_old_enough(self):
        import datetime as _dt
        mod = self._patched_module()
        today = _dt.date.today().isoformat()
        _run(mod.compact_store(information="today", project="proj", label="l", date=today))

        result = _run(mod.compact_prune(project="proj", older_than_days=1, dry_run=False))
        self.assertIn("Nothing to prune", result)

    # ── case-insensitive project matching ────────────────────────────────

    def test_prune_finds_entries_stored_under_different_casing(self):
        mod = self._patched_module()
        _run(mod.compact_store(information="old", project="MyProject", label="l1", date="2026-07-01"))
        _run(mod.compact_store(information="new", project="MyProject", label="l2", date="2026-08-01"))

        # Query with different casing -- should still find and prune correctly
        result = _run(mod.compact_prune(project="myproject", keep_last_n=1, dry_run=False))
        self.assertIn("Deleted 1 compact(s)", result)

    # ── explicit collection override ─────────────────────────────────────

    def test_explicit_collection_uses_exact_match_filter(self):
        mod = self._patched_module()
        shared_col = "some-shared-collection"
        _run(mod.compact_store(information="proj-a content", project="proj-a", label="l1",
                               date="2026-07-01", collection=shared_col))
        _run(mod.compact_store(information="proj-b content", project="proj-b", label="l2",
                               date="2026-07-02", collection=shared_col))

        # Prune only proj-a's entry from the shared collection; proj-b must survive
        result = _run(mod.compact_prune(
            project="proj-a", keep_last_n=0, dry_run=False, collection=shared_col,
        ))
        self.assertIn("Deleted 1 compact(s)", result)

        # proj-b's entry must still be present
        remaining = list(mod.QdrantClient().collections[shared_col].values())
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].payload["project"], "proj-b")

    # ── result includes actionable info ─────────────────────────────────

    def test_deleted_entry_summary_includes_date_label_and_id(self):
        mod = self._patched_module()
        _run(mod.compact_store(information="old", project="proj", label="my-label", date="2026-07-01"))
        _run(mod.compact_store(information="new", project="proj", label="l2", date="2026-08-01"))

        result = _run(mod.compact_prune(project="proj", keep_last_n=1, dry_run=False))
        self.assertIn("2026-07-01", result)
        self.assertIn("my-label", result)
        self.assertIn("(id: ", result)

    # ── no entries to prune ──────────────────────────────────────────────

    def test_no_entries_at_all_returns_nothing_to_prune(self):
        mod = self._patched_module()
        # Create a collection but do not store anything in it by storing
        # something under a DIFFERENT project that creates its own collection,
        # then query a non-existent project.
        result = _run(mod.compact_prune(project="ghost-project", keep_last_n=1))
        self.assertIn("No compacts collection found", result)

    # ── robustness: malformed dates (Copilot review, round 1) ────────────

    def test_keep_last_n_unparseable_date_does_not_shadow_newer_valid_entry(self):
        """Regression: a malformed date string like "2026-99-99" sorts ahead
        of every valid 2026 date lexicographically, so including it in the
        sort would silently treat it as "newest" and delete a real, valid
        compact instead. Entries with unparseable dates must be preserved
        outside the keep_last_n quota, not treated as newer than valid ones.
        """
        mod = self._patched_module()
        client_cls = mod.QdrantClient
        canonical_col = f"{mod.COMPACT_COLLECTION}-{mod._sanitize_project('proj')}"
        # A point with a malformed date -- direct insert, bypassing the
        # compact_store validator which enforces ISO format.
        bad_point = mod.qdrant_models.PointStruct(
            id=str(mod.uuid.uuid4()), vector={"fake-vector": [0.1, 0.2, 0.3]},
            payload={"project": "proj", "label": "bad-date", "date": "2026-99-99", "information": "bad"},
        )
        client_cls().upsert(collection_name=canonical_col, points=[bad_point])
        _run(mod.compact_store(information="newest valid", project="proj", label="real", date="2026-08-01"))

        # keep_last_n=1 should keep "newest valid" and delete... the bad-date
        # entry is simply preserved outside the quota (not treated as newest),
        # so after pruning we should still have the real one.
        _run(mod.compact_prune(project="proj", keep_last_n=1, dry_run=False))

        remaining = list(mod.QdrantClient().collections[canonical_col].values())
        real_dates = [p.payload.get("date") for p in remaining]
        # The genuinely newest valid compact must survive.
        self.assertIn("2026-08-01", real_dates)

    def test_older_than_days_very_large_value_does_not_crash(self):
        """Regression: older_than_days=1_000_000 raises OverflowError from
        datetime.timedelta instead of returning a tool result (Copilot review,
        round 1). The fix catches OverflowError and returns a safe message.
        """
        mod = self._patched_module()
        _run(mod.compact_store(information="content", project="proj", label="l", date="2026-08-01"))

        result = _run(mod.compact_prune(project="proj", older_than_days=1_000_000, dry_run=False))
        # Must return a string, not raise, and must not claim anything was deleted.
        self.assertIsInstance(result, str)
        self.assertNotIn("Deleted", result)


# ---------------------------------------------------------------------------
# The server's own tool set, qdrant-find / qdrant-store, and compact_store's
# compression step (everything above was moved from the compress server's tests)
# ---------------------------------------------------------------------------


class RegisteredToolSet(unittest.TestCase):
    """Pins the tool names. `qdrant-find` and `qdrant-store` must stay exactly
    as the third-party server served them so `mcp__qdrant__qdrant-find` and every
    permission rule written against it keep working; the compact tools come
    with the server the compact skills now call."""

    def test_the_registered_tool_set_is_the_documented_one(self):
        mod = _load_qdrant_mcp_server()
        names = {info.name for info in mod.mcp._tool_manager.list_tools()}
        self.assertEqual(
            names,
            {"qdrant-find", "qdrant-store", "compact_store", "compact_find", "compact_prune"},
        )

    def test_the_server_is_registered_under_the_name_qdrant(self):
        # The tool prefix Claude Code shows is mcp__<this name>__, and existing
        # permission rules and the compress hook match on `mcp__qdrant__`.
        self.assertEqual(_load_qdrant_mcp_server().mcp.name, "qdrant")

    def test_find_and_store_expose_the_same_arguments_as_the_replaced_server(self):
        import asyncio

        mod = _load_qdrant_mcp_server()
        tools = {t.name: t for t in asyncio.run(mod.mcp.list_tools())}
        self.assertEqual(set(tools["qdrant-find"].input_schema["properties"]), {"query"})
        self.assertEqual(tools["qdrant-find"].input_schema["required"], ["query"])
        self.assertEqual(set(tools["qdrant-store"].input_schema["properties"]), {"information", "metadata"})
        self.assertEqual(tools["qdrant-store"].input_schema["required"], ["information"])

    def test_descriptions_carry_the_guidance_that_used_to_live_in_the_env_block(self):
        import asyncio

        mod = _load_qdrant_mcp_server()
        tools = {t.name: t for t in asyncio.run(mod.mcp.list_tools())}
        find = tools["qdrant-find"].description or ""
        store = tools["qdrant-store"].description or ""
        self.assertIn("BEFORE grepping", find)
        self.assertIn("Do NOT use for exact string", find)
        self.assertIn("file_path and line_range", store)
        self.assertIn("Do not store speculative", store)

    def test_this_server_does_not_register_the_compression_tools(self):
        names = {info.name for info in _load_qdrant_mcp_server().mcp._tool_manager.list_tools()}
        self.assertFalse(names & {"compress_file", "compress_command_output", "fetch_url", "compress_text"})


class ServerStartsOverStdio(unittest.TestCase):
    """Launches the script the way .mcp.json does and talks MCP to it. The
    in-process tests above import the module and never reach its entry point, so
    only this proves the script actually serves (it needs no Qdrant and no
    model: nothing is built until the first tool call)."""

    def test_initialize_and_list_tools_over_a_real_stdio_process(self):
        import asyncio

        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(REPO_ROOT / "tools" / "qdrant_mcp_server.py")],
            env={"QDRANT_URL": "http://127.0.0.1:9", "COLLECTION_NAME": "never-contacted"},
        )

        async def go():
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return sorted(t.name for t in (await session.list_tools()).tools)

        names = asyncio.run(asyncio.wait_for(go(), timeout=60))
        self.assertEqual(names, ["compact_find", "compact_prune", "compact_store", "qdrant-find", "qdrant-store"])


class FastembedCachePersistenceTests(unittest.TestCase):
    """Module init must pin FASTEMBED_CACHE_PATH before any FastEmbedProvider
    is constructed, or the tools fall back to a wipeable OS temp dir."""

    def test_module_load_sets_persistent_cache_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "FASTEMBED_CACHE_PATH"}
        with mock.patch.dict(os.environ, env, clear=True):
            _load_qdrant_mcp_server()
            self.assertEqual(
                os.environ["FASTEMBED_CACHE_PATH"],
                str(Path.home() / ".claude" / "claude-runway" / "fastembed-cache"),
            )

    def test_module_load_respects_existing_value(self):
        with mock.patch.dict(os.environ, {"FASTEMBED_CACHE_PATH": "/custom/cache"}):
            _load_qdrant_mcp_server()
            self.assertEqual(os.environ["FASTEMBED_CACHE_PATH"], "/custom/cache")


class EmbeddingModelsAreConfiguredIndependently(unittest.TestCase):
    """EMBEDDING_MODEL is the code-index model; COMPACT_EMBEDDING_MODEL is the
    compact model. They must not leak into each other: changing the compact one
    orphans stored compacts, and an install that never set it was stored under
    BAAI/bge-small-en whatever its code-index model is."""

    def _load_with_env(self, **env):
        keys = ("EMBEDDING_MODEL", "COMPACT_EMBEDDING_MODEL")
        base = {k: v for k, v in os.environ.items() if k not in keys}
        base.update(env)
        with mock.patch.dict(os.environ, base, clear=True):
            return _load_qdrant_mcp_server()

    def test_compact_model_falls_back_to_the_legacy_model_when_unset(self):
        mod = self._load_with_env(EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(mod.COMPACT_EMBEDDING_MODEL, "BAAI/bge-small-en")
        self.assertEqual(mod.EMBEDDING_MODEL, "sentence-transformers/all-MiniLM-L6-v2")

    def test_compact_model_comes_from_its_own_variable(self):
        mod = self._load_with_env(EMBEDDING_MODEL="a/code-model", COMPACT_EMBEDDING_MODEL="b/compact-model")
        self.assertEqual(mod.COMPACT_EMBEDDING_MODEL, "b/compact-model")
        self.assertEqual(mod.EMBEDDING_MODEL, "a/code-model")

    def test_blank_values_count_as_unset(self):
        mod = self._load_with_env(EMBEDDING_MODEL="", COMPACT_EMBEDDING_MODEL="")
        self.assertEqual(mod.COMPACT_EMBEDDING_MODEL, "BAAI/bge-small-en")
        self.assertEqual(mod.EMBEDDING_MODEL, "sentence-transformers/all-MiniLM-L6-v2")

    def test_compact_tools_embed_with_the_compact_model_not_the_code_model(self):
        mod = self._load_with_env(EMBEDDING_MODEL="a/code-model", COMPACT_EMBEDDING_MODEL="b/compact-model")
        mod.QdrantClient = _make_fake_qdrant_client_class()
        built = []

        def _provider(model_name):
            built.append(model_name)
            return _FakeEmbeddingProvider(model_name)

        mod.FastEmbedProvider = _provider
        _run(mod.compact_store(information="x", project="p", label="l", date="2026-08-27"))
        _run(mod.compact_find(project="p", query="x"))
        self.assertEqual(built, ["b/compact-model", "b/compact-model"])


class _VectorProvider:
    """A deterministic embedding provider for the real QdrantConnector: one
    unit vector per text, so a query for the same text is the nearest hit."""

    def get_vector_name(self):
        return "fast-test-model"

    def get_vector_size(self):
        return 4

    def _vec(self, text):
        v = [0.0] * 4
        v[sum(map(ord, text)) % 4] = 1.0
        return v

    async def embed_documents(self, documents):
        return [self._vec(d) for d in documents]

    async def embed_query(self, query):
        return self._vec(query)


class QdrantFindAndStoreTools(unittest.TestCase):
    """qdrant-find / qdrant-store against a real QdrantConnector on qdrant-client's
    in-process `:memory:` mode, so the tool bodies and the connector are
    exercised together without a server or a model."""

    def _server(self, collection="proj"):
        from qdrant_connector import QdrantConnector

        mod = _load_qdrant_mcp_server()
        mod.COLLECTION_NAME = collection
        mod._connector = QdrantConnector(":memory:", None, collection, _VectorProvider())
        return mod

    def test_store_then_find_round_trips_content_and_metadata(self):
        mod = self._server()
        reply = _run(mod.qdrant_store("auth is handled in middleware", {"file_path": "src/auth.py", "line_range": "10-40"}))
        self.assertEqual(reply, "Remembered: auth is handled in middleware in collection proj")
        results = _run(mod.qdrant_find("auth is handled in middleware"))
        self.assertEqual(results[0], "Results for the query 'auth is handled in middleware'")
        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[1],
            "<entry><content>auth is handled in middleware</content><metadata>"
            + json.dumps({"file_path": "src/auth.py", "line_range": "10-40"})
            + "</metadata></entry>",
        )

    def test_an_entry_stored_without_metadata_has_an_empty_metadata_element(self):
        mod = self._server()
        _run(mod.qdrant_store("note", None))
        results = _run(mod.qdrant_find("note"))
        self.assertEqual(results[1], "<entry><content>note</content><metadata></metadata></entry>")

    def test_find_with_nothing_stored_says_so(self):
        mod = self._server()
        self.assertEqual(_run(mod.qdrant_find("anything")), ["No information found for the query 'anything'"])

    def test_find_returns_at_most_the_search_limit(self):
        mod = self._server()
        for i in range(mod.SEARCH_LIMIT + 5):
            _run(mod.qdrant_store(f"note {i}", None))
        results = _run(mod.qdrant_find("note 0"))
        self.assertEqual(len(results) - 1, mod.SEARCH_LIMIT)

    def test_both_tools_report_a_missing_collection_name_instead_of_crashing(self):
        mod = self._server()
        mod.COLLECTION_NAME = None
        self.assertTrue(_run(mod.qdrant_find("q"))[0].startswith("Error: COLLECTION_NAME is not set"))
        self.assertTrue(_run(mod.qdrant_store("x", None)).startswith("Error: COLLECTION_NAME is not set"))

    def test_tools_use_the_default_collection_of_this_project_only(self):
        mod = self._server("project-a")
        _run(mod.qdrant_store("only in a", None))
        # A second server for another project sharing the same in-memory client
        # must not see it: each tool is scoped to its own COLLECTION_NAME.
        other = self._server("project-b")
        other._connector.client = mod._connector.client
        self.assertEqual(_run(other.qdrant_find("only in a")), ["No information found for the query 'only in a'"])


class QdrantToolsDoNotLoadTheModelAtStartup(unittest.TestCase):
    def test_importing_the_module_constructs_no_embedding_model(self):
        # Loading the ONNX model takes seconds (and a download on a cold cache);
        # the server must answer `initialize` first. Any construction at import
        # would call fastembed's TextEmbedding.
        import qdrant_embeddings

        with mock.patch.object(qdrant_embeddings, "TextEmbedding", side_effect=AssertionError("model loaded at import")):
            _load_qdrant_mcp_server()

    def test_the_connector_is_built_on_first_use_and_then_reused(self):
        mod = _load_qdrant_mcp_server()
        mod.COLLECTION_NAME = "proj"
        mod.EMBEDDING_MODEL = "some/model"
        built = []

        class _Connector:
            def __init__(self, url, key, collection, provider):
                built.append((url, key, collection, provider.model_name))

            async def search(self, query, *, collection_name, limit):
                return []

        mod.FastEmbedProvider = _FakeEmbeddingProvider
        mod.QdrantConnector = _Connector
        self.assertEqual(built, [])
        _run(mod.qdrant_find("a"))
        _run(mod.qdrant_find("b"))
        self.assertEqual(built, [(mod.QDRANT_URL, mod.QDRANT_API_KEY, "proj", "some/model")])


class _FakeLmStudioClientModule:
    """Stands in for libs/lmstudio_client.py, which imports `openai`."""

    @staticmethod
    def build_client(base_url):
        return ("fake-client", base_url)


COMPRESSED = "[compressed 3000 -> 1200 chars across 1 chunk(s), ~60% smaller]\n\n## What we were working on\nshort"


class CompactStoreCompressesBeforeStoring(unittest.TestCase):
    """compact_store now compresses the summary itself (the /my-compact skill
    used to do that in a separate compress_text call), falling back to storing
    it as given when compression cannot happen."""

    INFORMATION = (
        "PROJECT: p\nDATE: 2026-08-27\nLABEL:\n\n"
        "## What we were working on\nRefactoring the cache layer. More follows.\n\n"
        "## Key decisions made\n- keep libs/cache_db.py\n"
    )

    def _module(self, compress):
        mod = _load_qdrant_mcp_server(stub_compression=False)
        mod.QdrantClient = _make_fake_qdrant_client_class()
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        mod._compress_impl = compress
        return mod

    def _store(self, mod, **kwargs):
        with mock.patch.dict(sys.modules, {"lmstudio_client": _FakeLmStudioClientModule}):
            return _run(mod.compact_store(
                information=kwargs.pop("information", self.INFORMATION), project="p", date="2026-08-27", **kwargs
            ))

    def _stored_payload(self, mod):
        col = next(iter(mod.QdrantClient().collections.values()))
        [point] = list(col.values())
        return point.payload

    def test_the_compressed_text_is_what_gets_stored_and_the_reply_says_so(self):
        async def compress(*args, **kwargs):
            return COMPRESSED

        mod = self._module(compress)
        reply = self._store(mod)
        payload = self._stored_payload(mod)
        self.assertEqual(payload["information"], COMPRESSED)
        self.assertEqual(payload["document"], COMPRESSED)
        self.assertTrue(reply.startswith("Stored compact in "))
        self.assertIn("compressed 3000 -> 1200 chars", reply)

    def test_it_asks_for_the_handoff_focus_with_both_preserve_flags_and_a_client(self):
        import local_compress_lib

        seen = {}

        async def compress(text, focus, **kwargs):
            seen.update(text=text, focus=focus, **kwargs)
            return COMPRESSED

        mod = self._module(compress)
        self._store(mod)
        self.assertEqual(seen["text"], self.INFORMATION)
        self.assertEqual(seen["focus"], local_compress_lib.COMPACT_FOCUS)
        self.assertIs(seen["preserve_identifiers"], True)
        self.assertIs(seen["preserve_sections"], True)
        self.assertEqual(seen["client"][0], "fake-client")

    def test_the_label_is_derived_from_what_the_caller_wrote_not_the_rewrite(self):
        async def compress(*args, **kwargs):
            return "[compressed 3000 -> 100 chars across 1 chunk(s), ~97% smaller]\n\nunrelated rewrite"

        mod = self._module(compress)
        self._store(mod)
        self.assertEqual(self._stored_payload(mod)["label"], "Refactoring the cache layer")

    def test_an_error_result_stores_the_summary_as_given(self):
        async def compress(*args, **kwargs):
            return "Error: could not reach LM Studio at http://localhost:1234/v1"

        mod = self._module(compress)
        reply = self._store(mod)
        self.assertEqual(self._stored_payload(mod)["information"], self.INFORMATION)
        self.assertIn("stored uncompressed", reply)
        self.assertIn("could not reach LM Studio", reply)
        self.assertTrue(reply.startswith("Stored compact in "))

    def test_a_total_outage_notice_is_dropped_but_the_content_kept(self):
        async def compress(*args, **kwargs):
            # The shape compress() really returns: notice, then its own
            # statistics header (about a run that did nothing), then the content.
            return (
                "[LM Studio appears unreachable -- no section was compressed; original content preserved]"
                "\n\n[compressed 120 -> 119 chars across 2 section(s), 0 compressed, 2 kept verbatim after a failed "
                "call, ~1% smaller]\n\n" + self.INFORMATION
            )

        mod = self._module(compress)
        reply = self._store(mod)
        stored = self._stored_payload(mod)["information"]
        self.assertEqual(stored, self.INFORMATION)
        self.assertNotIn("[compressed ", stored)  # no false "compressed" claim next to an uncompressed body
        self.assertIn("stored uncompressed", reply)

    def test_an_all_timeouts_notice_is_dropped_too(self):
        async def compress(*args, **kwargs):
            return (
                "[LM Studio timed out on every request -- LM Studio may be busy or unreachable; "
                "no section was compressed; original content preserved]\n\n" + self.INFORMATION
            )

        mod = self._module(compress)
        self._store(mod)
        self.assertEqual(self._stored_payload(mod)["information"], self.INFORMATION)

    def test_a_summary_under_the_threshold_is_reported_as_stored_uncompressed(self):
        async def compress(text, *args, **kwargs):
            return text  # what compress() returns for text under its size threshold

        mod = self._module(compress)
        reply = self._store(mod)
        self.assertEqual(self._stored_payload(mod)["information"], self.INFORMATION)
        self.assertIn("stored uncompressed", reply)
        self.assertIn("threshold", reply)

    def test_an_exception_from_compression_never_loses_the_compact(self):
        async def compress(*args, **kwargs):
            raise RuntimeError("boom")

        mod = self._module(compress)
        reply = self._store(mod)
        self.assertEqual(self._stored_payload(mod)["information"], self.INFORMATION)
        self.assertIn("compression failed", reply)
        self.assertIn("boom", reply)

    def test_a_missing_openai_client_still_stores_the_compact(self):
        # A Qdrant-only install has no `openai`, so libs/lmstudio_client.py
        # cannot be imported there; compact_store must still work.
        async def compress(*args, **kwargs):
            raise AssertionError("compression must not be attempted without a client")

        mod = self._module(compress)
        with mock.patch.dict(sys.modules, {"lmstudio_client": None}):
            reply = _run(mod.compact_store(information=self.INFORMATION, project="p", date="2026-08-27"))
        self.assertEqual(self._stored_payload(mod)["information"], self.INFORMATION)
        self.assertIn("LM Studio client is unavailable", reply)

    def test_nothing_reaches_qdrant_when_the_date_is_invalid_so_nothing_is_compressed(self):
        called = []

        async def compress(*args, **kwargs):
            called.append(1)
            return COMPRESSED

        mod = self._module(compress)
        reply = _run(mod.compact_store(information=self.INFORMATION, project="p", date="not-a-date"))
        self.assertTrue(reply.startswith("Error:"))
        self.assertEqual(called, [])

    def test_a_retry_with_the_same_identity_still_overwrites_after_compression(self):
        async def compress(*args, **kwargs):
            return COMPRESSED

        mod = self._module(compress)
        self._store(mod, label="same")
        self._store(mod, label="same")
        col = next(iter(mod.QdrantClient().collections.values()))
        self.assertEqual(len(col), 1)


class CompactStoreSurvivesAConcurrentFirstWrite(unittest.TestCase):
    """Two server processes saving a project's first compact can both see the
    collection absent; the loser's create_collection then conflicts. It must
    carry on into the collection the winner made (checking its model), and only
    re-raise when the collection is still absent."""

    def _module(self, client_cls):
        mod = _load_qdrant_mcp_server()
        mod.QdrantClient = client_cls
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod

    @staticmethod
    def _store(mod):
        return _run(mod.compact_store(information="first compact", project="my-project", label="l", date="2026-08-27"))

    def test_the_loser_of_the_race_still_stores_into_the_winners_collection(self):
        base = _make_fake_qdrant_client_class()

        class _Racing(base):
            def create_collection(self, collection_name, vectors_config):
                super().create_collection(collection_name, vectors_config)  # the other process wins...
                raise RuntimeError("409 Conflict: collection already exists")  # ...and ours conflicts

        mod = self._module(_Racing)
        reply = self._store(mod)
        self.assertTrue(reply.startswith("Stored compact in "), reply)
        [collection] = _Racing().collections.values()
        [point] = list(collection.values())
        self.assertEqual(point.payload["information"], "first compact")

    def test_a_create_failure_with_no_collection_afterwards_is_not_swallowed(self):
        base = _make_fake_qdrant_client_class()

        class _Broken(base):
            def create_collection(self, collection_name, vectors_config):
                raise RuntimeError("storage backend unavailable")

        with self.assertRaisesRegex(RuntimeError, "storage backend unavailable"):
            self._store(self._module(_Broken))

    def test_the_winner_using_another_model_is_reported_not_written_into(self):
        from types import SimpleNamespace

        from qdrant_client import models

        base = _make_fake_qdrant_client_class()

        class _RacingOtherModel(base):
            def create_collection(self, collection_name, vectors_config):
                super().create_collection(collection_name, vectors_config)
                raise RuntimeError("409 Conflict: collection already exists")

            def get_collection(self, collection_name):
                other = {"fast-some-other-model": models.VectorParams(size=3, distance=models.Distance.COSINE)}
                return SimpleNamespace(config=SimpleNamespace(metadata=None, params=SimpleNamespace(vectors=other)))

        mod = self._module(_RacingOtherModel)
        reply = self._store(mod)
        self.assertIn("embedding model mismatch", reply)
        [collection] = _RacingOtherModel().collections.values()
        self.assertEqual(collection, {})

    def test_an_ordinary_first_write_is_unchanged(self):
        base = _make_fake_qdrant_client_class()
        mod = self._module(base)
        self.assertTrue(self._store(mod).startswith("Stored compact in "))


class CompactsStoredBeforeTheServerMovedStayReadable(unittest.TestCase):
    """A compact written by the old compact_store (on the local-compress server)
    must be found, listed and pruned by this server, because the on-disk shape
    is unchanged: payload fields `project`/`label`/`date`/`information`/
    `document`, under the model-named vector."""

    LEGACY_PAYLOAD = {
        "document": "legacy compact body",
        "information": "legacy compact body",
        "project": "my-project",
        "label": "old label",
        "date": "2026-01-02",
        "metadata": {"source": "compact"},
    }

    def _module_with_legacy_point(self):
        from qdrant_client import models

        mod = _load_qdrant_mcp_server()
        client_cls = _make_fake_qdrant_client_class()
        mod.QdrantClient = client_cls
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        client = client_cls()
        # The collection name and vector name the old tool used by default.
        client.upsert(
            "conversation-compacts-my-project",
            [models.PointStruct(id=mod._compact_point_id("my-project", "old label", "2026-01-02"),
                                vector={"fast-bge-small-en": [0.1, 0.2, 0.3]}, payload=dict(self.LEGACY_PAYLOAD))],
        )
        return mod, client

    def test_compact_find_lists_it(self):
        mod, _client = self._module_with_legacy_point()
        result = _run(mod.compact_find(project="my-project"))
        self.assertIn("Found 1 compact(s) for project 'my-project'", result)
        self.assertIn("old label", result)
        self.assertIn("legacy compact body", result)

    def test_compact_find_with_a_query_reads_it_too(self):
        mod, _client = self._module_with_legacy_point()
        result = _run(mod.compact_find(project="my-project", query="legacy"))
        self.assertIn("legacy compact body", result)

    def test_a_new_compact_lands_in_the_same_collection_beside_it(self):
        mod, client = self._module_with_legacy_point()
        _run(mod.compact_store(information="new body", project="my-project", label="new", date="2026-02-03"))
        self.assertEqual(list(client.collections), ["conversation-compacts-my-project"])
        self.assertEqual(len(client.collections["conversation-compacts-my-project"]), 2)
        result = _run(mod.compact_find(project="my-project"))
        self.assertIn("Found 2 compact(s)", result)

    def test_compact_prune_can_delete_it(self):
        mod, client = self._module_with_legacy_point()
        reply = _run(mod.compact_prune(project="my-project", older_than_days=0, dry_run=False))
        self.assertIn("Deleted 1 compact(s)", reply)
        self.assertEqual(client.collections["conversation-compacts-my-project"], {})


if __name__ == "__main__":
    unittest.main()
