#!/usr/bin/env python3
"""Tests for tools/compress_mcp_server.py's _log_fixed_overhead().

Stdlib-only (unittest, no pytest) and no network -- _log_fixed_overhead()
never calls out to LM Studio; it only inspects already-registered tool
functions and estimates a token count from static text (name + signature +
docstring), so this needs neither a live model nor a real database:

    .venv/bin/python -m unittest discover -s tests

Why this file exists (issue #22): _log_fixed_overhead() used to enumerate a
hardcoded 5-function tuple. compact_store/compact_find/savings_summary/
savings_detail shipped later and were never added to it, so the /my-savings
"tool overhead" annotation silently understated the real cost for months --
nothing errored, the estimate was just quietly wrong. The fix replaces the
tuple with a runtime enumeration of the server's own registered tools; the
point of test_new_tool_is_picked_up_without_code_changes below is to make
that exact failure mode impossible to reintroduce silently again.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "tools"))


def _load_compress_mcp_server():
    """
    Loads a FRESH copy of the module per call (its own MCPServer instance, its
    own registered tools) so tests that register an extra ad-hoc tool can't
    leak that tool into other tests. savings_ledger is still the one shared
    module instance every load imports (Python caches by module name), which
    is what makes mock.patch.object(mod.savings_ledger, "set_meta", ...)
    reliably patch the real thing regardless of which load called it.
    """
    spec = importlib.util.spec_from_file_location(
        "compress_mcp_server", REPO_ROOT / "tools" / "compress_mcp_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SanitizeProjectAvoidsPunctuationCollisions(unittest.TestCase):
    """Regression coverage for issue #45: _sanitize_project used to collapse
    "my.project" / "my project" / "my@project" / "my-project" into the
    identical sanitized string, so distinct projects could land in the same
    Qdrant collection. The fix appends a hash suffix derived from the
    ORIGINAL (pre-sanitization) string, which still differs even when the
    sanitized-into-hyphens portion doesn't.
    """

    def test_punctuation_variants_no_longer_collide(self):
        mod = _load_compress_mcp_server()
        variants = ["my.project", "my project", "my@project", "my-project"]
        sanitized = [mod._sanitize_project(v) for v in variants]
        # Every pairwise result must be distinct -- this is the actual
        # reported bug: all four used to sanitize to the same "my-project".
        self.assertEqual(len(sanitized), len(set(sanitized)))

    def test_deterministic_across_calls(self):
        # compact_store and compact_find each call _sanitize_project
        # independently and must agree on the same collection name for the
        # same project, or /my-resume would never find what /my-compact
        # just stored.
        mod = _load_compress_mcp_server()
        first = mod._sanitize_project("Acme.Support.TicketsApi")
        second = mod._sanitize_project("Acme.Support.TicketsApi")
        self.assertEqual(first, second)

    def test_output_contains_only_valid_qdrant_collection_characters(self):
        mod = _load_compress_mcp_server()
        for project in ["my.project", "my project", "my@project", "plain-name", "Mixed_Case-123"]:
            result = mod._sanitize_project(project)
            self.assertRegex(result, r"^[a-zA-Z0-9_-]+$")

    def test_plain_alphanumeric_name_keeps_readable_prefix(self):
        # The common case (no sanitization needed) should still read as
        # "<original>-<hash>", not become unrecognizable.
        mod = _load_compress_mcp_server()
        result = mod._sanitize_project("myproject")
        self.assertTrue(result.startswith("myproject-"))
        suffix = result[len("myproject-"):]
        self.assertEqual(len(suffix), 8)

    def test_casing_variants_collapse_to_the_same_collection(self):
        # Regression coverage for issue #37: unlike the punctuation variants
        # above (deliberately kept DISTINCT), casing variants of the exact
        # same project string must sanitize IDENTICALLY, or a casing drift
        # between /my-compact and /my-resume points at a nonexistent
        # collection.
        mod = _load_compress_mcp_server()
        variants = ["MyProject", "myproject", "MYPROJECT", "mYpRoJeCt"]
        sanitized = {mod._sanitize_project(v) for v in variants}
        self.assertEqual(len(sanitized), 1)


class _FakeEmbeddingProvider:
    """Stands in for mcp_server_qdrant.embeddings.fastembed.FastEmbedProvider
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
        mod = _load_compress_mcp_server()
        first = mod._compact_point_id("proj", "label", "2026-08-27")
        second = mod._compact_point_id("proj", "label", "2026-08-27")
        self.assertEqual(first, second)

    def test_different_inputs_produce_different_ids(self):
        mod = _load_compress_mcp_server()
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
        mod = _load_compress_mcp_server()
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
        mod = _load_compress_mcp_server()
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
        mod = _load_compress_mcp_server()
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
        mod = _load_compress_mcp_server()
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
        mod = _load_compress_mcp_server()
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
        mod = _load_compress_mcp_server()
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
        mod = _load_compress_mcp_server()

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
        mod = _load_compress_mcp_server()
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
        mod = _load_compress_mcp_server()
        with self.assertRaises(RuntimeError):
            mod._collections_for_project(self._BrokenGetCollectionsClient(), "some-project")

    def test_candidate_hint_degrades_gracefully_on_the_same_failure(self):
        mod = _load_compress_mcp_server()
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
        mod = _load_compress_mcp_server()
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
    """Regression coverage for a real truncation-before-filter bug flagged
    in PR review on #123 (rounds 4 AND 5): the semantic query path used to
    ask Qdrant for exactly `limit` results per collection and only THEN
    apply the case-insensitive per-entry project filter client-side. A
    shared collection holding more than `limit` entries that rank ahead of
    a project's own (lower-ranked) entries could fill the entire truncated
    page with OTHER projects' points, which the filter would then remove
    entirely -- silently reporting no matches even though the target
    project's real history still exists in that same collection, just
    past the truncation point.

    Round 4's first attempt at a fix (over-fetching a larger, but still
    fundamentally capped, number of results) was itself flagged as
    insufficient in round 5: it only moves the same bug to a bigger
    threshold, since ANY fixed cap can still be exceeded by enough
    higher-ranked OTHER-project points. The actual fix builds an EXACT,
    Qdrant-SIDE MatchAny filter (from every literal casing variant
    actually stored in that collection -- see _collection_project_values)
    for any non-canonical collection, so Qdrant itself only ever ranks and
    truncates ALREADY-isolated points -- there's no cap to exceed, because
    truncation happens after isolation, not before it. This test uses a
    count of other-project points well beyond round 4's old 12-point/200
    over-fetch-floor test case specifically to prove this isn't just a
    bigger buffer."""

    def _patched_module(self):
        mod = _load_compress_mcp_server()
        mod.QdrantClient = _make_fake_qdrant_client_class()
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod

    def test_target_projects_entry_still_found_behind_many_higher_ranked_others(self):
        mod = self._patched_module()
        client_cls = mod.QdrantClient
        shared_name = f"{mod.COMPACT_COLLECTION}-a-shared-collection-abcdef01"
        # Insert far more other-project points than any fixed over-fetch
        # cap would have covered (round 4's fix used a 200-point floor) --
        # the fake's query_points returns points in insertion order,
        # standing in for "rank", so these are all inserted FIRST.
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
        mod = _load_compress_mcp_server()
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
        mod = _load_compress_mcp_server()
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


class LogFixedOverheadEnumeratesRegisteredTools(unittest.TestCase):
    def test_overhead_covers_every_currently_registered_tool(self):
        mod = _load_compress_mcp_server()
        mod.TRACK_SAVINGS = True
        captured = {}
        with mock.patch.object(
            mod.savings_ledger, "set_meta", side_effect=lambda k, v: captured.setdefault(k, v)
        ):
            mod._log_fixed_overhead()

        tool_names = {info.name for info in mod.mcp._tool_manager.list_tools()}
        # Pinned to the module's actual current tool set, not just len() ==
        # 10, so a rename/removal is caught too, not just a count drift.
        self.assertEqual(
            tool_names,
            {
                "list_local_models",
                "compress_file",
                "compress_command_output",
                "fetch_url",
                "compress_text",
                "compact_store",
                "compact_find",
                "compact_prune",
                "savings_summary",
                "savings_detail",
                "savings_trend",
            },
        )
        self.assertIn("schema_overhead_tokens", captured)
        self.assertGreater(int(captured["schema_overhead_tokens"]), 0)

    def test_new_tool_is_picked_up_without_code_changes(self):
        """The actual issue #22 regression guard: a tool registered AFTER
        this function was last edited must still be included automatically,
        with zero changes to _log_fixed_overhead() itself.
        """
        mod = _load_compress_mcp_server()

        @mod.mcp.tool()
        def _newly_added_probe_tool() -> str:
            """A tool added after _log_fixed_overhead was last edited."""
            return "x"

        mod.TRACK_SAVINGS = True
        captured = {}
        with mock.patch.object(
            mod.savings_ledger, "set_meta", side_effect=lambda k, v: captured.setdefault(k, v)
        ):
            mod._log_fixed_overhead()

        tool_names = {info.name for info in mod.mcp._tool_manager.list_tools()}
        self.assertIn("_newly_added_probe_tool", tool_names)
        self.assertIn("schema_overhead_tokens", captured)

    def test_noop_when_tracking_disabled(self):
        mod = _load_compress_mcp_server()
        mod.TRACK_SAVINGS = False
        with mock.patch.object(mod.savings_ledger, "set_meta") as set_meta:
            mod._log_fixed_overhead()
        set_meta.assert_not_called()


class CompressCommandOutputDefaultsToRedaction(unittest.TestCase):
    """Regression coverage for issue #46: compress_command_output never
    exposed preserve_identifiers or preserve_sections, so its call into the
    shared compress() pipeline always used both flags' defaults
    (preserve_identifiers=False, preserve_sections=False) -- and
    redact_and_disclose() (libs/local_compress_lib.py) only runs credential
    redaction when at least one of those two is True (PR #135 review, round
    4 -- initially gated on preserve_identifiers alone, widened to either
    flag once round 4 caught preserve_sections=True falling through to the
    same ungated whole-text path on unstructured input). That left the tool
    most likely to see a raw secret (curl output, printenv, config dumps)
    never redacting one.

    Mocks at the _compress_impl boundary (the local_compress_lib.compress()
    function, imported under that alias) rather than running a real LM
    Studio call -- same reasoning as this repo's other MCP-server tests: no
    network needed to verify what kwargs get passed through.
    """

    def _capture_compress_impl_kwargs(self, mod):
        captured = {}

        async def _fake_compress_impl(
            text, focus, skip_if_under_chars, chunk_chars, max_total_chars, max_chars,
            model, base_url, preserve_identifiers=False, preserve_sections=False, ctx=None,
        ):
            captured["preserve_identifiers"] = preserve_identifiers
            captured["preserve_sections"] = preserve_sections
            return "[compressed 10 -> 5 chars across 1 chunk(s), ~50% smaller]\n\nfake summary"

        mod._compress_impl = _fake_compress_impl
        return captured

    def test_default_passes_preserve_identifiers_true(self):
        mod = _load_compress_mcp_server()
        captured = self._capture_compress_impl_kwargs(mod)
        _run(mod.compress_command_output(command="echo hello"))
        self.assertTrue(captured["preserve_identifiers"])

    def test_caller_can_still_opt_out(self):
        mod = _load_compress_mcp_server()
        captured = self._capture_compress_impl_kwargs(mod)
        _run(mod.compress_command_output(command="echo hello", preserve_identifiers=False))
        self.assertFalse(captured["preserve_identifiers"])

    def test_preserve_sections_still_defaults_off_and_is_passed_through(self):
        # preserve_sections isn't part of issue #46's finding, but since this
        # tool now exposes both flags (matching compress_text's pattern),
        # confirm the second flag's own default/pass-through isn't broken by
        # the same change.
        mod = _load_compress_mcp_server()
        captured = self._capture_compress_impl_kwargs(mod)
        _run(mod.compress_command_output(command="echo hello"))
        self.assertFalse(captured["preserve_sections"])
        _run(mod.compress_command_output(command="echo hello", preserve_sections=True))
        self.assertTrue(captured["preserve_sections"])

    def test_redaction_actually_runs_end_to_end_with_the_real_pipeline(self):
        # Goes one level deeper than the kwarg-capture tests above: uses the
        # REAL local_compress_lib.compress() (not a fake) with a fake LM
        # Studio client, so this proves the default genuinely results in a
        # secret being redacted from compress_command_output's output, not
        # just that a flag was threaded through correctly. Patches
        # local_compress_lib.client directly (not compress_mcp_server's own
        # `_client` alias) -- compress()'s internal call to `client(...)`
        # resolves that name against ITS OWN module globals at call time,
        # not against whatever compress_mcp_server imported it as.
        import local_compress_lib
        mod = _load_compress_mcp_server()
        secret = "ghp_" + ("a" * 36)
        # The secret comes from the command's STDOUT here, not its own
        # command line text, to keep this test focused on the compressed
        # BODY's redaction specifically -- the savings footer's "source"
        # field (a separate leak vector: the raw command line itself, which
        # is also now redacted -- see SavingsFooterRedactsSourceCredentials
        # below, PR #135 review round 3) is exercised by its own dedicated
        # test instead. Passing it via an env var keeps the command line
        # itself secret-free while still producing the secret in stdout.
        os.environ["CLAUDE_RUNWAY_TEST_SECRET"] = secret
        self.addCleanup(os.environ.pop, "CLAUDE_RUNWAY_TEST_SECRET", None)

        class _FakeChoice:
            def __init__(self, content):
                self.message = mock.Mock(content=content)

        class _FakeCompletion:
            def __init__(self, content):
                self.choices = [_FakeChoice(content)]

        class _FakeChatCompletions:
            def create(self, model, messages, temperature=0):
                # Echo the secret back, simulating a model that reproduces a
                # credential-shaped value verbatim from its input.
                return _FakeCompletion(f"Ran a command. Token seen: {secret}")

        class _FakeChat:
            def __init__(self):
                self.completions = _FakeChatCompletions()

        class _FakeClient:
            def __init__(self):
                self.chat = _FakeChat()

        with mock.patch.object(local_compress_lib, "client", lambda base_url: _FakeClient()):
            result = _run(mod.compress_command_output(
                command='printf "%s" "$CLAUDE_RUNWAY_TEST_SECRET"',
                skip_if_under_chars=0, model="fake-model",
            ))
        self.assertNotIn(secret, result)
        self.assertIn("credential-shaped value(s) redacted", result)

    def test_redaction_still_runs_under_the_real_default_threshold(self):
        # PR #135 review (Copilot): the end-to-end test above used
        # skip_if_under_chars=0 specifically to route through the main
        # compression pipeline, which masked a real gap -- command output
        # shorter than the DEFAULT 2000-char threshold hit compress()'s
        # early return, which used to hand back the original text with no
        # redaction at all, regardless of preserve_identifiers. Fixed in
        # local_compress_lib.compress() (the early-return branch now
        # redacts when preserve_identifiers is set); this test exercises
        # compress_command_output's actual DEFAULT skip_if_under_chars
        # (2000) with a short secret-bearing command, which is the common
        # case for real command output. No LM Studio mock needed here --
        # the early-return path returns before resolve_model/client are
        # ever touched.
        mod = _load_compress_mcp_server()
        secret = "ghp_" + ("a" * 36)
        os.environ["CLAUDE_RUNWAY_TEST_SECRET"] = secret
        self.addCleanup(os.environ.pop, "CLAUDE_RUNWAY_TEST_SECRET", None)

        result = _run(mod.compress_command_output(
            command='printf "%s" "$CLAUDE_RUNWAY_TEST_SECRET"',
        ))
        self.assertNotIn(secret, result)
        self.assertIn("credential-shaped value(s) redacted", result)


class SavingsFooterRedactsSourceCredentials(unittest.TestCase):
    """Regression coverage for PR #135 review, round 3 (Copilot):
    _append_savings_footer's "source" field echoed the raw caller-supplied
    value (compress_command_output passes the actual shell command)
    verbatim and unredacted into the savings footer -- so a command like
    `curl -H 'Authorization: Bearer ghp_...'` still leaked its embedded
    credential via that field even though the compressed BODY the command
    produced was properly redacted. That footer gets persisted into the
    savings ledger by hooks/compress_bash_output.py, turning a transient
    secret in a command line into a durable one in that database.
    """

    def test_source_credential_is_redacted_in_the_footer(self):
        mod = _load_compress_mcp_server()
        mod.TRACK_SAVINGS = True
        secret = "sk-ant-" + ("b" * 40)
        footer_json = mod._append_savings_footer(
            outer_text="[compressed 100 -> 10 chars across 1 chunk(s), ~90% smaller]\n\nsummary",
            inner_result="[compressed 100 -> 10 chars across 1 chunk(s), ~90% smaller]\n\nsummary",
            tool="compress_command_output", raw_text="raw", credited=True,
            source=f"curl -H 'Authorization: Bearer {secret}' https://example.com",
        )
        self.assertNotIn(secret, footer_json)
        self.assertIn("REDACTED-CREDENTIAL", footer_json)

    def test_clean_source_is_unaffected(self):
        # Confirms the fix doesn't mangle an ordinary command with no
        # credential in it -- redact_credentials() is a no-op when there's
        # nothing to redact.
        mod = _load_compress_mcp_server()
        mod.TRACK_SAVINGS = True
        result = mod._append_savings_footer(
            outer_text="[compressed 100 -> 10 chars across 1 chunk(s), ~90% smaller]\n\nsummary",
            inner_result="[compressed 100 -> 10 chars across 1 chunk(s), ~90% smaller]\n\nsummary",
            tool="compress_command_output", raw_text="raw", credited=True,
            source="pytest -q",
        )
        self.assertIn('"source": "pytest -q"', result)

    def test_credential_straddling_the_200_char_truncation_boundary_is_still_caught(self):
        # Redaction must run over the FULL source before truncating to 200
        # chars -- truncating first could split a credential in half,
        # leaving an unmatched (and un-redacted) fragment on one side of
        # the cut.
        mod = _load_compress_mcp_server()
        mod.TRACK_SAVINGS = True
        secret = "ghp_" + ("c" * 36)
        padding = "x" * 180
        source = f"{padding} {secret}"  # secret starts well past char 180
        self.assertGreater(len(source), 200)
        result = mod._append_savings_footer(
            outer_text="[compressed 100 -> 10 chars across 1 chunk(s), ~90% smaller]\n\nsummary",
            inner_result="[compressed 100 -> 10 chars across 1 chunk(s), ~90% smaller]\n\nsummary",
            tool="compress_command_output", raw_text="raw", credited=True,
            source=source,
        )
        self.assertNotIn(secret, result)


class _FakeStreamedResponse:
    """Stand-in for requests.Response in stream=True mode -- exposes just
    the surface fetch_url actually touches (.headers, .raise_for_status(),
    .iter_content(), .close(), .encoding, .apparent_encoding, ._content, and
    the context-manager protocol fetch_url now uses via
    `with requests.get(...) as resp:` -- PR #124 review) so no real network
    call happens. `body_chunks` is a list of byte strings handed back one at
    a time from .iter_content(), simulating a streamed download regardless
    of what Content-Length (if any) claimed up front. `raise_in_iter`, if
    set, is raised partway through iteration to simulate a broken chunked
    response / read timeout -- the case that exposed the pre-fix connection
    leak. `apparent_encoding_result` fakes what real requests derives via
    chardet/charset_normalizer over `self._content` -- fetch_url populates
    `_content` itself (mirroring how it uses the real attribute) before
    reading this, so this fake doesn't need to run real detection logic.
    """

    def __init__(
        self, body_chunks, content_length=None, encoding="utf-8", status_ok=True,
        raise_in_iter=None, apparent_encoding_result="utf-8",
    ):
        self._body_chunks = body_chunks
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}
        self.encoding = encoding
        self._status_ok = status_ok
        self._raise_in_iter = raise_in_iter
        self._apparent_encoding_result = apparent_encoding_result
        self._content = False
        self.closed = False

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("simulated HTTP error")

    def iter_content(self, chunk_size=65536):
        for chunk in self._body_chunks:
            yield chunk
        if self._raise_in_iter is not None:
            raise self._raise_in_iter

    @property
    def apparent_encoding(self):
        return self._apparent_encoding_result

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class FetchUrlEnforcesSizeCap(unittest.TestCase):
    """Regression coverage for issue #39: fetch_url used to call
    requests.get() with no stream=True, no Content-Length check, and no
    size cap of any kind -- resp.text/resp.content fully materialized the
    body into memory unconditionally before max_total_chars (a
    post-extraction cap) ever got a chance to run.
    """

    def test_declared_content_length_over_cap_is_refused_without_reading_body(self):
        mod = _load_compress_mcp_server()
        fake_resp = _FakeStreamedResponse(
            body_chunks=[b"<html>should never be read</html>"],
            content_length=50 * 1024 * 1024,  # 50 MB, over the 10 MB default cap
        )
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/huge-file"))

        self.assertIn("Error fetching", result)
        self.assertIn("Content-Length", result)
        self.assertTrue(fake_resp.closed)

    def test_missing_content_length_still_aborts_once_stream_exceeds_cap(self):
        """The case the issue explicitly calls out as the normal one: a
        server that never sends (or understates) Content-Length -- the
        incremental byte count during iter_content() is what has to catch
        this, since there's no header to check up front."""
        mod = _load_compress_mcp_server()
        oversized_chunk = b"x" * (11 * 1024 * 1024)  # 11 MB, over the 10 MB default cap
        fake_resp = _FakeStreamedResponse(
            body_chunks=[oversized_chunk],
            content_length=None,
        )
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/no-content-length"))

        self.assertIn("Error fetching", result)
        self.assertIn("exceeded", result)
        self.assertTrue(fake_resp.closed)

    def test_response_under_cap_is_fetched_and_extracted_normally(self):
        mod = _load_compress_mcp_server()
        html = b"<html><body><p>Small page, well under any size cap.</p></body></html>"
        fake_resp = _FakeStreamedResponse(body_chunks=[html], content_length=len(html))
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/small-page"))

        self.assertNotIn("Error fetching", result)
        self.assertIn("Small page", result)
        # PR #124 review fix: fetch_url now uses `with requests.get(...) as
        # resp:`, so the connection is released on every exit path,
        # including a normal successful fetch -- not just the early-return
        # error paths that used to call resp.close() explicitly.
        self.assertTrue(fake_resp.closed)

    def test_max_response_bytes_override_is_respected(self):
        """A caller-supplied max_response_bytes narrower than the default
        must be honored, not just the module-level default."""
        mod = _load_compress_mcp_server()
        body = b"<html>" + b"y" * 2000 + b"</html>"
        fake_resp = _FakeStreamedResponse(body_chunks=[body], content_length=len(body))
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/medium-page", max_response_bytes=1000))

        self.assertIn("Error fetching", result)
        self.assertIn("Content-Length", result)

    def test_connection_is_closed_when_iteration_raises_mid_stream(self):
        """PR #124 review (Copilot): a bare `resp = requests.get(...)` left
        the connection open on any exception from iter_content() (a normal
        read timeout or broken chunked response) -- those exceptions used to
        propagate straight past every resp.close() call in the function.
        Reproduced directly against the pre-fix code (a fake response whose
        iter_content() raised mid-stream left .closed False); this pins the
        fix, which switched to `with requests.get(...) as resp:` so the
        connection is released on any exit path, exception included.
        """
        mod = _load_compress_mcp_server()
        fake_resp = _FakeStreamedResponse(
            body_chunks=[b"partial data"],
            raise_in_iter=ConnectionError("simulated broken chunked response"),
        )
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/flaky"))

        self.assertIn("Error fetching", result)
        self.assertTrue(fake_resp.closed)

    def test_invalid_server_declared_charset_falls_back_to_utf8_instead_of_crashing(self):
        """PR #124 review (Copilot): resp.encoding is derived from a
        server-controlled Content-Type header -- a value like
        `charset=invalid-bogus-charset` makes requests set resp.encoding to
        that literal (unvalidated) string. Decoding with an unknown codec
        name raises LookupError, which used to be unhandled here (reproduced
        directly: the pre-fix code raised LookupError instead of returning
        an error string). requests' own resp.text property specifically
        catches (LookupError, TypeError) around this same decode and falls
        back to UTF-8 -- this pins that same fallback.
        """
        mod = _load_compress_mcp_server()
        html = b"<html><body><p>Still readable as UTF-8 bytes.</p></body></html>"
        fake_resp = _FakeStreamedResponse(
            body_chunks=[html], content_length=len(html), encoding="invalid-bogus-charset",
        )
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/bad-charset"))

        self.assertNotIn("Error fetching", result)
        self.assertIn("Still readable", result)

    def test_headerless_response_uses_detected_encoding_not_a_blind_utf8_default(self):
        """PR #124 review round 2 (Copilot): when resp.encoding is None (no
        Content-Type charset declared at all), the old resp.text path fell
        back to apparent_encoding (chardet/charset_normalizer detection over
        the full body), not a blind UTF-8 assumption. Reproduced directly:
        a headerless Windows-1252-encoded page containing a curly
        single-quote (0x92) decoded to a bare U+FFFD replacement character
        under straight UTF-8 before this fix. Pins that fetch_url now
        populates resp._content with the already-downloaded bytes and reads
        resp.apparent_encoding instead of defaulting straight to UTF-8.
        """
        mod = _load_compress_mcp_server()
        # "It’s" (curly apostrophe, U+2019) encoded as cp1252 -- 0x92 is not
        # valid standalone UTF-8, so a straight UTF-8 decode with
        # errors="replace" loses it as U+FFFD.
        body = "It’s a legacy page.".encode("cp1252")
        fake_resp = _FakeStreamedResponse(
            body_chunks=[body],
            content_length=len(body),
            encoding=None,
            apparent_encoding_result="cp1252",
        )
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/legacy-page"))

        self.assertNotIn("Error fetching", result)
        self.assertIn("It’s a legacy page", result)
        # Confirms fetch_url populated the cache with OUR bytes (bounded by
        # max_response_bytes) rather than trying to re-read the network.
        self.assertEqual(fake_resp._content, body)


class CompactPruneTests(unittest.TestCase):
    """Tests for compact_prune (issue #70): a retention/cleanup tool for
    stored conversation compacts, with dry_run-by-default safety.

    Uses the same _FakeQdrantClient/_FakeEmbeddingProvider pattern as
    compact_store/compact_find tests -- no real Qdrant or embedding model
    needed.
    """

    def _patched_module(self):
        mod = _load_compress_mcp_server()
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
        result = _run(mod.compact_prune(project="proj", keep_last_n=1, dry_run=False))

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


if __name__ == "__main__":
    unittest.main()
