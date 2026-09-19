"""
Embedding-model mismatch check shared by `tools/ingest_mcp_server.py`
(`find_in_collection`, `index_repo`, `sync_repo`) and `libs/memory_bank_lib.py`
(issue #175). Relocated here from `ingest_mcp_server.py` (where it was a private,
single-file function used only by `find_in_collection`) rather than
duplicated, so callers can't drift apart on this logic. The default
behavior is unchanged from that original home (fails open on an
inconclusive check), but two things were added since relocation that
change behavior for callers that opt into them: an optional
`recovery_hint` (lets a caller override the default remediation text) and
an optional `fail_closed` mode (inverts the default for a caller about to
perform a destructive operation -- see `check_embedding_model_mismatch`'s
own docstring).

Guards against a wrong `embedding_model` producing plausible-looking garbage
search results with no error: different models produce incompatible vector
spaces, so a caller pointed at a collection indexed with a different model
needs an actionable error, not a silent bad answer.
"""

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import models as _m

from qdrant_retry import call_with_retry


def check_embedding_model_mismatch(
    client: "QdrantClient",
    collection: str,
    provider,
    *,
    recovery_hint: Optional[str] = None,
    fail_closed: bool = False,
) -> Optional[str]:
    """
    Compare the embedding provider's expected vector name and dimension against
    what the collection actually stores, returning an error string on a
    definitive mismatch or None when the check passes (or is inconclusive).

    Fails open on any exception BY DEFAULT (`fail_closed=False`) -- a network
    error or unexpected config shape must never block a valid read-only query;
    the caller already verified the collection exists, and every other caller
    of this function (`find_in_collection`, `memory_bank_lib.py`'s remember/
    recall/ensure_collection) is fine proceeding on an inconclusive result.

    `fail_closed=True` (PR #178 review, fourth and eighth passes) inverts that
    for the two callers where it's wrong: `index_repo(reset=True)`'s and
    `sync_repo`'s pre-delete checks. Each exists specifically to stop a
    destructive delete before it runs -- fail-open there meant an
    inconclusive check (a transient network error,
    retries exhausted, an unexpected response shape) was silently treated the
    same as "confirmed compatible," letting the destructive delete proceed
    anyway on a schema it never actually verified. With `fail_closed=True`, an
    exception during the check returns an actionable error instead of None,
    so "couldn't verify" blocks the caller the same way a confirmed mismatch
    does -- the whole point of a PRE-delete safety check is worthless if it
    can be silently skipped by the exact failure it's supposed to guard
    against.

    Covers three Qdrant vector-config cases:
    - None -- sparse-only collection with no dense-vector config at all; definitive
      incompatibility (not inconclusive), same as empty dict.
    - Dict[str, VectorParams] -- named vectors (mcp-server-qdrant's own format,
      using a "fast-<model-slug>" key produced by FastEmbedProvider.get_vector_name()).
      Errors on missing name (including empty dict -- sparse-only) or wrong dimension.
    - VectorParams -- unnamed/default single vector (older or third-party ingestion).
      QdrantConnector.search always passes using=get_vector_name(), which Qdrant
      cannot resolve for an unnamed/default-vector collection.

    `recovery_hint`, if given, REPLACES the default `find_in_collection`/
    `index_repo`-oriented remediation text in the two error messages that carry
    one (the name-mismatch and unnamed-vector-format cases). Added for issue
    #175 (PR #178 review): `memory_bank_lib.py` also calls this shared check,
    where "use index_repo(reset=true)" is actively wrong advice -- `index_repo`
    refuses to ever touch the shared memory-bank collection -- and "check the
    other repo's .mcp.json"/"that find_in_collection expects" don't apply
    either, since memory-bank isn't a cross-repo lookup. Defaults to None,
    which preserves the original `find_in_collection`-oriented text exactly,
    so this is a purely additive change for that caller.
    """
    try:
        info = call_with_retry(client.get_collection, collection)
        vectors_config = info.config.params.vectors

        expected_name = provider.get_vector_name()
        expected_size = provider.get_vector_size()

        if vectors_config is None or (isinstance(vectors_config, dict) and expected_name not in vectors_config):
            # vectors_config is None: sparse-only collection with no dense-vector
            # config at all (a definitive incompatibility, not inconclusive).
            # Empty dict {}: sparse-only collection -- same incompatibility.
            # Non-empty dict without the expected key: wrong model was used.
            # All three cases: QdrantConnector.search would fail trying to resolve
            # using=expected_name against a collection that doesn't have it.
            present = sorted(vectors_config.keys()) if isinstance(vectors_config, dict) else []
            hint = (
                f", but the collection has: {present}" if present
                else " (collection has no dense vectors)"
            )
            return (
                f"Error: embedding model mismatch for collection '{collection}'. "
                f"The model '{provider.model_name}' produces a vector named "
                f"'{expected_name}'{hint}. "
                + (
                    recovery_hint
                    or "Pass the embedding_model that matches the one used when this "
                    "collection was indexed (check the other repo's .mcp.json)."
                )
            )
        elif isinstance(vectors_config, dict):
            stored_size = vectors_config[expected_name].size
            if stored_size != expected_size:
                return (
                    f"Error: embedding dimension mismatch for collection '{collection}'. "
                    f"Model '{provider.model_name}' produces {expected_size}-dim vectors, "
                    f"but the collection's '{expected_name}' vector has {stored_size} dims. "
                    f"Pass the embedding_model that matches the one used when indexing."
                )
        elif isinstance(vectors_config, _m.VectorParams):
            # Single unnamed VectorParams: QdrantConnector.search always passes
            # using=get_vector_name(), which Qdrant cannot resolve for an unnamed/
            # default-vector collection. Return an error immediately rather than
            # letting it fail downstream with an opaque Qdrant error.
            # Recovery (find_in_collection's default text): previously
            # recommended index_repo(reset=true) to wipe and recreate with a
            # named-vector schema -- no longer effective advice (PR #178
            # review, third pass): reset=true was changed to always preserve
            # the collection's existing schema (never delete_collection), so
            # it can no longer fix an unnamed-vector collection's format at
            # all. Recommend dropping/recreating the collection directly
            # instead -- the only thing that still actually works.
            return (
                f"Error: collection '{collection}' uses an unnamed/default vector "
                f"format (not compatible with mcp-server-qdrant's named-vector search). "
                + (
                    recovery_hint
                    or f"Drop this collection directly in Qdrant and re-index it from scratch "
                    f"so it stores vectors under the '{expected_name}' name that "
                    f"find_in_collection expects -- index_repo(reset=true) no longer "
                    f"recreates an existing collection's schema."
                )
            )
        else:
            # Found in PR #178 review (seventh pass): any vectors_config shape
            # OTHER than None, dict, or VectorParams (a future qdrant-client
            # type this function doesn't know about yet) fell through every
            # branch above with no return, silently reaching the shared
            # `return None` after this try/except -- treating "we don't know
            # what this is" as "confirmed compatible" even under
            # fail_closed=True, exactly the failure mode fail_closed exists
            # to prevent for a caller about to run a destructive delete.
            # Routing it through the exception path instead makes it behave
            # like every other inconclusive case: fails open by default,
            # fails closed (loud, actionable error) when the caller asked
            # for that.
            raise TypeError(f"unrecognized vectors_config shape: {type(vectors_config).__name__}")
    except Exception as e:
        if fail_closed:
            return (
                f"Error: could not verify embedding-model compatibility for collection "
                f"'{collection}' ({e}) -- refusing to proceed with a destructive operation "
                f"until this can be confirmed. This is likely transient (a dropped Qdrant "
                f"connection); retry, or investigate if it persists."
            )
        # Network error, unexpected config shape, model-description lookup
        # failure -- don't block a potentially valid query.
        return None
    return None
