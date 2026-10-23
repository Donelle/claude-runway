"""
Resolve the (url, api_key) pair an MCP tool or the ingest CLI should hand to
QdrantClient.

Several tools accept an optional `qdrant_url`/`qdrant_api_key` as TOOL INPUT,
i.e. text the model chooses. If a call passes its own qdrant_url but no key
and the code falls back to the configured key anyway, QdrantClient sends that
key in the `api-key` header to whatever host the call named. The realistic
trigger is prompt injection: text in a file, page or issue the model reads
tells it to call a tool with an attacker's URL, and the configured credential
goes with the request.

The standalone ingest CLI (tools/ingest_to_qdrant.py) has the same exposure
through its --qdrant-url flag: a model can run it through Bash, and a mistyped
or copied URL would otherwise receive the exported QDRANT_API_KEY.

This module is the one place that rule lives, so the ingest server, the
memory-bank server, the qdrant server's compact tools and the ingest CLI
(whose --qdrant-url/--qdrant-api-key play the same role) cannot drift apart:

  - an explicit key always wins (the caller chose both the host and the key);
  - the configured default key is used only when the resolved URL IS the
    configured default URL;
  - any other URL gets no key at all.

Standard library only, so any server can import it.
"""

from typing import Optional


def same_endpoint(a: str, b: str) -> bool:
    """Whether two Qdrant URLs name the same endpoint. Also how
    ingest_mcp_server decides that a call targets this project's own
    collection (which owns the checkout's sync manifest and its lock), so the
    key rule and that decision agree on what "the configured URL" is."""
    # A trailing slash names the same endpoint, so it must not make the
    # configured key silently disappear for this project's own Qdrant. This is
    # deliberately the ONLY normalisation: anything fuzzier (case folding,
    # default ports, resolving hostnames) widens what receives the key.
    return a.rstrip("/") == b.rstrip("/")


def resolve_qdrant_credentials(
    url: Optional[str],
    api_key: Optional[str],
    *,
    default_url: str,
    default_api_key: Optional[str],
) -> tuple[str, Optional[str]]:
    """Return the (url, api_key) to connect with.

    url/api_key are the tool's optional inputs; default_url/default_api_key are
    the project's configured QDRANT_URL/QDRANT_API_KEY. An empty string counts
    as "not given" for both inputs, like the `or` fallbacks this replaces.
    """
    resolved_url = url or default_url
    if api_key:
        return resolved_url, api_key
    if same_endpoint(resolved_url, default_url):
        return resolved_url, default_api_key
    return resolved_url, None
