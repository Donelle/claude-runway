"""
The one definition of the savings envelope that mcpservers/compress_mcp_server.py
returns and hooks/compress_output.py unwraps, so the two cannot drift apart on
what counts as one.

    {"compressed_text": "<exactly what Claude should see>",
     "metrics": {"tool": str, "raw_chars": int, "out_chars": int,
                 "credited": bool, "source": str}}

The hook has to decide from the text alone whether a result is an envelope, and
a result can be ANY text -- `compress_file` on a small file hands the file back
unchanged, and that file could itself be envelope-shaped JSON. The server
therefore calls `text_is_envelope` on every plain-text result of a tool the hook
unwraps and, in that one case, wraps the text in a genuine envelope whose counts
are null (the hook unwraps it and records nothing) instead of returning it bare.
That is what makes the format collision-safe: a bare result the hook would
mistake for an envelope is never returned. The check is defined here, next to
the parser it must agree with, because the two have to move together.

Stdlib only, no I/O.
"""

import json


def is_savings_envelope(value) -> bool:
    """True only for a dict with EXACTLY a string `compressed_text` and a dict
    `metrics` whose `tool`, `credited` and `source` have the types the server
    writes. The count fields are deliberately not checked: a malformed or null
    count must still unwrap the envelope (Claude must never see the JSON) and is
    then skipped, not recorded, by the hook."""
    if not isinstance(value, dict) or set(value) != {"compressed_text", "metrics"}:
        return False
    metrics = value["metrics"]
    return (
        isinstance(value["compressed_text"], str)
        and isinstance(metrics, dict)
        and isinstance(metrics.get("tool"), str)
        and isinstance(metrics.get("credited"), bool)
        and isinstance(metrics.get("source"), str)
    )


def parse_envelope_text(text):
    """If `text` is the savings envelope, return the parsed envelope dict;
    otherwise None.

    Decodes exactly the layers the real call path produces: the MCP layer wraps
    the string the server returned as `{"result": "<that string>"}`, and the
    server's string is itself the envelope's JSON. So `text` is either that
    wrapper (decode twice) or the envelope itself (decode once). Strict
    json.loads only: a parse failure at either layer, or a value that is not
    exactly the envelope (see is_savings_envelope), means ordinary tool output,
    and nothing is repaired, stripped or guessed at."""
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text)
        if isinstance(value, dict) and set(value) == {"result"} and isinstance(value["result"], str):
            value = json.loads(value["result"])
    except (ValueError, RecursionError):
        return None
    return value if is_savings_envelope(value) else None


def text_is_envelope(text) -> bool:
    """True if `text`, returned bare by a tool, would be taken by the hook for a
    savings envelope, in any shape the hook accepts: handed over as it is, or
    after the MCP layer wraps it as `{"result": "<text>"}`."""
    if not isinstance(text, str):
        return False
    if parse_envelope_text(text) is not None:
        return True
    try:
        wrapped = json.dumps({"result": text})
    except (ValueError, TypeError):
        return False
    return parse_envelope_text(wrapped) is not None
