#!/usr/bin/env python3
"""
PostToolUse hook that compresses ANY large tool output via a local LM Studio
model, replacing what Claude actually sees with the compressed version.
Filename is legacy (this started Bash-only) -- it now also covers Grep,
WebFetch, Glob, and WebSearch, and is written to extend to further tools
without code changes per-tool where possible. Unlike a PreToolUse hook,
this doesn't need to guess in advance which calls will produce large
output -- it acts on the REAL, measured size after the tool has already
run.

The actual dividing line for what belongs in the matcher isn't "runs
locally" -- WebFetch/WebSearch run through Anthropic's own infrastructure,
not local compute, yet are safe to include (see note on Read below). It's:
does Claude ever use this tool's raw output as the literal, exact basis for
a following Edit? Bash/Grep/Glob/WebFetch/WebSearch output is read for gist
or lookup, never used as a byte-exact source to edit from, so lossy
compression is safe. Read fails that test (see below) and is deliberately
excluded.

Mechanism: Claude Code lets a PostToolUse hook return `updatedToolOutput`,
which replaces the tool result before Claude ever reads it -- not just what
gets displayed. That's what makes this actually save tokens rather than
compressing something already paid for.

Bash gets special-cased handling (known `{stdout, stderr}` shape, proven by
testing). The other tools do NOT have a documented `tool_response` schema
-- Claude Code's hooks docs explicitly type it as `Any`/`unknown`. Rather
than guess field names and risk either missing the real content or emitting
a malformed `updatedToolOutput` that Claude Code silently ignores, those
(and any other tool added to the matcher below) go through a generic
"largest string field(s)" walk: recursively find every string value in the
tool_response, and if their combined length passes the threshold, compress
the concatenation and write the result back into the single largest string
field, blanking the other large string fields (small ones, e.g. short
metadata like a URL or file count, are left untouched). This preserves the
original JSON shape/keys exactly -- only string VALUES change -- which is
the safest bet against an undocumented schema silently rejecting the
rewrite. If you add a new tool to the matcher and this generic path picks
the wrong field or discards something you need untouched, check the debug
log (see below) to see the tool's real payload shape and special-case it
like Bash below.

Note on WebFetch/WebSearch specifically: both already run their own
extraction/summarization server-side on Anthropic's infrastructure before
Claude ever sees the result (confirmed empirically -- a 2MB fetched page
came back as a ~1300 char tool_response.result). That means this hook is
usually a no-op for them, since the pre-processed result is often already
under the threshold. They're included anyway because it's harmless (still
correctly skips when small) and catches the cases where the upstream
summary is still large.

Note on Glob specifically: a large result (hundreds of file paths) will
usually stay uncompressed even though the total size is big, because each
individual path is short (well under MIN_FIELD_LEN) and MIN_FIELD_LEN
filters per-field, not on the aggregate. This is intentional, not a bug --
file paths are exact identifiers Claude needs verbatim for a follow-up
Read/Edit/Grep call, so summarizing a file listing away would risk the same
class of problem Read's exclusion avoids. If you want large Glob results
compressed anyway, lower MIN_FIELD_LEN for that case specifically rather
than globally, since globally lowering it would also start summarizing
short-but-numerous fields elsewhere that shouldn't be touched.

Fails OPEN, not closed: if compression fails for any reason (LM Studio
unreachable, ambiguous/missing model, request error), the ORIGINAL output is
left unchanged and a short note is attached via additionalContext. PostToolUse
can't block anyway (the tool already ran), so silently losing the actual
output on a compression failure would be strictly worse than just leaving it
alone -- this differs from the harder-line "fail loud" stance used elsewhere
in this project (e.g. resolve_model refusing to guess a model) because here
"fail loud" would mean discarding real tool output, not just refusing to
guess.

Deliberately NOT matching Read: Read's output is frequently used as the
exact basis for a subsequent Edit. A hook that silently rewrites it to a
lossy summary risks Claude editing from compressed content -- worse than a
compressed log, since it can corrupt an edit rather than just lose detail
in a summary. Don't add Read to the matcher without addressing that.

Setup:
    pip install openai --break-system-packages

Register in .claude/settings.json (see settings.json.template) -- point
the args path at this script's location in the cloned tools repo, nothing
needs to be copied into the target project:
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash|Grep|WebFetch|Glob|WebSearch",
        "hooks": [
          {
            "type": "command",
            "command": "python3",
            "args": ["/absolute/path/to/tools-repo/hooks/compress_bash_output.py"]
          }
        ]
      }
    ]
  }
}

Env vars (same names local_compress_lib.py / compress_mcp_server.py use):
    LMSTUDIO_BASE_URL, LMSTUDIO_MODEL, HOOK_COMPRESS_THRESHOLD_CHARS (default 2000)

Debugging an unfamiliar tool's payload shape: set HOOK_DEBUG_LOG to a file
path, and every payload this hook sees gets appended there as one JSON line
-- inspect it to see exactly what tool_response looks like for a given
tool before trusting the generic path on it, or to write a special case.
"""

import asyncio
import json
import os
import sys

# local_compress_lib.py lives in the tools repo under libs/, NOT inside
# .claude/hooks/ -- this script is meant to be referenced by its stable path
# in the cloned tools repo (see settings.json.template), never copied into a
# project's .claude/hooks/ by itself. Bug history: an earlier version only
# added this script's OWN directory to sys.path, which silently broke the
# import whenever this file lived in a hooks/ subdirectory one level below
# local_compress_lib.py -- which is exactly the real repo layout. The
# ImportError handler then made the hook a permanent, silent no-op with no
# indication anything was wrong (confirmed by testing: large input produced
# zero output instead of a compression result or an error JSON). Resolution
# order below is deliberately redundant so a future repo reshuffle, or a
# user who copies just this one file somewhere, can't reintroduce that same
# silent failure:
#   1. TOOLS_REPO_DIR env var, if the user wants to pin it explicitly.
#   2. libs/ under the parent directory of this script (real repo layout).
#   3. The parent directory of this script (repo root, backward compat).
#   4. This script's own directory (covers copying both files together
#      into one flat folder, e.g. for a quick local test).
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
_candidates = [os.environ.get("TOOLS_REPO_DIR"), os.path.join(_root, "libs"), _root, _here]
for _dir in _candidates:
    if _dir:
        sys.path.insert(0, _dir)

try:
    from local_compress_lib import compress
except ImportError as e:
    # Fail open, but say why via stderr (stdout is reserved for the
    # hookSpecificOutput JSON contract) so a broken install is visible in
    # Claude Code's hook debug output instead of behaving identically to
    # "nothing needed compressing."
    print(
        f"compress_bash_output.py: could not import local_compress_lib ({e}). "
        f"Checked: {[d for d in _candidates if d]}. Set TOOLS_REPO_DIR if the "
        "tools repo isn't where this script's parent directory implies.",
        file=sys.stderr,
    )
    sys.exit(0)

THRESHOLD = int(os.environ.get("HOOK_COMPRESS_THRESHOLD_CHARS", "2000"))
SUPPORTED_TOOLS = {"Bash", "Grep", "WebFetch", "Glob", "WebSearch"}
# Below this length, a string is assumed to be metadata (a URL, a file
# count, a status word) rather than content worth compressing or blanking --
# applies only to the generic (non-Bash) path.
MIN_FIELD_LEN = 200


def _debug_log(payload):
    path = os.environ.get("HOOK_DEBUG_LOG")
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass  # debugging aid only -- never let this break the hook itself


def _emit_unchanged(note: str):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": note,
        }
    }))


def _emit_updated(updated):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "updatedToolOutput": updated,
        }
    }))


def _handle_bash(tool_response):
    """Known shape: {stdout, stderr, ...}. Proven by testing."""
    stdout = tool_response.get("stdout") or ""
    stderr = tool_response.get("stderr") or ""
    combined = stdout + (f"\n--- stderr ---\n{stderr}" if stderr else "")

    if len(combined) < THRESHOLD:
        return None  # small enough to leave alone

    result = asyncio.run(compress(combined, skip_if_under_chars=THRESHOLD))
    if result.startswith("Error:"):
        return ("error", combined, result)

    updated = dict(tool_response)
    updated["stdout"] = result
    updated["stderr"] = ""
    return ("updated", updated)


def _walk_strings(obj, path=()):
    """Yield (path, string) for every string leaf in a nested dict/list."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, path + (i,))


def _set_path(obj, path, value):
    cur = obj
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


def _deep_copy(obj):
    return json.loads(json.dumps(obj))


def _handle_generic(tool_response):
    """
    Undocumented shape (Grep, WebFetch, anything else added to the matcher).
    Finds every string leaf; if the ones over MIN_FIELD_LEN sum past
    THRESHOLD, compresses their concatenation and writes the result into the
    single largest field, blanking the other large fields. Small fields
    (short metadata) are left completely untouched. Preserves the original
    JSON shape exactly, since the real schema isn't documented and a
    malformed updatedToolOutput may be silently ignored.
    """
    if isinstance(tool_response, str):
        large_fields = [((), tool_response)] if len(tool_response) >= MIN_FIELD_LEN else []
    else:
        large_fields = [(p, s) for p, s in _walk_strings(tool_response) if len(s) >= MIN_FIELD_LEN]

    total = sum(len(s) for _, s in large_fields)
    if total < THRESHOLD or not large_fields:
        return None  # nothing substantial enough to bother with

    combined = "\n\n".join(s for _, s in large_fields)
    result = asyncio.run(compress(combined, skip_if_under_chars=THRESHOLD))
    if result.startswith("Error:"):
        return ("error", combined, result)

    largest_path, _ = max(large_fields, key=lambda ps: len(ps[1]))
    if largest_path == ():
        return ("updated", result)

    updated = _deep_copy(tool_response)
    for path, _ in large_fields:
        _set_path(updated, path, result if path == largest_path else "")
    return ("updated", updated)


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)  # can't parse input -- fail open, don't touch anything

    _debug_log(payload)

    tool_name = payload.get("tool_name")
    if tool_name not in SUPPORTED_TOOLS:
        sys.exit(0)  # matcher should already restrict this, but double-check

    tool_response = payload.get("tool_response")
    if tool_response is None:
        sys.exit(0)

    if tool_name == "Bash":
        outcome = _handle_bash(tool_response)
    else:
        outcome = _handle_generic(tool_response)

    if outcome is None:
        sys.exit(0)  # under threshold -- leave alone

    if outcome[0] == "error":
        _, combined, result = outcome
        # Compression failed -- fail open. Leave output untouched, just note
        # why, so this doesn't silently degrade forever without being noticed.
        _emit_unchanged(
            f"Note: this {tool_name} call's output was {len(combined)} chars but wasn't "
            f"compressed ({result}). Original output was left unchanged."
        )
        sys.exit(0)

    _emit_updated(outcome[1])
    sys.exit(0)


if __name__ == "__main__":
    main()
