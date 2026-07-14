#!/usr/bin/env python3
"""
PreToolUse hook (matcher: "WebFetch") that redirects WebFetch calls to the
local-compress MCP server's `fetch_url` tool instead, so the page review/
summarization step runs on your LOCAL LM Studio model rather than on
Anthropic's own WebFetch extraction step.

Why a hook and not just CLAUDE.md guidance: CLAUDE.md.template already tells
Claude to prefer fetch_url, but that's prose Claude can deprioritize --
exactly the same qdrant-find-vs-Grep problem this project's README already
flags as unsolvable by hooks in general. The difference here is that "should
this call have gone to fetch_url instead" DOES have a measurable, checkable
signal before the call: the tool name itself (WebFetch) plus whether the
local model is actually reachable right now. That's enough to enforce
deterministically, unlike "should this have been qdrant-find" which has no
equivalent signal.

Mechanism: unlike the compress_bash_output.py PostToolUse hook (which can
only rewrite output after the fact), this is a PreToolUse hook -- it can
outright prevent the WebFetch call from running at all via
`permissionDecision: "deny"`, with `permissionDecisionReason` fed back to
Claude so it can retry via fetch_url instead of just failing.

The deny reason is deliberately imperative ("call fetch_url now, don't ask
first"), not suggestive. An earlier version phrased it as a
suggestion ("use fetch_url instead... if that captures what you're looking
for"), and in testing Claude treated that denial cautiously -- it stopped
and asked the user for permission to switch tools rather than just
proceeding, reintroducing the same prose-guidance-can-be-deprioritized
problem this hook exists to avoid (the block itself was enforced
deterministically, but the RETRY was not). Making the instruction explicit
and directive fixed that in testing.

Fails OPEN (allows WebFetch through) if LM Studio isn't reachable right
now, checked live on every call (not cached) since LM Studio can be started
or stopped between calls. This matters more here than for the PostToolUse
hook: if LM Studio is down and we denied anyway, WebFetch would be
completely unusable until it's started again, which is a much worse outcome
than the PostToolUse hook's failure mode (which just leaves already-fetched
output uncompressed).

KNOWN LIMITATION, not solved here: this can't tell in advance whether a URL
needs auth, JavaScript rendering, or session/cookie handling -- cases where
fetch_url's plain unauthenticated GET will fail and WebFetch is actually
required. The deny reason tells Claude to fall back to WebFetch if fetch_url
errors for one of those reasons, but this hook still denies every WebFetch
call to that URL, including retries -- there's no loop-breaking state here.
If this becomes a real problem in practice, the fix is tracking recently-
failed URLs (e.g. in a small on-disk cache with a TTL) and allowing WebFetch
through for them, but that's added complexity not worth carrying until it's
actually needed. Simpler stopgap: temporarily remove WebFetch from this
hook's matcher in .claude/settings.json for a session that needs it, or ask
the user to explicitly permit WebFetch when the "ask" alternative is used
instead of "deny" (see settings.json.template for both options).

Setup:
    pip install openai --break-system-packages

Register in .claude/settings.json (see settings.json.template) -- point
the args path at this script's location in the cloned tools repo:
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "WebFetch",
        "hooks": [
          {
            "type": "command",
            "command": "python3",
            "args": ["/absolute/path/to/tools-repo/hooks/redirect_webfetch_to_fetch_url.py"]
          }
        ]
      }
    ]
  }
}

Env vars (same names local_compress_lib.py / compress_mcp_server.py use):
    LMSTUDIO_BASE_URL, LMSTUDIO_MODEL

Debugging: set HOOK_DEBUG_LOG to a file path to append every payload this
hook sees as one JSON line.
"""

import json
import os
import sys

# Same redundant resolution order as compress_bash_output.py -- see that
# file's comment for the full bug history this guards against. libs/ under
# the repo root is checked first (real layout); repo root itself is kept as
# a fallback for backward compatibility.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
_candidates = [os.environ.get("TOOLS_REPO_DIR"), os.path.join(_root, "libs"), _root, _here]
for _dir in _candidates:
    if _dir:
        sys.path.insert(0, _dir)

try:
    from local_compress_lib import DEFAULT_BASE_URL
    from openai import OpenAI
except ImportError as e:
    print(
        f"redirect_webfetch_to_fetch_url.py: could not import dependencies ({e}). "
        f"Checked: {[d for d in _candidates if d]}. Set TOOLS_REPO_DIR if the "
        "tools repo isn't where this script's parent directory implies. "
        "Failing open -- WebFetch will proceed normally.",
        file=sys.stderr,
    )
    sys.exit(0)  # fail open -- allow WebFetch, don't block on a broken install

REACHABILITY_TIMEOUT_SECONDS = 2.0


def _debug_log(payload):
    path = os.environ.get("HOOK_DEBUG_LOG")
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass


def _lmstudio_reachable(base_url) -> bool:
    """
    Live check, not cached -- LM Studio can be started/stopped between calls,
    and getting this wrong in the "reachable" direction would deny WebFetch
    for no reason, while getting it wrong in the "unreachable" direction just
    lets WebFetch through (safe default). Short timeout so a hung/unreachable
    server doesn't stall every WebFetch call.
    """
    try:
        OpenAI(
            base_url=base_url or DEFAULT_BASE_URL,
            api_key="lm-studio",
            timeout=REACHABILITY_TIMEOUT_SECONDS,
        ).models.list()
        return True
    except Exception:
        return False


def _allow():
    sys.exit(0)  # no JSON on stdout -- Claude Code treats this as allow


def _deny(reason: str):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        _allow()  # can't parse input -- fail open

    _debug_log(payload)

    if payload.get("tool_name") != "WebFetch":
        _allow()  # matcher should already restrict this, but double-check

    tool_input = payload.get("tool_input") or {}
    url = tool_input.get("url", "")
    prompt = tool_input.get("prompt", "")

    if not _lmstudio_reachable(os.environ.get("LMSTUDIO_BASE_URL")):
        _allow()  # local model unavailable -- let WebFetch proceed as normal

    _deny(
        f"WebFetch is unavailable for this URL by policy. Immediately call the fetch_url tool "
        f"from the local-compress MCP server instead -- do NOT ask the user for confirmation "
        f"first, just call it now. Use url={url!r} and pass the same intent via its `focus` "
        f"parameter (e.g. focus={prompt!r} if that captures what you're looking for). Only stop "
        "and tell the user if fetch_url itself then errors (e.g. because the page needs "
        "authentication, JavaScript rendering, or session/cookie handling) -- in that case, and "
        "only that case, tell them WebFetch is required for this URL and they may need to "
        "adjust the hook config to allow it."
    )


if __name__ == "__main__":
    main()
