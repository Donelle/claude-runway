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
output uncompressed). "Isn't reachable" also covers "is reachable but no
model could actually be resolved" (zero or 2+ models loaded, nothing
pinned) -- see _lmstudio_reachable's docstring (issue #29): denying WebFetch
and then having fetch_url fail too for that reason breaks this exact same
design goal just as badly as LM Studio being fully down.

KNOWN LIMITATION (now addressed): this can't tell in advance whether a URL
needs auth, JavaScript rendering, or session/cookie handling -- cases where
fetch_url's plain unauthenticated GET will fail and WebFetch is actually
required. The fix (issue #64): a small on-disk SQLite cache records each
URL the hook denies, with a timestamp. When the hook sees that same URL
again within the TTL (default: 1 hour, set by
CLAUDE_RUNWAY_WEBFETCH_FAILED_URL_TTL in seconds), it allows WebFetch
through, because the retry implies fetch_url was tried and failed between
the two calls. After allowing, the entry is deleted so future calls are
denied again normally. The cache reuses the shared libs/cache_db.py store
(same pattern as libs/qdrant_collection_hints.py). Any cache I/O error
fails open -- WebFetch proceeds as if the cache were empty.

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
    CLAUDE_RUNWAY_LMSTUDIO_URL, CLAUDE_RUNWAY_LMSTUDIO_MODEL

    Must be exported at the OS/shell level to reach this script -- hook entries
    in .claude/settings.json have no `env` field and can't see .mcp.json's env
    block, so these have to be set in BOTH places with matching values. Formerly
    LMSTUDIO_BASE_URL / LMSTUDIO_MODEL; those names are no longer read.

Debugging: set HOOK_DEBUG_LOG to a file path to append every payload this
hook sees as one JSON line.
"""

import datetime
import hashlib
import json
import os
import sqlite3
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
    from local_compress_lib import DEFAULT_BASE_URL, DEFAULT_MODEL, stale_env_warning
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

# How long (seconds) a denied URL stays in the cache before expiring. If a
# second WebFetch call for the same URL arrives within this window, the hook
# assumes fetch_url was already tried (and failed) and allows WebFetch through.
# Override via CLAUDE_RUNWAY_WEBFETCH_FAILED_URL_TTL (int or float seconds).
_DEFAULT_FAILED_URL_TTL_SECONDS = 3600.0


def _failed_url_ttl() -> float:
    """
    Returns the configured TTL in seconds, falling back to the default.
    Invalid values (non-numeric, nan, inf, negative, or zero) are silently
    ignored and fall back to the default -- nan/inf would make deny markers
    effectively never expire, while negative/zero would prevent any retry
    from being allowed.
    """
    raw = os.environ.get("CLAUDE_RUNWAY_WEBFETCH_FAILED_URL_TTL")
    if raw:
        try:
            value = float(raw)
            import math
            if math.isfinite(value) and value > 0:
                return value
        except ValueError:
            pass
    return _DEFAULT_FAILED_URL_TTL_SECONDS


class _DeniedUrlCache:
    """
    Records URLs this hook has denied (sent fetch_url redirect for), keyed by
    a SHA-256 digest of the URL (never the URL itself, since URLs can contain
    credentials or signed query strings), with a timestamp. When the hook sees
    the same URL again within the TTL, it allows WebFetch through as a one-shot
    skip — the retry implies fetch_url was tried and failed.

    Uses the shared cache_db.py SQLite store (disposable, fine to delete).
    Owns its own table `webfetch_denied_urls`. Fails open throughout:
    any SQLite or OS error is caught, logged to stderr, and treated as "no
    cached entry." For check_and_clear_if_retrying this means returning False
    (treat as miss), and main() then falls through to record_denied + _deny —
    consistent with the hook's overall principle that an unavailable cache must
    never make the retry-allow path unreachable (that would reintroduce the
    unresolvable loop this feature exists to prevent); the miss just means the
    skip can't fire this call. record_denied swallowing errors has no impact on
    the allow path, since check_and_clear_if_retrying already failed open.

    All database operations that both read AND write (check_and_clear_if_retrying)
    use BEGIN IMMEDIATE to prevent two concurrent hook processes from both
    reading the same row and both returning True — each process either wins the
    write lock and gets the one-shot skip, or blocks until the other finishes
    and then finds the row already deleted.
    """

    _TABLE_DDL = """
        CREATE TABLE IF NOT EXISTS webfetch_denied_urls (
            url_digest TEXT PRIMARY KEY,
            denied_at TEXT NOT NULL
        )
    """

    @staticmethod
    def _digest(url: str) -> str:
        """SHA-256 hex digest of the URL. Never stores the URL itself."""
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _connect(self):
        """Opens the shared cache DB and ensures this table exists."""
        # cache_db is in libs/, which is already on sys.path from the
        # redundant resolution block at the top of this module.
        from cache_db import connect as _connect_shared_cache  # noqa: PLC0415
        conn = _connect_shared_cache()
        conn.execute(self._TABLE_DDL)
        return conn

    def record_denied(self, url: str) -> bool:
        """
        Upserts a denial record for the SHA-256 digest of `url` with the
        current UTC timestamp. Called immediately before the hook emits the
        deny decision.

        Returns True on success, False if the write failed. The CALLER must
        allow WebFetch on False: if the record can't be written, the next
        WebFetch call for the same URL will find nothing in the cache and
        receive a deny again -- making the loop permanently unbreakable. An
        unwritable cache is strictly worse than no cache at all, so failing
        to write a denial record means the corresponding deny must be skipped.
        """
        try:
            digest = self._digest(url)
            conn = self._connect()
            with conn:
                conn.execute(
                    "INSERT INTO webfetch_denied_urls (url_digest, denied_at) VALUES (?, ?) "
                    "ON CONFLICT(url_digest) DO UPDATE SET denied_at = excluded.denied_at",
                    (digest, datetime.datetime.now(datetime.timezone.utc).isoformat()),
                )
            conn.close()
            return True
        except (sqlite3.Error, OSError, ImportError) as e:
            print(
                f"[claude-runway] could not record denied URL: {e}",
                file=sys.stderr,
            )
            return False

    def check_and_clear_if_retrying(self, url: str, ttl_seconds: float) -> bool:
        """
        Returns True if `url` is in the cache with a denied_at timestamp
        within `ttl_seconds` of now (meaning this is a retry after fetch_url
        failed). If True, the cache entry is deleted so the next WebFetch
        call for this URL is denied again normally.

        Uses BEGIN IMMEDIATE so the read and delete are atomic — two concurrent
        hook processes cannot both see the same entry and both return True.

        Returns False if the URL is absent, expired, or the DB is unavailable.
        Fails open on any error (see class docstring).
        """
        conn = None
        try:
            digest = self._digest(url)
            conn = self._connect()
            # BEGIN IMMEDIATE acquires a write lock immediately, preventing
            # concurrent readers from also seeing (and consuming) this entry.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT denied_at FROM webfetch_denied_urls WHERE url_digest = ?",
                (digest,),
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return False
            denied_at = datetime.datetime.fromisoformat(row[0])
            # Ensure timezone-aware comparison regardless of how the stored
            # value was serialized (always UTC from record_denied, but be safe).
            if denied_at.tzinfo is None:
                denied_at = denied_at.replace(tzinfo=datetime.timezone.utc)
            age = (datetime.datetime.now(datetime.timezone.utc) - denied_at).total_seconds()
            # Delete this entry unconditionally (whether expired or not). Each
            # lookup cleans up its own row -- the expected pattern is one
            # record_denied followed by exactly one check (the retry), so
            # this per-URL delete is sufficient to prevent the table from
            # growing unboundedly for URLs that DO get retried.
            conn.execute(
                "DELETE FROM webfetch_denied_urls WHERE url_digest = ?", (digest,)
            )
            conn.execute("COMMIT")
            if age > ttl_seconds:
                # Expired — treat as a miss (WebFetch will be denied normally).
                return False
            # Within TTL: this is a retry after fetch_url failed. One-shot skip.
            return True
        except (sqlite3.Error, OSError, ImportError, ValueError) as e:
            print(
                f"[claude-runway] could not check denied-URL cache: {e}",
                file=sys.stderr,
            )
            return False
        finally:
            # Always release the connection (and any held write lock) so the
            # next record_denied call in main() can open its own connection
            # without blocking on SQLite's busy timeout.
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass


_denied_url_cache = _DeniedUrlCache()


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
    True only when LM Studio is not just reachable, but ALSO in a state
    where fetch_url's own model resolution (local_compress_lib.resolve_model,
    called with no explicit model -- exactly how fetch_url resolves when
    Claude doesn't pass model=, which the deny message below never tells it
    to) would actually succeed. Denying WebFetch and then having fetch_url
    fail too -- zero or 2+ models loaded, with nothing pinned -- defeats
    this hook's "a stopped local model never makes WebFetch unusable" design
    goal exactly as badly as denying while LM Studio is fully down (issue #29).

    Mirrors resolve_model's own decision order rather than calling it
    directly or duplicating its logic wholesale:
      1. Reachability is checked first and unconditionally (the try/except
         below) -- if this raises, resolve_model is never even consulted.
         This matters because resolve_model's OWN pinned-model branch skips
         the reachability check entirely (it returns the pinned name without
         ever calling .models.list()) -- correct for resolve_model's actual
         job of picking a model string before a real completion call (where
         a real failure would surface then), but wrong here: this hook has
         no later call to fall back on, so a pin must not paper over LM
         Studio being fully down.
      2. An unmigrated old-named env var (stale_env_warning) means
         resolve_model would refuse with a hard error -- checked BEFORE the
         pinned-model check below, mirroring resolve_model's own order
         (issue #41 moved resolve_model's stale check to run unconditionally
         ahead of its explicit_model/DEFAULT_MODEL short-circuits; this hook
         must match that order or a pinned model with any other stale var
         set would report "reachable" here while fetch_url's own
         resolve_model call then hard-fails on the exact same stale check --
         denying native WebFetch and redirecting to a tool that immediately
         errors, worse than just letting WebFetch through).
      3. A pinned CLAUDE_RUNWAY_LMSTUDIO_MODEL (DEFAULT_MODEL) always
         resolves once reachable and non-stale, regardless of how many
         models are loaded -- matching resolve_model, which never checks the
         loaded count in that branch either.
      4. Otherwise, exactly one loaded model is required to auto-detect --
         zero or multiple both fail, same as resolve_model.
    Steps 2-4 reuse DEFAULT_MODEL/stale_env_warning directly from
    local_compress_lib (not re-derived) so this can't drift from
    resolve_model's real behavior, and reuse the SAME models.list() response
    already fetched for step 1 -- no second network round-trip.

    Live check, not cached -- LM Studio can be started/stopped between
    calls, and getting this wrong in the "resolvable" direction would deny
    WebFetch for no reason, while getting it wrong in the "not resolvable"
    direction just lets WebFetch through (safe default). Short timeout so a
    hung/unreachable server doesn't stall every WebFetch call.
    """
    try:
        models = OpenAI(
            base_url=base_url or DEFAULT_BASE_URL,
            api_key="lm-studio",
            timeout=REACHABILITY_TIMEOUT_SECONDS,
        ).models.list()
    except Exception:
        return False
    if stale_env_warning():
        return False
    if DEFAULT_MODEL:
        return True
    return len(models.data) == 1


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

    if not _lmstudio_reachable(os.environ.get("CLAUDE_RUNWAY_LMSTUDIO_URL")):
        _allow()  # local model unavailable -- let WebFetch proceed as normal

    # Check whether this URL was already denied recently (within the TTL). The
    # marker is written at deny-time (below), not at fetch_url-failure-time --
    # the hook can't directly observe fetch_url's outcome. The design trade-off:
    # if fetch_url succeeds, Claude doesn't retry WebFetch, so the stale marker
    # is never consumed; if fetch_url fails and Claude retries WebFetch as the
    # deny reason instructs, this check fires and allows it through. Read errors
    # are treated as a miss (no match) rather than a fail-open allow -- a read
    # error doesn't give us reason to believe fetch_url already failed.
    if _denied_url_cache.check_and_clear_if_retrying(url, _failed_url_ttl()):
        _allow()

    # Record the denial before emitting it, so a retry WebFetch call for this
    # URL (indicating fetch_url failed) is recognised as such by the check above.
    # If the write fails, allow WebFetch through: an unwritable cache means the
    # next WebFetch retry would find nothing and be denied again -- permanently
    # breaking the loop this feature exists to prevent. Better to let this call
    # through than to create a URL the hook can never unblock.
    if not _denied_url_cache.record_denied(url):
        _allow()

    _deny(
        f"WebFetch is unavailable for this URL by policy. Immediately call the fetch_url tool "
        f"from the local-compress MCP server instead -- do NOT ask the user for confirmation "
        f"first, just call it now. Use url={url!r} and pass the same intent via its `focus` "
        f"parameter (e.g. focus={prompt!r} if that captures what you're looking for). If "
        "fetch_url itself then errors because the page needs authentication, JavaScript "
        "rendering, or session/cookie handling, immediately retry the original WebFetch call "
        "for this URL -- the hook will allow it through on that retry. Do NOT stop and ask the "
        "user; just proceed: fetch_url first, and if that errors, WebFetch next."
    )


if __name__ == "__main__":
    main()
