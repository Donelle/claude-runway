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

When CLAUDE_RUNWAY_TRACK_SAVINGS=1, this hook is also the SOLE writer for
the opt-in savings tracker's per-session ledger (see libs/savings_ledger.py):
it always has session_id (hooks receive it on stdin; MCP tool calls do not),
so it's the one place that can attribute a compression event to a session.
For tools it compresses itself (Bash/Grep/Glob/etc. below), it logs directly.
For the local-compress MCP server's own credited tools (compress_file,
compress_command_output, fetch_url -- matcher extended to
mcp__local-compress__*), the server can't write the ledger itself (no
session_id), so it appends a small machine-readable footer to its result
instead; this hook strips that footer before Claude ever sees it and logs
the numbers it carries. When tracking is off, none of this runs and behavior
is unchanged from before this feature existed.

The actual dividing line for what belongs in the matcher isn't "runs
locally" -- WebFetch/WebSearch run through Anthropic's own infrastructure,
not local compute, yet are safe to include (see note on Read below). It's:
does Claude ever use this tool's raw output as the literal, exact basis for
a following Edit? Bash/Grep/Glob/WebFetch/WebSearch output is read for gist
or lookup, never used as a byte-exact source to edit from, so lossy
compression is safe. Read fails that test (see below) and is deliberately
excluded.

That same question gets asked a SECOND time, per-command, within Bash --
because unlike the other matched tools, "Bash output" is not one kind of
output. `dotnet build` is a log to skim; `git rev-parse HEAD` is a 40-char
identifier where one wrong nibble is a silent, confident lie. So before a
Bash call is compressed, its command string is checked by
_is_exactness_critical() (see the exempt-list block below), and an
exactness-critical command exits silently with the real output intact --
the same way an under-threshold call does, since raw output is the correct
result here, not something to announce. A size threshold cannot make this
distinction on its own: it sees only bytes, and the dangerous cases are
frequently the SMALL ones, where compression saves a few hundred tokens and
risks dropping the single line that carried the answer. A trailing
`# compress-ok` on the command opts any exempt command back in.

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
tool_response, group them by which sibling RECORD they belong to (e.g.
`results[3]` vs `results[4]` in a WebSearch-shaped payload -- see
_record_group_key), and compress/replace within each record independently.
A response with no sibling record structure at all (a flat shape, or one
dominant field) falls back to the original whole-blob behavior: compress
everything together and write the result back into the single largest
string field, blanking the other large string fields (small ones, e.g.
short metadata like a URL or file count, are left untouched either way).
This preserves the original JSON shape/keys exactly -- only string VALUES
change -- which is the safest bet against an undocumented schema silently
rejecting the rewrite. Per-record grouping exists because concatenating
across independent records and writing the result into just one of them
silently destroyed every other record's content (issue #38, reproduced
live against a 10-result WebSearch payload: 9 of 10 snippets got blanked
to "" with no error or indication). If you add a new tool to the matcher
and this generic path picks the wrong field or discards something you need
untouched, check the debug log (see below) to see the tool's real payload
shape and special-case it like Bash below.

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
unreachable, ambiguous/missing model, request error, OR any other unexpected
exception anywhere in main()'s dispatch after payload parsing -- see the
top-level try/except wrapping _dispatch() below, issue #44), the ORIGINAL
output is left unchanged and a short note is attached via additionalContext.
PostToolUse can't block anyway (the tool already ran), so silently losing the
actual output on a compression failure would be strictly worse than just
leaving it alone -- this differs from the harder-line "fail loud" stance used
elsewhere in this project (e.g. resolve_model refusing to guess a model)
because here "fail loud" would mean discarding real tool output, not just
refusing to guess.

Deliberately NOT matching Read: Read's output is frequently used as the
exact basis for a subsequent Edit. A hook that silently rewrites it to a
lossy summary risks Claude editing from compressed content -- worse than a
compressed log, since it can corrupt an edit rather than just lose detail
in a summary. Don't add Read to the matcher without addressing that.

Setup:
    pip install openai --break-system-packages

Register in .claude/settings.json (see settings.json.template) -- point
the args path at this script's location in the cloned tools repo, nothing
needs to be copied into the target project. The full recommended matcher
is in settings.json.template, which extends the base set below with vetted
MCP tool allowlists for GitHub (individually vetted against the edit-basis
rule -- get_file_contents and get_pull_request_files excluded, see dispatch
comments below) and Splunk:
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash|Grep|WebFetch|Glob|WebSearch|mcp__local-compress__.*",
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
    CLAUDE_RUNWAY_LMSTUDIO_URL, CLAUDE_RUNWAY_LMSTUDIO_MODEL,
    CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS (default 2000),
    CLAUDE_RUNWAY_TRACK_SAVINGS (opt-in savings tracker, off by default -- see
    libs/savings_ledger.py), CLAUDE_RUNWAY_SAVINGS_DB (optional override of
    the savings DB's location; must match the value set in compress_mcp_server.py's
    env if you set it at all -- default needs no coordination)

    ALL of these must be exported at the OS/shell level to reach this script --
    a hook entry in .claude/settings.json has no `env` field, so it inherits the
    parent environment and cannot see .mcp.json's env block. The MCP server
    actually inherits the same full shell environment too (not a small
    allowlist -- see README's "Why two places" section and issue #117), but
    only for a key genuinely absent from its own env block; this toolkit's
    shipped .mcp.json always declares CLAUDE_RUNWAY_LMSTUDIO_URL/
    CLAUDE_RUNWAY_LMSTUDIO_MODEL/CLAUDE_RUNWAY_TRACK_SAVINGS/
    CLAUDE_RUNWAY_SAVINGS_DB explicitly, so those four need setting in BOTH
    places with matching values. CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS is
    the exception -- this script is its only reader, so it's shell-only; an
    .mcp.json copy would just be inert. Formerly
    LMSTUDIO_BASE_URL / LMSTUDIO_MODEL / HOOK_COMPRESS_THRESHOLD_CHARS; those
    names are no longer read at all -- see local_compress_lib.stale_env_warning,
    which this script reports on stderr at import time.

Debugging an unfamiliar tool's payload shape: set HOOK_DEBUG_LOG to a file
path, and every payload this hook sees gets appended there as one JSON line
-- inspect it to see exactly what tool_response looks like for a given
tool before trusting the generic path on it, or to write a special case.
"""

import asyncio
import json
import os
import re
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
    from local_compress_lib import compress, estimate_tokens, stale_env_warning
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

# savings_ledger is optional -- an older checkout without libs/savings_ledger.py
# should keep working exactly as before, just without ledger logging.
try:
    import savings_ledger
    _SAVINGS_LEDGER_AVAILABLE = True
except ImportError:
    _SAVINGS_LEDGER_AVAILABLE = False


def _record_savings_event(session_id, tool, raw_tokens, out_tokens, credited, source="", project=None):
    """Best-effort -- a ledger-logging bug must never break the hook's real job
    (making sure Claude sees the right output). `project` is threaded through to
    record_event's own `project` field -- see that function's docstring and
    issue #35: this is what lets current_session_id()'s later project-filtered
    lookup avoid picking an unrelated live session from a different project."""
    if not (_SAVINGS_LEDGER_AVAILABLE and session_id and savings_ledger.tracking_enabled()):
        return
    try:
        savings_ledger.record_event(session_id, tool, raw_tokens, out_tokens, credited, source, project=project)
    except Exception:
        pass


THRESHOLD = int(os.environ.get("CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS", "2000"))

# Reported on stderr rather than as additionalContext, following the same
# precedent as the ImportError handler above: stdout is reserved for the
# hookSpecificOutput JSON contract, and this hook runs on every matched tool
# call, so surfacing a config warning to Claude every time would be pure noise.
# stderr shows up in Claude Code's hook debug output, which is where you'd look.
# The user-visible path is the fail-open note in main() -- see below.
_STALE_ENV = stale_env_warning()
if _STALE_ENV:
    print(f"compress_bash_output.py: {_STALE_ENV}", file=sys.stderr)
SUPPORTED_TOOLS = {"Bash", "Grep", "WebFetch", "Glob", "WebSearch"}
# MCP tools whose credited compression the server itself performs -- this hook's
# job for these is only to strip the machine-readable savings footer
# (SAVINGS_FOOTER_RE below) before Claude sees it, and log the
# numbers it carries. See compress_mcp_server.py's _append_savings_footer.
MCP_SAVINGS_TOOL_PREFIX = "mcp__local-compress__"
# Tool names (without the mcp__local-compress__ prefix) that already run
# their OWN compression -- with the caller's own focus/preserve_identifiers/
# preserve_sections -- regardless of whether a savings footer ends up on the
# result. compress_text never emits a footer at all (by design, since it's
# never credited toward savings); the other three omit one whenever
# CLAUDE_RUNWAY_TRACK_SAVINGS is off server-side even though real
# compression still happened. The no-footer fallback below must never
# re-compress these generically -- doing so would discard exactly what the
# first pass was told to preserve (PR #88 review: this is /my-compact's own
# real usage, which sets preserve_identifiers=preserve_sections=True).
_SELF_COMPRESSING_MCP_TOOLS = {"compress_file", "compress_command_output", "fetch_url", "compress_text"}
SAVINGS_FOOTER_RE = re.compile(r"\n?<!--CLAUDE_RUNWAY_SAVINGS:(\{.*?\})-->\s*\Z", re.DOTALL)
# Below this length, a string is assumed to be metadata (a URL, a file
# count, a status word) rather than content worth compressing or blanking --
# applies only to the generic (non-Bash) path.
MIN_FIELD_LEN = 200

# --- Exactness-critical Bash commands (never compressed) --------------------
#
# Extends the module docstring's Read-exclusion principle *within* Bash. The
# question there is "does Claude ever use this output as the literal, exact
# basis for what it does next?" -- answered globally for Read, but Bash is not
# one kind of output. `dotnet build` is a log to skim; `git rev-parse HEAD` is
# a 40-char identifier where a single wrong nibble is a silent, confident lie.
# A size threshold cannot tell those apart: it only sees bytes, and the
# dangerous outputs are frequently the SMALL ones, where a summary saves a few
# hundred tokens and risks dropping the one line that carried the answer.
#
# Bug history: a `git show HEAD:file | grep -c pattern; git log` call was
# compressed 1217 -> 924 chars and the grep count -- the entire point of the
# call -- was dropped from the summary. The surrounding prose was preserved,
# so the result read as complete. Compression that silently removes the
# answer while looking authoritative is worse than no compression, because
# there is nothing to notice.
#
# Matched at the START of a shell segment (after leading VAR=val assignments
# and sudo), so `git ...` is caught but `--message "regit"` is not.
_EXACT_CMDS = (
    # Version control is the tool reached for to VERIFY state, and a lossy
    # `git status`/`git diff` reads as authoritative while being wrong. The
    # transports are excluded: their output is progress noise, not state.
    r"git(?!\s+(?:clone|fetch|pull|push)\b)",
    # The output IS a count or a digest -- there is nothing to summarize.
    r"wc|cksum|md5|md5sum|sha1sum|sha256sum|sha512sum|shasum",
    # Exact identifiers, typically substituted straight into a follow-up call.
    r"pwd|realpath|readlink|basename|dirname|which|hostname|whoami|id",
    r"env|printenv",
    # Requested in a machine-readable form precisely because it gets parsed.
    r"jq|yq",
    # Version pins, where the digits are the entire payload.
    r"pip\s+freeze|npm\s+ls",
    # Encoded/parsed payloads: a summary is not decodable.
    r"base64|openssl|xxd|od",
)
# The leading-assignment part accepts quoted values, since `\S*` alone stops at
# the first space and would leave `FOO="a b" git status` unmatched -- silently
# dropping the exemption for a genuinely exactness-critical command. Quoted
# alternatives come first so `\S*` can't grab a bare `"a` and strand the rest.
_EXACT_CMD_RE = re.compile(
    r"^(?:\w+=(?:\"[^\"]*\"|'[^']*'|\S*)\s+)*(?:sudo\s+)?(?:" + "|".join(_EXACT_CMDS) + r")\b"
)

# Matched ANYWHERE in a segment: these flags are how a caller says "give me
# output I intend to parse exactly," regardless of which command they're on.
#
# The output-format VALUE (json/tsv/yaml) is matched case-insensitively via
# the scoped (?i:...) group, since `-o JSON` is realistic and would otherwise
# fall through. This is deliberately narrower than a blanket re.IGNORECASE on
# the whole regex: that would also fold `-h` into matching `-H`, which means
# something unrelated on real tools -- curl's custom-header flag, or df's
# human-readable-with-SI-units flag, or docker's daemon-host flag -- none of
# them a case variant of help.
_EXACT_FLAG_RE = re.compile(
    r"--porcelain\b|--json\b|--version\b|--query\b|--format[= ]"
    r"|-o\s+(?i:json|tsv|yaml)\b|--output[= ](?i:json|tsv|yaml)\b|\|\s*jq\b"
    # `grep -c` / `grep -rc` etc. -- counting, where the number is the answer.
    r"|\bgrep\b[^|;&]*\s-\w*c(?:\s|$)"
    # A dry run's entire output IS the preview of exactly what would happen --
    # the file list, the counts, the planned actions. Summarizing it defeats
    # the reason for running it. Found by dogfooding this repo: a summarized
    # `ingest_to_qdrant.py --dry-run` dropped the "Found 200 chunks" line and
    # every file_path/line_range row, keeping only the incidental file previews
    # quoted inside them -- structure discarded, payload retained.
    r"|--dry-run\b"
    # Help text is read to copy exact flag spellings into the next command, so
    # a rewrite that renames or omits a flag is worse than no help at all --
    # same reasoning as --version. Bundled forms (`ls -lh`) deliberately don't
    # match; only a standalone -h does.
    r"|--help\b|\s-h(?:\s|$)"
    # `gh api` (REST or GraphQL) always returns structured JSON meant to be
    # parsed field-by-field -- an exact ID substituted into a follow-up call,
    # or a comment/review body read verbatim to decide what to reply to or
    # fix -- never prose to skim. Same rationale as the `jq|yq` command
    # exemption above, just reached through `gh` instead of a literal pipe to
    # the `jq` binary (issue #190).
    #
    # Matched ANYWHERE rather than as a start-of-segment command (like
    # `jq|yq`) on purpose: the real call sites in this repo's own skills wrap
    # it in shell command substitution, e.g.
    #   COMMENT_IDS=$(gh api repos/.../pulls/<PR>/comments --paginate --jq "...")
    # A start-anchored match can't see past that `VAR=$(...)` wrapper, since
    # the leading-assignment sub-pattern only accepts a quoted string or a
    # whitespace-free token as the assigned value -- verified directly before
    # picking this approach, a start-anchored `gh\s+api` addition matched the
    # bare form but returned False on this exact wrapped form.
    #
    # Also deliberately broader than gating on `--jq`/`graphql` specifically:
    # issue #190's own suggested fix (those two flags only) was verified
    # incomplete before implementing -- this repo's `my-gh-pr-feedback` skill
    # calls `gh api .../comments --paginate` with neither flag at all, so a
    # narrower fix would have left that skill's primary feedback-fetch step
    # exactly as exposed as before. One `gh api` match covers all three real
    # shapes (plain, `--jq`-filtered, and `graphql`) plus any future one.
    r"|\bgh\s+api\b"
)

# Escape hatch: a genuinely large exempt output (a 5MB `git diff`) can opt back
# in with a trailing `# compress-ok` comment, so this list never becomes a
# reason to paste huge diffs into context.
#
# Anchored to the END of the command, because an escape hatch that fires from
# anywhere is a way to LOSE the exemption by accident: `git commit -m "fix the
# # compress-ok bug"` would otherwise opt a git command back into compression
# from inside a quoted argument. That's the exact failure this whole block
# exists to prevent, reintroduced through the opt-out. Requiring it trailing
# also matches how it's documented and how a real shell comment behaves.
_FORCE_COMPRESS_RE = re.compile(r"#\s*compress-ok\s*\Z")


def _is_exactness_critical(command: str) -> bool:
    """True if any segment of a (possibly compound) Bash command produces
    output whose value depends on being byte-exact.

    Checks every segment, not just the first: outputs of a pipeline or a
    `a; b` chain are interleaved into one stdout, so there is no way to
    compress part of it. One exactness-critical segment protects the whole.
    """
    if not command or _FORCE_COMPRESS_RE.search(command):
        return False
    for segment in re.split(r"\|\||&&|[;|\n]", command):
        segment = segment.strip()
        if segment and (_EXACT_CMD_RE.match(segment) or _EXACT_FLAG_RE.search(segment)):
            return True
    return False


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
    """Known shape: {stdout, stderr, ...}. Proven by testing.
    Returns ("updated", updated, raw_text, compressed_text) on success so the
    caller can log savings without redoing the compression work."""
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
    return ("updated", updated, combined, result)


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


def _record_group_key(path):
    """Return the path prefix identifying which sibling "record" a string
    field belongs to: everything up to and including the NEAREST (last,
    i.e. innermost/deepest) list index in its path -- not the first one.
    e.g. ('results', 3, 'snippet') -> ('results', 3), so every field under
    results[3] groups together, separately from results[4].

    Must be the last index, not the first: for a nested shape like
    ('sections', 0, 'results', 3, 'snippet'), stopping at the FIRST index
    (the 'sections' one) would give every result under sections[0] the same
    key ('sections', 0) regardless of its own 'results' index -- merging
    sections[0].results[3] and sections[0].results[4] back into one group,
    reintroducing the exact cross-record blending this grouping exists to
    prevent, just one level deeper (PR #107 review). Using the last index
    instead isolates the innermost repeating record, which is always the
    one actually holding these sibling fields.

    Fields with no list index anywhere in their path (flat top-level
    metadata, or a single nested object with no list ancestor at all) share
    the group key () -- there's no sibling record structure to separate
    them, so they fall back to being treated as one group.
    """
    last_index = None
    for i, key in enumerate(path):
        if isinstance(key, int):
            last_index = i
    if last_index is None:
        return ()
    return path[: last_index + 1]


def _compress_group(fields):
    """Compress one record's large string fields as a unit. Returns
    (raw_text, result) where result is either the compressed text or an
    "Error: ..." string from compress() -- both callers below unpack
    exactly these two values. Mirrors the single-blob logic _handle_generic
    used to apply globally -- now scoped to just this group's own fields."""
    combined = "\n\n".join(s for _, s in fields)
    result = asyncio.run(compress(combined, skip_if_under_chars=THRESHOLD))
    return combined, result


def _handle_generic(tool_response):
    """
    Undocumented shape (Grep, WebFetch, anything else added to the matcher).
    Finds every string leaf over MIN_FIELD_LEN; if they sum past THRESHOLD,
    compression proceeds -- but grouped by sibling record (see
    _record_group_key) rather than concatenated across the whole response,
    so multiple independent records (e.g. WebSearch's `results` list) never
    get blended into one combined summary that then gets written back into
    only ONE record while every sibling's own field is blanked to "" (issue
    #38 -- reproduced live: a 10-result WebSearch-shaped payload lost 9 of
    10 snippets this way, with no error or indication it happened).

    - Exactly one group (no sibling record structure, e.g. Grep's flat
      shape, or a single dominant field) -- unchanged from the original
      behavior: compress the concatenation, write it into the single
      largest field, blank the other large fields.
    - Multiple groups (sibling records) -- each group is compressed
      independently via compress()'s own skip_if_under_chars check, applied
      to that RECORD's own combined eligible content, same as the
      single-group case below but scoped to one record instead of the
      whole response -- NOT a per-field check, so two fields that are each
      individually under THRESHOLD can still get compressed together if
      their combined length crosses it (PR #107 review: the previous
      wording here read as a per-field guarantee, which it never was). A
      record whose combined content stays under THRESHOLD is left
      completely untouched rather than forcibly summarized just because ITS
      SIBLINGS pushed the RESPONSE-WIDE aggregate over the line. A record
      whose own combined content does cross THRESHOLD gets compressed and
      its largest field replaced, exactly like the single-group case,
      scoped to that record.
    Small fields (short metadata) are left completely untouched either way.
    Preserves the original JSON shape exactly, since the real schema isn't
    documented and a malformed updatedToolOutput may be silently ignored.

    Returns ("updated", updated, raw_text, compressed_text) on success,
    None if nothing ended up changing (every group was individually under
    threshold), or ("error", raw_text, message) if any group's compression
    call failed -- same three-shape contract _handle_bash uses, so the
    caller can log savings and fail open uniformly.
    """
    if isinstance(tool_response, str):
        large_fields = [((), tool_response)] if len(tool_response) >= MIN_FIELD_LEN else []
    else:
        large_fields = [(p, s) for p, s in _walk_strings(tool_response) if len(s) >= MIN_FIELD_LEN]

    total = sum(len(s) for _, s in large_fields)
    if total < THRESHOLD or not large_fields:
        return None  # nothing substantial enough to bother with

    groups = {}
    for path, s in large_fields:
        groups.setdefault(_record_group_key(path), []).append((path, s))

    if len(groups) <= 1:
        # No sibling record structure -- same whole-blob behavior as before
        # this fix: compress everything together, write the result into the
        # single largest field, blank the rest.
        combined, result = _compress_group(large_fields)
        if result.startswith("Error:"):
            return ("error", combined, result)

        largest_path, _ = max(large_fields, key=lambda ps: len(ps[1]))
        if largest_path == ():
            return ("updated", result, combined, result)

        updated = _deep_copy(tool_response)
        for path, _ in large_fields:
            _set_path(updated, path, result if path == largest_path else "")
        return ("updated", updated, combined, result)

    # Sibling records exist -- compress each one independently so one
    # record's summary never displaces another record's content.
    updated = _deep_copy(tool_response)
    raw_parts, compressed_parts = [], []
    for fields in groups.values():
        combined, result = _compress_group(fields)
        raw_parts.append(combined)
        if result.startswith("Error:"):
            # Fail open for the WHOLE response, same policy as every other
            # path here -- a partial rewrite (some records compressed, one
            # group's call failed) is worse than none, since it's not
            # obvious to Claude which fields are still trustworthy.
            return ("error", "\n\n".join(raw_parts), result)
        compressed_parts.append(result)
        if result == combined:
            continue  # this record's own fields were under THRESHOLD -- untouched
        largest_path, _ = max(fields, key=lambda ps: len(ps[1]))
        for path, _ in fields:
            _set_path(updated, path, result if path == largest_path else "")

    raw_text = "\n\n".join(raw_parts)
    compressed_text = "\n\n".join(compressed_parts)
    if compressed_text == raw_text:
        return None  # every group was individually under threshold -- true no-op
    return ("updated", updated, raw_text, compressed_text)


def _extract_savings_footer(text):
    """If text ends with a CLAUDE_RUNWAY_SAVINGS footer, return (parsed_dict,
    text_with_footer_stripped); otherwise (None, text) unchanged."""
    if not isinstance(text, str):
        return None, text
    m = SAVINGS_FOOTER_RE.search(text)
    if not m:
        return None, text
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None, text
    return data, text[:m.start()]


def _handle_mcp_savings_footer(tool_response):
    """
    Strips compress_mcp_server.py's machine-readable savings footer from a
    credited MCP tool's output (see compress_mcp_server.py's
    _append_savings_footer) before Claude ever sees it, returning the
    parsed savings data alongside the footer-stripped payload. The real
    shape of an MCP tool's tool_response isn't documented any more than any
    other non-Bash tool's is (see module docstring), so this walks every
    string leaf the same way _handle_generic does, rather than assuming a
    specific shape, and stops at the first leaf carrying the footer.
    Returns (None, None) if no footer is present -- i.e. tracking was off
    server-side, or nothing was actually compressed, so there's nothing to
    strip or log.
    """
    if isinstance(tool_response, str):
        data, stripped = _extract_savings_footer(tool_response)
        return (data, stripped) if data is not None else (None, None)

    for path, s in _walk_strings(tool_response):
        data, stripped = _extract_savings_footer(s)
        if data is not None:
            updated = _deep_copy(tool_response)
            _set_path(updated, path, stripped)
            return data, updated
    return None, None


def _finish_compression_outcome(outcome, tool_name, source, session_id, project):
    """
    Shared tail for turning an _handle_bash/_handle_generic `outcome` into
    the right hook response -- used by both the ordinary Bash/Grep/etc. path
    and the MCP-tool no-footer fallback below (issue #25). Always calls
    sys.exit(0) itself, matching every other terminal branch in main().
    """
    if outcome is None:
        sys.exit(0)  # under threshold -- leave alone

    if outcome[0] == "error":
        _, combined, result = outcome
        # Compression failed -- fail open. Leave output untouched, just note
        # why, so this doesn't silently degrade forever without being noticed.
        _emit_unchanged(
            f"Note: this {tool_name} call's output was {len(combined)} chars but wasn't "
            f"compressed ({result}). Original output was left unchanged."
            # Only attached on the failure path, where it's likely to be the actual
            # cause and the user is already being interrupted anyway.
            + (f" {_STALE_ENV}" if _STALE_ENV else "")
        )
        sys.exit(0)

    _, updated, raw_text, compressed_text = outcome
    _record_savings_event(
        session_id, f"hook:{tool_name}", estimate_tokens(raw_text), estimate_tokens(compressed_text),
        True, source, project=project,
    )
    _emit_updated(updated)
    sys.exit(0)


def _dispatch(payload):
    """Everything main() does after the payload has been parsed off stdin --
    split out so main() can wrap this whole body in one try/except (issue
    #44) without also swallowing the json.load try/except above it, which
    has to stay narrowly scoped to the parse step itself."""
    _debug_log(payload)

    tool_name = payload.get("tool_name")
    session_id = payload.get("session_id")
    tool_response = payload.get("tool_response")
    # Same source session_end_savings.py already uses for the identical
    # purpose -- derived once here so both _record_savings_event call sites
    # below tag their JSONL entries with it (issue #35: this is what lets
    # current_session_id()'s later project-filtered lookup tell this
    # session's events apart from an unrelated project's).
    project = savings_ledger.project_name_from_cwd(payload.get("cwd", "")) if _SAVINGS_LEDGER_AVAILABLE else None

    if tool_name and tool_name.startswith(MCP_SAVINGS_TOOL_PREFIX):
        if tool_response is None:
            sys.exit(0)
        data, updated = _handle_mcp_savings_footer(tool_response)
        if data is not None:
            _record_savings_event(
                session_id, data.get("tool", tool_name), data.get("raw_tokens", 0),
                data.get("out_tokens", 0), data.get("credited", True), data.get("source", ""),
                project=project,
            )
            _emit_updated(updated)
            sys.exit(0)
        # No footer. If this is one of the tools that already ran its OWN
        # compression (see _SELF_COMPRESSING_MCP_TOOLS's comment), leave it
        # alone regardless of size -- re-compressing it generically here
        # would discard exactly what that first pass was told to preserve.
        # Otherwise, this is one of the local-compress tools that never
        # compresses anything itself (compact_find, compact_store,
        # list_local_models, savings_summary/detail) -- fall back to the
        # same size-based compression every other matched tool gets,
        # otherwise every one of THOSE calls is a complete no-op regardless
        # of size, the opposite of what this hook exists to do (issue #25).
        # compact_find in particular can return up to 10 full stored
        # conversation compacts.
        bare_tool_name = tool_name[len(MCP_SAVINGS_TOOL_PREFIX):]
        if bare_tool_name in _SELF_COMPRESSING_MCP_TOOLS:
            sys.exit(0)
        outcome = _handle_generic(tool_response)
        _finish_compression_outcome(outcome, tool_name, tool_name, session_id, project)

    # Generic MCP tools from other servers (GitHub, Splunk, JIRA, etc.) --
    # NOT mcp__local-compress__, whose path comes first above. All of these
    # go through _handle_generic: the generic walker is schema-agnostic and
    # the dividing line for whether a tool belongs in the matcher at all
    # (does Claude use this output as the exact basis for a following Edit?)
    # is enforced at the settings.json.template matcher level, not here.
    # Each tool in the matcher has been individually vetted against that rule:
    # - get_file_contents: excluded -- returns raw file bytes (edit-basis).
    # - get_pull_request_files: excluded -- returns per-file patch hunks
    #   (edit-basis; the patch IS the code being reviewed or implemented,
    #   not a summary of statistics).
    # - write/mutation calls (create_*/push_files/fork_repository): excluded
    #   -- response is created-object metadata, which is never large enough
    #   to compress, and these paths don't benefit from being in the matcher.
    # - add_issue_comment, create_pull_request_review: included despite being
    #   write calls -- their response is the newly created object's JSON
    #   (id, url, body, timestamps), read for confirmation or follow-up, never
    #   the byte-exact basis for an Edit. The "NOT write mutations" framing in
    #   earlier drafts was an oversimplification; the real test is edit-basis,
    #   not read-vs-write.
    if tool_name.startswith("mcp__"):
        outcome = _handle_generic(tool_response)
        _finish_compression_outcome(outcome, tool_name, tool_name, session_id, project)

    if tool_name not in SUPPORTED_TOOLS:
        sys.exit(0)  # matcher should already restrict this, but double-check

    if tool_response is None:
        sys.exit(0)

    if tool_name == "Bash":
        source = (payload.get("tool_input") or {}).get("command", "")
        if _is_exactness_critical(source):
            # Silent, like the under-threshold path: Claude gets the real
            # output, which is the correct result, not something to announce.
            sys.exit(0)
        outcome = _handle_bash(tool_response)
    else:
        outcome = _handle_generic(tool_response)
        source = tool_name

    _finish_compression_outcome(outcome, tool_name, source, session_id, project)


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)  # can't parse input -- fail open, don't touch anything

    # Top-level guard for everything AFTER the parse step above (issue #44):
    # _dispatch() calls into _handle_bash/_handle_generic, which call
    # compress() via asyncio.run -- compress()'s own docstring promises
    # "never raises," but that's upheld only incidentally (the one
    # network-calling piece, complete(), is what actually catches
    # everything; nothing else in the pipeline has its own try/except). Any
    # other unexpected error anywhere in this path -- here, or in a future
    # change to _dispatch() itself -- must fail open the same way every KNOWN
    # failure mode in this file already does, rather than propagate as a raw
    # traceback with no _STALE_ENV context and no explanation.
    #
    # Deliberately `except Exception`, not `BaseException`: every terminal
    # branch inside _dispatch()/_finish_compression_outcome() already calls
    # sys.exit(0), which raises SystemExit -- a BaseException subclass that
    # must keep propagating past this guard unharmed, or every existing
    # "done, exit cleanly" path below would get rerouted into the error note
    # this except block is only meant for genuinely unexpected failures.
    try:
        _dispatch(payload)
    except Exception as e:
        _emit_unchanged(
            f"Note: this hook failed unexpectedly ({type(e).__name__}: {e}). "
            "Original output was left unchanged."
            + (f" {_STALE_ENV}" if _STALE_ENV else "")
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
