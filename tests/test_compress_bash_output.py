#!/usr/bin/env python3
"""Table tests for hooks/compress_bash_output.py's exactness-critical skipping.

Stdlib-only (unittest, no pytest) and no network -- `_is_exactness_critical`
is pure regex over a command string, so this needs neither LM Studio nor
Qdrant and runs in milliseconds:

    .venv/bin/python -m unittest discover -s tests

The venv Python is required only because importing the hook module pulls in
local_compress_lib -> openai. Nothing in these tests calls it.

Why this file exists: the exempt list is the one place in this repo where a
regex mistake is SILENT in both directions. A false positive forfeits savings
with no signal; a false negative lets a `git rev-parse` summary through, which
is the exact bug that motivated the feature. Neither shows up in normal use,
so it has to be pinned here.
"""

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))

import compress_bash_output as hook  # noqa: E402
from compress_bash_output import _is_exactness_critical  # noqa: E402


# Commands whose output must reach Claude byte-exact (hook skips compression).
MUST_SKIP = [
    # --- git: the tool reached for to verify state -----------------------
    ("git status --short", "porcelain-ish state"),
    ("git diff --stat", "counts per file"),
    ("git rev-parse HEAD", "a 40-char identifier"),
    ("git log --format=%H", "--format anywhere in segment"),
    ("git show HEAD:f.json | grep -c foo", "the original bug: count in a pipeline"),
    ("FOO=1 git log", "leading VAR=val assignment"),
    ('FOO="a b" git status', "quoted assignment value containing a space"),
    ("FOO='a b' git rev-parse HEAD", "single-quoted assignment value"),
    ('A=1 B="x y" sudo git diff', "several assignments, quoted and bare, plus sudo"),
    ("sudo git status", "leading sudo"),
    # The escape hatch must not fire from inside a quoted argument -- that
    # would lose the exemption by accident, which is the failure this list
    # exists to prevent, reintroduced through the opt-out.
    ('git commit -m "fix the # compress-ok bug"', "'# compress-ok' inside an argument"),
    ('git log --grep="# compress-ok"', "same, in a --grep pattern"),
    # --- counts and digests: nothing to summarize ------------------------
    ("cat x | wc -l", "exempt segment is not the first one"),
    ("sha256sum f", "digest"),
    ("md5sum f", "md5 vs md5sum alternation must backtrack"),
    ("grep -rc TODO src/", "grep -rc, bundled flags"),
    # --- exact identifiers substituted into a follow-up call -------------
    ("which python3", ""),
    ("printenv PATH", ""),
    ("pwd", "single bare token, no arguments"),
    ("id", "shortest token in the list"),
    # --- machine-readable by request -------------------------------------
    ("jq .foo < x.json", ""),
    ("cat x.json | jq .a", "| jq matched anywhere"),
    ("az role assignment list -o json", "-o json on a non-exempt command"),
    ("kubectl get pods -o yaml", ""),
    ("npm --version", "version pin"),
    ("pip freeze", "two-word command name"),
    ("az group show --query id -o tsv", "--query plus -o tsv"),
    # --- long-form --output, and case variance in the format value (#33) ---
    ("aws s3api list-buckets --output json", "--output long form, space-separated"),
    ("az group show --output=yaml", "--output long form, = separated"),
    ("az role assignment list -o JSON", "uppercase format value on the short form"),
    ("aws s3api list-buckets --output JSON", "uppercase format value on the long form"),
    # --- previews and help: the output IS the exact thing being asked for ---
    ("python tools/ingest_to_qdrant.py --dry-run", "a dry run's output is the preview"),
    ("kubectl apply -f x.yaml --dry-run=client", "--dry-run carrying a value"),
    ("./configure --help", "flag spellings get copied into the next command"),
    ("python x.py -h", "standalone -h"),
    ("df -h", "-h as its own token; tabular output that reads exact anyway"),
    ("gtk-app --help-all", "--help-all is still help output, so matching it is right"),
    # --- gh api (issue #190): structured REST/GraphQL data, never prose ---
    (
        "gh api repos/CVNA-SandboxOrg/claude-runway/pulls/190/comments --paginate",
        "plain gh api call with no --jq/--json at all (my-gh-pr-feedback's shape)",
    ),
    (
        "gh api graphql -f query='query { repository(name: \"x\") { pullRequest(number: 1) "
        "{ reviewThreads(first: 100) { nodes { id comments(first: 1) { nodes { body } } } } } } }'",
        "graphql body-fetch shape",
    ),
    (
        'COMMENT_IDS=$(gh api repos/CVNA-SandboxOrg/claude-runway/pulls/190/comments '
        '--paginate --jq ".[] | select(.user.login != \\"me\\") | \\"comment:\\" + (.id|tostring)") '
        "|| FETCH_FAILED=1",
        "the real VAR=$(...) command-substitution shape used by my-gh-autowork's NEW_IDS diffing "
        "-- a start-anchored fix would have missed this",
    ),
]

# Commands that should still be compressed normally.
MUST_COMPRESS = [
    # --- ordinary bulk output, the whole point of the hook ---------------
    ("dotnet build", "log to skim; also guards `od` against `d`-prefixed names"),
    ("npm run test", "`npm ls` must not match `npm run`"),
    ("cat SKILL.md", "doc-shaped read -- the big historical win"),
    ("az deployment group create --template-file t.json", "no exact-output flag"),
    ("grep -rn \"pattern\" src/", "grep WITHOUT -c stays compressible"),
    # --- git transports: progress noise, not state -----------------------
    ("git clone https://example.com/r.git", ""),
    ("git fetch origin", ""),
    ("git pull --rebase", ""),
    ("git push origin main", ""),
    # --- the escape hatch ------------------------------------------------
    ("git diff # compress-ok", "opts a huge exempt output back in"),
    ("git status --short # compress-ok", "escape hatch beats an exact flag too"),
    ("git diff #compress-ok", "no space after the hash"),
    ("git diff # compress-ok   ", "trailing whitespace after the marker"),
    # --- near-misses that must NOT trip the regexes ----------------------
    ("dotnet run --env x", "`env` is segment-anchored, not matched mid-command"),
    ("grep --color=always foo x", "`--color` must not read as `grep -c`"),
    ("grep -rn x src/ | cut -c1-80", "`cut -c` is a different segment from grep"),
    ("ideviceinfo", "`id` must not match inside a longer word"),
    ("identify img.png", "same, with a real-world command name"),
    ("echo \"run git status\"", "exempt name only inside a quoted argument"),
    ("echo hi || dotnet build", "|| must split as one delimiter, not two empties"),
    ("", "empty command"),
    # --- near-misses on the -h / --dry-run additions ---------------------
    ("ls -lh /var/log", "bundled -lh must not read as a standalone -h"),
    ("du -sh .", "same, bundled -sh"),
    ("grep -rh pattern src/", "same, bundled -rh"),
    ('echo "-h"', "-h inside a quoted argument, not preceded by whitespace"),
    ("python x.py --dry-run # compress-ok", "escape hatch beats the dry-run exemption"),
    # --- case-insensitivity is scoped to the output-format value only (#33) ---
    ('curl -H "Accept: application/json" https://x', "-H is curl's header flag, not -h/help"),
    # --- gh, but not `gh api`: unaffected by the issue #190 addition ---
    ('gh pr comment 190 --body "thanks!"', "gh subcommand other than api stays compressible"),
    ("gh pr view 190 -R x/y", "no --json, no api subcommand -- ordinary human-readable text"),
]


class ExactnessCritical(unittest.TestCase):
    def test_must_skip(self):
        for command, why in MUST_SKIP:
            with self.subTest(command=command):
                self.assertTrue(
                    _is_exactness_critical(command),
                    f"{command!r} should be exempt from compression ({why})",
                )

    def test_must_compress(self):
        for command, why in MUST_COMPRESS:
            with self.subTest(command=command):
                self.assertFalse(
                    _is_exactness_critical(command),
                    f"{command!r} should still be compressed ({why})",
                )


class _StubSavingsLedger:
    """Stands in for the real libs/savings_ledger.py so these tests never
    touch a real JSONL/SQLite file -- only the call args reaching
    record_event() matter here."""

    def __init__(self):
        self.calls = []

    def tracking_enabled(self):
        return True

    def project_name_from_cwd(self, cwd):
        # Same logic as the real function (Path(cwd).name if cwd else
        # "unknown") without importing pathlib for a one-liner test double.
        return cwd.rstrip("/").rsplit("/", 1)[-1] if cwd else "unknown"

    def record_event(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class RecordSavingsEventForwardsProject(unittest.TestCase):
    """Issue #35: hooks/compress_bash_output.py is savings_ledger's SOLE
    writer, so _record_savings_event forwarding its `project` argument
    through to record_event is what lets current_session_id()'s later
    project-filtered lookup tell this session's events apart from an
    unrelated project's."""

    def setUp(self):
        self.stub = _StubSavingsLedger()
        self._real_ledger = hook.savings_ledger
        self._real_available = hook._SAVINGS_LEDGER_AVAILABLE
        hook.savings_ledger = self.stub
        hook._SAVINGS_LEDGER_AVAILABLE = True

    def tearDown(self):
        hook.savings_ledger = self._real_ledger
        hook._SAVINGS_LEDGER_AVAILABLE = self._real_available

    def test_forwards_project_kwarg_to_record_event(self):
        hook._record_savings_event(
            "sess-1", "hook:Bash", 1000, 100, True, "some command", project="my-repo",
        )
        self.assertEqual(len(self.stub.calls), 1)
        _, kwargs = self.stub.calls[0]
        self.assertEqual(kwargs.get("project"), "my-repo")

    def test_project_defaults_to_none_when_not_passed(self):
        # Regression guard: an omitted project must stay None, not silently
        # become an empty string or the literal word "None" -- either would
        # still (wrongly) match a real project filter's string comparison.
        hook._record_savings_event("sess-1", "hook:Bash", 1000, 100, True, "cmd")
        _, kwargs = self.stub.calls[0]
        self.assertIsNone(kwargs.get("project"))


class MainDerivesProjectFromPayloadCwd(unittest.TestCase):
    """End-to-end through main() itself -- confirms the actual wiring (not
    just record_event's signature): payload["cwd"] really does reach
    record_event as `project`. Drives the MCP-footer-strip path specifically
    (a tool_name starting with mcp__local-compress__) because it never calls
    compress()/LM Studio -- it only parses an already-compressed footer a
    credited MCP tool call would have appended, so this needs neither a
    live model nor network access."""

    def setUp(self):
        self.stub = _StubSavingsLedger()
        self._real_ledger = hook.savings_ledger
        self._real_available = hook._SAVINGS_LEDGER_AVAILABLE
        hook.savings_ledger = self.stub
        hook._SAVINGS_LEDGER_AVAILABLE = True

    def tearDown(self):
        hook.savings_ledger = self._real_ledger
        hook._SAVINGS_LEDGER_AVAILABLE = self._real_available

    def _run_main(self, payload):
        real_stdin = sys.stdin
        sys.stdin = io.StringIO(json.dumps(payload))
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                try:
                    hook.main()
                except SystemExit:
                    pass
        finally:
            sys.stdin = real_stdin
        return out.getvalue()

    def test_cwd_derived_project_reaches_record_event(self):
        footer = json.dumps({
            "tool": "compress_file", "raw_tokens": 1000, "out_tokens": 100,
            "credited": True, "source": "some/file.log",
        })
        payload = {
            "session_id": "sess-1",
            "cwd": "/Users/someone/repos/my-actual-project",
            "tool_name": "mcp__local-compress__compress_file",
            "tool_response": f"[compressed 1000 -> 100 chars]\n<!--CLAUDE_RUNWAY_SAVINGS:{footer}-->",
        }
        self._run_main(payload)
        self.assertEqual(len(self.stub.calls), 1)
        _, kwargs = self.stub.calls[0]
        self.assertEqual(kwargs.get("project"), "my-actual-project")


async def _fake_compress(text, skip_if_under_chars=2000, **kwargs):
    """Deterministic stand-in for local_compress_lib.compress() -- avoids
    needing a live LM Studio model for tests that exercise the actual
    compression call, not just the pieces around it. Mirrors compress()'s
    real under-threshold contract (return the text unchanged) so tests can
    exercise both branches through the SAME fake."""
    if len(text) < skip_if_under_chars:
        return text
    return f"[compressed {len(text)} -> 4 chars across 1 chunk(s), ~99% smaller]\nGIST"


class MainFallsBackToGenericForNonFooterMcpTools(unittest.TestCase):
    """Issue #25: an mcp__local-compress__* tool call that carries NO savings
    footer (compact_find, compact_store, list_local_models,
    savings_summary/detail -- none of these ever call
    _append_savings_footer) must still get the same size-based compression
    every other matched tool gets, not a silent no-op regardless of size."""

    def setUp(self):
        self.stub = _StubSavingsLedger()
        self._real_ledger = hook.savings_ledger
        self._real_available = hook._SAVINGS_LEDGER_AVAILABLE
        self._real_compress = hook.compress
        hook.savings_ledger = self.stub
        hook._SAVINGS_LEDGER_AVAILABLE = True
        hook.compress = _fake_compress

    def tearDown(self):
        hook.savings_ledger = self._real_ledger
        hook._SAVINGS_LEDGER_AVAILABLE = self._real_available
        hook.compress = self._real_compress

    def _run_main(self, payload):
        real_stdin = sys.stdin
        sys.stdin = io.StringIO(json.dumps(payload))
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                try:
                    hook.main()
                except SystemExit:
                    pass
        finally:
            sys.stdin = real_stdin
        return out.getvalue()

    def test_large_no_footer_response_gets_compressed_and_recorded(self):
        # compact_find can return up to 10 full stored compacts -- simulated
        # here as one large string well over THRESHOLD, no footer attached.
        big_response = "STORED COMPACT CONTENT. " * 200  # well over THRESHOLD
        self.assertGreater(len(big_response), hook.THRESHOLD)
        payload = {
            "session_id": "sess-1",
            "cwd": "/repos/my-project",
            "tool_name": "mcp__local-compress__compact_find",
            "tool_response": big_response,
        }
        printed = self._run_main(payload)
        # main() must have emitted an updatedToolOutput, not silently exited.
        self.assertIn("updatedToolOutput", printed)
        self.assertIn("GIST", printed)
        self.assertEqual(len(self.stub.calls), 1)
        args, kwargs = self.stub.calls[0]
        self.assertEqual(args[1], "hook:mcp__local-compress__compact_find")
        self.assertEqual(kwargs.get("project"), "my-project")

    def test_small_no_footer_response_is_left_untouched(self):
        # Regression guard: this fallback must not start compressing small
        # compact_find results that were never a problem in the first place.
        small_response = "just one short compact"
        self.assertLess(len(small_response), hook.THRESHOLD)
        payload = {
            "session_id": "sess-1",
            "cwd": "/repos/my-project",
            "tool_name": "mcp__local-compress__compact_find",
            "tool_response": small_response,
        }
        printed = self._run_main(payload)
        self.assertEqual(printed, "")  # no hookSpecificOutput at all -- true no-op
        self.assertEqual(len(self.stub.calls), 0)

    def test_self_compressing_tool_without_footer_is_never_double_compressed(self):
        # PR #88 review (Copilot): compress_text never emits a footer at
        # all, and compress_file/compress_command_output/fetch_url omit one
        # whenever tracking is off server-side even though real compression
        # already ran -- with the caller's own focus/preserve_identifiers/
        # preserve_sections (this is /my-compact's own real usage shape).
        # None of these four must ever fall through to the generic
        # fallback: doing so would silently discard exactly what that first
        # pass was told to preserve.
        already_compressed = "[compressed 9000 -> 3000 chars across 3 chunk(s), ~67% smaller]\n" + (
            "PRESERVED_IDENTIFIER_TOKEN " * 150  # still well over THRESHOLD
        )
        self.assertGreater(len(already_compressed), hook.THRESHOLD)
        for bare_name in sorted(hook._SELF_COMPRESSING_MCP_TOOLS):
            with self.subTest(tool=bare_name):
                self.stub.calls.clear()
                payload = {
                    "session_id": "sess-1",
                    "cwd": "/repos/my-project",
                    "tool_name": f"mcp__local-compress__{bare_name}",
                    "tool_response": already_compressed,
                }
                printed = self._run_main(payload)
                self.assertEqual(printed, "", f"{bare_name} must be a true no-op here")
                self.assertEqual(len(self.stub.calls), 0)

    def test_footer_path_still_takes_priority_over_the_fallback(self):
        # Regression guard for issue #35's existing footer-path behavior --
        # confirms the refactor into _finish_compression_outcome() didn't
        # change what happens when a footer IS present.
        footer = json.dumps({
            "tool": "compress_file", "raw_tokens": 1000, "out_tokens": 100,
            "credited": True, "source": "some/file.log",
        })
        payload = {
            "session_id": "sess-1",
            "cwd": "/repos/my-project",
            "tool_name": "mcp__local-compress__compress_file",
            "tool_response": f"[compressed 1000 -> 100 chars]\n<!--CLAUDE_RUNWAY_SAVINGS:{footer}-->",
        }
        self._run_main(payload)
        self.assertEqual(len(self.stub.calls), 1)
        args, _ = self.stub.calls[0]
        # tool name comes from the footer's own "tool" field here, not
        # prefixed with "hook:" -- that's the credited-footer path's
        # existing (unchanged) behavior, distinct from the fallback path's
        # "hook:<tool_name>" naming asserted above.
        self.assertEqual(args[1], "compress_file")


class CompactFindMultiEntryIsNeverGenericallyCompressed(unittest.TestCase):
    """Issue #193: compact_find's output is structured data /my-resume
    parses for control flow (a "Found {N} compact(s)" header + N discrete
    dated/labeled entries), not prose to skim. Generic size-based
    compression collapses multiple entries into one flowing narrative
    before /my-resume ever sees discrete entries to count, breaking its
    0/1/2+ branching. A response whose own header reports N > 1 must be
    left completely untouched, regardless of size -- the same true no-op
    _SELF_COMPRESSING_MCP_TOOLS gets. N <= 1 (including no header at all,
    e.g. an error string) must still fall through to the ordinary
    size-based path, unchanged from before this fix (issue #25's original
    intent: compact_find isn't unconditionally exempt)."""

    def setUp(self):
        self.stub = _StubSavingsLedger()
        self._real_ledger = hook.savings_ledger
        self._real_available = hook._SAVINGS_LEDGER_AVAILABLE
        self._real_compress = hook.compress
        hook.savings_ledger = self.stub
        hook._SAVINGS_LEDGER_AVAILABLE = True
        hook.compress = _fake_compress

    def tearDown(self):
        hook.savings_ledger = self._real_ledger
        hook._SAVINGS_LEDGER_AVAILABLE = self._real_available
        hook.compress = self._real_compress

    def _run_main(self, payload):
        real_stdin = sys.stdin
        sys.stdin = io.StringIO(json.dumps(payload))
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                try:
                    hook.main()
                except SystemExit:
                    pass
        finally:
            sys.stdin = real_stdin
        return out.getvalue()

    def _multi_entry_response(self, n):
        # Scaled off hook.THRESHOLD (not the code's documented 2000
        # default) rather than a fixed repeat count -- this repo's own
        # dogfood shell overrides CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS to
        # 4000, and a hardcoded assumption would fail confusingly here.
        per_entry_chars = (hook.THRESHOLD // max(n, 1)) + 200
        lines = [f"Found {n} compact(s) for project 'claude-runway':\n"]
        for i in range(1, n + 1):
            lines.append(f"--- {i}. 2026-09-{i:02d} — session {i} (id: abcd1234) ---")
            lines.append("STORED COMPACT CONTENT. " * (per_entry_chars // 25 + 1))
            lines.append("")
        return "\n".join(lines)

    def test_multi_entry_response_over_threshold_is_left_completely_untouched(self):
        response = self._multi_entry_response(5)
        self.assertGreater(len(response), hook.THRESHOLD)
        payload = {
            "session_id": "sess-1",
            "cwd": "/repos/claude-runway",
            "tool_name": "mcp__local-compress__compact_find",
            "tool_response": response,
        }
        printed = self._run_main(payload)
        self.assertEqual(printed, "", "a multi-entry compact_find result must be a true no-op")
        self.assertEqual(len(self.stub.calls), 0)

    def test_single_entry_response_over_threshold_still_gets_compressed(self):
        # N == 1 has no multi-entry structure to lose -- must still follow
        # the ordinary size-based fallback path (issue #25's intent).
        response = self._multi_entry_response(1)
        self.assertGreater(len(response), hook.THRESHOLD)
        payload = {
            "session_id": "sess-1",
            "cwd": "/repos/claude-runway",
            "tool_name": "mcp__local-compress__compact_find",
            "tool_response": response,
        }
        printed = self._run_main(payload)
        self.assertIn("updatedToolOutput", printed)
        self.assertIn("GIST", printed)
        self.assertEqual(len(self.stub.calls), 1)

    def test_headerless_response_over_threshold_still_gets_compressed(self):
        # Regression guard: an error string or any other compact_find
        # response shape with no "Found N compact(s)" header at all (e.g.
        # "No compacts found for project ...") must behave exactly as it
        # did before this fix -- fall through to size-based compression.
        response = "No compacts found for project 'claude-runway'. " * 100
        self.assertGreater(len(response), hook.THRESHOLD)
        payload = {
            "session_id": "sess-1",
            "cwd": "/repos/claude-runway",
            "tool_name": "mcp__local-compress__compact_find",
            "tool_response": response,
        }
        printed = self._run_main(payload)
        self.assertIn("updatedToolOutput", printed)
        self.assertEqual(len(self.stub.calls), 1)

    def test_multi_entry_response_under_threshold_is_a_true_noop_either_way(self):
        # Small multi-entry response: already a no-op via the normal
        # under-threshold path, but confirms the new header check doesn't
        # change that outcome (no ledger event either way).
        response = self._multi_entry_response(2)[:100]
        payload = {
            "session_id": "sess-1",
            "cwd": "/repos/claude-runway",
            "tool_name": "mcp__local-compress__compact_find",
            "tool_response": response,
        }
        printed = self._run_main(payload)
        self.assertEqual(printed, "")
        self.assertEqual(len(self.stub.calls), 0)


class CompactFindEntryCountHelper(unittest.TestCase):
    """Direct unit coverage for _compact_find_entry_count, independent of
    the full hook dispatch path exercised above."""

    def test_extracts_count_from_header(self):
        text = "Found 3 compact(s) for project 'foo':\n--- 1. ... ---"
        self.assertEqual(hook._compact_find_entry_count(text), 3)

    def test_returns_none_when_no_header_present(self):
        self.assertIsNone(hook._compact_find_entry_count("just some other text"))

    def test_ignores_header_phrase_when_not_at_the_leaf_start(self):
        # PR #194 review: the header regex must be anchored to the START of
        # the leaf, not just matched anywhere in it -- otherwise a STORED
        # compact's own content quoting this exact phrase mid-string (e.g. a
        # past /my-compact session about debugging this very hook) would be
        # mistaken for the real leading header. Reproduced live before the
        # fix: this returned 5, not None.
        fake = "Some earlier stored compact discusses: Found 5 compact(s) for project 'other':\nmore text"
        self.assertIsNone(hook._compact_find_entry_count(fake))

    def test_finds_header_nested_in_a_dict_or_list(self):
        nested = {"content": [{"type": "text", "text": "Found 2 compact(s) for project 'x':\n..."}]}
        self.assertEqual(hook._compact_find_entry_count(nested), 2)


class HandleGenericGroupsBySiblingRecord(unittest.TestCase):
    """Issue #38: sibling records (e.g. WebSearch's `results` list) must be
    compressed independently, not concatenated into one blob whose result
    then overwrites only the largest field while every other record's field
    is silently blanked to "". Uses the same hook.compress monkeypatch
    pattern as MainFallsBackToGenericForNonFooterMcpTools above, but calls
    _handle_generic directly since that's where the grouping logic lives."""

    def setUp(self):
        self._real_compress = hook.compress
        hook.compress = _fake_compress

    def tearDown(self):
        hook.compress = self._real_compress

    def test_individually_small_siblings_are_never_blanked(self):
        # Reproduces the reported bug directly: 10 sibling records, each
        # individually under THRESHOLD, but the aggregate total is over it
        # -- the exact shape that used to blank 9 of 10 snippets.
        snippet_len = hook.MIN_FIELD_LEN + 50
        self.assertLess(snippet_len, hook.THRESHOLD, "snippet must stay under THRESHOLD individually")
        # Enough records that the aggregate clears THRESHOLD regardless of
        # its configured value (CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS can
        # override the 2000 default -- this repo's own dogfood shell sets
        # it to 4000), while each individual snippet stays under it.
        num_results = (hook.THRESHOLD // snippet_len) + 3
        results = [
            {
                "title": f"Title {i}",
                "url": f"https://example.com/{i}",
                "snippet": f"MARKER_{i} " + ("x" * snippet_len),
            }
            for i in range(num_results)
        ]
        tool_response = {"results": results}
        total = sum(len(r["snippet"]) for r in results)
        self.assertGreater(total, hook.THRESHOLD, "aggregate must exceed THRESHOLD to trigger the old bug")

        outcome = hook._handle_generic(tool_response)

        # True no-op: every record was individually under its own threshold,
        # so nothing should change even though the aggregate passed.
        self.assertIsNone(outcome)
        for i, r in enumerate(results):
            self.assertIn(f"MARKER_{i}", r["snippet"])

    def test_one_oversized_sibling_compresses_alone(self):
        small_len = hook.MIN_FIELD_LEN + 20
        self.assertLess(small_len, hook.THRESHOLD)
        big_len = hook.THRESHOLD + 500
        small0 = "SMALL_MARKER_0 " + ("x" * small_len)
        small2 = "SMALL_MARKER_2 " + ("x" * small_len)
        results = [
            {"title": "small 0", "snippet": small0},
            {"title": "big 1", "snippet": "BIG_MARKER_1 " + ("y" * big_len)},
            {"title": "small 2", "snippet": small2},
        ]
        tool_response = {"results": results}

        outcome = hook._handle_generic(tool_response)

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome[0], "updated")
        _, updated, _, _ = outcome
        updated_results = updated["results"]
        # The oversized record was compressed on its own...
        self.assertIn("[compressed", updated_results[1]["snippet"])
        self.assertNotIn("BIG_MARKER_1", updated_results[1]["snippet"])
        # ...while its siblings kept their exact original content --
        # not blanked, not touched at all, regardless of the aggregate
        # having passed THRESHOLD.
        self.assertEqual(updated_results[0]["snippet"], small0)
        self.assertEqual(updated_results[2]["snippet"], small2)

    def test_no_sibling_structure_falls_back_to_whole_blob_behavior(self):
        # Flat shape, no list index anywhere in any field's path (e.g.
        # Grep's undocumented tool_response) -- must behave exactly like
        # the pre-fix code: compress everything together and blank every
        # large field except the single largest one.
        matches_len = hook.THRESHOLD
        context_len = hook.THRESHOLD + 500
        tool_response = {
            "matches": "A" * matches_len,
            "context": "B" * context_len,  # the larger of the two fields
        }

        outcome = hook._handle_generic(tool_response)

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome[0], "updated")
        _, updated, _, _ = outcome
        # The larger field ("context") received the compressed result; the
        # smaller sibling field was blanked -- unchanged from before this
        # fix, since there's no sibling RECORD structure here (no list
        # index anywhere), so this collapses to a single group.
        self.assertIn("[compressed", updated["context"])
        self.assertEqual(updated["matches"], "")

    def test_nested_sibling_records_group_by_innermost_index(self):
        # PR #107 review (Copilot): grouping by the FIRST list index in a
        # path (e.g. the 'sections' index) rather than the LAST/nearest one
        # would merge sections[0].results[3] and sections[0].results[4]
        # back into one group, reintroducing the exact cross-record
        # blending this whole fix exists to prevent -- just one level
        # deeper. Confirms results[3] and results[4] under the same
        # sections[0] parent are still treated as separate records.
        snippet_len = hook.MIN_FIELD_LEN + 50
        self.assertLess(snippet_len, hook.THRESHOLD)
        results = [
            {"snippet": f"MARKER_{i} " + ("x" * snippet_len)}
            for i in range(3)
        ]
        tool_response = {"sections": [{"results": results}]}

        key0 = hook._record_group_key(("sections", 0, "results", 0, "snippet"))
        key1 = hook._record_group_key(("sections", 0, "results", 1, "snippet"))
        self.assertNotEqual(key0, key1, "results[0] and results[1] must be separate records")

        outcome = hook._handle_generic(tool_response)

        # Every record individually under THRESHOLD -- still a true no-op,
        # exactly like the flat (non-nested) version of this same guard.
        self.assertIsNone(outcome)
        for i, r in enumerate(results):
            self.assertIn(f"MARKER_{i}", r["snippet"])


async def _raising_compress(text, skip_if_under_chars=2000, **kwargs):
    """Simulates an unexpected, non-network failure deep in compress()'s own
    pipeline (e.g. a bug in chunk_text/classify_relevant/split_sections) --
    unlike a network error, compress() has no try/except anywhere in its own
    body to catch this (issue #44's related finding: compress()'s docstring
    promises "never raises," but that guarantee is upheld only incidentally,
    since it's complete()'s own internal try/except -- not anything in
    compress() itself -- that actually holds it)."""
    if len(text) < skip_if_under_chars:
        return text
    raise RuntimeError("boom: unexpected failure deep in compress()")


class MainFailsOpenOnUnexpectedException(unittest.TestCase):
    """Issue #44: main() only wrapped json.load(sys.stdin) in try/except --
    everything after (_handle_bash/_handle_generic/_record_savings_event/
    _emit_updated, now factored into _dispatch()) ran completely unguarded,
    so any unexpected exception propagated as a raw traceback instead of the
    same fail-open additionalContext note every OTHER known failure mode in
    this file already gets."""

    def setUp(self):
        self._real_compress = hook.compress
        hook.compress = _raising_compress

    def tearDown(self):
        hook.compress = self._real_compress

    def _run_main(self, payload):
        real_stdin = sys.stdin
        sys.stdin = io.StringIO(json.dumps(payload))
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                try:
                    hook.main()
                except SystemExit:
                    pass
                except Exception:
                    self.fail("main() must not let an unexpected exception escape uncaught")
        finally:
            sys.stdin = real_stdin
        return out.getvalue()

    def test_unexpected_exception_in_bash_path_fails_open_with_note(self):
        big_command_output = "x" * (hook.THRESHOLD + 500)
        payload = {
            "session_id": "sess-1",
            "cwd": "/repos/my-project",
            "tool_name": "Bash",
            "tool_input": {"command": "dotnet build"},
            "tool_response": {"stdout": big_command_output, "stderr": ""},
        }
        printed = self._run_main(payload)
        self.assertIn("additionalContext", printed)
        self.assertIn("failed unexpectedly", printed)
        # Fail open means the note is the only thing emitted -- the original
        # output must NOT come back rewritten as if compression succeeded.
        self.assertNotIn("updatedToolOutput", printed)

    def test_unexpected_exception_in_generic_path_fails_open_with_note(self):
        big_field = "y" * (hook.THRESHOLD + 500)
        payload = {
            "session_id": "sess-1",
            "cwd": "/repos/my-project",
            "tool_name": "Grep",
            "tool_response": {"matches": big_field},
        }
        printed = self._run_main(payload)
        self.assertIn("additionalContext", printed)
        self.assertIn("failed unexpectedly", printed)
        self.assertNotIn("updatedToolOutput", printed)

    def test_ordinary_sys_exit_paths_still_work_after_the_guard(self):
        # Regression guard for the guard itself: an ordinary "under
        # threshold, leave alone" outcome must still exit cleanly with NO
        # output at all -- confirms the new `except Exception` (deliberately
        # not `except BaseException`) doesn't also swallow the SystemExit
        # these paths already raise via sys.exit(0).
        small_output = "short"
        payload = {
            "session_id": "sess-1",
            "cwd": "/repos/my-project",
            "tool_name": "Bash",
            "tool_input": {"command": "echo hi"},
            "tool_response": {"stdout": small_output, "stderr": ""},
        }
        printed = self._run_main(payload)
        self.assertEqual(printed, "")


class MainHandlesGenericMcpTools(unittest.TestCase):
    """Issue #63: MCP tools from servers other than local-compress (e.g.
    mcp__github__search_code, mcp__claude_ai_Splunk__*) must go through
    _handle_generic when included in the matcher -- large responses get
    compressed, small ones are left untouched. The dispatch path is the
    new `if tool_name.startswith("mcp__"):` branch in _dispatch(), which
    runs AFTER the mcp__local-compress__ check and BEFORE the SUPPORTED_TOOLS
    guard that would otherwise block every MCP tool name not in that set.

    Uses the same hook.compress monkeypatch pattern as other test classes --
    no live LM Studio required."""

    def setUp(self):
        self.stub = _StubSavingsLedger()
        self._real_ledger = hook.savings_ledger
        self._real_available = hook._SAVINGS_LEDGER_AVAILABLE
        self._real_compress = hook.compress
        hook.savings_ledger = self.stub
        hook._SAVINGS_LEDGER_AVAILABLE = True
        hook.compress = _fake_compress

    def tearDown(self):
        hook.savings_ledger = self._real_ledger
        hook._SAVINGS_LEDGER_AVAILABLE = self._real_available
        hook.compress = self._real_compress

    def _run_main(self, payload):
        real_stdin = sys.stdin
        sys.stdin = io.StringIO(json.dumps(payload))
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                try:
                    hook.main()
                except SystemExit:
                    pass
        finally:
            sys.stdin = real_stdin
        return out.getvalue()

    def test_large_github_search_result_gets_compressed(self):
        # mcp__github__search_code returns a list of matching file excerpts --
        # read for relevance, never the exact basis for a following Edit.
        # Use a flat string response here (no sibling record structure) so
        # _handle_generic takes the whole-blob path: the total aggregate is
        # one group, compressed together and written to the largest field.
        # This avoids the per-record-over-threshold quirk that makes the
        # sibling-record path leave small individual records untouched.
        big_response = "GITHUB SEARCH RESULT " * (hook.THRESHOLD // 15 + 50)
        self.assertGreater(len(big_response), hook.THRESHOLD)
        payload = {
            "session_id": "sess-1",
            "cwd": "/repos/my-project",
            "tool_name": "mcp__github__search_code",
            "tool_response": big_response,
        }
        printed = self._run_main(payload)
        self.assertIn("updatedToolOutput", printed)

    def test_small_github_list_issues_is_left_untouched(self):
        # A small response (no issues, or a few short titles) must be a true
        # no-op -- compression should not run just because the tool name matches.
        small_response = {"total_count": 0, "items": []}
        payload = {
            "session_id": "sess-1",
            "cwd": "/repos/my-project",
            "tool_name": "mcp__github__list_issues",
            "tool_response": small_response,
        }
        printed = self._run_main(payload)
        # No hookSpecificOutput at all -- genuine under-threshold no-op.
        self.assertEqual(printed, "")
        self.assertEqual(len(self.stub.calls), 0)

    def test_large_splunk_search_result_gets_compressed(self):
        # mcp__claude_ai_Splunk__search_datadog_logs returns rows of log
        # events -- read for gist (did the deploy succeed? any errors?),
        # never used as a byte-exact source to edit from.
        # Sized to guarantee the aggregate is over THRESHOLD regardless of
        # the configured value (CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS can
        # override the 2000 default -- this repo's dogfood shell sets 4000).
        line_count = hook.THRESHOLD // 50 + 10
        big_log = "\n".join(
            f"2025-01-01T00:00:00Z level=info msg=SPLUNK_LOG_LINE_{i} " + "x" * 40
            for i in range(line_count)
        )
        self.assertGreater(len(big_log), hook.THRESHOLD)
        payload = {
            "session_id": "sess-1",
            "cwd": "/repos/my-project",
            "tool_name": "mcp__claude_ai_Splunk__search_datadog_logs",
            "tool_response": big_log,
        }
        printed = self._run_main(payload)
        self.assertIn("updatedToolOutput", printed)

    def test_github_search_code_savings_event_recorded(self):
        # Confirms the generic mcp__ path records a savings event via
        # _finish_compression_outcome, which logs the tool as "hook:{tool_name}"
        # -- the same "hook:" prefix used for all non-footer paths (Bash, Grep,
        # etc. included), mirroring the existing assertion at test case
        # test_large_no_footer_response_gets_compressed_and_recorded (line 308).
        # Sized dynamically so this passes regardless of the configured threshold.
        big_response = "SEARCH RESULTS " * (hook.THRESHOLD // 10 + 50)
        self.assertGreater(len(big_response), hook.THRESHOLD)
        payload = {
            "session_id": "sess-2",
            "cwd": "/repos/my-project",
            "tool_name": "mcp__github__search_issues",
            "tool_response": big_response,
        }
        self._run_main(payload)
        self.assertEqual(len(self.stub.calls), 1)
        args, kwargs = self.stub.calls[0]
        # _finish_compression_outcome logs the tool as "hook:{tool_name}".
        self.assertIn("mcp__github__search_issues", args[1])
        self.assertEqual(kwargs.get("project"), "my-project")

    def test_mcp_local_compress_path_still_takes_precedence(self):
        # Regression guard: the new generic mcp__ branch must come AFTER
        # the mcp__local-compress__ check, not before it -- a local-compress
        # tool with a savings footer must still go through the footer-strip
        # path and NOT be re-processed generically.
        footer = json.dumps({
            "tool": "compress_file", "raw_tokens": 500, "out_tokens": 50,
            "credited": True, "source": "some/file.log",
        })
        payload = {
            "session_id": "sess-3",
            "cwd": "/repos/my-project",
            "tool_name": "mcp__local-compress__compress_file",
            "tool_response": f"[compressed]\n<!--CLAUDE_RUNWAY_SAVINGS:{footer}-->",
        }
        self._run_main(payload)
        self.assertEqual(len(self.stub.calls), 1)
        args, _ = self.stub.calls[0]
        # Footer path tags the event with the footer's own tool name, not
        # "hook:mcp__local-compress__compress_file" -- unchanged behavior.
        self.assertEqual(args[1], "compress_file")


if __name__ == "__main__":
    unittest.main()
