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
import re
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))

import compress_bash_output as hook  # noqa: E402
from compress_bash_output import _is_exactness_critical  # noqa: E402

# The other three stdin-reading hooks, imported for the issue #263 stdin-encoding
# regression tests at the bottom of this file. Each is a standalone script with
# its own main(); none does network I/O on import.
import record_session_id as record_session_id_hook  # noqa: E402
import redirect_webfetch_to_fetch_url as redirect_hook  # noqa: E402
import session_end_savings as session_end_hook  # noqa: E402


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
        "gh api repos/Donelle/claude-runway/pulls/190/comments --paginate",
        "plain gh api call with no --jq/--json at all (my-gh-pr-feedback's shape)",
    ),
    (
        "gh api graphql -f query='query { repository(name: \"x\") { pullRequest(number: 1) "
        "{ reviewThreads(first: 100) { nodes { id comments(first: 1) { nodes { body } } } } } } }'",
        "graphql body-fetch shape",
    ),
    (
        'COMMENT_IDS=$(gh api repos/Donelle/claude-runway/pulls/190/comments '
        '--paginate --jq ".[] | select(.user.login != \\"me\\") | \\"comment:\\" + (.id|tostring)") '
        "|| FETCH_FAILED=1",
        "the real VAR=$(...) command-substitution shape used by my-gh-autowork's NEW_IDS diffing "
        "-- a start-anchored fix would have missed this",
    ),
    # --- issue #268: wrappers that used to hide the command from `^` anchoring ---
    ('FILES=$(git diff --name-only); echo "$FILES"', "VAR=$(cmd) command substitution"),
    ("HASH=$(git rev-parse HEAD)", "captured hash in a substitution"),
    ("(git diff) 2>&1", "subshell wrapper"),
    ("{ git diff; }", "brace group wrapper"),
    ('echo "$(git rev-parse HEAD)"', "substitution inside double quotes"),
    ("X=`git rev-parse HEAD`", "backtick substitution"),
    ("N=$(cat f | wc -l)", "wc inside a substitution"),
    ("A=1 B=$(sha256sum f)", "leading assignment before a substitution"),
    ("npm list", "documented alias of npm ls"),
    ("npm list --depth=0", "npm list with flags"),
    ("kubectl get pods -o=json", "-o=json spelling"),
    ("kubectl get pods -ojson", "attached -ojson spelling"),
    ("kubectl get pods -o=YAML", "case-insensitive value on the = spelling"),
    # --- issue #269: the exemptions that must SURVIVE the false-positive fixes ---
    ("env", "bare env listing"),
    ("env | sort", "env piped -- still a variable listing"),
    ("env -i FOO=1 git status", "env launching an exempt command"),
    ("env -u HOME pwd", "env with -u NAME before an exempt command"),
    ("cat <<EOF | jq .\n{}\nEOF", "heredoc opener line still carries its own pipeline"),
    ("cat <<EOF\nbody\nEOF\ngit status", "exempt command AFTER a heredoc terminator"),
    ("kubectl get pods -o json | tee x.txt", "-o json as a real flag"),
    ('git status "x y"', "quoted arg must not hide a command-start exemption"),
    # --- review round 1 on PR #390: forms the first fix wrongly un-exempted ---
    ("env --unset HOME git diff", "env option with a separate operand (--unset)"),
    ("env -C /tmp git diff", "env -C DIR operand"),
    ("env --chdir /tmp git diff", "env --chdir DIR operand"),
    ("env -- git status", "-- ends env's options"),
    ("env FOO=1 env BAR=2 git diff", "nested env wrappers"),
    ("cat <<EOF\n$(git diff)\nEOF", "unquoted heredoc: substitution in the body executes"),
    ("cat <<EOF\n`git diff`\nEOF", "unquoted heredoc: backtick in the body executes"),
    ("bash <<'EOF'\ngit diff\nEOF", "heredoc body fed to a shell interpreter is a script"),
    ("bash -c 'gh api repos/x/y/issues --paginate'", "gh api inside a quoted payload still executes"),
    ("cat <<EOF\nEOF\ngit diff\ncat <<EOF\nnotes\nEOF", "empty heredoc must end at its FIRST terminator"),
    # --- review round 2 on PR #390 ---
    ("cat <<EOF\n$(\ngit diff\n)\nEOF", "multiline substitution in an unquoted heredoc"),
    ("cat <<EOF\n`\ngit diff\n`\nEOF", "multiline backtick substitution"),
    ("cat <<'EOF' | bash\ngit diff\nEOF", "heredoc piped into an interpreter on the opener line"),
    ("env -S 'git diff' HEAD", "env -S command string: uncertain parse keeps the exemption"),
    ("bash -c 'kubectl get pods -o json'", "quoted shell payload with a real -o json"),
    ("bash -c 'aws s3api list-buckets --output json'", "quoted shell payload with --output json"),
    ("bash -c 'git diff'", "quoted shell payload is itself an exempt command"),
    # --- review round 3 on PR #390 ---
    ('echo "<<EOF"\ngit diff\ncat <<EOF\nnotes\nEOF', "quoted << is not a heredoc opener"),
    ("echo hi # <<EOF\ngit diff\ncat <<EOF\nnotes\nEOF", "<< inside a comment is not an opener"),
    ("cat <<EOF-ONE\ngit diff\nEOF\nnotes\nEOF", "delimiter must be a complete token"),
    ("ssh host <<'EOF'\ngit diff\nEOF", "ssh feeds the heredoc to a remote shell"),
    ("cat <<'EOF' \\\n| bash\ngit diff\nEOF", "backslash-continued opener"),
    ('env PATH="$PATH":/opt/bin git diff', "composite env assignment value"),
    ("env 'FOO=a b' git diff", "fully quoted env assignment word"),
    ('env CI=1 "git" diff', "quoted launched executable"),
    ('env GIT_CONFIG_GLOBAL="$(printf /dev/null)" git diff', "assignment value split at $("),
    # --- review round 4 on PR #390 ---
    ('env NOTE="say \\"hello world\\" now" git diff', "escaped quotes in an assignment value"),
    ("env LABEL=a\\ b git diff", "backslash-escaped space in an assignment value"),
    ("cat <<'DOC-TEXT'\nexample <<EOF\nDOC-TEXT\ngit diff\ncat <<EOF\nnotes\nEOF", "non-word delimiter"),
    ("bash \\\n<<'EOF'\ngit diff\nEOF", "interpreter on a continued preceding line"),
    ('echo "\n<<EOF\n"\ngit diff\ncat <<EOF\nnotes\nEOF', "quote state carried across lines"),
    ('env -C "$PWD"/. git diff', "adjacent quoted/unquoted env operand"),
    ("env CI=1 /usr/bin/git diff", "launched executable given by path"),
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
    # --- issue #268 near-misses: new splitting must not expose fake segments ---
    ('dotnet test --filter "(id=1)"', "mid-segment paren must not split off `id=1`"),
    ("npm listen", "`npm list` must not match a longer word"),
    ("tool --foo-ojson x", "-ojson only counts as its own flag"),
    ("X=$(dotnet build)", "substitution around a non-exempt command"),
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
    # --- issue #269: false positives that used to forfeit compression ---
    ("env CI=1 npm test", "env as a launcher, not a listing"),
    ("env -i PATH=/bin dotnet build", "env with options launching a non-exempt command"),
    ("cat <<EOF\nfoo\ngit status is a verification\nEOF", "heredoc body line starting with an exempt word"),
    ("cat <<'EOF' > f.sh\nenv\nwc -l x\nEOF", "quoted-delimiter heredoc, several exempt-looking body lines"),
    ("cat <<-EOF\n\tgit log\n\tEOF", "<<- heredoc with indented terminator"),
    ('echo "try --json here" && dotnet build', "flag text inside a quoted argument"),
    ("echo 'use --porcelain' && npm test", "flag text inside single quotes"),
    ("curl -o json.txt https://example.com", "-o with a filename that starts with json"),
    ("curl --output yaml.out https://example.com", "--output with a yaml-prefixed filename"),
    ("echo $((1<<3)) && dotnet build", "arithmetic << is not a heredoc"),
    ("env -C /tmp npm test", "env with an operand option launching a non-exempt command"),
    ("env FOO=1 env BAR=2 npm test", "nested env launching a non-exempt command"),
    ("cat <<EOF\nEOF\ndotnet build", "empty heredoc followed by a non-exempt command"),
    ("bash -c 'dotnet build'", "quoted shell payload that is not exempt"),
    ("cat <<EOF\n$HOME git status\nEOF", "unquoted heredoc without a substitution is data"),
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


class ExactnessCriticalIsLinearTime(unittest.TestCase):
    def test_adversarial_assignment_prefix_does_not_backtrack_exponentially(self):
        # `0="" 0="" ...` made the old `\S*`-based assignment prefix exponential
        # (CodeQL flagged the env twin of it on PR #390); 30 words took many
        # seconds before, and is instant now. Bound is generous to avoid flakes.
        import time

        for cmd in ("0=" + '"" 0=' * 30, "env " + '0="" ' * 30 + "x"):
            start = time.monotonic()
            _is_exactness_critical(cmd)
            self.assertLess(time.monotonic() - start, 2.0, cmd[:40])


class CompileExtraExactPatterns(unittest.TestCase):
    """Issue #192: CLAUDE_RUNWAY_EXTRA_EXACT_PATTERNS lets a project add its
    own exactness-critical patterns on top of the built-in list, without
    editing this shared hook file. Tests _compile_extra_exact_patterns()
    directly against the real env var, restoring whatever was there
    beforehand so this doesn't leak into other tests in the same process."""

    def setUp(self):
        self._real_env = os.environ.get("CLAUDE_RUNWAY_EXTRA_EXACT_PATTERNS")

    def tearDown(self):
        if self._real_env is None:
            os.environ.pop("CLAUDE_RUNWAY_EXTRA_EXACT_PATTERNS", None)
        else:
            os.environ["CLAUDE_RUNWAY_EXTRA_EXACT_PATTERNS"] = self._real_env

    def test_unset_yields_no_patterns(self):
        os.environ.pop("CLAUDE_RUNWAY_EXTRA_EXACT_PATTERNS", None)
        self.assertEqual(hook._compile_extra_exact_patterns(), [])

    def test_blank_string_yields_no_patterns(self):
        os.environ["CLAUDE_RUNWAY_EXTRA_EXACT_PATTERNS"] = ""
        self.assertEqual(hook._compile_extra_exact_patterns(), [])

    def test_semicolon_separated_patterns_all_compiled(self):
        os.environ["CLAUDE_RUNWAY_EXTRA_EXACT_PATTERNS"] = r"internal-cli;my-hash-tool\b"
        patterns = hook._compile_extra_exact_patterns()
        self.assertEqual(len(patterns), 2)
        self.assertTrue(any(p.search("run internal-cli --dump") for p in patterns))
        self.assertTrue(any(p.search("my-hash-tool") for p in patterns))

    def test_blank_entries_between_semicolons_are_ignored(self):
        # A trailing/leading/doubled semicolon (easy to type by hand in a
        # shell export) must not produce an empty-string pattern that would
        # match every command via re.search("", ...).
        os.environ["CLAUDE_RUNWAY_EXTRA_EXACT_PATTERNS"] = " ;internal-cli; ; "
        patterns = hook._compile_extra_exact_patterns()
        self.assertEqual(len(patterns), 1)
        self.assertTrue(patterns[0].search("internal-cli"))

    def test_invalid_pattern_is_skipped_not_fatal(self):
        # Fail open PER-PATTERN: a typo in one entry must not take down the
        # other, valid entries alongside it -- same precedent as the
        # ImportError/stale-env-var fail-open paths elsewhere in this file.
        os.environ["CLAUDE_RUNWAY_EXTRA_EXACT_PATTERNS"] = r"good-one;(unbalanced["
        buf = io.StringIO()
        with redirect_stderr(buf):
            patterns = hook._compile_extra_exact_patterns()
        self.assertEqual(len(patterns), 1)
        self.assertTrue(patterns[0].search("good-one"))
        self.assertIn("skipping invalid", buf.getvalue())


class ExtraExactPatternsAffectIsExactnessCritical(unittest.TestCase):
    """End-to-end through _is_exactness_critical() itself, monkeypatching the
    module-level compiled list the same way other tests here monkeypatch
    hook.compress/hook.savings_ledger -- avoids needing a real module reload
    just to exercise a different env var value."""

    def setUp(self):
        self._real_patterns = hook._EXTRA_EXACT_PATTERNS

    def tearDown(self):
        hook._EXTRA_EXACT_PATTERNS = self._real_patterns

    def test_extra_pattern_exempts_a_project_specific_command(self):
        hook._EXTRA_EXACT_PATTERNS = [re.compile(r"internal-hash-tool")]
        self.assertTrue(_is_exactness_critical("internal-hash-tool --for record.json"))

    def test_extra_patterns_are_additive_builtin_list_still_works(self):
        # No extra patterns configured must not regress the builtin list --
        # this env var can only ADD checks, never replace the existing ones.
        hook._EXTRA_EXACT_PATTERNS = []
        self.assertTrue(_is_exactness_critical("git status --short"))

    def test_unrelated_command_is_unaffected_by_an_extra_pattern(self):
        hook._EXTRA_EXACT_PATTERNS = [re.compile(r"internal-hash-tool")]
        self.assertFalse(_is_exactness_critical("dotnet build"))

    def test_compress_ok_escape_hatch_still_beats_an_extra_pattern(self):
        # The existing opt-out must apply uniformly -- a project's own
        # pattern doesn't get a separate, un-overridable exemption path.
        hook._EXTRA_EXACT_PATTERNS = [re.compile(r"internal-hash-tool")]
        self.assertFalse(_is_exactness_critical("internal-hash-tool --for record.json # compress-ok"))

    def test_extra_pattern_matches_anywhere_in_a_segment(self):
        # Matched anywhere (like _EXACT_FLAG_RE), not start-anchored (like
        # _EXACT_CMD_RE) -- so a project's pattern still fires even when
        # wrapped in shell command substitution, the same VAR=$(...) blind
        # spot issue #190 found for a start-anchored `gh api` addition.
        hook._EXTRA_EXACT_PATTERNS = [re.compile(r"internal-hash-tool")]
        self.assertTrue(_is_exactness_critical('ID=$(internal-hash-tool --for record.json)'))


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
    through to record_event is what tags each transient event with its
    project. Originally (issue #35) this is what let current_session_id()'s
    project-filtered lookup tell this session's events apart from an
    unrelated project's; as of issue #213 that lookup reads session_id_lib's
    shadow markers instead, so this forwarding is now informational/
    debugging coverage only -- see savings_ledger.record_event's docstring."""

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
        # become an empty string or the literal word "None" -- honesty of
        # the stored value matters even though (issue #213) this field no
        # longer drives current_session_id()'s project-filtered lookup.
        hook._record_savings_event("sess-1", "hook:Bash", 1000, 100, True, "cmd")
        _, kwargs = self.stub.calls[0]
        self.assertIsNone(kwargs.get("project"))

    def test_forwards_agent_fields_to_record_event(self):
        # Issue #365: which agent earned the saving rides along on the event.
        hook._record_savings_event(
            "sess-1", "hook:Bash", 1000, 100, True, "cmd",
            project="my-repo", agent_id="agent-1", agent_type="Explore",
        )
        _, kwargs = self.stub.calls[0]
        self.assertEqual(kwargs.get("agent_id"), "agent-1")
        self.assertEqual(kwargs.get("agent_type"), "Explore")

    def test_agent_fields_default_to_none(self):
        # A main-session event must be recorded exactly as before #365.
        hook._record_savings_event("sess-1", "hook:Bash", 1000, 100, True, "cmd")
        _, kwargs = self.stub.calls[0]
        self.assertIsNone(kwargs.get("agent_id"))
        self.assertIsNone(kwargs.get("agent_type"))


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


class MainThreadsAgentFieldsToRecordEvent(unittest.TestCase):
    """Issue #365: end-to-end through main() -- the payload's agent_id/agent_type
    reach record_event on EVERY recording path (MCP footer, generic MCP, Bash),
    not just the one _record_savings_event unit test above covers. Uses the same
    stub ledger / fake compress as the neighboring classes, so no live model."""

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

    def _single_call_kwargs(self):
        self.assertEqual(len(self.stub.calls), 1)
        return self.stub.calls[0][1]

    def test_mcp_footer_path(self):
        footer = json.dumps({"tool": "compress_file", "raw_tokens": 1000, "out_tokens": 100,
                             "credited": True, "source": "f.log"})
        self._run_main({
            "session_id": "s", "cwd": "/repos/p",
            "agent_id": "agent-9", "agent_type": "general-purpose",
            "tool_name": "mcp__local-compress__compress_file",
            "tool_response": f"[compressed 1000 -> 100 chars]\n<!--CLAUDE_RUNWAY_SAVINGS:{footer}-->",
        })
        kwargs = self._single_call_kwargs()
        self.assertEqual(kwargs.get("agent_id"), "agent-9")
        self.assertEqual(kwargs.get("agent_type"), "general-purpose")

    def test_generic_mcp_path(self):
        self._run_main({
            "session_id": "s", "cwd": "/repos/p",
            "agent_id": "agent-9", "agent_type": "Explore",
            "tool_name": "mcp__local-compress__compact_find",
            "tool_response": "STORED COMPACT CONTENT. " * 200,
        })
        kwargs = self._single_call_kwargs()
        self.assertEqual(kwargs.get("agent_id"), "agent-9")
        self.assertEqual(kwargs.get("agent_type"), "Explore")

    def test_bash_path(self):
        self._run_main({
            "session_id": "s", "cwd": "/repos/p",
            "agent_id": "agent-9", "agent_type": "Explore",
            "tool_name": "Bash",
            "tool_input": {"command": "cat big.log"},
            "tool_response": {"stdout": "LOG LINE. " * 600, "stderr": ""},
        })
        kwargs = self._single_call_kwargs()
        self.assertEqual(kwargs.get("agent_id"), "agent-9")
        self.assertEqual(kwargs.get("agent_type"), "Explore")

    def test_main_session_payload_without_agent_fields_passes_none(self):
        self._run_main({
            "session_id": "s", "cwd": "/repos/p",
            "tool_name": "Bash",
            "tool_input": {"command": "cat big.log"},
            "tool_response": {"stdout": "LOG LINE. " * 600, "stderr": ""},
        })
        kwargs = self._single_call_kwargs()
        self.assertIsNone(kwargs.get("agent_id"))
        self.assertIsNone(kwargs.get("agent_type"))

    def test_main_thread_started_with_agent_flag_carries_type_without_id(self):
        # `claude --agent X`: agent_type present, agent_id absent -- forwarded as-is
        # (the ledger's aggregation, not the hook, decides this is NOT a subagent).
        self._run_main({
            "session_id": "s", "cwd": "/repos/p", "agent_type": "my-agent",
            "tool_name": "Bash",
            "tool_input": {"command": "cat big.log"},
            "tool_response": {"stdout": "LOG LINE. " * 600, "stderr": ""},
        })
        kwargs = self._single_call_kwargs()
        self.assertIsNone(kwargs.get("agent_id"))
        self.assertEqual(kwargs.get("agent_type"), "my-agent")


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


class TimeoutErrorFlowsIntoTheWasntCompressedNote(unittest.TestCase):
    """Issue #368: the hook embeds compress()'s "Error: ..." string verbatim, so
    a timeout's "may be busy" wording must reach the user, not the misleading
    "check it's still running"."""

    def test_timeout_wording_is_in_the_note(self):
        message = (
            "Error: LM Studio request timed out after 60s (chunk 1/2) -- LM Studio is "
            "may be busy with other requests or unreachable; see "
            "CLAUDE_RUNWAY_LMSTUDIO_TIMEOUT_SECONDS."
        )
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            hook._finish_compression_outcome(
                ("error", "x" * 5000, message), "Bash", "Bash", "sess-1", "/repos/my-project",
            )
        self.assertEqual(cm.exception.code, 0)
        note = json.loads(buf.getvalue())["hookSpecificOutput"]["additionalContext"]
        self.assertIn("wasn't compressed", note)
        self.assertIn("may be busy", note)
        self.assertNotIn("check it's still running", note)


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
    mcp__github__search_code) must go through
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


def _windows_style_stdin(payload_bytes):
    """Return a text stream that mimics how Python opens stdin on a Western
    Windows install: a TextIOWrapper over the raw payload bytes whose encoding
    is cp1252, NOT utf-8 (issue #263). Building it explicitly (rather than
    relying on the test runner actually being Windows) makes these tests
    deterministic on all three CI OSes -- cp1252 is a built-in codec
    everywhere. A hook's main() is expected to reconfigure this stream to
    utf-8 before reading it; these tests verify it does."""
    return io.TextIOWrapper(io.BytesIO(payload_bytes), encoding="cp1252", newline="")


def _utf8_stdin_bytes(payload):
    """Serialize a payload the way Claude Code actually writes it to a hook's
    stdin: raw UTF-8 JSON with non-ASCII left as literal multi-byte sequences
    (ensure_ascii=False), NOT backslash-uXXXX-escaped ASCII. The escaped form
    would make every byte ASCII and hide the encoding bug entirely (json.dumps
    defaults to ensure_ascii=True)."""
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


# A sentinel exercising BOTH failure modes the issue describes at once:
#   - multi-byte UTF-8 that cp1252 silently mojibakes: U+00E9 (e-acute) and
#     U+2192 (right arrow) -- routine in build logs / fetched pages; and
#   - U+0081, whose UTF-8 encoding is 0xC2 0x81. 0x81 is one of cp1252's five
#     unmapped bytes, so reading these bytes as cp1252 raises UnicodeDecodeError
#     on the 0x81, while a correct utf-8 read keeps it. That unmapped byte is
#     what makes the cp1252-vs-utf-8 difference an observable RAISE, not merely
#     cosmetic mojibake. All three chars are written as \u escapes so no
#     editor/terminal round-trip can silently drop them.
_E_ACUTE = "é"
_ARROW = "→"
_CTRL_0081 = ""  # UTF-8: C2 81 -> 0x81 unmapped in cp1252
_UTF8_SENTINEL = "caf" + _E_ACUTE + " " + _ARROW + " " + _CTRL_0081 + " end"


class HooksReconfigureStdinToUtf8(unittest.TestCase):
    """Issue #263: all four stdin-reading hooks must force stdin to utf-8
    before json.load, because Windows Python defaults stdin to the locale
    codepage (cp1252). Left unfixed, multi-byte UTF-8 decodes as mojibake
    (which compress_bash_output would then splice back as the tool's output)
    and cp1252-unmapped bytes raise UnicodeDecodeError that the fail-open
    json.load guard swallows -- silently skipping compression / marker writes
    for exactly the riskiest payloads. These tests feed a simulated
    Windows-cp1252 stdin carrying real UTF-8 bytes and confirm each hook
    decodes them correctly rather than corrupting or dropping the payload."""

    def test_vehicle_is_real_cp1252_would_raise(self):
        # Guards the test itself: prove the simulated stdin genuinely misbehaves
        # under cp1252 the way the bug describes, so a passing per-hook test
        # below means the reconfigure actually fired rather than the vehicle
        # being a no-op. The 0xC2 0x81 (U+0081) sequence is valid utf-8 but
        # 0x81 is unmapped in cp1252 -- reading the stream as cp1252 raises.
        raw = _utf8_stdin_bytes({"x": _UTF8_SENTINEL})
        with self.assertRaises(UnicodeDecodeError):
            _windows_style_stdin(raw).read()

    def _run_hook_main(self, main_callable, payload):
        """Drive a hook's main() with a Windows-cp1252 stdin carrying `payload`
        as real UTF-8 bytes. Re-raises anything that isn't SystemExit, so a
        hook that crashes on the UTF-8 payload (the pre-fix behavior) fails the
        test loudly instead of silently."""
        raw = _utf8_stdin_bytes(payload)
        real_stdin = sys.stdin
        sys.stdin = _windows_style_stdin(raw)
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                try:
                    main_callable()
                except SystemExit:
                    pass  # hooks exit(0) on their fail-open / no-op paths
        finally:
            sys.stdin = real_stdin
        return out.getvalue()

    def test_compress_hook_decodes_utf8_and_does_not_mojibake(self):
        # The highest-stakes hook: it compresses tool_response and can splice
        # the result back as Claude's view of the output. Capture what reaches
        # the compress call and assert it's the real UTF-8 text, not mojibake.
        seen = {}

        async def _capture_compress(text, *a, **k):
            seen["text"] = text
            return "[compressed %d -> 4 chars across 1 chunk(s), ~99%% smaller]\nGIST" % len(text)

        real_compress = hook.compress
        hook.compress = _capture_compress
        try:
            payload = {
                "session_id": "s",
                "cwd": "/repos/p",
                "tool_name": "Bash",
                "tool_input": {"command": "run-build"},
                # _handle_bash expects the {stdout, stderr} dict shape. Pad
                # stdout past the compression threshold so it actually calls
                # compress() rather than taking the under-threshold short-circuit.
                "tool_response": {
                    "stdout": _UTF8_SENTINEL + ("x" * (hook.THRESHOLD + 100)),
                    "stderr": "",
                },
            }
            self._run_hook_main(hook.main, payload)
        finally:
            hook.compress = real_compress
        self.assertIn("text", seen, "compress() was never reached -- payload failed to parse")
        # The decoded text must contain the real UTF-8 chars, never their
        # cp1252 mojibake (e.g. 'caf' + mojibake for the e-acute).
        self.assertIn(_E_ACUTE, seen["text"])
        self.assertIn(_ARROW, seen["text"])
        self.assertNotIn("Ã©", seen["text"])  # 'Ã©', the cp1252 mojibake of é

    def test_record_session_id_hook_parses_utf8_payload_without_crashing(self):
        # A SessionEnd payload (no tool_name) whose cwd carries non-ASCII.
        # Under the bug this raised UnicodeDecodeError, swallowed into a silent
        # skip of the shadow-marker write; after the fix it parses and
        # dispatches cleanly. The hook is fail-open by design, so a clean
        # (non-raising) run through main() is the observable success here.
        payload = {"hook_event_name": "SessionEnd", "session_id": "s", "cwd": "/repos/" + _UTF8_SENTINEL}
        self._run_hook_main(record_session_id_hook.main, payload)

    def test_redirect_hook_parses_utf8_payload_without_crashing(self):
        # A non-WebFetch payload: the hook parses it, sees tool_name != WebFetch,
        # and allows. The point is that the parse itself survives the UTF-8 bytes
        # on a cp1252-default stdin rather than raising into fail-open.
        payload = {"tool_name": "Bash", "tool_input": {"command": _UTF8_SENTINEL}}
        self._run_hook_main(redirect_hook.main, payload)

    def test_session_end_savings_hook_parses_utf8_payload_without_crashing(self):
        # Tracking is off by default, so main() parses then no-ops at the
        # tracking_enabled() gate -- but it must REACH that gate, i.e. the UTF-8
        # payload must parse under the cp1252-default stdin first.
        payload = {"session_id": "s", "cwd": "/repos/" + _UTF8_SENTINEL, "transcript_path": None}
        self._run_hook_main(session_end_hook.main, payload)


class InvalidThresholdEnvTests(unittest.TestCase):
    """Issue #270: a malformed CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS used to
    raise ValueError at import time, killing the hook on every matched call.
    Run in a real subprocess because the crash happens at module import."""

    HOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks", "compress_bash_output.py")

    def _run(self, env_value):
        import subprocess

        env = dict(os.environ, CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS=env_value)
        payload = {"tool_name": "Bash", "tool_input": {"command": "echo hi"},
                   "tool_response": {"stdout": "hi", "stderr": ""}}
        return subprocess.run([sys.executable, self.HOOK], input=json.dumps(payload),
                              capture_output=True, text=True, env=env, timeout=60)

    def test_non_numeric_values_fall_back_with_stderr_warning(self):
        for bad in ("4,000", "2k", "abc", ""):
            with self.subTest(value=bad):
                r = self._run(bad)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertNotIn("Traceback", r.stderr)
                self.assertIn("invalid CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS", r.stderr)

    def test_parse_threshold_function(self):
        old = os.environ.get("CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS")
        try:
            os.environ["CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS"] = "4,000"
            with redirect_stderr(io.StringIO()):
                self.assertEqual(hook._parse_threshold(), hook._DEFAULT_THRESHOLD)
            os.environ["CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS"] = " 4000 "
            self.assertEqual(hook._parse_threshold(), 4000)
        finally:
            if old is None:
                os.environ.pop("CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS", None)
            else:
                os.environ["CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS"] = old


if __name__ == "__main__":
    unittest.main()
