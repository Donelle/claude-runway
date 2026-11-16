#!/usr/bin/env python3
"""Tests for skills/cr-load-context: the skill that loads a project's
context at the start of a session, and the `facts.sh` script it runs.

A skill has no unit test of its behaviour, so three things are pinned:

- the instructions in SKILL.md that the design depends on (git and the MCP tools are
  the only prerequisites, `gh`/`az` are optional, no ticket lookup, the three rounds, never reading
  `.mcp.json`, `--porcelain` on the script, `/cr-doctor` suggested and never run);
- that `init --install-skills` ships the whole folder, script included;
- `facts.sh` itself, run for real against throwaway git repos with STUB `gh` and `az`
  executables on PATH (no network, no real forge): GitHub and Azure DevOps remotes in
  every URL form, a missing or signed-out CLI, a disabled repo (`isDisabled` and
  `TF401019`), no ticket section, and that no credential or `.mcp.json` content ever
  reaches the output.

    .venv/bin/python -m unittest discover -s tests
"""

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO_ROOT / "skills" / "cr-load-context"
SKILL_PATH = SKILL_DIR / "SKILL.md"
FACTS = SKILL_DIR / "facts.sh"

sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "hooks"))

import setup_project_lib as lib  # noqa: E402
from compress_output import _is_exactness_critical  # noqa: E402

BASH = "/bin/bash" if Path("/bin/bash").exists() else shutil.which("bash")


def _section(text: str, start: str, end: str | None = None) -> str:
    i = text.index(start)
    j = text.index(end, i + len(start)) if end else len(text)
    return text[i:j]


class CrLoadContextSkillTest(unittest.TestCase):
    def setUp(self):
        self.text = SKILL_PATH.read_text(encoding="utf-8")

    def test_uses_the_repos_skill_layout(self):
        self.assertTrue(self.text.startswith("# Skill: cr-load-context\n"))
        for heading in ("## Prerequisites", "## Usage", "## Steps"):
            self.assertIn(heading, self.text)
        # Installing is the installer's job (`init --install-skills`), not the skill's.
        self.assertNotIn("## Installation", self.text)

    def test_prerequisites_are_git_and_the_mcp_tools_only(self):
        prereq = _section(self.text, "## Prerequisites", "## Usage")
        self.assertIn("`git`", prereq)
        for server in ("compact", "codebase-indexer", "local-compress", "memory-bank"):
            self.assertIn(f"`{server}`", prereq)
        # gh and az appear only as optional extras, never as requirements.
        self.assertIn("optional", prereq)
        self.assertIn("Nothing else is required", prereq)
        # Everything named before the "optional extras" sentence is the required list.
        required = prereq[: prereq.index("Nothing else is required")]
        for optional in ("`gh`", "`az`"):
            self.assertNotIn(optional, required)

    def test_the_hand_run_fallback_never_prints_the_origin_with_its_credential(self):
        # facts.sh strips userinfo from the origin URL, so the by-hand fallback must
        # too: a plain `git remote get-url origin` prints an HTTPS password or token
        # straight into the model context.
        found = 0
        for line in self.text.splitlines():
            for m in re.finditer(r"git remote get-url origin|remote\.origin\.url", line):
                found += 1
                guarded = (
                    "| sed -E" in line[m.end():]      # piped through the userinfo strip
                    or ") -replace" in line[m.end():]  # the PowerShell form
                    or "Never run a plain" in line     # the warning itself
                )
                self.assertTrue(guarded, f"unsanitized origin read in SKILL.md: {line.strip()[:120]}")
        self.assertGreaterEqual(found, 3)  # sed form, PowerShell form, and the warning

    def test_a_missing_optional_tool_skips_one_section_and_stops_nothing(self):
        self.assertIn("`skipped: <reason>`", self.text)
        self.assertIn("No step below may stop because of them", self.text)

    def test_round_one_is_one_message_with_all_five_calls(self):
        round1 = _section(self.text, "### 1.", "### 2.")
        self.assertIn("ONE message", round1)
        for call in ("cb_indexer_sync_repo", "facts.sh", "cb_indexer_get_collection_info", "list_local_models", "recall"):
            self.assertIn(call, round1)
        # cb_indexer_sync_repo runs in the background and nothing waits on it.
        self.assertIn("background", round1)
        self.assertIn("Nothing waits on it", round1)

    def test_sync_repo_is_not_started_for_a_project_with_no_index(self):
        # Option A: with no manifest, cb_indexer_sync_repo would be a full rebuild, which the
        # epic says is the user's to ask for. The background subagent checks first and does not call it.
        round1 = _section(self.text, "### 1.", "### 2.")
        self.assertIn("only when the project already has an index", round1)
        self.assertIn("Bash `test -f`", round1)
        self.assertIn("do NOT call `cb_indexer_sync_repo`", round1)
        self.assertIn("`no manifest: sync not started`", round1)
        # Without the Agent tool the skill calls it itself, and only after the facts say present.
        self.assertIn("`sync manifest: present`", round1)
        report = _section(self.text, "### 4. Report")
        self.assertIn("suggest `cb_indexer_index_repo`", report)

    def test_windows_runs_facts_ps1_and_never_assumes_bash_or_wsl(self):
        text = _section(self.text, "### Which script to run", "### 2.")
        self.assertIn("facts.ps1", text)
        self.assertIn("-ExecutionPolicy Bypass -File", text)
        self.assertIn("--porcelain", text)
        self.assertIn("Do not assume bash exists on Windows", text)
        self.assertIn("never use WSL", text)
        prereq = _section(self.text, "## Prerequisites", "## Usage")
        self.assertIn("bash is not a prerequisite on Windows", prereq)

    def test_the_by_hand_fallback_quotes_the_json_list_for_powershell(self):
        # An unquoted `--json a,b` is an array in PowerShell and gh rejects it.
        text = _section(self.text, "### Which script to run", "### 2.")
        self.assertIn("--json 'number,title'", text)

    def test_facts_sh_is_pinned_to_lf_line_endings(self):
        # A CRLF checkout (git autocrlf on Windows) breaks a bash script, so the repo says LF.
        attrs = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("*.sh text eol=lf", attrs)

    def test_the_script_is_run_with_a_bare_porcelain_flag(self):
        self.assertIn('bash "<skill folder>/facts.sh" --porcelain', self.text)
        # The hook exempts the real command line, so the output is never summarized.
        self.assertTrue(_is_exactness_critical('bash "/home/u/.claude/skills/cr-load-context/facts.sh" --porcelain'))

    def test_a_tool_guide_for_each_live_server_is_in_the_skill_but_not_recited(self):
        # A project may lack claude-runway's CLAUDE.md guidance, so the skill carries it, scoped
        # to the servers that answered, for the model's own use.
        guide = _section(self.text, "### 3. Learn the live tools", "### 4. Report")
        self.assertIn("live", guide)
        self.assertIn("Skip the entry of a group that is missing or errored", guide)
        self.assertIn("do not recite it in the report", guide)
        for server in ("compact", "codebase-indexer", "memory-bank", "local-compress"):
            self.assertIn(f"**`{server}`**", guide)
        # The rules that change what the model does, one or two per server.
        for rule in (
            "use Grep for exact identifiers",                      # cb_indexer_find_repo vs Grep
            "Read only that span",                                 # results are leads
            "never a raw file dump",                               # cb_indexer_store_repo
            "run it at the start of a session",                    # cb_indexer_sync_repo
            "different** already-indexed repo",                    # cb_indexer_find_in_collection
            "call `cb_indexer_list_collections` first, never guess",
            "`recall` before finalizing any non-trivial change",
            "Ask first",                                           # remember
            "`general=True` only when the user explicitly says",
            "It needs the user's explicit confirmation before `confirm=True`",
            "Never compress source code you are about to edit",    # local-compress
            "Do not use `fetch_url` for a page that needs login",
            "read and run things directly",
        ):
            with self.subTest(rule=rule):
                self.assertIn(rule, guide)

    # Each MCP server's file and the group heading the skill's guide uses for it.
    SERVERS = {
        "compact_mcp_server.py": "compact",
        "codebase_indexer_mcp_server.py": "codebase-indexer",
        "memory_bank_mcp_server.py": "memory-bank",
        "compress_mcp_server.py": "local-compress",
        "analytics_mcp_server.py": "analytics",
        "codebase_reviewer_mcp_server.py": "codebase-reviewer",
    }

    @staticmethod
    def _tools_defined_in(path: Path) -> list[str]:
        """Names of the functions a server registers with `@mcp.tool(...)` (an explicit
        `name=` wins over the function name), read from source so nothing is imported."""
        import ast

        names = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(target, ast.Attribute) and target.attr == "tool":
                    explicit = None
                    if isinstance(dec, ast.Call):
                        for kw in dec.keywords:
                            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                                explicit = kw.value.value
                    names.append(explicit or node.name)
        return names

    def test_the_guide_names_every_tool_each_server_defines(self):
        # The guide is a copy of what the servers offer, and the servers have grown since the
        # first version of it. Reading the real server files keeps it from drifting again: a new
        # tool, or a new server, fails here until the guide covers it.
        guide = _section(self.text, "### 3. Learn the live tools", "### 4. Report")
        servers = sorted(p.name for p in (REPO_ROOT / "mcpservers").glob("*_mcp_server.py"))
        self.assertEqual(servers, sorted(self.SERVERS), "a server file is not mapped to a guide group")
        for filename, group in self.SERVERS.items():
            start = guide.index(f"- **`{group}`**")
            nxt = guide.find("\n- **`", start + 1)
            entry = guide[start: nxt if nxt != -1 else len(guide)]
            tools = self._tools_defined_in(REPO_ROOT / "mcpservers" / filename)
            self.assertTrue(tools, f"found no tools in {filename}")
            for tool in tools:
                with self.subTest(group=group, tool=tool):
                    self.assertIn(f"`{tool}`", entry)

    def test_the_analytics_entry_lists_every_metric_id_the_template_names(self):
        # The four ids come from templates/CLAUDE.md.template's get_metrics text, so the two
        # cannot disagree about which metric ids have a writer.
        guide = _section(self.text, "- **`analytics`**", "- **`codebase-reviewer`**")
        template = (REPO_ROOT / "templates" / "CLAUDE.md.template").read_text(encoding="utf-8")
        for metric_id in ("autowork", "memory-bank", "codebase-reviewer", "pr-review-rounds"):
            with self.subTest(metric_id=metric_id):
                self.assertIn(f"`{metric_id}`", template)
                self.assertIn(f"`{metric_id}`", guide)

    def test_the_report_keeps_tool_status_only(self):
        report = _section(self.text, "### 4. Report")
        self.assertIn("do not recite step 3's guide", report)

    def test_never_reads_mcp_json_and_reads_docs_with_read(self):
        self.assertIn("Never read or print `.mcp.json`", self.text)
        self.assertIn("Use Read for every file, never `cat`, `sed` or `head` through Bash", self.text)

    def test_no_ticket_tracking(self):
        # The skill reports pull requests only; it looks up no issue or ticket anywhere.
        for gone in ("JIRA", "Jira", "jra_", "ticket key", "keys to look up"):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, self.text)

    def test_forge_facts_are_pull_requests_only(self):
        self.assertIn('do not write "no issues"', self.text)

    def test_suggests_cr_doctor_and_never_runs_it(self):
        self.assertIn("suggest `/cr-doctor`", self.text)
        self.assertIn("do not run it", self.text)

    def test_warns_before_a_first_index_and_discloses_the_one_side_effect(self):
        self.assertIn("suggest `cb_indexer_index_repo`", self.text)
        self.assertIn("do not start it silently", self.text)
        self.assertIn(".qdrant_index_manifest.json", self.text)

    def test_install_skills_ships_the_script_with_the_skill(self):
        with tempfile.TemporaryDirectory() as d:
            plan = {p.name: p for p in lib.plan_skill_installs(REPO_ROOT, Path(d))}
        self.assertIn("cr-load-context", plan)
        self.assertEqual(plan["cr-load-context"].action, "install")
        self.assertEqual([e.name for e in plan["cr-load-context"].extra_files], ["facts.ps1", "facts.sh"])

    def test_the_wheel_ships_the_script_too(self):
        # setup.py's data_files() globs each skill folder; a script beside SKILL.md must be in it,
        # or a pipx/uv install would have the skill and no script. setup.py runs `setup()` at import,
        # so its source is executed with that last call cut off.
        source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8").replace("setup(data_files=data_files())", "")
        namespace: dict = {"__name__": "setup_under_test"}
        previous = os.getcwd()
        os.chdir(REPO_ROOT)  # setup.py's globs are relative to the repo root
        try:
            exec(compile(source, "setup.py", "exec"), namespace)
            targets = dict(namespace["data_files"]())
        finally:
            os.chdir(previous)
        self.assertEqual(
            targets["src/skills/cr-load-context"],
            ["skills/cr-load-context/SKILL.md", "skills/cr-load-context/facts.ps1", "skills/cr-load-context/facts.sh"],
        )
        self.assertEqual(targets["src/skills/cr-doctor"], ["skills/cr-doctor/SKILL.md"])

    def test_the_docs_tables_list_it(self):
        files = (REPO_ROOT / "docs" / "files.md").read_text(encoding="utf-8")
        skills = (REPO_ROOT / "docs" / "skills-and-hooks.md").read_text(encoding="utf-8")
        self.assertIn("`skills/cr-load-context/SKILL.md`", files)
        self.assertIn("`skills/cr-load-context/facts.sh`", files)
        self.assertIn("`skills/cr-load-context/facts.ps1`", files)
        self.assertIn("| `cr-load-context` |", skills)
        self.assertIn("cr-load-context", (REPO_ROOT / "docs" / "installation.md").read_text(encoding="utf-8"))
        self.assertIn("/cr-load-context", (REPO_ROOT / "templates" / "CLAUDE.md.template").read_text(encoding="utf-8"))


@unittest.skipIf(BASH is None, "no bash")
class FactsScriptStaticTest(unittest.TestCase):
    def setUp(self):
        self.text = FACTS.read_text(encoding="utf-8")

    def code(self) -> str:
        return "\n".join(ln for ln in self.text.splitlines() if not ln.lstrip().startswith("#"))

    def test_parses_under_every_bash(self):
        for bash in {BASH, "/bin/bash", shutil.which("bash")}:
            if bash and Path(bash).exists():
                with self.subTest(bash=bash):
                    subprocess.run([bash, "-n", str(FACTS)], check=True)

    def test_avoids_gnu_and_bash4_constructs(self):
        forbidden = {
            "mapfile/readarray": r"\b(mapfile|readarray)\b",
            "associative array": r"declare\s+-A",
            "case-conversion expansion": r"\$\{[A-Za-z_]+(,,|\^\^)",
            "sed -i": r"\bsed\s+-i",
            "readlink -f": r"readlink\s+-f",
            "date -d": r"\bdate\s+-d",
            "grep -P": r"\bgrep\s+(-\w*P|--perl)",
            "xargs -r": r"\bxargs\s+-r",
            "\\b in a BSD grep/sed pattern": r"(grep|sed)\b[^|\n]*\\b",
        }
        code = self.code()
        for label, pattern in forbidden.items():
            with self.subTest(label):
                self.assertIsNone(re.search(pattern, code), f"{label} is not portable to macOS bash 3.2")

    def test_never_touches_mcp_json_and_writes_no_file(self):
        code = self.code()
        self.assertNotIn(".mcp.json", code)
        stripped = code.replace("2>/dev/null", "").replace(">/dev/null", "").replace("2>&1", "")
        self.assertIsNone(re.search(r">\s*\S", re.sub(r"'[^']*'|\"[^\"]*\"", "", stripped)),
                          "facts.sh must not redirect output into a file")
        self.assertNotIn("mktemp", code)

    def test_every_gh_pr_list_has_a_limit(self):
        for line in self.code().splitlines():
            if re.search(r"\bgh pr list -R", line):
                self.assertIn("--limit", line)


class _Env(unittest.TestCase):
    """A throwaway git repo plus stub `gh` and `az` on PATH."""

    GH = """#!/bin/sh
echo "gh $*" >> "$STUB_LOG"
case "$1 $2" in
  "auth status")
    [ "${STUB_GH_AUTH:-ok}" = ok ] && exit 0
    echo "You are not logged into any GitHub hosts" >&2; exit 1 ;;
  "pr list")
    if [ -n "${STUB_GH_ERR:-}" ]; then echo "$STUB_GH_ERR" >&2; exit 1; fi
    [ -n "${STUB_GH_STDERR:-}" ] && echo "/x/site-packages/ext.py:12: SyntaxWarning: invalid escape sequence" >&2
    [ -n "${STUB_GH_PRS:-}" ] && printf '%s\\n' "$STUB_GH_PRS"
    exit 0 ;;
esac
exit 0
"""

    AZ = """#!/bin/sh
echo "az $*" >> "$STUB_LOG"
case "$1 $2" in
  "account show")
    [ "${STUB_AZ_AUTH:-ok}" = ok ] && exit 0
    echo "ERROR: Please run 'az login' to setup account." >&2; exit 1 ;;
  "repos list")
    if [ -n "${STUB_AZ_LIST_ERR:-}" ]; then echo "$STUB_AZ_LIST_ERR" >&2; exit 1; fi
    [ -n "${STUB_AZ_REPOS:-}" ] && printf '%s\\n' "$STUB_AZ_REPOS"
    exit 0 ;;
  "repos pr")
    if [ -n "${STUB_AZ_PR_ERR:-}" ]; then echo "$STUB_AZ_PR_ERR" >&2; exit 1; fi
    [ -n "${STUB_AZ_STDERR:-}" ] && echo "/x/site-packages/azext_devops/foo.py:12: SyntaxWarning: invalid escape sequence" >&2
    [ -n "${STUB_AZ_PRS:-}" ] && printf '%s\\n' "$STUB_AZ_PRS"
    exit 0 ;;
esac
exit 0
"""

    def setUp(self):
        if BASH is None:
            self.skipTest("no bash")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()
        self.repo = self.tmp / "project"
        self.repo.mkdir()
        self.stubs = self.tmp / "stubs"
        self.stubs.mkdir()
        self.log = self.tmp / "calls.log"
        self.log.write_text("")
        self.shadow_dirs = []

    # -- building blocks ----------------------------------------------------

    def git(self, *args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
            cwd=self.repo, check=True, capture_output=True,
        )

    def make_repo(self, branch="main", origin=None, files=None):
        self.git("init", "-q")
        self.git("symbolic-ref", "HEAD", f"refs/heads/{branch}")
        (self.repo / "README.md").write_text("# Project\n", encoding="utf-8")
        for name, body in (files or {}).items():
            p = self.repo / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "first commit")
        if origin:
            self.git("remote", "add", "origin", origin)

    def install_stub(self, name, body):
        path = self.stubs / name
        path.write_text(body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def path_without(self, *names):
        """The caller's PATH with `names` hidden: a directory holding one of them is
        replaced by a directory of symlinks to everything else in it."""
        # On Windows an executable is `gh.exe` / `az.cmd`, never a bare `gh`, so each name
        # is hidden under every extension the shell would resolve it with.
        names = tuple(n + ext for n in names for ext in ("", ".exe", ".cmd", ".bat"))
        dirs = []
        for d in os.environ.get("PATH", "").split(os.pathsep):
            if d and any((Path(d) / n).exists() for n in names):
                shadow = Path(tempfile.mkdtemp(dir=self.tmp))
                for entry in Path(d).iterdir():
                    if entry.name not in names:
                        try:
                            (shadow / entry.name).symlink_to(entry)
                        except OSError:
                            pass
                dirs.append(str(shadow))
            elif d:
                dirs.append(d)
        return os.pathsep.join(dirs)

    def run_facts(self, *, with_gh=True, with_az=True, env=None, cwd=None):
        """Runs facts.sh; returns (stdout, returncode). With a stub present it is first on
        PATH; without one, the real gh/az (if any) are hidden."""
        if with_gh:
            self.install_stub("gh", self.GH)
        if with_az:
            self.install_stub("az", self.AZ)
        hidden = [n for n, present in (("gh", with_gh), ("az", with_az)) if not present]
        base = self.path_without(*hidden) if hidden else os.environ.get("PATH", "")
        full_env = dict(os.environ)
        full_env["PATH"] = os.pathsep.join([str(self.stubs), base]) if (with_gh or with_az) else base
        if not with_gh:
            (self.stubs / "gh").unlink(missing_ok=True)
        if not with_az:
            (self.stubs / "az").unlink(missing_ok=True)
        full_env["STUB_LOG"] = str(self.log)
        full_env.update(env or {})
        proc = subprocess.run(
            [BASH, str(FACTS), "--porcelain"], cwd=cwd or self.repo, env=full_env,
            capture_output=True, text=True, timeout=120,
        )
        return proc.stdout, proc.returncode

    def calls(self) -> str:
        return self.log.read_text()


class RunLimitedTest(_Env):
    """`run_limited` in facts.sh, taken out of the script and run on its own (the script's
    own limits are 20 to 40 seconds, too long to wait for in a test). The fallback path, used
    when neither `timeout` nor `gtimeout` is installed, has to bound a command that starts a
    child WITHOUT exec, which is what Homebrew's az wrapper does."""

    def _function(self) -> str:
        text = FACTS.read_text(encoding="utf-8")
        start = text.index("run_limited() {")
        return text[start: text.index("\n}\n", start) + 3]

    def _run(self, body: str, *, hide_timeout: bool):
        # Hide timeout/gtimeout by overriding `command -v` inside the shell, not by editing
        # PATH: on Windows Git Bash `timeout` lives in /usr/bin beside bash itself, so
        # shadowing that directory would hide everything else the script needs too.
        hide = ('command() { case "$2" in timeout|gtimeout) return 1;; esac; builtin command "$@"; }\n'
                if hide_timeout else "")
        start = time.monotonic()
        proc = subprocess.run([BASH, "-c", hide + self._function() + "\n" + body], env=dict(os.environ),
                              capture_output=True, text=True, timeout=60)
        return proc, time.monotonic() - start

    def _wrapper(self) -> Path:
        # Like the az wrapper: starts the real work as a child and does not exec it.
        script = self.tmp / "wrapper.sh"
        script.write_text(f"#!/bin/bash\nsleep 30 &\necho $! > {self.tmp}/child.pid\nwait\n", encoding="utf-8")
        script.chmod(0o755)
        return script

    def test_the_perl_fallback_kills_the_whole_process_group_not_just_the_wrapper(self):
        if shutil.which("perl") is None:
            self.skipTest("no perl")
        if sys.platform == "win32":
            # Git for Windows always ships timeout(1), so this fallback never runs there; its
            # msys perl cannot exec a Windows-path script (rc=127), which says nothing about macOS.
            self.skipTest("the perl fallback is a macOS-only path; Git Bash has timeout")
        proc, elapsed = self._run(f'run_limited 2 "{self._wrapper()}"; echo "rc=$?"', hide_timeout=True)
        self.assertIn("rc=142", proc.stdout)
        self.assertLess(elapsed, 15, "the call was not bounded by its limit")
        pid = int((self.tmp / "child.pid").read_text())
        deadline = time.monotonic() + 5
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break          # the child is gone
            if time.monotonic() > deadline:
                os.kill(pid, 9)
                self.fail("the wrapper's child was left running after the time limit")
            time.sleep(0.1)

    def test_the_perl_fallback_passes_output_and_exit_status_through(self):
        if shutil.which("perl") is None:
            self.skipTest("no perl")
        proc, _ = self._run('out=$(run_limited 5 sh -c "echo hello; exit 3"); echo "rc=$? out=$out"', hide_timeout=True)
        self.assertIn("rc=3 out=hello", proc.stdout)

    def test_timeout_when_installed_is_used_and_reports_124(self):
        if shutil.which("timeout") is None and shutil.which("gtimeout") is None:
            self.skipTest("no timeout or gtimeout")
        proc, elapsed = self._run(f'run_limited 2 "{self._wrapper()}"; echo "rc=$?"', hide_timeout=False)
        self.assertIn("rc=124", proc.stdout)
        self.assertLess(elapsed, 15)
        try:                       # tidy up: timeout(1) kills only the wrapper
            os.kill(int((self.tmp / "child.pid").read_text()), 9)
        except (OSError, ValueError):
            pass


class FactsCoreTest(_Env):
    def test_core_sections_are_present_and_the_exit_code_is_zero(self):
        self.make_repo(branch="main", origin="https://github.com/acme/widgets.git",
                       files={"CLAUDE.md": "rules\n", "docs/design.md": "# The design\nbody\n"})
        out, rc = self.run_facts(env={"STUB_GH_PRS": ""})
        self.assertEqual(rc, 0)
        for heading in ("## git", "## entry docs", "## index", "## forge"):
            self.assertIn(heading, out)
        self.assertNotIn("## ticket", out)
        self.assertNotIn("JIRA", out)
        self.assertIn("branch: main", out)
        self.assertIn("README: README.md (1 lines)", out)
        self.assertIn("CLAUDE.md: present (1 lines)", out)
        self.assertIn("docs/design.md (2 lines): The design", out)

    def test_sync_manifest_absent_warns_about_a_long_first_embed_and_present_does_not(self):
        self.make_repo(origin="https://github.com/acme/widgets.git")
        out, _ = self.run_facts()
        self.assertIn("sync manifest: ABSENT", out)
        self.assertIn("first embed is long", out)
        self.assertIn("cb_indexer_sync_repo was not started", out)
        (self.repo / ".qdrant_index_manifest.json").write_text("{}", encoding="utf-8")
        out, _ = self.run_facts()
        self.assertIn("sync manifest: present", out)
        self.assertNotIn("ABSENT", out)

    def test_no_claude_md_and_no_docs_are_reported_not_errors(self):
        self.make_repo(origin="https://github.com/acme/widgets.git")
        out, rc = self.run_facts()
        self.assertEqual(rc, 0)
        self.assertIn("CLAUDE.md: absent", out)
        self.assertIn("docs/: none", out)

    def test_mcp_json_is_never_read_or_printed(self):
        self.make_repo(origin="https://github.com/acme/widgets.git")
        (self.repo / ".mcp.json").write_text('{"QDRANT_API_KEY": "sk-very-secret-key"}', encoding="utf-8")
        out, _ = self.run_facts()
        self.assertNotIn("sk-very-secret-key", out)
        self.assertNotIn("QDRANT_API_KEY", out)

    def test_a_folder_that_is_not_a_git_repo_still_reports_and_exits_zero(self):
        out, rc = self.run_facts(cwd=self.repo)
        self.assertEqual(rc, 0)
        self.assertIn("not a git repository", out)
        self.assertIn("## entry docs", out)
        self.assertIn("## forge", out)

    def test_a_password_that_contains_an_at_sign_is_removed_whole(self):
        # git itself mis-parses this URL, but whatever it reports must not leak a fragment.
        self.make_repo(origin="https://user:p@ss@github.com/acme/widgets.git")
        out, _ = self.run_facts()
        self.assertNotIn("ss@", out)
        self.assertNotIn("user", out)
        self.assertIn("origin: https://github.com/acme/widgets.git", out)

    def test_credentials_in_the_remote_url_are_never_printed(self):
        self.make_repo(origin="https://someone:ghp_SECRETTOKEN@github.com/acme/widgets.git")
        out, _ = self.run_facts()
        self.assertNotIn("SECRETTOKEN", out)
        self.assertNotIn("someone", out)
        self.assertIn("origin: https://github.com/acme/widgets.git", out)


class FactsGitHubTest(_Env):
    PRS = "\t".join(["12", "PROJ-4610", "open", "false", "PROJ-4610: fix the thing, see OPS-6064"]) + "\n" + \
          "\t".join(["13", "feature/other", "draft", "false", "Other work"])

    def test_lists_open_pull_requests_and_marks_the_current_branch(self):
        self.make_repo(branch="PROJ-4610", origin="https://github.com/acme/widgets.git")
        out, _ = self.run_facts(env={"STUB_GH_PRS": self.PRS})
        self.assertIn("forge: GitHub (acme/widgets)", out)
        self.assertIn("open pull requests: 2", out)
        self.assertIn("#12 [PROJ-4610] (open) PROJ-4610: fix the thing, see OPS-6064  <- current branch", out)
        self.assertIn("#13 [feature/other] (draft) Other work", out)
        self.assertIn("scope: pull requests only\n", out)
        self.assertNotIn("no issues", out)
        # Pointed at origin's repo explicitly, with an explicit limit.
        self.assertIn("gh pr list -R acme/widgets --state open --limit 30", self.calls())

    def test_a_stderr_line_with_a_successful_call_is_not_a_pull_request(self):
        # `2>&1` merges the streams so an error can be read; on success a stray warning line
        # (python's SyntaxWarning from an extension, say) must not be counted or listed as a PR.
        self.make_repo(branch="main", origin="https://github.com/acme/widgets.git")
        out, _ = self.run_facts(env={"STUB_GH_PRS": "\t".join(["5", "feat", "open", "false", "Real PR"]),
                                     "STUB_GH_STDERR": "1"})
        self.assertIn("open pull requests: 1", out)
        self.assertIn("#5 [feat] (open) Real PR", out)
        self.assertNotIn("SyntaxWarning", out)

    def test_a_fork_pr_from_a_same_named_branch_is_not_marked_as_this_branch(self):
        self.make_repo(branch="main", origin="https://github.com/acme/widgets.git")
        fork = "\t".join(["9", "main", "open", "true", "DEMO-3907: a fork's main"])
        out, _ = self.run_facts(env={"STUB_GH_PRS": fork})
        self.assertIn("#9 [main] (open) DEMO-3907: a fork's main", out)
        self.assertNotIn("<- current branch", out)

    def test_a_long_title_is_cut_with_a_marker(self):
        self.make_repo(branch="main", origin="https://github.com/acme/widgets.git")
        title = "T" * 120
        out, _ = self.run_facts(env={"STUB_GH_PRS": "\t".join(["5", "feat", "open", "false", title])})
        self.assertIn("T" * 90 + "...", out)
        self.assertNotIn("T" * 91, out)
        short = "S" * 90   # exactly the limit: nothing cut, no marker
        out, _ = self.run_facts(env={"STUB_GH_PRS": "\t".join(["6", "feat", "open", "false", short])})
        self.assertIn("S" * 90, out)
        self.assertNotIn("S" * 90 + "...", out)

    def test_ssh_remote_is_understood(self):
        self.make_repo(origin="git@github.com:acme/widgets.git")
        out, _ = self.run_facts()
        self.assertIn("forge: GitHub (acme/widgets)", out)

    def test_gh_missing_skips_only_the_forge(self):
        self.make_repo(origin="https://github.com/acme/widgets.git")
        out, rc = self.run_facts(with_gh=False)
        self.assertEqual(rc, 0)
        self.assertIn("forge facts skipped: gh is not installed", out)
        for heading in ("## git", "## entry docs", "## index"):
            self.assertIn(heading, out)
        self.assertNotIn("scope: pull requests only", out)

    def test_gh_not_signed_in_skips_only_the_forge(self):
        self.make_repo(origin="https://github.com/acme/widgets.git")
        out, rc = self.run_facts(env={"STUB_GH_AUTH": "no"})
        self.assertEqual(rc, 0)
        self.assertIn("forge facts skipped: gh is not signed in to github.com", out)
        self.assertNotIn("pr list", self.calls())

    def test_gh_failing_skips_only_the_forge_with_its_first_error_line(self):
        self.make_repo(origin="https://github.com/acme/widgets.git")
        out, rc = self.run_facts(env={"STUB_GH_ERR": "HTTP 502: bad gateway\nsecond line"})
        self.assertEqual(rc, 0)
        self.assertIn("forge facts skipped: gh pr list failed: HTTP 502: bad gateway", out)
        self.assertNotIn("second line", out)

    def test_other_host_and_no_origin_are_skipped_with_a_reason(self):
        self.make_repo(origin="https://gitlab.com/acme/widgets.git")
        out, _ = self.run_facts()
        self.assertIn("forge facts skipped: origin host gitlab.com is neither GitHub nor Azure DevOps", out)

    def test_no_origin_remote(self):
        self.make_repo(origin=None)
        out, rc = self.run_facts()
        self.assertEqual(rc, 0)
        self.assertIn("forge facts skipped: no origin remote", out)


class FactsAzureDevOpsTest(_Env):
    VS_URL = "https://contoso.visualstudio.com/DefaultCollection/Contoso.Scm/_git/Contoso.Widgets.Api"
    PRS = "\t".join(["425199", "refs/heads/fix/stale-etag", "true", "fix(reconciliation): skip upsert"]) + "\n" + \
          "\t".join(["419499", "refs/heads/feature/PROJ-2135-fix", "false", "PROJ-2135: fix Pending Decision"])

    def listing(self, disabled="false"):
        return f"Contoso.Widgets.Api\t{disabled}\nOther.Repo\tfalse"

    def test_visualstudio_remote_with_default_collection(self):
        self.make_repo(branch="master", origin=self.VS_URL)
        out, rc = self.run_facts(env={"STUB_AZ_REPOS": self.listing(), "STUB_AZ_PRS": self.PRS})
        self.assertEqual(rc, 0)
        self.assertIn("forge: Azure DevOps (org contoso, project Contoso.Scm, repo Contoso.Widgets.Api)", out)
        self.assertIn("active pull requests: 2", out)
        # refs/heads/ is dropped and the draft flag shown.
        self.assertIn("#425199 [fix/stale-etag] (draft) fix(reconciliation): skip upsert", out)
        self.assertIn("#419499 [feature/PROJ-2135-fix] (active) PROJ-2135: fix Pending Decision", out)
        self.assertIn("scope: pull requests only\n", out)
        calls = self.calls()
        # org and project are passed explicitly, never taken from `az devops configure`.
        self.assertIn("--org https://dev.azure.com/contoso --project Contoso.Scm", calls)
        self.assertIn("--repository Contoso.Widgets.Api --status active", calls)
        self.assertNotIn("devops configure", calls)
        self.assertIn("--only-show-errors", calls)

    def test_dev_azure_com_remote_with_userinfo_hides_it_and_is_parsed(self):
        self.make_repo(origin="https://someone:AZPAT@dev.azure.com/myorg/My%20Project/_git/my-repo")
        out, _ = self.run_facts(env={"STUB_AZ_REPOS": "my-repo\tfalse", "STUB_AZ_PRS": ""})
        self.assertNotIn("AZPAT", out)
        self.assertNotIn("someone", out)
        self.assertIn("org myorg, project My Project, repo my-repo", out)
        self.assertIn("--org https://dev.azure.com/myorg --project My Project", self.calls())

    def test_ssh_remote(self):
        self.make_repo(origin="git@ssh.dev.azure.com:v3/myorg/myproject/my-repo")
        out, _ = self.run_facts(env={"STUB_AZ_REPOS": "my-repo\tfalse"})
        self.assertIn("org myorg, project myproject, repo my-repo", out)

    def test_a_stderr_line_with_a_successful_az_call_is_not_a_pull_request(self):
        self.make_repo(origin=self.VS_URL)
        prs = "\t".join(["77", "refs/heads/feat", "false", "PROJ-1: a real PR"])
        out, _ = self.run_facts(env={"STUB_AZ_REPOS": self.listing(), "STUB_AZ_PRS": prs, "STUB_AZ_STDERR": "1"})
        self.assertIn("active pull requests: 1", out)
        self.assertIn("#77 [feat] (active) PROJ-1: a real PR", out)
        self.assertNotIn("SyntaxWarning", out)

    def test_a_disabled_repo_is_reported_and_its_pull_requests_are_not_asked_for(self):
        self.make_repo(origin=self.VS_URL)
        out, rc = self.run_facts(env={"STUB_AZ_REPOS": self.listing(disabled="true")})
        self.assertEqual(rc, 0)
        self.assertIn("forge facts skipped: repo Contoso.Widgets.Api is disabled (isDisabled: true)", out)
        self.assertNotIn("repos pr list", self.calls())

    def test_tf401019_from_the_pr_list_means_disabled_removed_or_no_access(self):
        self.make_repo(origin=self.VS_URL)
        err = "ERROR: TF401019: The Git repository with name or identifier X does not exist"
        out, rc = self.run_facts(env={"STUB_AZ_REPOS": self.listing(), "STUB_AZ_PR_ERR": err})
        self.assertEqual(rc, 0)
        self.assertIn("is disabled, removed, or you have no access to it (TF401019)", out)

    def test_az_missing_skips_only_the_forge(self):
        self.make_repo(origin=self.VS_URL)
        out, rc = self.run_facts(with_az=False)
        self.assertEqual(rc, 0)
        self.assertIn("forge facts skipped: az is not installed", out)
        for heading in ("## git", "## entry docs", "## index"):
            self.assertIn(heading, out)

    def test_az_not_signed_in_skips_only_the_forge(self):
        self.make_repo(origin=self.VS_URL)
        out, _ = self.run_facts(env={"STUB_AZ_AUTH": "no"})
        self.assertIn("forge facts skipped: az is not signed in (run: az login)", out)
        self.assertNotIn("repos list", self.calls())

    def test_missing_azure_devops_extension_is_named(self):
        self.make_repo(origin=self.VS_URL)
        err = "ERROR: 'repos' is misspelled or not recognized by the system."
        out, _ = self.run_facts(env={"STUB_AZ_LIST_ERR": err})
        self.assertIn("the azure-devops extension is not installed", out)

    def test_other_az_failure_gives_its_first_line(self):
        self.make_repo(origin=self.VS_URL)
        out, _ = self.run_facts(env={"STUB_AZ_REPOS": self.listing(), "STUB_AZ_PR_ERR": "ERROR: boom\nmore"})
        self.assertIn("forge facts skipped: az repos pr list failed: ERROR: boom", out)
        self.assertNotIn("more", out.split("forge facts skipped:")[1])


class FactsNoTicketTest(_Env):
    def test_a_key_shaped_branch_and_pr_title_produce_no_ticket_output(self):
        # Branch names and PR titles that look like ticket keys are listed as pull requests
        # and nothing more: the script has no ticket section.
        self.make_repo(branch="PROJ-4610", origin="https://github.com/acme/widgets.git")
        prs = "\t".join(["12", "PROJ-4610", "open", "false", "PROJ-4610: the work"])
        out, rc = self.run_facts(env={"STUB_GH_PRS": prs})
        self.assertEqual(rc, 0)
        self.assertIn("#12 [PROJ-4610] (open) PROJ-4610: the work  <- current branch", out)
        for gone in ("## ticket", "JIRA", "ticket key", "keys to look up"):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, out)

    def test_ado_current_branch_is_still_marked(self):
        self.make_repo(branch="feature/x", origin=FactsAzureDevOpsTest.VS_URL)
        prs = "\t".join(["9", "refs/heads/feature/x", "false", "PROJ-2135: the thing"])
        out, _ = self.run_facts(env={"STUB_AZ_REPOS": "Contoso.Widgets.Api\tfalse", "STUB_AZ_PRS": prs})
        self.assertIn("<- current branch", out)
        self.assertNotIn("## ticket", out)


POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
GH_FACTS_PS1 = REPO_ROOT / ".claude" / "skills" / "cr-gh-load-context" / "facts.ps1"
FACTS_PS1 = SKILL_DIR / "facts.ps1"


class PowerShellScriptsTest(unittest.TestCase):
    """The two facts.ps1 files are the Windows default (bash is not assumed there). They
    are checked as text on every platform, and RUN for real wherever PowerShell exists."""

    def test_the_scripts_are_ascii_only(self):
        # Windows PowerShell 5.1 reads a BOM-less file as the ANSI codepage, so a non-ASCII
        # character (an em dash in a comment) would corrupt it there.
        for path in (FACTS_PS1, GH_FACTS_PS1):
            with self.subTest(path=path.name):
                path.read_bytes().decode("ascii")

    def test_the_scripts_avoid_powershell_7_only_syntax(self):
        # No ternary, null-coalescing or pipeline chains: Windows PowerShell 5.1 must parse them.
        for path in (FACTS_PS1, GH_FACTS_PS1):
            text = path.read_text(encoding="ascii")
            code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
            for bad in (" ?? ", "&& ", " ? $", "?."):
                with self.subTest(path=path.name, bad=bad):
                    self.assertNotIn(bad, code)

    def test_the_scripts_force_utf8_and_never_read_mcp_json(self):
        for path in (FACTS_PS1, GH_FACTS_PS1):
            text = path.read_text(encoding="ascii")
            self.assertIn("[Console]::OutputEncoding", text)
            code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
            self.assertNotIn(".mcp.json", code)

    def test_the_contributor_skill_points_windows_at_its_script(self):
        text = (REPO_ROOT / ".claude" / "skills" / "cr-gh-load-context" / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("facts.ps1", text)
        self.assertIn("Do not assume bash exists on Windows", text)
        self.assertIn("never use WSL", text)


@unittest.skipIf(POWERSHELL is None, "no PowerShell")
class FactsPowerShellTest(_Env):
    """facts.ps1 run for real against throwaway repos, with the real gh hidden or stubbed
    out of the way (no network, no real forge). Same cases as the bash script, same words."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()
        self.repo = self.tmp / "project"
        self.repo.mkdir()
        self.stubs = self.tmp / "stubs"
        self.stubs.mkdir()
        self.log = self.tmp / "calls.log"
        self.log.write_text("")
        self.shadow_dirs = []

    def run_ps(self, *, hide=("gh", "az"), env=None, prepend=None):
        path = self.path_without(*hide)
        if prepend:
            path = os.pathsep.join([str(prepend), path])
        full_env = dict(os.environ, PATH=path)
        full_env.update(env or {})
        proc = subprocess.run(
            [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(FACTS_PS1), "--porcelain"],
            cwd=self.repo, env=full_env, capture_output=True, text=True, encoding="utf-8", timeout=180,
        )
        return proc.stdout.replace("\r\n", "\n"), proc.returncode

    def test_no_origin_prints_every_core_section_and_skips_the_forge(self):
        self.make_repo()
        out, rc = self.run_ps()
        self.assertEqual(rc, 0)
        for heading in ("## git", "## entry docs", "## index", "## forge"):
            self.assertIn(heading, out)
        self.assertIn("branch: main", out)
        self.assertIn("README: README.md", out)
        self.assertIn("CLAUDE.md: absent", out)
        self.assertIn("sync manifest: ABSENT", out)
        self.assertIn("forge facts skipped: no origin remote", out)

    def test_not_a_git_repository_still_exits_zero_with_its_sections(self):
        out, rc = self.run_ps()
        self.assertEqual(rc, 0)
        self.assertIn("not a git repository", out)
        self.assertIn("## entry docs", out)

    def test_gh_missing_skips_only_the_forge_and_never_prints_the_credential(self):
        self.make_repo(origin="https://user:s3cr3t-token@github.com/acme/widgets.git")
        out, rc = self.run_ps()
        self.assertEqual(rc, 0)
        self.assertIn("forge: GitHub (acme/widgets)", out)
        self.assertIn("forge facts skipped: gh is not installed", out)
        self.assertIn("origin: https://github.com/acme/widgets.git", out)
        self.assertNotIn("s3cr3t-token", out)
        for heading in ("## git", "## entry docs", "## index"):
            self.assertIn(heading, out)

    def test_azure_devops_origin_is_parsed_and_az_missing_skips_only_the_forge(self):
        self.make_repo(origin="https://user:tok@contoso.visualstudio.com/DefaultCollection/Proj%20X/_git/repo")
        out, _ = self.run_ps()
        self.assertIn("forge: Azure DevOps (org contoso, project Proj X, repo repo)", out)
        self.assertIn("forge facts skipped: az is not installed", out)
        self.assertNotIn("tok@", out)

    def test_an_unknown_host_is_skipped_with_its_name(self):
        self.make_repo(origin="https://git.example.com/a/b.git")
        out, _ = self.run_ps()
        self.assertIn("forge facts skipped: origin host git.example.com is neither GitHub nor Azure DevOps", out)

    # -- Azure DevOps: the same `az` stub drives facts.sh (tested against real az on macOS) and
    # facts.ps1, so the PowerShell translation has to give the same answer, word for word. --

    def _install_az_stub(self):
        self.install_stub("az", self.AZ)
        if sys.platform == "win32":
            # PowerShell resolves `az` to az.cmd; it hands the call to the same sh stub.
            (self.stubs / "az.cmd").write_text(f'@"{BASH}" "%~dp0az" %*\r\n', encoding="ascii")

    def _ado_both(self, env, *, branch="feature/x"):
        if BASH is None:
            self.skipTest("no bash to compare against")
        self.make_repo(branch=branch, origin=FactsAzureDevOpsTest.VS_URL)
        self._install_az_stub()
        env = dict(env, STUB_LOG=str(self.log))
        ps_out, _ = self.run_ps(hide=("gh",), env=env, prepend=self.stubs)
        full = dict(os.environ, PATH=os.pathsep.join([str(self.stubs), self.path_without("gh")]), **env)
        sh = subprocess.run([BASH, str(FACTS), "--porcelain"], cwd=self.repo, env=full,
                            capture_output=True, text=True, encoding="utf-8", timeout=180)
        sh_out = sh.stdout.replace("\r\n", "\n")
        self.assertEqual(ps_out, sh_out)
        return ps_out

    def test_ado_pull_requests_match_facts_sh_and_mark_the_current_branch(self):
        prs = "\n".join(["\t".join(["9", "refs/heads/feature/x", "false", "PROJ-2135: the thing"]),
                         "\t".join(["10", "refs/heads/other", "true", "A draft"])])
        out = self._ado_both({"STUB_AZ_REPOS": "Contoso.Widgets.Api\tfalse", "STUB_AZ_PRS": prs})
        self.assertIn("active pull requests: 2", out)
        self.assertIn("#9 [feature/x] (active) PROJ-2135: the thing  <- current branch", out)
        self.assertIn("#10 [other] (draft) A draft", out)
        self.assertIn("scope: pull requests only", out)

    def test_ado_disabled_repo_matches_facts_sh(self):
        out = self._ado_both({"STUB_AZ_REPOS": "contoso.widgets.api\ttrue"})
        self.assertIn("is disabled (isDisabled: true)", out)

    def test_ado_tf401019_matches_facts_sh(self):
        out = self._ado_both({"STUB_AZ_REPOS": "Contoso.Widgets.Api\tfalse",
                              "STUB_AZ_PR_ERR": "ERROR: TF401019: The Git repository was not found"})
        self.assertIn("(TF401019)", out)

    def test_ado_not_signed_in_and_list_errors_match_facts_sh(self):
        out = self._ado_both({"STUB_AZ_AUTH": "no"})
        self.assertIn("forge facts skipped: az is not signed in (run: az login)", out)

    def test_ado_missing_extension_matches_facts_sh(self):
        out = self._ado_both({"STUB_AZ_LIST_ERR": "ERROR: 'repos' is misspelled or not recognized by the system."})
        self.assertIn("the azure-devops extension is not installed", out)

    def test_it_matches_facts_sh_section_for_section(self):
        # The report must not depend on which script ran, so on a repo both can read the
        # two outputs are identical.
        if BASH is None:
            self.skipTest("no bash to compare against")
        self.make_repo(
            origin="git@ssh.dev.azure.com:v3/contoso/Proj/repo",
            files={"CLAUDE.md": "x\n", "docs/a.md": "# Page A\n\nbody\n", "docs/b.md": "no heading\n"},
        )
        ps_out, _ = self.run_ps()
        env = dict(os.environ, PATH=self.path_without("gh", "az"))
        sh = subprocess.run([BASH, str(FACTS), "--porcelain"], cwd=self.repo, env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
        self.assertEqual(ps_out, sh.stdout.replace("\r\n", "\n"))


if __name__ == "__main__":
    unittest.main()
