#!/usr/bin/env python3
"""Checks for scripts/bootstrap.sh.

The script creates a venv and installs packages, which doesn't fit this
suite's no-network rule, so the install half is verified by hand (see
docs/development-workflow.md). What can be tested here is pinned:

- the file's executable bit in git (a lost mode makes `./scripts/bootstrap.sh`
  fail with "permission denied" on a fresh clone), its bash shebang, and that
  it parses under every bash on the machine, including macOS's 3.2;
- the env step: its Python is embedded in the script, so it is
  extracted and run directly, and `--env-only` runs the real bash glue
  against a temp repo, a stub doctor and a fake HOME. Nothing here touches the
  real HOME or the real ~/.claude.
"""
import contextlib
import io
import json
import os
import platform
import re
import runpy
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "bootstrap.sh"
sys.path.insert(0, str(REPO_ROOT / "libs"))

_HEREDOC = re.compile(r"cat <<'PY'\n(.*?)\nPY\n", re.DOTALL)


def script_text():
    return SCRIPT.read_text(encoding="utf-8")


def embedded_python():
    matches = _HEREDOC.findall(script_text())
    assert len(matches) == 1, "expected exactly one embedded Python block"
    return matches[0]


def bash_only_text():
    """The script's executable bash: the embedded Python and the comments are cut out."""
    code = _HEREDOC.sub("", script_text())
    return "\n".join(line for line in code.splitlines() if not line.lstrip().startswith("#"))


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()


class BootstrapShTests(unittest.TestCase):
    def test_exists_with_bash_shebang(self):
        self.assertTrue(SCRIPT.is_file())
        self.assertEqual(script_text().splitlines()[0], "#!/usr/bin/env bash")

    def test_executable_bit_committed(self):
        if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
            self.skipTest("needs a git checkout")
        out = subprocess.run(
            ["git", "ls-files", "-s", "scripts/bootstrap.sh"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
        if not out:
            self.skipTest("not yet tracked by git")
        self.assertTrue(out.startswith("100755"), out)

    def test_parses(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash not available")
        subprocess.run([bash, "-n", str(SCRIPT)], check=True)

    def test_parses_under_system_bash(self):
        # macOS's /bin/bash is 3.2, the oldest bash this has to run under.
        if not Path("/bin/bash").exists():
            self.skipTest("no /bin/bash")
        subprocess.run(["/bin/bash", "-n", str(SCRIPT)], check=True)

    def test_bash_part_avoids_gnu_and_bash4_constructs(self):
        text = bash_only_text()
        forbidden = {
            "mapfile/readarray": r"\b(mapfile|readarray)\b",
            "associative array": r"declare\s+-A",
            "case-conversion expansion": r"\$\{[A-Za-z_]+(,,|\^\^)",
            "sed -i": r"\bsed\s+-i",
            "readlink -f": r"readlink\s+-f",
            "date -d": r"\bdate\s+-d",
            "realpath": r"\brealpath\b",
            "mktemp": r"\bmktemp\b",
        }
        for label, pattern in forbidden.items():
            with self.subTest(label):
                self.assertIsNone(re.search(pattern, text), f"{label} is not portable to macOS bash 3.2")

    def test_no_heredoc_inside_command_substitution(self):
        # bash 3.2 mis-parses a heredoc inside $( ); the Python lives in a function.
        for line in bash_only_text().splitlines():
            if "$(" in line:
                self.assertNotIn("<<", line)


class EnvStepTestCase(_TempDirCase):
    """Loads the embedded Python once per test as a plain namespace."""

    def setUp(self):
        super().setUp()
        source = self.tmp / "env_step.py"
        source.write_text(embedded_python(), encoding="utf-8")
        self.ns = runpy.run_path(str(source), run_name="env_step_under_test")
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.repo = self.tmp / "checkout"
        (self.repo / "libs").mkdir(parents=True)

    def write_mcp(self, servers):
        (self.repo / ".mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")

    def render(self, *, environ=None, declared=None):
        notes = []
        text = self.ns["render_env_file"](self.repo, self.home, environ or {}, declared or {}, notes)
        return text, notes

    def source(self, text, env=None):
        """What a real POSIX shell has after sourcing `text`, with only `env` exported."""
        if shutil.which("sh") is None:
            self.skipTest("needs a POSIX shell")
        path = self.tmp / "env.sh"
        path.write_text(text, encoding="utf-8")
        base = {"HOME": str(self.home), "PATH": os.environ.get("PATH", "")}
        base.update(env or {})
        out = subprocess.run(
            ["sh", "-euc", '. "$1"; exec "$2" -c "import json, os; print(json.dumps(dict(os.environ)))"',
             "sh", str(path), sys.executable],
            env=base, capture_output=True, text=True, check=True,
        ).stdout
        return json.loads(out)


class ReadDeclaredTests(EnvStepTestCase):
    def read(self):
        notes = []
        return self.ns["read_declared"](self.repo / ".mcp.json", notes), notes

    def test_missing_file_declares_nothing_and_says_so(self):
        declared, notes = self.read()
        self.assertEqual(declared, {})
        self.assertTrue(any("not found" in n for n in notes))

    def test_local_compress_values_are_read(self):
        self.write_mcp({"local-compress": {"env": {
            "CLAUDE_RUNWAY_LMSTUDIO_URL": "http://box:1/v1",
            "CLAUDE_RUNWAY_LMSTUDIO_MODEL": "m1",
            "CLAUDE_RUNWAY_TRACK_SAVINGS": "1",
        }}})
        declared, _ = self.read()
        self.assertEqual(declared, {
            "CLAUDE_RUNWAY_LMSTUDIO_URL": "http://box:1/v1",
            "CLAUDE_RUNWAY_LMSTUDIO_MODEL": "m1",
            "CLAUDE_RUNWAY_TRACK_SAVINGS": "1",
        })

    def test_qdrant_only_project_still_declares_its_lm_studio_values(self):
        self.write_mcp({"qdrant": {"env": {"CLAUDE_RUNWAY_LMSTUDIO_MODEL": "pinned"}}})
        declared, _ = self.read()
        self.assertEqual(declared, {"CLAUDE_RUNWAY_LMSTUDIO_MODEL": "pinned"})

    def test_first_declaring_block_wins(self):
        self.write_mcp({
            "qdrant": {"env": {"CLAUDE_RUNWAY_LMSTUDIO_MODEL": "from-qdrant"}},
            "local-compress": {"env": {"CLAUDE_RUNWAY_LMSTUDIO_MODEL": "from-local-compress"}},
        })
        declared, _ = self.read()
        self.assertEqual(declared["CLAUDE_RUNWAY_LMSTUDIO_MODEL"], "from-local-compress")

    def test_savings_db_comes_from_the_analytics_block_only(self):
        self.write_mcp({
            "local-compress": {"env": {"CLAUDE_RUNWAY_SAVINGS_DB": "/inert/copy.db"}},
            "analytics": {"env": {"CLAUDE_RUNWAY_SAVINGS_DB": "/real/savings.db"}},
        })
        declared, _ = self.read()
        self.assertEqual(declared, {"CLAUDE_RUNWAY_SAVINGS_DB": "/real/savings.db"})

    def test_unfilled_placeholder_is_ignored_with_a_note(self):
        self.write_mcp({"local-compress": {"env": {"CLAUDE_RUNWAY_LMSTUDIO_MODEL": "REPLACE-WITH-MODEL"}}})
        declared, notes = self.read()
        self.assertEqual(declared, {})
        self.assertTrue(any("REPLACE-WITH-MODEL" in n for n in notes))

    def test_blank_value_is_a_real_declaration(self):
        self.write_mcp({"local-compress": {"env": {"CLAUDE_RUNWAY_LMSTUDIO_URL": ""}}})
        declared, _ = self.read()
        self.assertEqual(declared, {"CLAUDE_RUNWAY_LMSTUDIO_URL": ""})

    def test_malformed_file_falls_back_to_defaults_with_a_note(self):
        (self.repo / ".mcp.json").write_text("{not json", encoding="utf-8")
        declared, notes = self.read()
        self.assertEqual(declared, {})
        self.assertTrue(any("Falling back to defaults" in n for n in notes))


class RenderEnvFileTests(EnvStepTestCase):
    def test_defaults_when_nothing_is_declared_or_exported(self):
        text, _ = self.render()
        env = self.source(text)
        self.assertEqual(env["CLAUDE_RUNWAY_DIR"], str(self.repo))
        self.assertEqual(env["CLAUDE_RUNWAY_LMSTUDIO_URL"], "http://localhost:1234/v1")
        self.assertEqual(env["CLAUDE_RUNWAY_LMSTUDIO_MODEL"], "")
        self.assertEqual(env["CLAUDE_RUNWAY_TRACK_SAVINGS"], "0")
        self.assertEqual(env["CLAUDE_RUNWAY_SAVINGS_DB"], str(self.home / ".claude" / "claude-runway" / "savings.db"))
        self.assertEqual(env["CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS"], "2000")
        self.assertEqual(env["CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS"], "0")
        self.assertEqual(env["CLAUDE_RUNWAY_TRACK_MEMORY_EVENTS"], "1")

    def test_an_existing_export_wins_over_every_default_even_a_blank_one(self):
        text, _ = self.render()
        env = self.source(text, {
            "CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS": "9000",
            "CLAUDE_RUNWAY_SAVINGS_DB": "/elsewhere.db",
            "CLAUDE_RUNWAY_LMSTUDIO_URL": "",
        })
        self.assertEqual(env["CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS"], "9000")
        self.assertEqual(env["CLAUDE_RUNWAY_SAVINGS_DB"], "/elsewhere.db")
        self.assertEqual(env["CLAUDE_RUNWAY_LMSTUDIO_URL"], "")

    def test_declared_values_and_the_checkout_path_are_set_whatever_was_exported(self):
        text, _ = self.render(declared={"CLAUDE_RUNWAY_TRACK_SAVINGS": "1", "CLAUDE_RUNWAY_LMSTUDIO_URL": ""})
        env = self.source(text, {
            "CLAUDE_RUNWAY_TRACK_SAVINGS": "0",
            "CLAUDE_RUNWAY_LMSTUDIO_URL": "http://stale/v1",
            "CLAUDE_RUNWAY_DIR": "/old/checkout",
        })
        self.assertEqual(env["CLAUDE_RUNWAY_TRACK_SAVINGS"], "1")
        self.assertEqual(env["CLAUDE_RUNWAY_LMSTUDIO_URL"], "")
        self.assertEqual(env["CLAUDE_RUNWAY_DIR"], str(self.repo))

    def test_variable_references_in_a_declared_value_are_expanded_by_the_shell(self):
        # Claude Code expands ${NAME} and ${NAME:-default} in .mcp.json; the shell must match.
        cases = {
            "${HOME}/custom/savings.db": f"{self.home}/custom/savings.db",
            "${RUNWAY_ROOT:-/fallback}/s.db": "/fallback/s.db",
            "${RUNWAY_ROOT:-${HOME}/d}/s.db": f"{self.home}/d/s.db",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                text, _ = self.render(declared={"CLAUDE_RUNWAY_SAVINGS_DB": raw})
                self.assertEqual(self.source(text)["CLAUDE_RUNWAY_SAVINGS_DB"], expected)
        text, _ = self.render(declared={"CLAUDE_RUNWAY_SAVINGS_DB": "${RUNWAY_ROOT:-/fallback}/s.db"})
        self.assertEqual(self.source(text, {"RUNWAY_ROOT": "/set"})["CLAUDE_RUNWAY_SAVINGS_DB"], "/set/s.db")

    def test_a_declared_value_can_never_run_a_command(self):
        # .mcp.json is committed with the repo and env.sh runs in every new terminal.
        marker = self.tmp / "ran"
        for raw in (
            f"$(touch {marker})", f"`touch {marker}`", f"${{X:-$(touch {marker})}}",
            f"${{X:-`touch {marker}`}}", f'${{X:-"$(touch {marker})"}}', f"$((1+1))$(touch {marker})",
        ):
            with self.subTest(raw=raw):
                text, _ = self.render(declared={"CLAUDE_RUNWAY_LMSTUDIO_MODEL": raw})
                self.source(text)
                self.assertFalse(marker.exists(), raw)

    def test_special_characters_in_a_declared_value_reach_the_shell_unchanged(self):
        nasty = 'a "quoted" $HOME `echo hi` back\\slash $(x) $'
        text, _ = self.render(declared={"CLAUDE_RUNWAY_LMSTUDIO_MODEL": nasty})
        self.assertEqual(self.source(text)["CLAUDE_RUNWAY_LMSTUDIO_MODEL"], nasty)

    def test_a_default_path_with_spaces_and_a_dollar_sign_reaches_the_shell_unchanged(self):
        self.home = self.tmp / "my home $x"
        self.home.mkdir()
        text, _ = self.render()
        self.assertEqual(
            self.source(text)["CLAUDE_RUNWAY_SAVINGS_DB"],
            str(self.home / ".claude" / "claude-runway" / "savings.db"),
        )

    def test_a_home_with_quotes_and_braces_reaches_the_shell_unchanged(self):
        for name in ("O'Connor", "Work{Personal}", 'say "hi"'):
            with self.subTest(home=name):
                self.home = self.tmp / name
                self.home.mkdir()
                text, _ = self.render()
                expected = str(self.home / ".claude" / "claude-runway" / "savings.db")
                self.assertEqual(self.source(text)["CLAUDE_RUNWAY_SAVINGS_DB"], expected)
                exported = self.source(text, {"CLAUDE_RUNWAY_SAVINGS_DB": "/mine.db"})
                self.assertEqual(exported["CLAUDE_RUNWAY_SAVINGS_DB"], "/mine.db")

    @unittest.skipIf(not Path("/bin/bash").exists(), "no /bin/bash")
    def test_the_file_parses_under_bash_and_zsh(self):
        self.home = self.tmp / "O'Connor {x}"
        self.home.mkdir()
        text, _ = self.render()
        path = self.tmp / "env.sh"
        path.write_text(text, encoding="utf-8")
        for shell in ("/bin/bash", shutil.which("zsh")):
            if shell:
                with self.subTest(shell=shell):
                    subprocess.run([shell, "-n", str(path)], check=True)

    def test_env_local_is_loaded_last_and_can_override(self):
        local = self.home / ".claude" / "claude-runway" / "env.local.sh"
        local.parent.mkdir(parents=True)
        local.write_text('export CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS="42"\nexport MINE="yes"\n', encoding="utf-8")
        text, _ = self.render()
        env = self.source(text)
        self.assertEqual(env["CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS"], "42")
        self.assertEqual(env["MINE"], "yes")

    def test_a_missing_env_local_is_not_an_error_under_set_e(self):
        text, _ = self.render()
        self.assertIn("CLAUDE_RUNWAY_DIR", self.source(text))  # source() runs `sh -eu`

    def test_a_conflicting_shell_export_is_reported(self):
        _text, notes = self.render(
            environ={"CLAUDE_RUNWAY_TRACK_SAVINGS": "0"}, declared={"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"}
        )
        self.assertTrue(any("CLAUDE_RUNWAY_TRACK_SAVINGS is exported in this shell" in n for n in notes))

    def test_agreeing_values_that_are_spelled_differently_are_not_a_conflict(self):
        _text, notes = self.render(
            environ={"CLAUDE_RUNWAY_TRACK_SAVINGS": "yes", "CLAUDE_RUNWAY_SAVINGS_DB": f"{self.home}/x.db"},
            declared={"CLAUDE_RUNWAY_TRACK_SAVINGS": "1", "CLAUDE_RUNWAY_SAVINGS_DB": "~/x.db"},
        )
        self.assertEqual(notes, [])

    def test_the_output_does_not_depend_on_the_shell_it_was_rendered_in(self):
        declared = {"CLAUDE_RUNWAY_LMSTUDIO_MODEL": "m"}
        bare, _ = self.render(declared=declared)
        busy, _ = self.render(declared=declared, environ={"CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS": "1"})
        self.assertEqual(bare, busy)

    def test_every_line_is_a_comment_an_export_or_the_env_local_line(self):
        text, _ = self.render()
        for line in text.splitlines():
            self.assertTrue(
                line == "" or line.startswith("#") or line.startswith("export ") or line.startswith("if [ "),
                line,
            )


@unittest.skipIf(shutil.which("bash") is None, "needs bash")
class CheckParsesTests(EnvStepTestCase):
    def test_a_normal_file_parses(self):
        text, _ = self.render(declared={"CLAUDE_RUNWAY_SAVINGS_DB": "${HOME}/x.db"})
        self.assertIsNone(self.ns["check_parses"](text))

    def test_an_apostrophe_in_a_live_default_is_caught_and_the_variable_named(self):
        text, _ = self.render(declared={"CLAUDE_RUNWAY_SAVINGS_DB": "${RUNWAY_ROOT:-/Users/O'Connor}/s.db"})
        problem = self.ns["check_parses"](text)
        self.assertIsNotNone(problem)
        self.assertIn("CLAUDE_RUNWAY_SAVINGS_DB", problem)

    def test_a_file_that_does_not_parse_is_never_installed(self):
        self.write_mcp({"analytics": {"env": {"CLAUDE_RUNWAY_SAVINGS_DB": "${R:-/Users/O'Connor}/s.db"}}})
        env_file = self.home / ".claude" / "claude-runway" / "env.sh"
        env_file.parent.mkdir(parents=True)
        env_file.write_text("# the previous, working file\n", encoding="utf-8")
        for libs_file in (REPO_ROOT / "libs").glob("*.py"):
            (self.repo / "libs" / libs_file.name).write_text(libs_file.read_text(encoding="utf-8"), encoding="utf-8")
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = str(self.home)
        self.addCleanup(lambda: os.environ.__setitem__("HOME", old_home) if old_home else os.environ.pop("HOME"))
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(self.ns["main"]([str(self.repo)]), 1)
        self.assertEqual(env_file.read_text(encoding="utf-8"), "# the previous, working file\n")
        self.assertIn("CLAUDE_RUNWAY_SAVINGS_DB would not parse", err.getvalue())


class WriteIfChangedTests(EnvStepTestCase):
    def test_writes_then_leaves_an_identical_file_alone(self):
        path = self.home / ".claude" / "claude-runway" / "env.sh"
        self.assertTrue(self.ns["write_if_changed"](path, "x\n"))
        stamp = path.stat().st_mtime_ns
        self.assertFalse(self.ns["write_if_changed"](path, "x\n"))
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        self.assertTrue(self.ns["write_if_changed"](path, "y\n"))
        self.assertEqual(path.read_text(encoding="utf-8"), "y\n")

    def test_keeps_the_existing_mode_and_creates_new_files_private(self):
        old_umask = os.umask(0o022)
        self.addCleanup(os.umask, old_umask)
        path = self.home / "env.sh"
        path.write_text("old\n", encoding="utf-8")
        for mode in (0o600, 0o640):
            with self.subTest(mode=oct(mode)):
                os.chmod(path, mode)
                self.ns["write_if_changed"](path, f"new{mode}\n")
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), mode)
        fresh = self.home / "d" / "env.sh"
        self.ns["write_if_changed"](fresh, "x\n")
        self.assertEqual(stat.S_IMODE(fresh.stat().st_mode), 0o600)

    def test_leaves_no_temp_file_behind(self):
        path = self.home / "d" / "env.sh"
        self.ns["write_if_changed"](path, "x\n")
        self.assertEqual([p.name for p in path.parent.iterdir()], ["env.sh"])


class ProfileTargetTests(EnvStepTestCase):
    def target(self, shell):
        return self.ns["profile_target"](shell, self.home)

    def test_zsh_uses_zshrc(self):
        self.assertEqual(self.target("/bin/zsh"), self.home / ".zshrc")

    def test_bash_with_nothing_present_creates_bash_profile(self):
        self.assertEqual(self.target("/bin/bash"), self.home / ".bash_profile")

    def test_bash_prefers_the_first_existing_login_file(self):
        (self.home / ".profile").write_text("", encoding="utf-8")
        self.assertEqual(self.target("/bin/bash"), self.home / ".profile")
        (self.home / ".bash_login").write_text("", encoding="utf-8")
        self.assertEqual(self.target("/bin/bash"), self.home / ".bash_login")
        (self.home / ".bash_profile").write_text("", encoding="utf-8")
        self.assertEqual(self.target("/bin/bash"), self.home / ".bash_profile")

    def test_other_or_missing_shells_have_no_target(self):
        self.assertIsNone(self.target("/usr/bin/fish"))
        self.assertIsNone(self.target(""))


class WireProfileTests(EnvStepTestCase):
    def wire(self, *, shell="/bin/zsh", system="Darwin", update=False, answer=None, ask=True):
        out, asked = [], []

        def fake_ask(prompt):
            asked.append(prompt)
            return answer or ""

        result = self.ns["wire_profile"](
            self.home, shell, system, update, fake_ask if ask else None, out=out.append
        )
        return result, "\n".join(out), asked

    def block(self):
        return self.ns["PROFILE_BLOCK"]

    def test_yes_appends_one_block_and_creates_the_file(self):
        result, out, asked = self.wire(answer="y")
        self.assertEqual(result, "wired")
        self.assertEqual((self.home / ".zshrc").read_text(encoding="utf-8"), self.block())
        self.assertEqual(asked, ["Add the claude-runway environment to ~/.zshrc? [y/N] "])
        self.assertIn("new terminal", out)

    def test_yes_in_any_case_and_the_long_form(self):
        for answer in ("Y", "yes", " YES "):
            with self.subTest(answer=answer):
                (self.home / ".zshrc").unlink(missing_ok=True)
                self.assertEqual(self.wire(answer=answer)[0], "wired")

    def test_empty_answer_and_no_write_nothing_and_print_the_block(self):
        for answer in ("", "n", "no", "maybe"):
            with self.subTest(answer=answer):
                result, out, _ = self.wire(answer=answer)
                self.assertEqual(result, "declined")
                self.assertFalse((self.home / ".zshrc").exists())
                self.assertIn(self.block().rstrip("\n"), out)

    def test_no_terminal_writes_nothing_and_prints_the_block(self):
        result, out, asked = self.wire(ask=False)
        self.assertEqual(result, "printed")
        self.assertEqual(asked, [])
        self.assertFalse((self.home / ".zshrc").exists())
        self.assertIn(self.block().rstrip("\n"), out)

    def test_update_profile_writes_without_asking(self):
        result, _out, asked = self.wire(update=True, ask=False)
        self.assertEqual(result, "wired")
        self.assertEqual(asked, [])
        self.assertEqual((self.home / ".zshrc").read_text(encoding="utf-8"), self.block())

    def test_existing_content_is_kept_byte_for_byte_before_the_block(self):
        before = "export PATH=$PATH:/x\nalias ll='ls -l'"  # no trailing newline
        (self.home / ".zshrc").write_text(before, encoding="utf-8")
        self.wire(answer="y")
        after = (self.home / ".zshrc").read_text(encoding="utf-8")
        self.assertTrue(after.startswith(before))
        self.assertEqual(after, before + "\n\n" + self.block())

    def test_a_second_run_finds_the_block_and_adds_nothing(self):
        self.wire(answer="y")
        once = (self.home / ".zshrc").read_text(encoding="utf-8")
        result, out, asked = self.wire(answer="y")
        self.assertEqual(result, "already")
        self.assertEqual(asked, [])
        self.assertIn("already loads", out)
        self.assertEqual((self.home / ".zshrc").read_text(encoding="utf-8"), once)

    def test_a_hand_written_line_counts_as_already_wired(self):
        (self.home / ".zshrc").write_text('. "$HOME/.claude/claude-runway/env.sh"\n', encoding="utf-8")
        self.assertEqual(self.wire(update=True, ask=False)[0], "already")

    def test_a_symlinked_profile_is_written_through_to_its_target(self):
        real = self.tmp / "dotfiles" / "zshrc"
        real.parent.mkdir()
        real.write_text("# mine\n", encoding="utf-8")
        (self.home / ".zshrc").symlink_to(real)
        result, out, _ = self.wire(answer="y")
        self.assertEqual(result, "wired")
        self.assertTrue((self.home / ".zshrc").is_symlink())
        self.assertEqual(real.read_text(encoding="utf-8"), "# mine\n\n" + self.block())
        self.assertIn(str(real), out)

    def test_bash_adds_to_the_existing_profile_not_a_new_bash_profile(self):
        (self.home / ".profile").write_text("# existing\n", encoding="utf-8")
        self.assertEqual(self.wire(shell="/bin/bash", answer="y")[0], "wired")
        self.assertFalse((self.home / ".bash_profile").exists())
        self.assertIn("claude-runway/env.sh", (self.home / ".profile").read_text(encoding="utf-8"))

    def test_other_platforms_never_write_even_with_the_flag(self):
        result, out, asked = self.wire(system="Linux", update=True, answer="y")
        self.assertEqual(result, "printed")
        self.assertEqual(asked, [])
        self.assertEqual(list(self.home.iterdir()), [])
        self.assertIn(self.block().rstrip("\n"), out)

    def test_an_unknown_shell_prints_the_block(self):
        result, out, _ = self.wire(shell="/usr/bin/fish", answer="y")
        self.assertEqual(result, "printed")
        self.assertIn("fish", out)
        self.assertEqual(list(self.home.iterdir()), [])

    def test_a_profile_that_is_not_utf8_is_wired_without_an_error(self):
        latin1 = b"# caf\xe9\nexport X=1\n"
        (self.home / ".zshrc").write_bytes(latin1)
        self.assertEqual(self.wire(answer="y")[0], "wired")
        self.assertEqual((self.home / ".zshrc").read_bytes(), latin1 + b"\n" + self.block().encode("utf-8"))
        self.assertEqual(self.wire(answer="y")[0], "already")

    def test_an_unwritable_profile_reports_and_prints_the_block(self):
        (self.home / ".zshrc").mkdir()  # a directory where the file should be
        result, out, _ = self.wire(answer="y")
        self.assertEqual(result, "failed")
        self.assertIn(self.block().rstrip("\n"), out)


@unittest.skipIf(shutil.which("git") is None, "needs git")
class EnvOnlyScriptTests(_TempDirCase):
    """Runs the real bash glue: `--env-only` against a temp repo, a stub doctor and a fake HOME."""

    def setUp(self):
        super().setUp()
        self.repo = self.tmp / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        shutil.copy(SCRIPT, self.repo / "scripts" / "bootstrap.sh")
        (self.repo / "libs").symlink_to(REPO_ROOT / "libs")
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        self.home = self.tmp / "home"
        self.home.mkdir()

    def make_venv(self):
        bin_dir = self.repo / ".venv" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "python").symlink_to(sys.executable)
        doctor = bin_dir / "claude-runway-doctor"
        # Echoes what it can see, to prove env.sh was loaded for the check.
        doctor.write_text(
            '#!/bin/sh\necho "stub doctor ran on $1 with dir=$CLAUDE_RUNWAY_DIR"\n', encoding="utf-8"
        )
        doctor.chmod(doctor.stat().st_mode | stat.S_IXUSR)

    def run_script(self, *args, bash="/bin/bash", shell="/bin/zsh", extra_env=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_RUNWAY_")}
        env.update({"HOME": str(self.home), "SHELL": shell})
        env.update(extra_env or {})
        return subprocess.run(
            [bash, str(self.repo / "scripts" / "bootstrap.sh"), *args],
            cwd=self.repo, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        )

    @property
    def env_file(self):
        return self.home / ".claude" / "claude-runway" / "env.sh"

    def bashes(self):
        found = [p for p in ("/bin/bash", shutil.which("bash")) if p and Path(p).exists()]
        if not found:
            self.skipTest("bash not available")
        return sorted(set(found))

    def test_env_only_without_a_venv_explains_what_to_do(self):
        for bash in self.bashes():
            with self.subTest(bash=bash):
                result = self.run_script("--env-only", bash=bash)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("without --env-only first", result.stderr)
                self.assertFalse(self.env_file.exists())

    def test_unknown_argument_is_refused(self):
        result = self.run_script("--bogus", bash=self.bashes()[0])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown argument '--bogus'", result.stderr)

    def test_env_only_writes_the_file_wires_the_profile_and_runs_the_doctor(self):
        self.make_venv()
        (self.repo / ".mcp.json").write_text(
            '{"mcpServers": {"local-compress": {"env": {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"}}}}',
            encoding="utf-8",
        )
        for bash in self.bashes():
            with self.subTest(bash=bash):
                zshrc = self.home / ".zshrc"
                zshrc.unlink(missing_ok=True)
                self.env_file.unlink(missing_ok=True)
                result = self.run_script("--env-only", "--update-profile", bash=bash)
                self.assertEqual(result.returncode, 0, result.stderr)
                text = self.env_file.read_text(encoding="utf-8")
                self.assertIn('export CLAUDE_RUNWAY_TRACK_SAVINGS="1"', text)
                self.assertIn(f'export CLAUDE_RUNWAY_DIR="{self.repo.resolve()}"', text)
                # The doctor runs with env.sh loaded: this process never sourced it.
                self.assertIn(f"stub doctor ran on {self.repo.resolve()} with dir={self.repo.resolve()}", result.stdout)
                if platform.system() == "Darwin":
                    self.assertIn("claude-runway/env.sh", zshrc.read_text(encoding="utf-8"))
                    self.assertIn("Dock", result.stdout)
                else:
                    self.assertFalse(zshrc.exists())

    def test_a_second_run_changes_nothing(self):
        self.make_venv()
        for bash in self.bashes():
            with self.subTest(bash=bash):
                self.assertEqual(self.run_script("--env-only", "--update-profile", bash=bash).returncode, 0)
                before = self.env_file.read_bytes()
                zshrc = self.home / ".zshrc"
                profile_before = zshrc.read_bytes() if zshrc.exists() else None
                result = self.run_script("--env-only", "--update-profile", bash=bash)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.env_file.read_bytes(), before)
                self.assertEqual(zshrc.read_bytes() if zshrc.exists() else None, profile_before)
                self.assertIn("Already up to date", result.stdout)

    def test_without_the_flag_and_without_a_terminal_the_profile_is_not_touched(self):
        self.make_venv()
        result = self.run_script("--env-only", bash=self.bashes()[0])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.env_file.is_file())
        self.assertFalse((self.home / ".zshrc").exists())
        self.assertIn("claude-runway/env.sh", result.stdout)  # the block, printed for the user

    def test_an_existing_shell_export_wins_over_a_default(self):
        # The doctor runs with env.sh loaded; the stub prints what it sees.
        self.make_venv()
        doctor = self.repo / ".venv" / "bin" / "claude-runway-doctor"
        doctor.write_text('#!/bin/sh\necho "threshold=$CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS"\n', encoding="utf-8")
        result = self.run_script(
            "--env-only", bash=self.bashes()[0],
            extra_env={"CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS": "123"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("threshold=123", result.stdout)

    def test_a_hand_edited_env_file_is_replaced_and_env_local_is_never_touched(self):
        self.make_venv()
        self.env_file.parent.mkdir(parents=True)
        self.env_file.write_text('export MY_CERT="-----BEGIN-----\nabc\n-----END-----"\n', encoding="utf-8")
        local = self.env_file.with_name("env.local.sh")
        local.write_text('export MINE="kept"\n', encoding="utf-8")
        result = self.run_script("--env-only", bash=self.bashes()[0])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("MY_CERT", self.env_file.read_text(encoding="utf-8"))
        self.assertEqual(local.read_text(encoding="utf-8"), 'export MINE="kept"\n')


if __name__ == "__main__":
    unittest.main()
