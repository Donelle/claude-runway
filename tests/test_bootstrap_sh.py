#!/usr/bin/env python3
"""Light static checks for scripts/bootstrap.sh (issue #31).

The script creates a venv and installs packages, which doesn't fit this
suite's no-network rule, so only what can regress silently without that is
pinned: the file's executable bit in git (a lost mode makes
`./scripts/bootstrap.sh` fail with "permission denied" on a fresh clone),
its bash shebang, and that it still parses. The real run is verified by hand
per the issue; see docs/development-workflow.md.
"""
import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "bootstrap.sh"


class BootstrapShTests(unittest.TestCase):
    def test_exists_with_bash_shebang(self):
        self.assertTrue(SCRIPT.is_file())
        self.assertEqual(SCRIPT.read_text().splitlines()[0], "#!/usr/bin/env bash")

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


if __name__ == "__main__":
    unittest.main()
