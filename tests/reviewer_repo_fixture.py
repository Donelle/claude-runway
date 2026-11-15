"""A throwaway git repository for the tests of the reviewer's tool executor and context gathering.

`build_repo(root)` makes the repository under `root` and returns its head commit. Besides the committed files it leaves
three things behind on purpose, because the executor must refuse each of them: an UNTRACKED file with a definition in it,
an untracked `.env`, and a file that is STAGED but not in the head commit (tracked in the index, missing at the head).
"""

import subprocess
from pathlib import Path

ORDERS = '''\
from app.pricing import total_price


def checkout(items):
    discount = apply_discount(items)
    return total_price(items) - discount
'''

PRICING = '''\
def total_price(items):
    total = 0
    for item in items:
        total += item.price
    return total


def apply_discount(items):
    return 0
'''

# A test file that defines the same name: the real definition must win over it.
TEST_PRICING = '''\
from app.pricing import total_price


def total_price(items):  # a stand-in used by the tests
    return 0


def test_total():
    assert total_price([]) == 0
'''

REPORT = '''\
from app.pricing import total_price
from app.orders import checkout


def summary(rows):
    return total_price(rows)


def cart(rows):
    return checkout(rows)
'''

UTIL_JS = '''\
function formatMoney(value) {
  return value.toFixed(2);
}
'''

GO_FILE = '''\
package main

func (s *Server) handle(w int) int {
    return w
}
'''


def git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=tester", "-c", "user.email=tester@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=root, capture_output=True, text=True, check=True,
    )
    return done.stdout.strip()


def write(path: Path, text: str) -> None:
    """Writes `text` byte for byte. `Path.write_text` turns every newline into CRLF on Windows, so a fixture file would
    commit with CRLF line ends there and a test comparing what the tools read against the text here would fail."""
    path.write_bytes(text.encode("utf-8"))


def build_repo_with(root: Path, files: dict) -> str:
    """A repository holding exactly `files` (path -> text), committed; returns the head commit. For a test that needs
    a file of its own shape rather than the standard fixture."""
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, text)
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "files")
    return git(root, "rev-parse", "HEAD")


def build_repo(root: Path) -> str:
    """Creates the repository and returns the head commit (a full hash)."""
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    files = {
        "app/__init__.py": "",
        "app/orders.py": ORDERS,
        "app/pricing.py": PRICING,
        "app/report.py": REPORT,
        "tests/test_pricing.py": TEST_PRICING,
        "web/util.js": UTIL_JS,
        "server/main.go": GO_FILE,
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, text)
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "first")
    head = git(root, "rev-parse", "HEAD")
    # After the head commit: an untracked definition, an untracked secret, and a staged-only file.
    write(root / "app" / "scratch.py", "def scratch_only():\n    return 1\n")
    write(root / ".env", "TOKEN=hunter2\n")
    write(root / "late.py", "def late_function():\n    return 2\n")
    git(root, "add", "late.py")
    # And an edit in the working tree that the head does not have: the executor must keep reading the head's version.
    write(root / "app" / "pricing.py", PRICING.replace("return total", "return total + 1000", 1))
    return head
