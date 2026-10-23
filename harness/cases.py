"""Seeded code with known answers, shared by the four live review harnesses.

Each `Case` is a tiny git history: a `base` tree and a `head` tree. The diff
between them is what a reviewer sees. What is wrong in `head` is written down
here, so a run can be scored instead of eyeballed:

- `Bug`   a real defect. A reviewer that reports it has found it.
- `Decoy` code that LOOKS suspicious but is correct (a shell call with a
          constant command, a locked counter). A finding on a decoy is a false
          positive, and is what the verifier stage exists to reject.

Bugs and decoys name the line by a `marker`, a string that occurs exactly once
in the head file. That keeps the answer next to the code when someone edits
the code; no line numbers to recount.

The verifier and classifier fixtures at the bottom are built from the same
cases, so all four harnesses agree on what is true.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Bug:
    id: str
    file: str
    marker: str
    cls: str
    claim: str
    failing_input: str
    # A literal 1-based line in the reviewed file, for answers recorded from history (a Copilot
    # comment has a line, not a marker). 0 means: find the line holding `marker` in `case.head`.
    line: int = 0


@dataclass(frozen=True)
class Decoy:
    id: str
    file: str
    marker: str
    why_fine: str
    # What a careless reviewer would claim about it (used for verifier fixtures).
    false_claim: str


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    base: dict
    head: dict
    bugs: tuple = ()
    decoys: tuple = ()


def line_of(text: str, marker: str) -> int:
    """1-based line of the one line containing `marker`; ValueError when it is missing or ambiguous."""
    hits = [number for number, line in enumerate(text.splitlines(), 1) if marker in line]
    if len(hits) != 1:
        raise ValueError(f"marker {marker!r} matches {len(hits)} lines, expected exactly 1")
    return hits[0]


def bug_line(case: "Case", bug: Bug) -> int:
    """The line a bug is on: its recorded line, else where its marker is in the head file."""
    return bug.line or line_of(case.head[bug.file], bug.marker)


def snippet(text: str, line: int, radius: int = 4) -> str:
    """The lines around `line`, numbered, as a verifier would be shown the code a finding cites."""
    lines = text.splitlines()
    start, end = max(1, line - radius), min(len(lines), line + radius)
    return "\n".join(f"{number:>4} {lines[number - 1]}" for number in range(start, end + 1))


# ---------------------------------------------------------------------------
# seeded-three: one small file, three unrelated defects
# ---------------------------------------------------------------------------

_SEEDED_BASE = {
    "backup.py": (
        'import subprocess\n\n\ndef archive(name):\n'
        '    subprocess.run(["tar", "czf", name + ".tgz", "data"], check=True)\n'
    ),
}
_SEEDED_HEAD = {
    "backup.py": (
        "import subprocess\n\n\ndef archive(name):\n"
        "    # name comes straight from a web form field\n"
        '    subprocess.run("tar czf " + name + ".tgz data", shell=True, check=True)\n\n\n'
        "def last_items(items, n):\n"
        '    """The last n items of the list."""\n'
        "    return items[len(items) - n + 1:]\n\n\n"
        "def average(values):\n"
        "    return sum(values) / len(values)\n"
    ),
}

SEEDED_THREE = Case(
    name="seeded-three",
    description="One file, three unrelated defects: shell injection, an off-by-one slice, division by zero.",
    base=_SEEDED_BASE,
    head=_SEEDED_HEAD,
    bugs=(
        Bug("shell-injection", "backup.py", 'shell=True, check=True', "shell-injection",
            "name is concatenated into a shell command line run with shell=True",
            "name = 'x; rm -rf ~'"),
        Bug("off-by-one", "backup.py", "items[len(items) - n + 1:]", "off-by-one",
            "the slice starts one element too late, so last_items returns n-1 items",
            "items = [1, 2, 3, 4, 5], n = 2 returns [5], expected [4, 5]"),
        Bug("division-by-zero", "backup.py", "return sum(values) / len(values)", "division-by-zero",
            "an empty list divides by zero",
            "values = []"),
    ),
)

# ---------------------------------------------------------------------------
# clean: the same file written correctly. Anything reported is a false positive.
# ---------------------------------------------------------------------------

CLEAN = Case(
    name="clean",
    description="The seeded-three file rewritten correctly. Every finding on it is a false positive.",
    base=_SEEDED_BASE,
    head={
        "backup.py": (
            "import subprocess\n\n\ndef archive(name):\n"
            '    """Archive data/ as <name>.tgz; name must be a plain file name."""\n'
            '    if "/" in name or name.startswith("-"):\n'
            '        raise ValueError(f"bad archive name: {name!r}")\n'
            '    subprocess.run(["tar", "czf", name + ".tgz", "data"], check=True)\n\n\n'
            "def last_items(items, n):\n"
            '    """The last n items of the list."""\n'
            "    return items[-n:] if n > 0 else []\n\n\n"
            "def average(values):\n"
            "    if not values:\n"
            "        return 0.0\n"
            "    return sum(values) / len(values)\n"
        ),
    },
)

# ---------------------------------------------------------------------------
# orders: a four-file package, six defects of different classes, three decoys
# ---------------------------------------------------------------------------

_ORDERS_BASE = {
    "orders/__init__.py": '"""Order handling."""\n',
    "orders/pricing.py": (
        '"""Price calculations for an order."""\n\n\n'
        "def line_total(unit_price, quantity):\n"
        "    return round(unit_price * quantity, 2)\n"
    ),
    "orders/store.py": '"""Reading and writing orders."""\n',
    "orders/queue.py": '"""Batching of pending orders."""\n',
    "orders/report.py": '"""Order reports."""\n',
}
_ORDERS_HEAD = {
    "orders/__init__.py": '"""Order handling."""\n',
    "orders/pricing.py": (
        '"""Price calculations for an order."""\n\n\n'
        "def line_total(unit_price, quantity):\n"
        "    return round(unit_price * quantity, 2)\n\n\n"
        "def apply_discount(price, percent):\n"
        '    """Price after a percentage discount; percent is 0-100."""\n'
        "    return price * (1 - percent)\n"
    ),
    "orders/store.py": (
        '"""Reading and writing orders."""\n\n'
        "import os\n\n"
        'ATTACHMENTS = "/srv/orders/attachments"\n\n\n'
        "def find_orders(db, customer):\n"
        "    cursor = db.execute(f\"SELECT id, total FROM orders WHERE customer = '{customer}'\")\n"
        "    return cursor.fetchall()\n\n\n"
        "def read_attachment(name):\n"
        '    with open(os.path.join(ATTACHMENTS, name), "rb") as handle:\n'
        "        return handle.read()\n\n\n"
        "def count_orders(db):\n"
        '    return db.execute("SELECT COUNT(*) FROM orders").fetchone()[0]\n'
    ),
    "orders/queue.py": (
        '"""Batching of pending orders."""\n\n'
        "import threading\n\n\n"
        "def pop_batch(pending, size):\n"
        '    """Remove and return up to `size` orders from the front of `pending`."""\n'
        "    batch = pending[: size + 1]\n"
        "    del pending[: size + 1]\n"
        "    return batch\n\n\n"
        "class Counter:\n"
        "    def __init__(self):\n"
        "        self.value = 0\n"
        "        self._lock = threading.Lock()\n\n"
        "    def increment(self):\n"
        "        with self._lock:\n"
        "            self.value += 1\n"
    ),
    "orders/report.py": (
        '"""Order reports."""\n\n'
        "import subprocess\n\n\n"
        "def average_wait(waits):\n"
        '    """Mean wait in minutes."""\n'
        "    return sum(waits) / len(waits)\n\n\n"
        'def add_tag(order, tags=[]):\n'
        '    tags.append(order["id"])\n'
        "    return tags\n\n\n"
        "def build_stamp():\n"
        '    return subprocess.run("date +%F", shell=True, capture_output=True, text=True).stdout.strip()\n'
    ),
}

ORDERS = Case(
    name="orders",
    description="Four files, six defects across injection, path, off-by-one, formula, empty-input and shared-state classes, plus three correct-but-suspicious decoys.",
    base=_ORDERS_BASE,
    head=_ORDERS_HEAD,
    bugs=(
        Bug("discount-formula", "orders/pricing.py", "return price * (1 - percent)", "logic",
            "percent is documented as 0-100 but is used as a fraction, so the result is wildly wrong",
            "price = 100, percent = 10 returns -900, expected 90"),
        Bug("sql-injection", "orders/store.py", "SELECT id, total FROM orders WHERE customer", "sql-injection",
            "customer is interpolated into the SQL text",
            "customer = \"x' OR '1'='1\" returns every order"),
        Bug("path-traversal", "orders/store.py", "open(os.path.join(ATTACHMENTS, name)", "path-traversal",
            "name is joined onto the attachments directory without being checked, so ../ escapes it",
            "name = '../../etc/passwd'"),
        Bug("batch-off-by-one", "orders/queue.py", "batch = pending[: size + 1]", "off-by-one",
            "takes and deletes size + 1 orders instead of size",
            "pending = [1, 2, 3], size = 1 returns [1, 2], expected [1]"),
        Bug("empty-average", "orders/report.py", "return sum(waits) / len(waits)", "division-by-zero",
            "an empty list divides by zero",
            "waits = []"),
        Bug("mutable-default", "orders/report.py", "def add_tag(order, tags=[])", "shared-state",
            "the default list is created once and shared by every call that omits tags",
            "add_tag({'id': 1}) then add_tag({'id': 2}) returns [1, 2]"),
    ),
    decoys=(
        Decoy("constant-sql", "orders/store.py", 'SELECT COUNT(*) FROM orders',
              "the query is a constant string, nothing the caller controls reaches it",
              "SQL injection in count_orders"),
        Decoy("locked-counter", "orders/queue.py", "self.value += 1",
              "the increment runs inside `with self._lock`, so concurrent callers cannot lose updates",
              "race condition: Counter.increment is not thread safe"),
        Decoy("constant-shell", "orders/report.py", 'shell=True, capture_output=True',
              "shell=True is used with a constant command and no caller input",
              "shell injection in build_stamp"),
    ),
)

ALL_CASES = {case.name: case for case in (SEEDED_THREE, CLEAN, ORDERS)}


# ---------------------------------------------------------------------------
# Verifier fixtures: a finding plus the code it cites, and whether it is real
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifierCase:
    id: str
    file: str
    line: int
    cls: str
    claim: str
    failing_input: str
    code: str
    real: bool


def verifier_cases(case: Case = ORDERS) -> list:
    """Every bug as a true finding (with its failing input) and every decoy as a false one."""
    out = []
    for bug in case.bugs:
        text = case.head[bug.file]
        line = line_of(text, bug.marker)
        out.append(VerifierCase(f"{case.name}/{bug.id}", bug.file, line, bug.cls, bug.claim,
                                bug.failing_input, snippet(text, line), True))
    for decoy in case.decoys:
        text = case.head[decoy.file]
        line = line_of(text, decoy.marker)
        out.append(VerifierCase(f"{case.name}/{decoy.id}", decoy.file, line, "decoy", decoy.false_claim,
                                "", snippet(text, line), False))
    return out


# ---------------------------------------------------------------------------
# Classifier fixtures: a rubric, and the entry each finding should land on
# ---------------------------------------------------------------------------

# The same shape as the `rubric` array in a reviewers file.
RUBRIC = (
    {"name": "injection-or-traversal", "level": "high", "score": 3,
     "definition": "Untrusted input reaches a shell, a SQL statement or a filesystem path without validation."},
    {"name": "crash-on-input", "level": "moderate", "score": 3,
     "definition": "A valid or empty input makes the code raise an unhandled exception."},
    {"name": "wrong-result", "level": "moderate", "score": 3,
     "definition": "The code returns or changes the wrong data for valid input: a bad formula, an off-by-one, state shared between calls."},
    {"name": "style-only", "level": "low", "score": 4,
     "definition": "Naming, formatting or readability, with no effect on behaviour."},
)


@dataclass(frozen=True)
class ClassifierCase:
    id: str
    claim: str
    failing_input: str
    # Rubric entry name the classifier should pick; None = fits nothing, must come back unclassified.
    expected: object


_BUG_ENTRY = {
    "shell-injection": "injection-or-traversal",
    "sql-injection": "injection-or-traversal",
    "path-traversal": "injection-or-traversal",
    "division-by-zero": "crash-on-input",
    "empty-average": "crash-on-input",
    "off-by-one": "wrong-result",
    "batch-off-by-one": "wrong-result",
    "discount-formula": "wrong-result",
    "mutable-default": "wrong-result",
}


def classifier_cases() -> list:
    out = []
    for case in (SEEDED_THREE, ORDERS):
        for bug in case.bugs:
            out.append(ClassifierCase(f"{case.name}/{bug.id}", bug.claim, bug.failing_input, _BUG_ENTRY[bug.id]))
    out.append(ClassifierCase("style/short-name", "the variable name `n` is too short to be readable", "", "style-only"))
    out.append(ClassifierCase("none/readme", "the project README could use a longer introduction", "", None))
    out.append(ClassifierCase(
        "overlap/path-traversal-crash",
        "a name containing ../ escapes the attachments directory, and a missing file raises FileNotFoundError",
        "name = '../../etc/passwd'",
        "injection-or-traversal",  # fits two entries; the higher level wins
    ))
    return out
