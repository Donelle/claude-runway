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
    # The failing input that careless reviewer would offer with the claim. A false finding from a model
    # carries a plausible one, so the verifier fixtures give decoys one too: a decoy with none would be
    # told apart from a real finding by that alone.
    false_input: str = ""


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
              "SQL injection in count_orders",
              false_input="count_orders(db) where the customer name is \"'; DROP TABLE orders; --\""),
        Decoy("locked-counter", "orders/queue.py", "self.value += 1",
              "the increment runs inside `with self._lock`, so concurrent callers cannot lose updates",
              "race condition: Counter.increment is not thread safe",
              false_input="two threads calling Counter.increment() at the same moment"),
        Decoy("constant-shell", "orders/report.py", 'shell=True, capture_output=True',
              "shell=True is used with a constant command and no caller input",
              "shell injection in build_stamp",
              false_input="build_stamp() run with a PATH that has a malicious `date` first"),
    ),
)

# ---------------------------------------------------------------------------
# More cases. Every comparison between models or reply formats used to rest on `seeded-three` and
# `orders`, so a difference could be luck. These add defect kinds (resource leaks, error handling, concurrency,
# off-by-one, injection, a removed guard) and shapes (one file, several files, a deletion-only change), plus a
# correct sibling for the ones a reviewer is most likely to over-report, so false alarms are measured on more than
# one input. Each stays stdlib-only and deterministic, and keeps its bugs more than LINE_TOLERANCE (3) lines apart
# so a finding is credited to one bug, not its neighbour.
# ---------------------------------------------------------------------------

# leaks: one file, resources that are not released on every path --------------------------------------------------

_LEAKS_BASE = {
    "fileutil.py": (
        '"""File helpers."""\n\n\n'
        "def read_text(path):\n"
        '    with open(path, encoding="utf-8") as handle:\n'
        "        return handle.read()\n"
    ),
}

LEAKS = Case(
    name="leaks",
    description="One file, four resources not released on every path: a file, a database connection, a scratch directory and a child process.",
    base=_LEAKS_BASE,
    head={
        "fileutil.py": (
            '"""File helpers."""\n\n'
            "import shutil\n"
            "import sqlite3\n"
            "import subprocess\n"
            "import tempfile\n\n\n"
            "def read_text(path):\n"
            '    with open(path, encoding="utf-8") as handle:\n'
            "        return handle.read()\n\n\n"
            "_recent = []\n\n\n"
            "def first_line(path):\n"
            '    handle = open(path, encoding="utf-8")\n'
            "    _recent.append(handle)  # remembered so a later call can reread it\n"
            '    return handle.readline().rstrip("\\n")\n\n\n'
            "def count_rows(db_path):\n"
            "    conn = sqlite3.connect(db_path)\n"
            '    rows = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]\n'
            "    conn.close()\n"
            "    return rows\n\n\n"
            "def build_in_scratch(command):\n"
            "    scratch = tempfile.mkdtemp()\n"
            "    subprocess.run(command, cwd=scratch, check=True)\n"
            "    shutil.rmtree(scratch)\n\n\n"
            "def start_worker(command):\n"
            "    proc = subprocess.Popen(command)\n"
            "    return proc.pid\n"
        ),
    },
    bugs=(
        Bug("retained-handle", "fileutil.py", 'handle = open(path, encoding="utf-8")', "resource-leak",
            "every opened handle is appended to the module-level _recent list and never closed, so the descriptors pile up until the process runs out",
            "first_line(path) called once per file over thousands of files raises OSError: too many open files"),
        Bug("connection-leak", "fileutil.py", 'conn.execute("SELECT COUNT(*) FROM items")', "resource-leak",
            "the connection is closed after the query, so a query that raises (no items table) never closes it",
            "db_path = a database with no items table"),
        Bug("scratch-leak", "fileutil.py", "subprocess.run(command, cwd=scratch, check=True)", "resource-leak",
            "check=True raises when the command fails, so the scratch directory is never removed",
            "command = ['false']"),
        Bug("child-never-waited", "fileutil.py", "proc = subprocess.Popen(command)", "resource-leak",
            "only the pid is returned, so the Popen object is dropped and the child is never waited on, leaving a zombie",
            "start_worker(['true']) called in a loop"),
    ),
    decoys=(
        Decoy("with-open", "fileutil.py", 'with open(path, encoding="utf-8") as handle:',
              "the with block closes the file on every path, including an exception",
              "file handle not closed in read_text",
              false_input="read_text(path) where handle.read() raises UnicodeDecodeError"),
    ),
)

CLEAN_LEAKS = Case(
    name="clean-leaks",
    description="The leaks file with every resource released on every path. Every finding on it is a false positive.",
    base=_LEAKS_BASE,
    head={
        "fileutil.py": (
            '"""File helpers."""\n\n'
            "import sqlite3\n"
            "import subprocess\n"
            "import tempfile\n"
            "from contextlib import closing\n\n\n"
            "def read_text(path):\n"
            '    with open(path, encoding="utf-8") as handle:\n'
            "        return handle.read()\n\n\n"
            "def first_line(path):\n"
            '    with open(path, encoding="utf-8") as handle:\n'
            "        line = handle.readline()\n"
            '    return line.rstrip("\\n") if line else None\n\n\n'
            "def count_rows(db_path):\n"
            "    with closing(sqlite3.connect(db_path)) as conn:\n"
            '        return conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]\n\n\n'
            "def build_in_scratch(command):\n"
            "    with tempfile.TemporaryDirectory() as scratch:\n"
            "        subprocess.run(command, cwd=scratch, check=True)\n\n\n"
            "def start_worker(command):\n"
            '    """The caller owns the process and must wait on it."""\n'
            "    return subprocess.Popen(command)\n"
        ),
    },
)

# errors: one file, wrong error handling -------------------------------------------------------------------------

_ERRORS_BASE = {
    "jobs.py": '"""Job runner."""\n\n\ndef run(job):\n    return job()\n',
}

ERRORS = Case(
    name="errors",
    description="One file, four wrong ways to handle an error (bare except, swallowed exception, wrong exception type, return in finally) and one handler that is correct.",
    base=_ERRORS_BASE,
    head={
        "jobs.py": (
            '"""Job runner."""\n\n'
            "import json\n\n\n"
            "def run(job):\n"
            "    return job()\n\n\n"
            "def run_quietly(job):\n"
            "    try:\n"
            "        return job()\n"
            "    except:\n"
            "        return None\n\n\n"
            "def load_config(path):\n"
            "    try:\n"
            '        with open(path, encoding="utf-8") as handle:\n'
            "            return json.load(handle)\n"
            "    except Exception:\n"
            "        pass\n\n\n"
            "def lookup(settings, key):\n"
            "    try:\n"
            "        return settings[key]\n"
            "    except IndexError:\n"
            "        return None\n\n\n"
            "def run_and_note(job, log):\n"
            "    try:\n"
            "        return job()\n"
            "    finally:\n"
            '        return log.append("done")\n\n\n'
            "def run_logged(job, log):\n"
            "    try:\n"
            "        return job()\n"
            "    except Exception as error:\n"
            "        log.append(str(error))\n"
            "        raise\n"
        ),
    },
    bugs=(
        Bug("bare-except", "jobs.py", "except:", "error-handling",
            "a bare except also catches KeyboardInterrupt and SystemExit, and hides every failure as None",
            "job raises KeyboardInterrupt on Ctrl-C and run_quietly returns None"),
        Bug("swallowed-error", "jobs.py", "except Exception:", "error-handling",
            "a missing or malformed config file is swallowed and None is returned, so the caller cannot tell it from an empty config",
            "path = a file that holds invalid JSON"),
        Bug("wrong-exception", "jobs.py", "except IndexError:", "error-handling",
            "a dict raises KeyError, not IndexError, so a missing key escapes instead of returning None",
            "settings = {}, key = 'a' raises KeyError"),
        Bug("return-in-finally", "jobs.py", 'return log.append("done")', "error-handling",
            "a return in finally discards the job's result and swallows any exception it raised, returning None",
            "job raises ValueError and run_and_note returns None"),
    ),
    decoys=(
        Decoy("log-and-reraise", "jobs.py", "except Exception as error:",
              "the broad handler only records the message and re-raises, so nothing is swallowed",
              "a broad except Exception swallows errors in run_logged",
              false_input="job raises ValueError('boom') and the caller never sees it"),
    ),
)

CLEAN_ERRORS = Case(
    name="clean-errors",
    description="The errors file with each failure handled correctly. Every finding on it is a false positive.",
    base=_ERRORS_BASE,
    head={
        "jobs.py": (
            '"""Job runner."""\n\n'
            "import json\n\n\n"
            "def run(job):\n"
            "    return job()\n\n\n"
            "def run_quietly(job):\n"
            '    """A job reports a failure it expects by raising RuntimeError; anything else is a bug and propagates."""\n'
            "    try:\n"
            "        return job()\n"
            "    except RuntimeError:\n"
            "        return None\n\n\n"
            "def load_config(path):\n"
            "    try:\n"
            '        with open(path, encoding="utf-8") as handle:\n'
            "            return json.load(handle)\n"
            "    except (OSError, ValueError) as error:\n"
            '        raise RuntimeError(f"cannot load {path}: {error}") from error\n\n\n'
            "def lookup(settings, key):\n"
            "    try:\n"
            "        return settings[key]\n"
            "    except KeyError:\n"
            "        return None\n\n\n"
            "def run_and_note(job, log):\n"
            "    try:\n"
            "        return job()\n"
            "    finally:\n"
            '        log.append("done")\n\n\n'
            "def run_logged(job, log):\n"
            "    try:\n"
            "        return job()\n"
            "    except Exception as error:\n"
            "        log.append(str(error))\n"
            "        raise\n"
        ),
    },
)

# concurrency: two files, shared state touched without the lock -----------------------------------------------------

_CONCURRENCY_BASE = {
    "counter.py": '"""Shared counters."""\n',
    "registry.py": '"""Registry of named resources."""\n',
}

CONCURRENCY = Case(
    name="concurrency",
    description="Two files, four races (an unlocked update, a check-then-act, a lock never released on error, iteration over a dict others change) and two thread-safe constructs that look risky.",
    base=_CONCURRENCY_BASE,
    head={
        "counter.py": (
            '"""Shared counters."""\n\n'
            "import queue\n"
            "import threading\n\n\n"
            "class Stats:\n"
            "    def __init__(self):\n"
            "        self.hits = {}\n"
            "        self._lock = threading.Lock()\n\n"
            "    def record(self, key):\n"
            "        self.hits[key] = self.hits.get(key, 0) + 1\n\n"
            "    def reset(self):\n"
            "        with self._lock:\n"
            "            self.hits.clear()\n\n\n"
            "_jobs = queue.Queue()\n\n\n"
            "def submit(job):\n"
            "    _jobs.put(job)\n\n\n"
            "_local = threading.local()\n\n\n"
            "def current_user():\n"
            '    return getattr(_local, "user", None)\n'
        ),
        "registry.py": (
            '"""Registry of named resources."""\n\n'
            "import threading\n\n"
            "_lock = threading.Lock()\n"
            "_items = {}\n\n\n"
            "def get_or_create(name, factory):\n"
            "    if name not in _items:\n"
            "        _items[name] = factory()\n"
            "    return _items[name]\n\n\n"
            "def remove(name):\n"
            "    _lock.acquire()\n"
            "    del _items[name]\n"
            "    _lock.release()\n\n\n"
            "def names():\n"
            "    result = []\n"
            "    for name in _items:\n"
            "        result.append(name)\n"
            "    return result\n"
        ),
    },
    bugs=(
        Bug("unlocked-update", "counter.py", "self.hits[key] = self.hits.get(key, 0) + 1", "race-condition",
            "the read-modify-write is not under self._lock (reset takes it), so two threads can lose an update",
            "two threads record('a') 1000 times each and hits['a'] ends below 2000"),
        Bug("check-then-act", "registry.py", "if name not in _items:", "race-condition",
            "two threads can both see the name missing and each call the factory, creating two resources and keeping one",
            "two threads call get_or_create('db', connect) at the same moment"),
        Bug("lock-not-released", "registry.py", "_lock.acquire()", "race-condition",
            "del raises KeyError for an unknown name, and the lock is never released, so every later caller blocks forever",
            "remove('missing') and then any other remove(...)"),
        Bug("iterate-while-mutated", "registry.py", "for name in _items:", "race-condition",
            "another thread adding or removing an item during the loop raises RuntimeError: dictionary changed size during iteration",
            "names() running while get_or_create adds a name from another thread"),
    ),
    decoys=(
        Decoy("thread-safe-queue", "counter.py", "_jobs.put(job)",
              "queue.Queue does its own locking, so concurrent puts are safe",
              "race condition: concurrent submit calls corrupt the queue",
              false_input="two threads calling submit() at the same moment"),
        Decoy("thread-local", "counter.py", 'return getattr(_local, "user", None)',
              "threading.local gives each thread its own attribute, so threads do not share it",
              "shared mutable global read without a lock in current_user",
              false_input="two threads reading current_user() while one of them sets a user"),
    ),
)

CLEAN_CONCURRENCY = Case(
    name="clean-concurrency",
    description="The concurrency files with every shared access under the lock. Every finding on it is a false positive.",
    base=_CONCURRENCY_BASE,
    head={
        "counter.py": (
            '"""Shared counters."""\n\n'
            "import queue\n"
            "import threading\n\n\n"
            "class Stats:\n"
            "    def __init__(self):\n"
            "        self.hits = {}\n"
            "        self._lock = threading.Lock()\n\n"
            "    def record(self, key):\n"
            "        with self._lock:\n"
            "            self.hits[key] = self.hits.get(key, 0) + 1\n\n"
            "    def reset(self):\n"
            "        with self._lock:\n"
            "            self.hits.clear()\n\n\n"
            "_jobs = queue.Queue()\n\n\n"
            "def submit(job):\n"
            "    _jobs.put(job)\n\n\n"
            "_local = threading.local()\n\n\n"
            "def current_user():\n"
            '    return getattr(_local, "user", None)\n'
        ),
        "registry.py": (
            '"""Registry of named resources."""\n\n'
            "import threading\n\n"
            "_lock = threading.RLock()  # reentrant: a factory may resolve another resource from this registry\n"
            "_items = {}\n\n\n"
            "def get_or_create(name, factory):\n"
            "    with _lock:\n"
            "        if name not in _items:\n"
            "            _items[name] = factory()\n"
            "        return _items[name]\n\n\n"
            "def remove(name):\n"
            "    with _lock:\n"
            "        del _items[name]\n\n\n"
            "def names():\n"
            "    with _lock:\n"
            "        return list(_items)\n"
        ),
    },
)

# inventory: three files, comparison and range mistakes ----------------------------------------------------------

_INVENTORY_BASE = {
    "inventory/__init__.py": '"""Stock keeping."""\n',
    "inventory/stock.py": '"""Stock levels."""\n',
    "inventory/pricing.py": '"""Prices and totals."""\n',
    "inventory/report.py": '"""Stock reports."""\n',
}

INVENTORY = Case(
    name="inventory",
    description="Four files, five defects (a wrong comparison, a range one past the end, float equality, `is` on a string, the wrong sort key) and two correct lines that look suspicious.",
    base=_INVENTORY_BASE,
    head={
        "inventory/__init__.py": '"""Stock keeping."""\n',
        "inventory/stock.py": (
            '"""Stock levels."""\n\n\n'
            "def reorder_list(levels, threshold):\n"
            '    """Items whose level is at or below the threshold."""\n'
            "    return [item for item, level in levels.items() if level < threshold]\n\n\n"
            "def last_n(entries, n):\n"
            "    out = []\n"
            "    for index in range(len(entries) - n, len(entries) + 1):\n"
            "        out.append(entries[index])\n"
            "    return out\n\n\n"
            "def copy_levels(levels):\n"
            "    return dict(levels)\n"
        ),
        "inventory/pricing.py": (
            '"""Prices and totals."""\n\n\n'
            "def is_free(price):\n"
            "    return price == 0.0\n\n\n"
            "def total(prices):\n"
            "    result = 0.0\n"
            "    for price in prices:\n"
            "        result += price\n"
            "    return result\n\n\n"
            "def same_total(a, b):\n"
            "    return total(a) == total(b)\n\n\n"
            "def is_sale(code):\n"
            '    return code is "SALE"\n'
        ),
        "inventory/report.py": (
            '"""Stock reports."""\n\n'
            "from inventory.stock import reorder_list\n\n\n"
            "def lowest(levels):\n"
            "    return sorted(levels.items(), key=lambda pair: pair[0])[0]\n\n\n"
            "def summary(levels, threshold):\n"
            "    low = reorder_list(levels, threshold)\n"
            '    return f"{len(low)} of {len(levels)} items need reordering"\n'
        ),
    },
    bugs=(
        Bug("wrong-comparison", "inventory/stock.py", "if level < threshold]", "logic",
            "the docstring says at or below the threshold, but < leaves out items exactly at it",
            "levels = {'bolt': 5}, threshold = 5 returns [], expected ['bolt']"),
        Bug("range-past-end", "inventory/stock.py", "range(len(entries) - n, len(entries) + 1)", "off-by-one",
            "the range runs one past the last index, so entries[len(entries)] raises IndexError",
            "entries = [1, 2, 3], n = 2 raises IndexError"),
        Bug("float-equality", "inventory/pricing.py", "return total(a) == total(b)", "logic",
            "float sums depend on order, so the same prices in another order can compare unequal",
            "a = [0.1, 0.2, 0.3], b = [0.3, 0.2, 0.1] returns False"),
        Bug("is-on-string", "inventory/pricing.py", 'return code is "SALE"', "logic",
            "`is` compares identity, not value, so a code built at run time is not the interned literal and the test fails",
            "code = ''.join(['SA', 'LE']) returns False"),
        Bug("wrong-sort-key", "inventory/report.py", "key=lambda pair: pair[0]", "logic",
            "pair[0] is the item name, so this returns the first name alphabetically, not the item with the lowest level",
            "levels = {'a': 9, 'b': 1} returns ('a', 9), expected ('b', 1)"),
    ),
    decoys=(
        Decoy("zero-sentinel", "inventory/pricing.py", "return price == 0.0",
              "0.0 is an exact sentinel for a free item that is stored as given, not the result of arithmetic",
              "float equality comparison is unreliable in is_free",
              false_input="price = 0.1 + 0.2 - 0.3 compared against 0.0"),
        Decoy("copy-of-ints", "inventory/stock.py", "return dict(levels)",
              "the levels are ints, which are immutable, so a shallow copy cannot share changeable state with the original",
              "shallow copy: changes to the copy leak into the original",
              false_input="copy = copy_levels(levels); copy['a'] = 2 changes levels"),
    ),
)

# handlers: two files, untrusted input reaching dangerous calls ---------------------------------------------------

_HANDLERS_BASE = {
    "web/__init__.py": '"""Web handlers."""\n',
    "web/handlers.py": '"""Request handlers."""\n',
    "web/jobs.py": '"""Background jobs."""\n',
}

HANDLERS = Case(
    name="handlers",
    description="Three files, five injection and secret-handling defects, plus three constant or parsed calls that look dangerous and are not (two of them with a hypothetical-environment attack, the kind that fooled the line format).",
    base=_HANDLERS_BASE,
    head={
        "web/__init__.py": '"""Web handlers."""\n',
        "web/handlers.py": (
            '"""Request handlers."""\n\n'
            "import ast\n"
            "import os\n"
            "import pickle\n\n"
            'API_TOKEN = "sk-live-4f9c1d7a2b"\n\n\n'
            "def ping(host):\n"
            '    os.system("ping -c 1 " + host)\n\n\n'
            "def load_session(body):\n"
            "    return pickle.loads(body)\n\n\n"
            "def calculate(expression):\n"
            "    return eval(expression)\n\n\n"
            "def find_user(db, name):\n"
            "    return db.execute(f\"SELECT id FROM users WHERE name = '{name}'\").fetchone()\n\n\n"
            "def parse_filter(text):\n"
            "    return ast.literal_eval(text)\n"
        ),
        "web/jobs.py": (
            '"""Background jobs."""\n\n'
            "import subprocess\n\n\n"
            "def disk_report():\n"
            '    return subprocess.run("df -h", shell=True, capture_output=True, text=True).stdout\n\n\n'
            "def git_head():\n"
            '    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()\n'
        ),
    },
    bugs=(
        Bug("hardcoded-secret", "web/handlers.py", 'API_TOKEN = "sk-live-4f9c1d7a2b"', "secret",
            "a live API token is committed in source, so anyone with the repository has it",
            "the token appears in git history and every clone"),
        Bug("os-system-injection", "web/handlers.py", 'os.system("ping -c 1 " + host)', "shell-injection",
            "host comes from the request and is concatenated into a shell command line",
            "host = '8.8.8.8; cat /etc/passwd'"),
        Bug("unsafe-deserialize", "web/handlers.py", "return pickle.loads(body)", "deserialization",
            "pickle.loads on a request body runs code chosen by the sender",
            "body = a pickle whose __reduce__ calls os.system"),
        Bug("eval-input", "web/handlers.py", "return eval(expression)", "code-injection",
            "eval runs whatever the request sends as Python",
            "expression = \"__import__('os').system('id')\""),
        Bug("fstring-sql", "web/handlers.py", "WHERE name = '{name}'", "sql-injection",
            "name is interpolated into the SQL text",
            "name = \"x' OR '1'='1\" returns the first user"),
    ),
    decoys=(
        Decoy("literal-eval", "web/handlers.py", "return ast.literal_eval(text)",
              "ast.literal_eval only evaluates Python literals and never calls anything, unlike eval",
              "code injection through ast.literal_eval in parse_filter",
              false_input="text = \"__import__('os').system('id')\""),
        Decoy("constant-shell-df", "web/jobs.py", '"df -h", shell=True',
              "shell=True is used with a constant command and no caller input",
              "shell injection in disk_report",
              false_input="disk_report() run with a PATH that has a malicious `df` first"),
        Decoy("list-args-git", "web/jobs.py", '["git", "rev-parse", "HEAD"]',
              "the command is a constant argument list with no shell, so nothing the caller controls reaches it",
              "command injection in git_head",
              false_input="git_head() run in a repository whose config sets a malicious core.pager"),
    ),
)

# guards-removed: a deletion-only change ---------------------------------------------------------------------------

_ACCOUNTS_BASE = (
    '"""Account operations."""\n\n\n'
    "def withdraw(balance, amount):\n"
    "    if amount <= 0:\n"
    '        raise ValueError("amount must be positive")\n'
    "    if amount > balance:\n"
    '        raise ValueError("insufficient funds")\n'
    "    return balance - amount\n\n\n"
    "def percent_of(part, whole):\n"
    "    if whole == 0:\n"
    '        raise ValueError("whole must not be zero")\n'
    "    return 100 * part / whole\n\n\n"
    "def first_name(full_name):\n"
    "    if not full_name.strip():\n"
    '        raise ValueError("name is empty")\n'
    "    return full_name.split()[0]\n\n\n"
    "def describe(account):\n"
    "    # kept for the old report format\n"
    "    return f\"{account['id']}: {account['balance']}\"\n\n\n"
    "def total(balances):\n"
    "    return sum(balances)\n"
)

GUARDS_REMOVED = Case(
    name="guards-removed",
    description="A deletion-only change: three validation guards are removed (the diff has no added lines), and a comment is removed that changes nothing.",
    base={"accounts.py": _ACCOUNTS_BASE},
    head={
        "accounts.py": (
            '"""Account operations."""\n\n\n'
            "def withdraw(balance, amount):\n"
            "    if amount <= 0:\n"
            '        raise ValueError("amount must be positive")\n'
            "    return balance - amount\n\n\n"
            "def percent_of(part, whole):\n"
            "    return 100 * part / whole\n\n\n"
            "def first_name(full_name):\n"
            "    return full_name.split()[0]\n\n\n"
            "def describe(account):\n"
            "    return f\"{account['id']}: {account['balance']}\"\n\n\n"
            "def total(balances):\n"
            "    return sum(balances)\n"
        ),
    },
    bugs=(
        Bug("overdraft-guard-removed", "accounts.py", "return balance - amount", "missing-validation",
            "the insufficient-funds check was deleted, so a withdrawal larger than the balance returns a negative balance",
            "withdraw(10, 50) returns -40"),
        Bug("zero-guard-removed", "accounts.py", "return 100 * part / whole", "missing-validation",
            "the zero check was deleted, so a whole of 0 now raises ZeroDivisionError instead of a clear ValueError",
            "percent_of(1, 0)"),
        Bug("empty-guard-removed", "accounts.py", "return full_name.split()[0]", "missing-validation",
            "the empty-name check was deleted, so an empty or blank name raises IndexError",
            "first_name('   ')"),
    ),
    decoys=(
        Decoy("comment-removed", "accounts.py", "return f\"{account['id']}: {account['balance']}\"",
              "only a comment was deleted here; the behaviour of describe is unchanged",
              "a deleted check leaves describe without validation of the account",
              false_input="describe({}) raises KeyError"),
    ),
)

CLEAN_GUARDS_REMOVED = Case(
    name="clean-guards-removed",
    description="A deletion-only change that is fine: an unused helper and a comment are removed and every guard stays. Every finding on it is a false positive.",
    base={"accounts.py": _ACCOUNTS_BASE},
    head={
        "accounts.py": (
            '"""Account operations."""\n\n\n'
            "def withdraw(balance, amount):\n"
            "    if amount <= 0:\n"
            '        raise ValueError("amount must be positive")\n'
            "    if amount > balance:\n"
            '        raise ValueError("insufficient funds")\n'
            "    return balance - amount\n\n\n"
            "def percent_of(part, whole):\n"
            "    if whole == 0:\n"
            '        raise ValueError("whole must not be zero")\n'
            "    return 100 * part / whole\n\n\n"
            "def first_name(full_name):\n"
            "    if not full_name.strip():\n"
            '        raise ValueError("name is empty")\n'
            "    return full_name.split()[0]\n\n\n"
            "def describe(account):\n"
            "    return f\"{account['id']}: {account['balance']}\"\n"
        ),
    },
)

# paging: one file, off-by-one in four shapes -----------------------------------------------------------------------

PAGING = Case(
    name="paging",
    description="One file, four off-by-one defects in pagination and chunking: a 1-based page used as 0-based, a floored page count, an exclusive end for an inclusive range, and a range that drops the last chunk.",
    base={"paging.py": '"""Pagination helpers."""\n'},
    head={
        "paging.py": (
            '"""Pagination helpers."""\n\n\n'
            "def page_slice(items, page, size):\n"
            '    """Items on a 1-based page."""\n'
            "    start = page * size\n"
            "    return items[start:start + size]\n\n\n"
            "def page_count(total, size):\n"
            '    """Number of pages needed for `total` items."""\n'
            "    return total // size\n\n\n"
            "def window(items, start, end):\n"
            '    """Items from start to end, both inclusive."""\n'
            "    return items[start:end]\n\n\n"
            "def chunks(items, size):\n"
            '    """Split items into lists of `size`; the last may be shorter."""\n'
            "    return [items[i:i + size] for i in range(0, len(items) - size, size)]\n"
        ),
    },
    bugs=(
        Bug("page-offset", "paging.py", "start = page * size", "off-by-one",
            "pages are 1-based, so page 1 should start at 0; this skips the first page and runs past the end",
            "items = list(range(10)), page = 1, size = 5 returns [5..9], expected [0..4]"),
        Bug("page-count-floor", "paging.py", "return total // size", "off-by-one",
            "floor division drops a partly filled last page",
            "total = 11, size = 5 returns 2, expected 3"),
        Bug("inclusive-end", "paging.py", "return items[start:end]", "off-by-one",
            "a slice excludes its end, but the docstring says the end is inclusive",
            "items = [1, 2, 3, 4], start = 1, end = 2 returns [2], expected [2, 3]"),
        Bug("dropped-chunk", "paging.py", "range(0, len(items) - size, size)", "off-by-one",
            "stopping at len(items) - size leaves out the last chunk",
            "items = [1, 2, 3, 4], size = 2 returns [[1, 2]], expected [[1, 2], [3, 4]]"),
    ),
)

ALL_CASES = {case.name: case for case in (
    SEEDED_THREE, CLEAN, ORDERS,
    LEAKS, CLEAN_LEAKS, ERRORS, CLEAN_ERRORS, CONCURRENCY, CLEAN_CONCURRENCY,
    INVENTORY, HANDLERS, GUARDS_REMOVED, CLEAN_GUARDS_REMOVED, PAGING,
)}


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
    # The line the finding accuses, as a reviewer would copy it (the verifier stage's grounding check
    # looks for it in the file).
    quote: str = ""


def verifier_cases(case: Case = ORDERS) -> list:
    """Every bug as a true finding (with its failing input) and every decoy as a false one."""
    out = []
    for bug in case.bugs:
        text = case.head[bug.file]
        line = line_of(text, bug.marker)
        out.append(VerifierCase(f"{case.name}/{bug.id}", bug.file, line, bug.cls, bug.claim,
                                bug.failing_input, snippet(text, line), True, text.splitlines()[line - 1].strip()))
    for decoy in case.decoys:
        text = case.head[decoy.file]
        line = line_of(text, decoy.marker)
        out.append(VerifierCase(f"{case.name}/{decoy.id}", decoy.file, line, "decoy", decoy.false_claim,
                                decoy.false_input, snippet(text, line), False, text.splitlines()[line - 1].strip()))
    return out


def fixture_cases() -> dict:
    """The cases that have answers to build verifier fixtures from (at least one bug or decoy), by name."""
    return {name: case for name, case in ALL_CASES.items() if case.bugs or case.decoys}


def merged_case(cases: list) -> Case:
    """One case holding every given case's trees, so a single repository serves all their fixtures.

    A path two cases share must hold the same text in both, or the fixtures of one would cite code from the other."""
    if len(cases) == 1:
        return cases[0]
    base: dict = {}
    head: dict = {}
    for case in cases:
        for tree, merged in ((case.base, base), (case.head, head)):
            for path, text in tree.items():
                if merged.setdefault(path, text) != text:
                    raise ValueError(f"cases {[c.name for c in cases]} share the path {path!r} with different text")
    return Case("+".join(case.name for case in cases), "merged cases", base, head,
                tuple(bug for case in cases for bug in case.bugs), tuple(decoy for case in cases for decoy in case.decoys))


def verifier_cases_for(cases: list) -> list:
    """The verifier fixtures of each case in turn (ids are `<case>/<bug or decoy id>`, so they never collide)."""
    return [fixture for case in cases for fixture in verifier_cases(case)]


# ---------------------------------------------------------------------------
# Classifier fixtures: a rubric, and the entry each finding should land on
# ---------------------------------------------------------------------------

# The same shape as the `rubric` array in a reviewers file.
RUBRIC = (
    {"name": "injection-or-traversal", "level": "high", "score": 0.5,
     "definition": "Untrusted input reaches a shell, a SQL statement or a filesystem path without validation."},
    {"name": "crash-on-input", "level": "moderate", "score": 0.5,
     "definition": "A valid or empty input makes the code raise an unhandled exception."},
    {"name": "wrong-result", "level": "moderate", "score": 0.5,
     "definition": "The code returns or changes the wrong data for valid input: a bad formula, an off-by-one, state shared between calls."},
    {"name": "style-only", "level": "low", "score": 0.75,
     "definition": "Naming, formatting or readability, with no effect on behaviour."},
)


@dataclass(frozen=True)
class ClassifierCase:
    id: str
    claim: str
    failing_input: str
    # Rubric entry name the classifier should pick; None = fits nothing, must come back unclassified.
    expected: object
    # True when the claim fits more than one entry, so `expected` is the one with the higher level.
    overlap: bool = False


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
        overlap=True,
    ))
    # Borderline claims: each only partly resembles an entry's wording, or no entry's. A model that says it is sure of
    # everything gives these a high probability too, which is what shows the scores dropping nothing.
    out.append(ClassifierCase("weak/vague-name", "the function is called `get`, which says nothing about what it fetches", "", "style-only"))
    out.append(ClassifierCase("weak/log-line", "a log line prints the customer's full name, which may be sensitive", "", None))
    out.append(ClassifierCase("weak/slow", "the loop is quadratic, so it may be slow for very large lists", "", None))
    out.append(ClassifierCase(
        "weak/partial-wrong",
        "the discount is rounded to whole cents before tax instead of after, so totals can be a cent off",
        "", "wrong-result",
    ))
    return out
