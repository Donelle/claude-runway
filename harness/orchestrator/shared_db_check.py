#!/usr/bin/env python3
"""Check that several containers can record into one shared reviewer database, as the harness's sessions do.

Every Pi session writes its own run record straight into one database on the host, through a folder Docker mounts into its
container. Whether that is safe depends on how Docker shares files on the machine, which a unit test cannot show. This runs
it: the host creates a database, then several containers record into it at the same time with the reviewer's real recording
code (`reviewer_db_lib.new_run_recorder`) and synthetic runs. No model is involved. A trial passes when every writer exits
0, every record is in the database and SQLite's integrity check says "ok".

What it holds to, as a real run must:

  * The database is created BEFORE any container starts (containers creating it together lost records).
  * The host does not open it again until every container has exited (host writes during a run lost records and damaged it;
    reading during a run has not been tested).
  * The folder is mounted, never the single file.

It uses a throwaway folder under `harness/live-tests/logs/`, so the same file sharing as the harness database, and never
that database itself. A passing folder is deleted; a failing one is kept and named. The folder of a trial that could not be
made, or was interrupted, is deleted too, since nothing was checked in it. When a container cannot be confirmed gone, the
script says so, does not read the database and keeps the folder, which that container may still be using. Run it once on
each machine whose harness database containers will share (it was designed on Windows with Docker Desktop; macOS has to be
run too):

    .venv/bin/python harness/orchestrator/shared_db_check.py            # .venv/Scripts/python.exe on Windows

Needs Docker and the harness image (`docker build -t claude-runway-pi:dev harness/pi`). A development tool, not product code;
`tests/test_harness_shared_db_check.py` covers its logic with no Docker. Exit code: 0 = every trial passed; 1 = a trial
failed; 2 = it could not run (a bad argument, no docker, no image).
"""

import argparse
import contextlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
# The driver says how a container of the harness is run (image, mounts, user, the `timeout` backstop) and how one is removed.
for _directory in (REPO_ROOT / "libs", REPO_ROOT / "harness" / "pi"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import pi_driver  # noqa: E402  # pyright: ignore[reportMissingImports]  (harness/pi is put on sys.path just above)

DEFAULT_WRITERS = 6
DEFAULT_RECORDS = 25
DEFAULT_TRIALS = 5
# How long one writer may take, containers starting included. A trial takes seconds; this only ends a stuck one.
DEFAULT_TIMEOUT_SECONDS = 300
# `docker run`'s own exit codes: 125 the daemon refused (no image, a bad mount), 126/127 the command could not be run.
DOCKER_COULD_NOT_RUN = (125, 126, 127)
# The key of each synthetic run's settings that says which writer saved which record.
MARKER = "shared_db_check"
C_LIBS = f"{pi_driver.C_REVIEWER}/libs"

# What one writer runs, passed with `python -c`, so the container needs no mount beyond `libs/` and the database folder. It
# waits until every writer is up (each leaves a file in the shared folder), because containers start some time apart, and
# without the wait the first could finish before the last began and nothing would be shared.
WRITER = """
import os, sys, time
from pathlib import Path
sys.path.insert(0, os.environ["CHECK_LIBS"])
from reviewer_db_lib import new_run_recorder
barrier = Path(os.environ["CHECK_BARRIER"])
index, writers, records = (int(value) for value in sys.argv[1:4])
(barrier / ("ready-%d" % index)).touch()
deadline = time.monotonic() + float(os.environ.get("CHECK_BARRIER_SECONDS", "120"))
while len(list(barrier.glob("ready-*"))) < writers:
    if time.monotonic() > deadline:
        sys.exit("writer %d: not every writer started in time" % index)
    time.sleep(0.02)
for number in range(records):
    recorder = new_run_recorder(source="mcp", review_type="reasoning", repo_path="/nowhere", base_ref="base",
                                head_ref="head", settings={"shared_db_check": "%d-%d" % (index, number)})
    recorder.close("finished")
print("writer %d: closed %d runs" % (index, records))
"""


class CannotRun(Exception):
    """The check could not be made at all, which is not a failed check."""


@dataclass
class TrialResult:
    expected: int
    saved: int = 0  # distinct synthetic records in the database
    integrity: str = ""
    exit_codes: list = field(default_factory=list)
    writer_errors: list = field(default_factory=list)  # lines the writers printed to stderr (a failed write is only a warning)
    problem: str = ""  # why the database could not be read afterwards

    @property
    def passed(self) -> bool:
        return (not self.problem and self.saved == self.expected and self.integrity == "ok"
                and bool(self.exit_codes) and all(code == 0 for code in self.exit_codes))


def writer_command(*, docker: str, image: str, name: str, libs: Path, database_dir: Path, index: int, writers: int,
                   records: int, timeout: int) -> list:
    """The `docker run` command of one writer: no network, `libs/` read-only and the database folder writable, nothing else."""
    return [
        docker, "run", "--rm", "--init", "--name", name, "--network", "none",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        *pi_driver.user_arguments(),
        "--env", "HOME=/tmp",  # the recorder also writes under ~/.claude, which here is thrown away with the container
        "--env", f"CHECK_LIBS={C_LIBS}",
        "--env", f"CHECK_BARRIER={pi_driver.C_DB}",
        "--env", f"CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB={pi_driver.C_DB}/{pi_driver.DB_FILE_NAME}",
        "--mount", f"type=bind,source={libs},target={C_LIBS},readonly",
        "--mount", f"type=bind,source={database_dir},target={pi_driver.C_DB}",
        "--entrypoint", pi_driver.C_TIMEOUT, image, f"--kill-after={pi_driver.KILL_AFTER_SECONDS}", str(timeout),
        pi_driver.C_VENV_PYTHON, "-c", WRITER, str(index), str(writers), str(records),
    ]


def _spawn(command: list, stderr_file) -> subprocess.Popen:
    try:
        return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr_file, stdin=subprocess.DEVNULL)
    except OSError as exc:
        raise CannotRun(f"could not start {command[0]!r}: {exc}") from exc


def read_back(path: Path) -> tuple:
    """`(distinct synthetic records, integrity)` of the database. Only to be called when no container is writing it."""
    with contextlib.closing(sqlite3.connect(path)) as connection:
        integrity = "; ".join(row[0] for row in connection.execute("PRAGMA integrity_check"))
        markers = set()
        for (settings,) in connection.execute("SELECT settings FROM runs"):
            marker = json.loads(settings or "{}").get(MARKER)
            if marker:
                markers.add(marker)
    return len(markers), integrity


def run_trial(
    database_dir: Path, *, writers: int, records: int, timeout: int, docker: str, image: str, spawn: Callable = _spawn,
    remove: Optional[Callable[[str], Optional[str]]] = None, unaccounted: Optional[dict] = None,
) -> TrialResult:
    """One trial in `database_dir`, which must not hold a database yet. `spawn(command, stderr_file)` returns an object with
    `wait(timeout)` and `kill()`; `remove(name)` removes a container, returning None when it is gone, else what to tell the
    user. `unaccounted` is filled with the containers not known to be gone, each with what to tell the user. It is empty when
    every one has ended or been removed, however the trial ends, and while it is not, one may still be writing: the database
    is not read. Raises CannotRun."""
    remove = remove or (lambda name: pi_driver.remove_container(docker, name))
    unaccounted = {} if unaccounted is None else unaccounted

    def remove_and_note(name: str) -> None:
        problem = remove(name)
        if problem:
            unaccounted[name] = problem
        else:
            unaccounted.pop(name, None)

    try:
        # Before any container starts, and closed again at once: the host does not hold it open while they write.
        database = pi_driver.ensure_harness_db(database_dir)
    except pi_driver.SessionError as exc:
        raise CannotRun(str(exc)) from exc
    run_id = uuid.uuid4().hex[:8]
    result = TrialResult(expected=writers * records)
    running: list = []  # (name, process, stderr file)
    # ignore_cleanup_errors: a log Windows still holds must not replace the error (or Ctrl+C) that is on its way out.
    with tempfile.TemporaryDirectory(prefix="claude-runway-db-check-", ignore_cleanup_errors=True) as logs:
        try:
            for index in range(writers):
                name = f"claude-runway-db-check-{run_id}-{index}"
                command = writer_command(docker=docker, image=image, name=name, libs=REPO_ROOT / "libs", database_dir=database_dir,
                                         index=index, writers=writers, records=records, timeout=timeout)
                stderr_file = open(Path(logs) / f"{index}.log", "w+b")
                try:
                    process = spawn(command, stderr_file)
                except BaseException:
                    stderr_file.close()
                    raise
                running.append((name, process, stderr_file))
                # Until its client ends or it is removed. Noted now, so that a cleanup that is itself interrupted leaves it here.
                unaccounted[name] = f"the container {name} was not confirmed gone; check `docker ps -a`"
            for name, process, _stderr in running:
                try:
                    # The container's own `timeout` fires first; this only covers a docker client that does not end.
                    result.exit_codes.append(process.wait(timeout=timeout + pi_driver.DOCKER_COMMAND_TIMEOUT_SECONDS))
                    unaccounted.pop(name, None)
                except subprocess.TimeoutExpired:
                    process.kill()
                    remove_and_note(name)
                    result.exit_codes.append(None)
        except BaseException:
            # Ctrl+C or a writer that could not start: remove every container, so that none is left writing into the folder.
            for name, process, _stderr in running:
                with contextlib.suppress(OSError):
                    process.kill()
                remove_and_note(name)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=10)  # Windows keeps a file open until the process that has it is gone
            raise
        finally:
            for _name, _process, stderr_file in running:
                stderr_file.seek(0)
                result.writer_errors += [line for line in stderr_file.read().decode("utf-8", "replace").splitlines() if line.strip()]
                stderr_file.close()
    if any(code in DOCKER_COULD_NOT_RUN for code in result.exit_codes):
        said = result.writer_errors[0] if result.writer_errors else "no message"
        raise CannotRun(f"docker could not run a writer container: {said}")
    if unaccounted:
        result.problem = ("the database was not read, since a container may still be writing it: "
                          + "; ".join(unaccounted.values()))
        return result
    # Every container has exited, so the host may open the database again.
    try:
        result.saved, result.integrity = read_back(database)
    except (sqlite3.Error, ValueError) as exc:
        result.problem = f"the database cannot be read ({type(exc).__name__}: {exc})"
    return result


def describe(number: int, trials: int, result: TrialResult) -> str:
    exited = sum(1 for code in result.exit_codes if code == 0)
    # With no integrity answer the database was never read, so there is no count to report.
    saved = f"{result.saved} of {result.expected} records saved" if result.integrity else "records not counted"
    text = (f"trial {number}/{trials}: {saved}, integrity {result.integrity or 'unknown'}, "
            f"{exited} of {len(result.exit_codes)} writers exited 0: {'PASS' if result.passed else 'FAIL'}")
    if result.problem:
        text += f"\n  {result.problem}"
    if not result.passed:
        text += "".join(f"\n  writer: {line}" for line in result.writer_errors[:10])
    return text


def _discard(folder: Path) -> None:
    """Deletes a throwaway folder there is no reason to keep, and names it when it cannot."""
    try:
        pi_driver._remove_tree(folder)
    except OSError as exc:
        print(f"warning: could not delete {folder}: {exc}", file=sys.stderr)


def _trial_in_new_folder(parent: Path, **trial) -> tuple:
    """`(folder, result)` of one trial in a new folder under `parent`; `trial` is what `run_trial` takes. A trial that gives no
    result (it could not be made, or was interrupted, Ctrl+C included) takes its folder with it, since nothing was checked in
    it. Unless a container could not be confirmed gone: it may have the folder mounted, so the folder is kept and named."""
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="db-check-", dir=parent))
    unaccounted: dict = {}
    try:
        return folder, run_trial(folder / "db", unaccounted=unaccounted, **trial)
    except BaseException:
        if unaccounted:
            for said in unaccounted.values():
                print(f"warning: {said}", file=sys.stderr)
            print(f"warning: kept {folder}, which a container may still be using", file=sys.stderr)
        else:
            _discard(folder)
        raise


def main(argv: Optional[list] = None, *, spawn: Callable = _spawn) -> int:
    parser = argparse.ArgumentParser(description="Check that containers can record into one shared reviewer database.")
    parser.add_argument("--writers", type=int, default=DEFAULT_WRITERS, help=f"containers writing at once (default {DEFAULT_WRITERS})")
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS, help=f"runs each records (default {DEFAULT_RECORDS})")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help=f"how many times, each on a new database (default {DEFAULT_TRIALS})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="seconds one writer container may take")
    parser.add_argument("--image", default=pi_driver.DEFAULT_IMAGE)
    parser.add_argument("--docker", default="docker", help="the docker executable")
    parser.add_argument("--folder", type=Path, help="where to make the throwaway folders (default harness/live-tests/logs/)")
    args = parser.parse_args(argv)
    if min(args.writers, args.records, args.trials, args.timeout) < 1:
        print("error: --writers, --records, --trials and --timeout must each be 1 or more", file=sys.stderr)
        return 2
    parent = (args.folder or REPO_ROOT.joinpath(*pi_driver.HARNESS_DB_SUBDIR[:-1])).expanduser().resolve()
    print(f"shared database check: {args.writers} containers x {args.records} records, {args.trials} trial(s), in {parent}")
    passed = 0
    for number in range(1, args.trials + 1):
        try:
            folder, result = _trial_in_new_folder(parent, writers=args.writers, records=args.records, timeout=args.timeout,
                                                  docker=args.docker, image=args.image, spawn=spawn)
        except (CannotRun, OSError) as exc:
            print(f"error: the check could not run: {exc}", file=sys.stderr)
            return 2
        print(describe(number, args.trials, result), flush=True)
        if result.passed:
            passed += 1
            _discard(folder)
        else:
            print(f"  kept for inspection: {folder}")
    print(f"result on {sys.platform}: {passed} of {args.trials} trial(s) passed")
    return 0 if passed == args.trials else 1


if __name__ == "__main__":
    sys.exit(main())
