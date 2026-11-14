#!/usr/bin/env python3
"""Tests for libs/reviewer_db_lib.py: the codebase-reviewer's own database of review runs.

Stdlib-only (unittest), no network. Every database, log folder, metrics store and savings ledger is
a temporary one (the environment is pointed at it in setUp), so nothing here touches the real home directory.

    .venv/bin/python -m unittest discover -s tests
"""

import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))

import metrics_lib  # noqa: E402
import reviewer_db_lib as db  # noqa: E402


class Clock:
    """A manual clock, so a test controls every start and end time."""

    def __init__(self, start=1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


class TempEnvTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.db_path = self.dir / "reviewer" / "codebase-reviewer.db"
        patcher = mock.patch.dict(os.environ, {
            "CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB": str(self.db_path),
            "CLAUDE_RUNWAY_METRICS_DB": str(self.dir / "metrics.db"),
            "CLAUDE_RUNWAY_SAVINGS_DB": str(self.dir / "savings.db"),
            "CLAUDE_RUNWAY_TRACK_SAVINGS": "",
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def rows(self, sql, *args):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(sql, args).fetchall()
        finally:
            conn.close()

    def recorder(self, clock=None, **fields):
        fields.setdefault("source", "test")
        fields.setdefault("review_type", "reasoning")
        return db.RunRecorder(clock=clock or Clock(), **fields)


class ResolvePath(unittest.TestCase):
    def test_default_override_and_relative_override(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB", None)
            self.assertEqual(db.resolve_db_path(), Path.home() / ".claude" / "claude-runway" / "codebase-reviewer.db")
            # An absolute path built from the platform's own temp directory: "/somewhere/r.db" has no
            # drive on Windows, so it is not absolute there and resolves against the home directory's drive.
            absolute = Path(tempfile.gettempdir()) / "somewhere" / "r.db"
            os.environ["CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB"] = str(absolute)
            self.assertEqual(db.resolve_db_path(), absolute)
            os.environ["CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB"] = "rel/r.db"
            self.assertEqual(db.resolve_db_path(), Path.home() / "rel" / "r.db")

    def test_the_log_folder_is_next_to_the_database(self):
        self.assertEqual(db.log_dir(Path("/x/y/r.db")), Path("/x/y") / db.LOG_DIR_NAME)


class SchemaAndRun(TempEnvTestCase):
    def test_connect_creates_every_table_and_is_idempotent(self):
        db.connect(self.db_path).close()
        db.connect(self.db_path).close()
        names = {r["name"] for r in self.rows("SELECT name FROM sqlite_master WHERE type = 'table'")}
        self.assertTrue({"machines", "models", "reviewer_setups", "runs", "stages", "requests", "model_loads"} <= names)
        self.assertEqual(self.rows("PRAGMA user_version")[0][0], db.SCHEMA_VERSION)
        self.assertEqual(self.rows("PRAGMA journal_mode")[0][0], "wal")

    def test_a_run_with_a_stage_a_request_and_a_load_is_written(self):
        clock = Clock()
        rec = self.recorder(clock, repo="/r", base_ref="main", head_ref="feat", branch="feat", pr=7, settings={"max_reviewers": 2})
        details = {"id": "model-a", "publisher": "pub", "arch": "qwen2", "compatibility_type": "gguf", "quantization": "Q4_K_M",
                   "max_context_length": 32768, "loaded_context_length": 8192}
        rec.model_loaded("model-a", clock.t, clock.advance(5), True, None, details)
        with rec.stage("reasoning", models=("model-a",)):
            with rec.request(reviewer="quoting", role="reasoning", model="model-a", subject=("chunk", 1), prompt_chars=900,
                             reply_format="lines", translator=None, prompt_hash="abc") as req:
                clock.advance(2.5)
                req.finish("a reply", usage={"prompt_tokens": 300, "completion_tokens": 40, "reasoning_tokens": 12})
        rec.model_unloaded("model-a", clock.t, clock.advance(1), True, None)
        rec.set_counts(chunks=1, findings=3)
        summary = rec.close("finished")
        assert summary is not None
        self.assertEqual(summary["combination"], "reasoning:model-a")
        run = self.rows("SELECT * FROM runs")[0]
        self.assertEqual((run["source"], run["review_type"], run["outcome"], run["pr"], run["branch"]), ("test", "reasoning", "finished", 7, "feat"))
        self.assertEqual((run["chunks"], run["findings"], run["wall_s"]), (1, 3, 8.5))
        self.assertEqual(json.loads(run["settings"]), {"max_reviewers": 2})
        self.assertIsNotNone(run["machine_id"])
        stage = self.rows("SELECT * FROM stages")[0]
        self.assertEqual((stage["stage"], stage["wall_s"], json.loads(stage["models"])), ("reasoning", 2.5, ["model-a"]))
        request = self.rows("SELECT * FROM requests")[0]
        self.assertEqual((request["outcome"], request["reply_chars"], request["prompt_chars"]), ("answered", 7, 900))
        self.assertEqual((request["prompt_tokens"], request["reply_tokens"], request["reasoning_tokens"]), (300, 40, 12))
        self.assertEqual((request["subject_kind"], request["subject_id"], request["role"], request["stage"]), ("chunk", "1", "reasoning", "reasoning"))
        loads = self.rows("SELECT * FROM model_loads ORDER BY id")
        self.assertEqual([(l["action"], l["wall_s"], l["ok"]) for l in loads], [("load", 5.0, 1), ("unload", 1.0, 1)])
        model = self.rows("SELECT * FROM models")[0]
        self.assertEqual((model["model_id"], model["quantization"], model["format"], model["max_context_length"]), ("model-a", "Q4_K_M", "gguf", 32768))
        setup = self.rows("SELECT * FROM reviewer_setups")[0]
        self.assertEqual((setup["model"], setup["reply_format"], setup["prompt_hash"]), ("model-a", "lines", "abc"))
        self.assertEqual(request["setup_id"], setup["id"])

    def test_latency_and_the_servers_timing_are_stored_in_milliseconds(self):
        clock = Clock()
        rec = self.recorder(clock)
        with rec.request(reviewer="r", role="reasoning", model="m") as req:
            clock.advance(2.5)
            req.finish("reply", usage={"prompt_tokens": 80, "completion_tokens": 40, "reasoning_tokens": 0, "ttft_ms": 148,
                                       "inter_token_ms": 20.23, "tokens_per_second": 49.43, "generation_ms": 1544,
                                       "stop_reason": "eosFound"})
        with rec.request(reviewer="r", role="reasoning", model="m") as req:
            clock.advance(0.0004)
            req.finish("quick")  # no timing from the server: the latency is still measured here
        rec.close()
        first, second = self.rows("SELECT * FROM requests ORDER BY started_at, rowid")
        self.assertEqual((first["latency_ms"], first["ttft_ms"], first["inter_token_ms"], first["tokens_per_second"]), (2500, 148, 20.23, 49.43))
        self.assertEqual((first["generation_ms"], first["stop_reason"], first["prompt_tokens"], first["reply_tokens"]), (1544, "eosFound", 80, 40))
        self.assertEqual((second["latency_ms"], second["ttft_ms"], second["inter_token_ms"], second["stop_reason"]), (0, None, None, None))

    def test_a_failed_request_still_has_its_latency(self):
        clock = Clock()
        rec = self.recorder(clock)
        with rec.request(reviewer="r", role="reasoning", model="m") as req:
            clock.advance(60.0)
            req.finish(None, failure_kind="timeout")
        rec.close()
        row = self.rows("SELECT latency_ms, outcome, ttft_ms FROM requests")[0]
        self.assertEqual(tuple(row), (60000, "timed_out", None))

    def test_a_database_from_before_the_timing_columns_gets_them_and_keeps_its_rows(self):
        self.db_path.parent.mkdir(parents=True)
        old = sqlite3.connect(str(self.db_path))
        old.executescript("""
            CREATE TABLE requests (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, stage TEXT, setup_id INTEGER, reviewer TEXT,
                model TEXT, role TEXT, subject_kind TEXT, subject_id TEXT, started_at REAL NOT NULL, ended_at REAL NOT NULL,
                wall_s REAL NOT NULL, prompt_chars INTEGER, reply_chars INTEGER, prompt_tokens INTEGER, reply_tokens INTEGER,
                reasoning_tokens INTEGER, outcome TEXT NOT NULL, problems INTEGER NOT NULL DEFAULT 0, log_ref TEXT);
            INSERT INTO requests (id, run_id, started_at, ended_at, wall_s, outcome) VALUES ('old', 'r0', 0, 1, 1, 'answered');
            PRAGMA user_version=1;
        """)
        old.commit()
        old.close()
        rec = self.recorder()
        with rec.request(reviewer="r", role="reasoning", model="m") as req:
            req.finish("x", usage={"ttft_ms": 99})
        rec.close()
        rows = {r["id"]: r for r in self.rows("SELECT * FROM requests")}
        self.assertEqual(len(rows), 2)  # the old row is kept, and the new one was written
        self.assertIsNone(rows["old"]["ttft_ms"])
        self.assertEqual(rows["old"]["latency_ms"], 1000)  # filled in from the seconds the old row already had
        self.assertEqual([r["ttft_ms"] for r in rows.values() if r["id"] != "old"], [99])
        self.assertEqual(self.rows("PRAGMA user_version")[0][0], db.SCHEMA_VERSION)

    def test_the_columns_of_an_older_file_are_added_under_the_write_lock(self):
        # Two sessions opening an older file at once must not both see a column missing: the check and the ALTER
        # happen inside a write transaction, so another session cannot write (or migrate) until it commits.
        conn = db.connect(self.db_path)
        conn.execute("PRAGMA user_version=3")  # an older file: the next connect migrates it
        conn.commit()
        conn.close()
        seen = []
        real = db._add_missing_columns

        def checked(conn):
            seen.append(conn.in_transaction)
            other = sqlite3.connect(str(self.db_path), timeout=0)
            try:
                with self.assertRaises(sqlite3.OperationalError) as caught:
                    other.execute("BEGIN IMMEDIATE")
                seen.append(str(caught.exception))
            finally:
                other.close()
            real(conn)

        with mock.patch.object(db, "_add_missing_columns", checked):
            db.connect(self.db_path).close()
        self.assertEqual(seen, [True, "database is locked"])

    def test_a_current_file_is_opened_without_waiting_for_the_write_lock(self):
        # Only the run's write takes the write lock, so a locked database costs a run one busy timeout, not two.
        db.connect(self.db_path).close()
        holder = sqlite3.connect(str(self.db_path))
        holder.execute("BEGIN IMMEDIATE")
        try:
            started = time.monotonic()
            db.connect(self.db_path, timeout=5).close()
            self.assertLess(time.monotonic() - started, 1.0)
        finally:
            holder.rollback()
            holder.close()

    def test_each_load_keeps_the_context_length_it_was_loaded_with(self):
        details = {"id": "m", "compatibility_type": "gguf", "quantization": "Q4", "max_context_length": 131072}
        for context in (8192, 32768):
            rec = self.recorder()
            rec.model_loaded("m", 0.0, 1.0, True, None, {**details, "loaded_context_length": context})
            rec.close()
        loads = self.rows("SELECT loaded_context_length, model_row_id FROM model_loads ORDER BY id")
        self.assertEqual([l["loaded_context_length"] for l in loads], [8192, 32768])
        self.assertEqual(len({l["model_row_id"] for l in loads}), 1)  # still one row for the model itself

    def test_a_stage_in_which_no_reviewer_answered_makes_the_run_incomplete(self):
        clock = Clock()
        rec = self.recorder(clock)
        with rec.stage("reasoning", models=("m",)):
            rec.model_loaded("m", clock.t, clock.advance(1), False, "out of memory", None)
        rec.close("finished")
        run = self.rows("SELECT outcome, error FROM runs")[0]
        self.assertEqual((run["outcome"], run["error"]), ("incomplete", "no reviewer answered in: reasoning"))

    def test_a_run_is_finished_when_each_stage_that_tried_got_an_answer(self):
        clock = Clock()
        rec = self.recorder(clock)
        with rec.stage("reasoning", models=("m",)):
            with rec.request(reviewer="a", role="reasoning", model="m") as req:
                req.finish(None, failure_kind="timeout")
            with rec.request(reviewer="b", role="reasoning", model="m") as req:
                req.finish("x")
        with rec.stage("verifier", models=("v",)):
            pass  # no finding to verify: nothing was tried, nothing went wrong
        rec.close("finished")
        self.assertEqual(self.rows("SELECT outcome FROM runs")[0]["outcome"], "finished")

    def test_a_setup_is_one_row_however_many_requests_use_it(self):
        rec = self.recorder()
        for n in range(3):
            with rec.request(reviewer="r", role="reasoning", model="m", subject=("chunk", n), reply_format="json") as req:
                req.finish("x")
        with rec.request(reviewer="r", role="reasoning", model="m", subject=("chunk", 9), reply_format="lines") as req:
            req.finish("x")
        rec.close()
        self.assertEqual(len(self.rows("SELECT * FROM reviewer_setups")), 2)
        self.assertEqual(len(self.rows("SELECT * FROM requests")), 4)

    def test_outcomes_answered_timed_out_refused_and_failed(self):
        rec = self.recorder()
        for kind, reply in (("x", "text"), ("timeout", None), ("bad_request", None), ("other", None), (None, None)):
            with rec.request(reviewer="r", role="reasoning", model="m") as req:
                req.finish(reply, failure_kind=kind)
        rec.close()
        outcomes = [r["outcome"] for r in self.rows("SELECT outcome FROM requests ORDER BY rowid")]
        self.assertEqual(outcomes, ["answered", "timed_out", "refused", "failed", "failed"])
        run = self.rows("SELECT timeouts, failures FROM runs")[0]
        self.assertEqual((run["timeouts"], run["failures"]), (1, 3))

    def test_a_request_that_raises_is_recorded_as_failed(self):
        rec = self.recorder()
        with self.assertRaises(RuntimeError):
            with rec.request(reviewer="r", role="reasoning", model="m"):
                raise RuntimeError("boom")
        rec.close()
        self.assertEqual(self.rows("SELECT outcome FROM requests")[0]["outcome"], "failed")

    def test_no_usage_leaves_the_token_columns_empty(self):
        rec = self.recorder()
        with rec.request(reviewer="r", role="reasoning", model="m") as req:
            req.finish("x")
        rec.close()
        row = self.rows("SELECT prompt_tokens, reply_tokens, reasoning_tokens FROM requests")[0]
        self.assertEqual(tuple(row), (None, None, None))

    def test_a_run_that_ends_in_an_error_is_still_written(self):
        with self.assertRaises(ValueError):
            with self.recorder() as rec:
                with rec.request(reviewer="r", role="reasoning", model="m") as req:
                    req.finish("x")
                raise ValueError("part-way failure")
        run = self.rows("SELECT outcome, error FROM runs")[0]
        self.assertEqual(run["outcome"], "errored")
        self.assertIn("part-way failure", run["error"])
        self.assertEqual(len(self.rows("SELECT * FROM requests")), 1)

    def test_close_is_idempotent(self):
        rec = self.recorder()
        self.assertIsNotNone(rec.close("finished"))
        self.assertIsNone(rec.close("errored"))
        self.assertEqual(len(self.rows("SELECT * FROM runs")), 1)

    def test_a_model_already_loaded_by_the_user_is_kept_without_a_load_row(self):
        rec = self.recorder()
        rec.model_seen({"id": "mine", "compatibility_type": "mlx", "quantization": "4bit", "max_context_length": 4096})
        rec.close()
        self.assertEqual(self.rows("SELECT model_id FROM models")[0]["model_id"], "mine")
        self.assertEqual(self.rows("SELECT * FROM model_loads"), [])

    def test_the_combination_lists_stages_in_run_order_and_models_sorted(self):
        clock = Clock()
        rec = self.recorder(clock)
        with rec.stage("reasoning", models=("b", "a")):
            clock.advance(1)
        with rec.stage("verifier", models=("c",)):
            clock.advance(1)
        self.assertEqual(rec.combination(), "reasoning:a+b|verifier:c")

    def test_a_stage_that_is_not_built_yet_records_through_the_same_calls(self):
        clock = Clock()
        rec = self.recorder(clock)
        with rec.stage("classifier", models=("judge",)):
            with rec.request(reviewer="c", role="classifier", model="judge", subject=("finding+rubric", "7:data-loss")) as req:
                clock.advance(3)
                req.finish("{}")
        rec.close()
        self.assertEqual(self.rows("SELECT combination FROM runs")[0]["combination"], "classifier:judge")
        row = self.rows("SELECT stage, role, subject_kind, subject_id FROM requests")[0]
        self.assertEqual(tuple(row), ("classifier", "classifier", "finding+rubric", "7:data-loss"))

    def test_several_runs_accumulate_and_share_the_machine_row(self):
        for _ in range(3):
            self.recorder().close()
        self.assertEqual(len(self.rows("SELECT * FROM runs")), 3)
        self.assertEqual(len(self.rows("SELECT * FROM machines")), 1)

    def test_the_machine_fingerprint_separates_machines(self):
        a = {"chip": "M1", "memory_bytes": 16, "os": "Darwin", "arch": "arm64", "lmstudio_version": None}
        b = {**a, "memory_bytes": 64}
        self.assertNotEqual(db.machine_fingerprint(a), db.machine_fingerprint(b))
        self.assertEqual(db.machine_fingerprint(a), db.machine_fingerprint(dict(a)))

    def test_the_lmstudio_build_is_read_from_the_banner_without_its_colours(self):
        banner = ("\x1b[38;5;166m   __   __  ___  ______          ___        _______   ____\x1b[0m\n"
                  "\x1b[38;5;214m  / /  /  |/  / / __/ /___ _____/ (_)__    / ___/ /  /  _/\x1b[0m\n\n"
                  "lms is LM Studio's CLI utility for your models, server, and inference runtime.\n"
                  "CLI commit: dcefd5a\n\nDocs: https://lmstudio.ai/docs/developer\n")
        self.assertEqual(db.parse_lmstudio_version(banner), "lms CLI commit dcefd5a")
        self.assertEqual(db.parse_lmstudio_version("lms 0.3.31+2\nCLI commit: abc123"), "0.3.31+2")
        self.assertIsNone(db.parse_lmstudio_version("some other output\n"))

    def test_a_remote_lm_studio_is_its_own_machine_and_a_local_one_is_this_machine(self):
        for url in (None, "", "http://localhost:1234/v1", "http://127.0.0.1:1234/v1", "http://[::1]:1234/v1"):
            with self.subTest(url=url):
                self.assertIsNone(db.lmstudio_host(url))
        self.assertEqual(db.lmstudio_host("http://GPU-Box.local:1234/v1"), "gpu-box.local")
        remote = db.machine_for("http://gpu-box.local:1234/v1")
        self.assertEqual((remote["host"], remote["chip"], remote["memory_bytes"]), ("gpu-box.local", None, None))
        # This machine's fingerprint is unchanged by the host column, so rows recorded before it still match.
        local = db.machine_for("http://localhost:1234/v1")
        self.assertNotIn("host", local)
        self.assertNotEqual(db.machine_fingerprint(remote), db.machine_fingerprint(local))

    def test_a_run_against_a_remote_lm_studio_is_filed_under_that_host(self):
        rec = self.recorder(lmstudio_url="http://gpu-box.local:1234/v1")
        rec.close()
        self.recorder(lmstudio_url="http://localhost:1234/v1").close()
        machines = self.rows("SELECT host, chip FROM machines ORDER BY id")
        self.assertEqual(sorted(str(m["host"]) for m in machines), ["None", "gpu-box.local"])
        remote = next(m for m in machines if m["host"])
        self.assertIsNone(remote["chip"])  # this machine cannot know the remote one's hardware
        self.assertEqual(len({r["machine_id"] for r in self.rows("SELECT machine_id FROM runs")}), 2)

    def test_machine_info_never_raises_and_has_every_field(self):
        info = db.machine_info()
        self.assertEqual(set(info), {"chip", "memory_bytes", "os", "arch", "lmstudio_version"})


class ProblemSummary(TempEnvTestCase):
    def test_a_summary_names_each_distinct_problem_once_with_its_count(self):
        summary = db.summarize_problems([
            "reviewer 'quoting': the reply was not JSON",
            "reviewer 'quoting': chunk 2/5: the reply was not JSON",
            "chunk 3/5: the translator exited with status 1: boom",
            "reviewer 'other': the reply was not JSON",
        ])
        self.assertEqual(summary, "the reply was not JSON (x3); the translator exited with status 1: boom")

    def test_the_reviewer_and_the_chunk_or_finding_number_are_left_out(self):
        self.assertEqual(db.summarize_problems(["reviewer 'v': finding 4/9: LM Studio request failed (timeout)"]),
                         "LM Studio request failed (timeout)")

    def test_no_problem_gives_none_and_blank_lines_are_ignored(self):
        self.assertIsNone(db.summarize_problems([]))
        self.assertIsNone(db.summarize_problems(["", "   ", "reviewer 'x': "]))

    def test_a_long_summary_is_cut_with_an_ellipsis(self):
        summary = db.summarize_problems([f"problem number {n} " + "x" * 80 for n in range(30)])
        assert summary is not None
        self.assertEqual(len(summary), db.MAX_SUMMARY_CHARS)
        self.assertTrue(summary.endswith("\u2026"))

    def test_the_summary_is_stored_in_the_request_row_and_outlives_the_log_files(self):
        rec = self.recorder(session_id="s1")
        with rec.request(reviewer="r", role="reasoning", model="m") as req:
            req.finish("x")
            req.problem("reviewer 'r': the reply was not JSON")
            req.problem("reviewer 'r': the reply was not JSON")
        with rec.request(reviewer="r", role="reasoning", model="m") as req:
            req.finish("fine")
        rec.close()
        for log in db.log_dir(self.db_path).glob("*.log"):
            log.unlink()  # the files are pruned...
        rows = self.rows("SELECT problems, problem_summary FROM requests ORDER BY started_at, rowid")
        self.assertEqual([tuple(r) for r in rows], [(2, "the reply was not JSON (x2)"), (0, None)])  # ...the row keeps it

    def test_a_database_from_before_the_summary_column_gets_it(self):
        conn = db.connect(self.db_path)
        conn.execute("ALTER TABLE requests DROP COLUMN problem_summary")  # what a version 2 file looked like
        conn.execute("PRAGMA user_version=2")
        conn.commit()
        conn.close()
        self.assertNotIn("problem_summary", {r["name"] for r in self.rows("PRAGMA table_info(requests)")})
        db.connect(self.db_path).close()
        self.assertIn("problem_summary", {r["name"] for r in self.rows("PRAGMA table_info(requests)")})
        self.assertEqual(self.rows("PRAGMA user_version")[0][0], db.SCHEMA_VERSION)


class Logs(TempEnvTestCase):
    """The session log: one readable file per session, every review a section in it, every stage writing the same way."""

    def log_text(self, session_id):
        return (db.log_dir(self.db_path) / db.log_file_name(session_id)).read_text(encoding="utf-8")

    def test_every_run_writes_a_section_to_the_file_named_after_its_session(self):
        clock = Clock()
        rec = self.recorder(clock, session_id="sess-abc", repo="/r", base_ref="main", head_ref="feat", branch="feat", pr=7,
                            settings={"max_reviewers": 2})
        with rec.stage("reasoning", models=("model-a",)):
            with rec.request(reviewer="quoting", role="reasoning", model="model-a", subject=("chunk", 2)) as req:
                clock.advance(12.7)
                req.finish("x", usage={"prompt_tokens": 629, "completion_tokens": 554, "reasoning_tokens": 473, "ttft_ms": 1062,
                                       "inter_token_ms": 21.4, "stop_reason": "eosFound"})
                req.problem("reviewer 'quoting': the reply was not JSON")
        rec.close("finished")
        files = list(db.log_dir(self.db_path).glob("*.log"))
        self.assertEqual([f.name for f in files], ["sess-abc.log"])
        text = self.log_text("sess-abc")
        self.assertIn(f"=== run {rec.run_id} |", text)
        self.assertIn("reasoning review via test ===", text)
        self.assertIn("repo ", text)
        self.assertIn("range main...feat  branch feat  pr 7", text)
        self.assertIn("settings: max_reviewers=2", text)
        request = self.rows("SELECT id FROM requests")[0]["id"]
        self.assertIn("reasoning | quoting (model-a) | chunk 2 | answered in 12.7s, first token 1062 ms, 21.4 ms/token, "
                      f"tokens 629 in / 554 out (473 thinking)  [request {request[:8]}]", text)
        self.assertNotIn("stopped:", text)  # a normal end is not worth a mention
        self.assertIn("problem: reviewer 'quoting': the reply was not JSON", text)
        self.assertRegex(text, r"=== finished in 12\.7s \| reasoning:model-a \| 1 request\(s\): 0 timed out, 0 failed ===\n")
        # The row has what the file has about the problem, so the file is not the only record.
        self.assertEqual(self.rows("SELECT problem_summary FROM requests")[0]["problem_summary"], "the reply was not JSON")

    def test_the_log_shows_what_each_stage_evaluated_in_time_order(self):
        clock = Clock()
        rec = self.recorder(clock, session_id="s", review_type="verifier")
        rec.model_loaded("model-a", clock.t, clock.advance(5), True, None, None)
        with rec.stage("reasoning", models=("model-a",)):
            with rec.request(reviewer="quoting", role="reasoning", model="model-a", subject=("chunk", 1)) as req:
                clock.advance(3)
                req.finish("x")
                req.note("raised 1 finding(s)", detail=("a.py:10-12 [injection] name goes into the shell", "  failing input: ; rm -rf", "  quote: run(name)"))
        with rec.stage("grounding"):
            rec.note("grounding", "1 of 2 finding(s) kept; 1 dropped as ungrounded")
            rec.note("grounding", "dropped: its quote is not in the file", detail=("b.py:3 [output] prints a constant",))
        with rec.stage("verifier", models=("model-v",)):
            with rec.request(reviewer="v", role="verifier", model="model-v", subject=("finding", 1)) as req:
                clock.advance(10)
                req.finish("x")
                req.note("rejected: the name is validated before it is used", detail=("a.py:10-12 [injection] name goes into the shell",))
        rec.set_result_text("Verified review: 0 confirmed.\nnothing to report")
        rec.close("finished")
        text = self.log_text("s")
        order = ["model | model-a | loaded in 5.0s", "reasoning | quoting (model-a) | chunk 1", "raised 1 finding(s)",
                 "a.py:10-12 [injection] name goes into the shell", "failing input: ; rm -rf", "quote: run(name)",
                 "grounding | 1 of 2 finding(s) kept", "dropped: its quote is not in the file", "b.py:3 [output]",
                 "verifier | v (model-v) | finding 1", "rejected: the name is validated before it is used",
                 "--- result returned to the caller ---", "Verified review: 0 confirmed.", "nothing to report", "--- end of result ---",
                 "=== finished in 18.0s | reasoning:model-a|verifier:model-v"]
        positions = [text.index(piece) for piece in order]
        self.assertEqual(positions, sorted(positions), "the log is not in the order things happened")
        self.assertIn("[    5.0s] reasoning |", text)  # time offsets are from the start of the run

    def test_a_stage_that_is_not_built_yet_writes_to_the_same_log(self):
        clock = Clock()
        rec = self.recorder(clock, session_id="s")
        with rec.stage("classifier", models=("judge",)):
            with rec.request(reviewer="c", role="classifier", model="judge", subject=("finding+rubric", "7:data-loss")) as req:
                clock.advance(2)
                req.finish("x")
                req.note("matched rubric entry data-loss, confidence 4")
            rec.note("classifier", "1 finding(s) classified, 0 unclassified")
        rec.close("finished")
        text = self.log_text("s")
        self.assertIn("classifier | c (judge) | finding+rubric 7:data-loss | answered", text)
        self.assertIn("matched rubric entry data-loss, confidence 4", text)
        self.assertIn("classifier | 1 finding(s) classified, 0 unclassified", text)

    def test_a_failed_run_says_so_in_its_footer_and_a_stopped_reply_is_flagged(self):
        clock = Clock()
        rec = self.recorder(clock, session_id="s")
        with rec.request(reviewer="r", role="reasoning", model="m", subject=("chunk", 1)) as req:
            clock.advance(60)
            req.finish(None, failure_kind="timeout")
        with rec.request(reviewer="r", role="reasoning", model="m", subject=("chunk", 2)) as req:
            clock.advance(1)
            req.finish("cut off", usage={"stop_reason": "maxPredictedTokensReached"})
        rec.close("errored", error="RuntimeError: boom")
        text = self.log_text("s")
        self.assertIn("| timed_out in 60.0s", text)
        self.assertIn("stopped: maxPredictedTokensReached", text)
        self.assertIn("=== errored in 61.0s |", text)
        self.assertIn("1 timed out", text)
        self.assertIn("error: RuntimeError: boom", text)

    def test_several_runs_of_one_session_share_one_file_in_order(self):
        first = self.recorder(session_id="sess-1")
        first.close("finished")
        second = self.recorder(session_id="sess-1")
        second.close("finished")
        files = list(db.log_dir(self.db_path).glob("*.log"))
        self.assertEqual([f.name for f in files], ["sess-1.log"])
        text = self.log_text("sess-1")
        self.assertLess(text.index(first.run_id), text.index(second.run_id))
        self.assertEqual(text.count("=== run "), 2)

    def test_different_sessions_never_write_the_same_file(self):
        self.recorder(session_id="sess-a").close()
        self.recorder(session_id="sess-b").close()
        self.assertEqual(sorted(f.name for f in db.log_dir(self.db_path).glob("*.log")), ["sess-a.log", "sess-b.log"])

    def test_a_run_with_nothing_wrong_still_writes_its_section(self):
        self.recorder(session_id="quiet").close()
        self.assertIn("=== finished in", self.log_text("quiet"))

    def test_a_session_id_is_made_safe_for_a_file_name(self):
        self.assertEqual(db.log_file_name("a/b\\..:c"), "a_b____c.log")
        self.assertEqual(db.log_file_name("x" * 500), "x" * 100 + ".log")
        for blank in (None, ""):
            self.assertEqual(db.log_file_name(blank), f"no-session-{os.getpid()}.log")

    def test_concurrent_runs_in_one_session_keep_their_sections_whole(self):
        import threading

        recorders = []
        for n in range(8):
            rec = self.recorder(session_id="busy")
            with rec.request(reviewer="r", role="reasoning", model="m") as req:
                req.finish("x")
                for k in range(20):
                    req.problem(f"run {n} problem {k} " + "y" * 200)
            recorders.append(rec)
        threads = [threading.Thread(target=rec.close) for rec in recorders]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        text = self.log_text("busy")
        sections = [part for part in text.split("\n\n") if part.strip()]
        self.assertEqual(len(sections), 8)
        for section in sections:
            self.assertTrue(section.startswith("=== run "), section[:80])
            self.assertIn("=== finished in", section.splitlines()[-1])
            owners = {line.split("run ")[1].split(" ")[0] for line in section.splitlines() if "problem: run " in line}
            self.assertEqual(len(owners), 1, "one run's section holds another run's lines")

    def test_long_text_is_cut_so_one_run_cannot_make_a_huge_file(self):
        rec = self.recorder(session_id="big")
        with rec.request(reviewer="r", role="reasoning", model="m") as req:
            req.finish("x")
            req.problem("p" * 5000)
            req.note("many lines", detail=tuple(f"detail line {n}" for n in range(100)))
        rec.set_result_text("r" * (db.LOG_RESULT_CHARS + 5000))
        rec.close()
        text = self.log_text("big")
        self.assertLess(len(text), db.LOG_RESULT_CHARS + 3000)
        self.assertIn("\u2026", text)
        self.assertIn(f"detail line {db.LOG_DETAIL_LINES - 1}", text)
        self.assertNotIn(f"detail line {db.LOG_DETAIL_LINES}", text)
        self.assertIn("\u2026 (cut)", text)

    def test_old_logs_are_pruned_by_count_and_by_age(self):
        folder = self.dir / "logs"
        folder.mkdir()
        now = time.time()
        for n in range(5):
            path = folder / f"{n}.log"
            path.write_text("x\n", encoding="utf-8")
            os.utime(path, (now - n * 10, now - n * 10))
        old = folder / "old.log"
        old.write_text("x\n", encoding="utf-8")
        os.utime(old, (now - 40 * 86400, now - 40 * 86400))
        (folder / "keep.txt").write_text("not a log", encoding="utf-8")
        removed = db.prune_logs(folder, keep=3, max_age_days=30, now=now)
        self.assertEqual(removed, 3)  # two beyond the newest three, and the 40-day-old one
        self.assertEqual(sorted(p.name for p in folder.glob("*.log")), ["0.log", "1.log", "2.log"])
        self.assertTrue((folder / "keep.txt").exists())

    def test_a_log_another_server_appends_to_while_pruning_is_kept(self):
        folder = self.dir / "logs"
        folder.mkdir()
        now = time.time()
        for n in range(3):
            path = folder / f"{n}.log"
            path.write_text("x\n", encoding="utf-8")
            os.utime(path, (now - 100 - n, now - 100 - n))
        late = folder / "2.log"  # the oldest: beyond keep=2 when the folder is listed

        def appended_after_listing():
            # prune_logs reads the clock after listing the folder: that is when the other server writes.
            with open(late, "a", encoding="utf-8") as handle:
                handle.write("a new section\n")
            return now

        with mock.patch.object(db.time, "time", side_effect=appended_after_listing):
            removed = db.prune_logs(folder, keep=2, max_age_days=30)
        self.assertEqual(removed, 0)
        self.assertIn("a new section", late.read_text(encoding="utf-8"))

    def test_pruning_a_missing_folder_does_not_raise(self):
        self.assertEqual(db.prune_logs(self.dir / "nope"), 0)


class FailOpen(TempEnvTestCase):
    def _quiet(self):
        return contextlib.redirect_stderr(io.StringIO())

    def test_an_unwritable_database_never_raises_and_says_so_on_stderr(self):
        self.db_path.parent.mkdir(parents=True)
        self.db_path.mkdir()  # a directory where the file should be
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rec = self.recorder()
            with rec.request(reviewer="r", role="reasoning", model="m") as req:
                req.finish("x")
            self.assertIsNotNone(rec.close())  # no exception, and the summary still comes back
        self.assertIn("could not write", err.getvalue())

    def test_a_connection_that_raises_never_stops_the_run(self):
        rec = self.recorder()
        with mock.patch.object(db, "connect", side_effect=sqlite3.OperationalError("disk I/O error")), self._quiet():
            self.assertIsNotNone(rec.close())

    def test_a_locked_database_costs_at_most_the_busy_timeout(self):
        db.connect(self.db_path).close()
        blocker = sqlite3.connect(str(self.db_path), isolation_level=None)
        blocker.execute("BEGIN EXCLUSIVE")
        self.addCleanup(blocker.close)
        rec = self.recorder()
        started = time.monotonic()
        with mock.patch.object(db, "BUSY_TIMEOUT_SECONDS", 0.2), self._quiet():
            rec.close()
        self.assertLess(time.monotonic() - started, 3.0)
        blocker.execute("ROLLBACK")
        self.assertEqual(self.rows("SELECT * FROM runs"), [])  # nothing was written, and nothing raised

    def test_a_failing_metrics_write_never_raises(self):
        with mock.patch.object(metrics_lib.MetricsStore, "record", side_effect=sqlite3.OperationalError("locked")), self._quiet():
            self.assertIsNotNone(self.recorder().close())
        self.assertEqual(len(self.rows("SELECT * FROM runs")), 1)  # the full record is still written

    def test_a_broken_request_helper_never_raises_into_the_review(self):
        rec = self.recorder()
        with mock.patch.object(rec, "add_request", side_effect=RuntimeError("buffer gone")), self._quiet():
            with rec.request(reviewer="r", role="reasoning", model="m") as req:
                req.finish("x")
                req.problem("p")

    def test_the_null_recorder_accepts_every_call_and_records_nothing(self):
        rec = db.NULL_RECORDER
        with rec.stage("reasoning", models=("m",)):
            with rec.request(reviewer="r", role="reasoning", model="m") as req:
                req.finish("x", usage={"prompt_tokens": 1})
                req.problem("p")
        rec.model_loaded("m", 0.0, 1.0, True, None, None)
        rec.model_unloaded("m", 0.0, 1.0, True, None)
        rec.model_seen(None)
        rec.set_counts(chunks=1)
        rec.set_sizes(1, 2)
        self.assertFalse(self.db_path.exists())


class SummaryEventAndSavings(TempEnvTestCase):
    def _run(self, **sizes):
        clock = Clock()
        rec = self.recorder(clock, session_id="sess-1", repo="/r", branch="feat", pr=12)
        with rec.stage("reasoning", models=("m",)):
            with rec.request(reviewer="r", role="reasoning", model="m") as req:
                clock.advance(4)
                req.finish("x")
        with rec.stage("verifier", models=("v",)):
            with rec.request(reviewer="v", role="verifier", model="v") as req:
                req.finish("x")
            with rec.request(reviewer="v", role="verifier", model="v") as req:
                clock.advance(6)
                req.finish(None, failure_kind="timeout")
        rec.set_counts(chunks=2, findings=5, confirmed=1)
        if sizes:
            rec.set_sizes(**sizes)
        rec.close("finished")

    def test_one_summary_event_lands_in_the_shared_metrics_store(self):
        self._run()
        store = metrics_lib.MetricsStore()
        events = store.detail("codebase-reviewer", limit=10)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual((event["event_type"], event["value"], event["session_id"]), ("review", 10.0, "sess-1"))
        meta = event["metadata"]
        self.assertEqual(meta["stage_s"], {"reasoning": 4.0, "verifier": 6.0})
        self.assertEqual((meta["findings"], meta["confirmed"], meta["timeouts"], meta["failures"]), (5, 1, 1, 0))
        self.assertEqual((meta["pr"], meta["branch"], meta["outcome"], meta["combination"]), (12, "feat", "finished", "reasoning:m|verifier:v"))
        self.assertEqual(meta["wall_s"], 10.0)
        # Per-request detail stays local: it is not copied to the shared store.
        self.assertNotIn("requests_detail", meta)

    def _ledger_files(self):
        return list((self.dir / "sessions").glob("*.jsonl")) if (self.dir / "sessions").exists() else []

    def test_savings_are_recorded_only_when_tracking_is_on(self):
        import savings_ledger

        with mock.patch.object(savings_ledger, "current_session_id", return_value="live-session"):
            self._run(diff_chars=9000, result_chars=700)
            self.assertEqual(self._ledger_files(), [])  # tracking is off
            with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"}):
                self._run(diff_chars=9000, result_chars=700)
        files = self._ledger_files()
        self.assertEqual(len(files), 1)
        entry = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual((entry["tool"], entry["raw_chars"], entry["out_chars"]), ("codebase-reviewer", 9000, 700))

    def test_savings_go_to_the_live_session_of_the_calling_project_only(self):
        import savings_ledger

        asked = []

        def lookup(project=None):
            asked.append(project)
            return "live-session" if project == "calling-project" else "another-projects-session"

        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"}), \
                mock.patch.object(savings_ledger, "current_session_id", side_effect=lookup), \
                mock.patch.object(db.os, "getcwd", return_value="/work/calling-project"):
            self._run(diff_chars=9000, result_chars=700)
        self.assertEqual(asked, ["calling-project"])
        files = self._ledger_files()
        self.assertEqual([f.stem for f in files], ["live-session"])
        entry = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(entry["project"], "calling-project")

    def test_savings_are_skipped_when_a_size_is_unknown_or_no_session_is_found(self):
        import savings_ledger

        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"}):
            with mock.patch.object(savings_ledger, "current_session_id", return_value="s"):
                self._run(diff_chars=100)  # no result size
            with mock.patch.object(savings_ledger, "current_session_id", return_value=None):
                self._run(diff_chars=100, result_chars=10)
        self.assertEqual(self._ledger_files(), [])

    def test_a_failing_savings_write_never_raises(self):
        import savings_ledger

        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"}), mock.patch.object(
            savings_ledger, "current_session_id", return_value="s"
        ), mock.patch.object(savings_ledger, "record_event", side_effect=OSError("disk full")), contextlib.redirect_stderr(io.StringIO()):
            self._run(diff_chars=100, result_chars=10)
        self.assertEqual(len(self.rows("SELECT * FROM runs")), 1)


class Labels(unittest.TestCase):
    def test_new_run_recorder_labels_the_run(self):
        with mock.patch.object(db, "branch_of", return_value="feat"):
            rec = db.new_run_recorder(source="mcp", review_type="verifier", repo_path=".", base_ref="main", head_ref="HEAD", pr=3,
                                      settings={"x": 1})
        self.assertEqual((rec.source, rec.review_type, rec.branch, rec.pr, rec.base_ref, rec.head_ref), ("mcp", "verifier", "feat", 3, "main", "HEAD"))
        self.assertTrue(os.path.isabs(rec.repo or ""))
        self.assertTrue(rec.session_id)

    def test_branch_of_is_none_outside_a_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(db.branch_of(tmp, "HEAD"))

    def test_hash_text_is_stable_and_short(self):
        self.assertEqual(db.hash_text("a prompt"), db.hash_text("a prompt"))
        self.assertNotEqual(db.hash_text("a prompt"), db.hash_text("another"))
        self.assertEqual(len(db.hash_text("x")), 16)


if __name__ == "__main__":
    unittest.main()
