#!/usr/bin/env python3
"""Tests for libs/reviewer_report_lib.py and tools/codebase_reviewer_report.py.

Stdlib-only (unittest), no network. The database is filled through `RunRecorder` with a manual clock, so every
time in it is known, and nothing touches the real home directory.

    .venv/bin/python -m unittest discover -s tests
"""

import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import reviewer_db_lib as db  # noqa: E402
import codebase_reviewer_report as cli  # noqa: E402
import reviewer_report_lib as rep  # noqa: E402

OTHER_MACHINE = {"chip": "Other Chip", "memory_bytes": 8, "os": "Other OS", "arch": "x", "lmstudio_version": "0"}


class Clock:
    def __init__(self, start=1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


class ReportTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.db_path = self.dir / "r.db"
        patcher = mock.patch.dict(os.environ, {
            "CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB": str(self.db_path),
            "CLAUDE_RUNWAY_METRICS_DB": str(self.dir / "metrics.db"),
            "CLAUDE_RUNWAY_TRACK_SAVINGS": "",
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def record(self, stages, *, load=0.0, other=0.0, outcome="finished", machine=None, review_type="verifier"):
        """One run. `stages` is [(stage, model, [(seconds, outcome)...])]; each request is one model request.
        A load of `load` seconds happens inside the first stage; `other` seconds pass outside every stage."""
        clock = Clock()
        rec = db.RunRecorder(source="test", review_type=review_type, clock=clock)
        first = True
        for stage, model, requests in stages:
            with rec.stage(stage, models=(model,)):
                if first and load:
                    rec.model_loaded(model, clock.t, clock.advance(load), True, None, None)
                first = False
                for seconds, result in requests:
                    with rec.request(reviewer="rv", role=stage, model=model) as req:
                        clock.advance(seconds)
                        if result == "answered":
                            req.finish("x")
                        else:
                            req.finish(None, failure_kind={"timed_out": "timeout", "refused": "bad_request"}.get(result))
        clock.advance(other)
        if machine is not None:
            with mock.patch.object(db, "machine_info", return_value=machine):
                rec.close(outcome)
        else:
            rec.close(outcome)
        return rec


class Percentile(unittest.TestCase):
    def test_nearest_rank(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(rep.percentile(values, 0.9), 9)
        self.assertEqual(rep.percentile(values, 1.0), 10)
        self.assertEqual(rep.percentile([5], 0.9), 5)


class ResponseTimes(ReportTestCase):
    def test_median_p90_and_max_per_model_and_role_with_the_failed_counted_beside(self):
        self.record([("reasoning", "model-a", [(s, "answered") for s in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)] + [(60, "timed_out")])])
        self.record([("verifier", "model-b", [(12, "answered"), (14, "answered"), (1, "refused")])])
        conn = rep.open_database()
        rows = {(r["model"], r["role"]): r for r in rep.response_times(conn)}
        a = rows[("model-a", "reasoning")]
        self.assertEqual((a["answered"], a["median_s"], a["p90_s"], a["max_s"], a["timed_out"], a["failed"]), (10, 5.5, 9, 10, 1, 0))
        b = rows[("model-b", "verifier")]
        self.assertEqual((b["answered"], b["median_s"], b["max_s"], b["timed_out"], b["failed"]), (2, 13.0, 14, 0, 1))
        conn.close()

    def test_one_model_can_be_picked(self):
        self.record([("reasoning", "a", [(1, "answered")])])
        self.record([("reasoning", "b", [(2, "answered")])])
        conn = rep.open_database()
        self.assertEqual([r["model"] for r in rep.response_times(conn, model="b")], ["b"])
        conn.close()

    def test_a_model_with_only_failures_has_no_times_but_is_listed(self):
        self.record([("reasoning", "slow", [(60, "timed_out"), (60, "timed_out")])])
        conn = rep.open_database()
        row = rep.response_times(conn)[0]
        self.assertEqual((row["answered"], row["median_s"], row["timed_out"]), (0, None, 2))
        self.assertIn("2 timed out", rep.format_response_times([row]))
        conn.close()

    def test_this_machine_leaves_out_other_machines(self):
        self.record([("reasoning", "a", [(1, "answered")])])
        self.record([("reasoning", "a", [(50, "answered")])], machine=OTHER_MACHINE)
        conn = rep.open_database()
        self.assertEqual(rep.response_times(conn)[0]["answered"], 2)
        here = rep.response_times(conn, this_machine=True)
        self.assertEqual((here[0]["answered"], here[0]["max_s"]), (1, 1))
        conn.close()


class ResponseTimeStats(ReportTestCase):
    """The millisecond timing and token figures beside a model's response times."""

    def _record_requests(self, model, requests, role="verifier"):
        clock = Clock()
        rec = db.RunRecorder(source="test", review_type="verifier", clock=clock)
        with rec.stage(role, models=(model,)):
            for seconds, usage in requests:
                with rec.request(reviewer="rv", role=role, model=model) as req:
                    clock.advance(seconds)
                    req.finish("x", usage=usage)
        rec.close()

    def test_medians_and_p90_of_latency_first_token_time_between_tokens_and_token_counts(self):
        usages = [{"prompt_tokens": 100 * n, "completion_tokens": 10 * n, "ttft_ms": 100 * n, "inter_token_ms": 10.0 + n,
                   "tokens_per_second": 1000 / (10.0 + n)} for n in range(1, 11)]
        self._record_requests("model-a", [(n / 10, usages[n - 1]) for n in range(1, 11)])  # latency 100 ms ... 1000 ms
        conn = rep.open_database()
        row = rep.response_times(conn)[0]
        conn.close()
        self.assertEqual(row["latency_ms"], {"median": 550, "p90": 900})
        self.assertEqual(row["ttft_ms"], {"median": 550, "p90": 900})
        self.assertEqual(row["inter_token_ms"], {"median": 15.5})  # the median of 11 ... 20
        self.assertEqual((row["prompt_tokens"]["median"], row["reply_tokens"]["median"]), (550, 55))
        text = rep.format_response_times([row])
        self.assertIn("latency median 550 ms (p90 900 ms)", text)
        self.assertIn("first token median 550 ms (p90 900 ms)", text)
        self.assertIn("between tokens median 15.5 ms", text)
        self.assertIn("tokens in/out median 550/55", text)

    def test_a_model_whose_server_reported_no_timing_still_shows_latency_and_tokens(self):
        self._record_requests("model-b", [(2.0, {"prompt_tokens": 80, "completion_tokens": 8}), (4.0, {"prompt_tokens": 90, "completion_tokens": 9})])
        conn = rep.open_database()
        row = rep.response_times(conn)[0]
        conn.close()
        self.assertEqual((row["ttft_ms"]["median"], row["inter_token_ms"]["median"]), (None, None))
        text = rep.format_response_times([row])
        self.assertIn("latency median 3000 ms", text)
        self.assertIn("tokens in/out median 85/9", text)  # the medians of (80, 90) and (8, 9); a half rounds up
        self.assertNotIn("first token", text)
        self.assertNotIn("between tokens", text)

    def test_with_no_usage_at_all_the_timing_line_is_the_measured_latency_only(self):
        self._record_requests("model-c", [(1.0, None)])
        conn = rep.open_database()
        row = rep.response_times(conn)[0]
        conn.close()
        self.assertEqual(row["latency_ms"]["median"], 1000)
        self.assertEqual((row["prompt_tokens"]["median"], row["reply_tokens"]["median"]), (None, None))
        self.assertEqual(rep._timing_line(row), "latency median 1000 ms (p90 1000 ms)")

    def test_failed_requests_do_not_count_toward_the_timing_figures(self):
        clock = Clock()
        rec = db.RunRecorder(source="test", review_type="verifier", clock=clock)
        with rec.stage("verifier", models=("m",)):
            with rec.request(reviewer="rv", role="verifier", model="m") as req:
                clock.advance(1.0)
                req.finish("x", usage={"ttft_ms": 200})
            with rec.request(reviewer="rv", role="verifier", model="m") as req:
                clock.advance(60.0)
                req.finish(None, failure_kind="timeout")
        rec.close()
        conn = rep.open_database()
        row = rep.response_times(conn)[0]
        conn.close()
        self.assertEqual((row["answered"], row["timed_out"], row["latency_ms"]["median"], row["ttft_ms"]["median"]), (1, 1, 1000, 200))

    def test_json_carries_the_figures(self):
        self._record_requests("model-a", [(1.0, {"ttft_ms": 150, "inter_token_ms": 20.0, "prompt_tokens": 7, "completion_tokens": 3})])
        data = json.loads(rep.report("response_times", as_json=True))
        self.assertEqual((data[0]["ttft_ms"]["median"], data[0]["inter_token_ms"]["median"], data[0]["latency_ms"]["median"]), (150, 20.0, 1000))


class ReviewTimes(ReportTestCase):
    def test_load_time_is_split_out_of_the_stage_and_the_parts_add_up(self):
        self.record([("reasoning", "a", [(10, "answered")]), ("verifier", "b", [(20, "answered")])], load=5, other=2)
        conn = rep.open_database()
        row = rep.review_times(conn)[0]
        self.assertEqual(row["combination"], "reasoning:a|verifier:b")
        self.assertEqual((row["reviews"], row["median_wall_s"]), (1, 37.0))
        split = row["middle_review"]
        self.assertEqual((split["load_s"], split["stage_s"], split["other_s"]), (5.0, {"reasoning": 10.0, "verifier": 20.0}, 2.0))
        self.assertEqual(split["load_s"] + sum(split["stage_s"].values()) + split["other_s"], split["wall_s"])
        conn.close()

    def test_the_split_is_one_real_review_so_it_adds_up_where_per_part_medians_would_not(self):
        # Each part's median here would be 1 s beside a 102 s review; the middle review's own parts add up.
        for load, reasoning, verifier in ((100, 1, 1), (1, 100, 1), (1, 1, 100)):
            self.record([("reasoning", "a", [(reasoning, "answered")]), ("verifier", "b", [(verifier, "answered")])], load=load)
        conn = rep.open_database()
        row = rep.review_times(conn)[0]
        conn.close()
        split = row["middle_review"]
        self.assertEqual((row["reviews"], row["median_wall_s"], split["wall_s"]), (3, 102.0, 102.0))
        self.assertEqual(split["load_s"] + sum(split["stage_s"].values()) + split["other_s"], 102.0)
        self.assertIn("middle review 102.0s = load", rep.format_review_times([row]))

    def test_combinations_are_ranked_fastest_first_with_the_review_count(self):
        for seconds in (30, 40, 50):
            self.record([("reasoning", "slow", [(seconds, "answered")])])
        self.record([("reasoning", "fast", [(5, "answered")])])
        conn = rep.open_database()
        rows = rep.review_times(conn)
        self.assertEqual([(r["combination"], r["reviews"], r["median_wall_s"]) for r in rows], [("reasoning:fast", 1, 5.0), ("reasoning:slow", 3, 40.0)])
        conn.close()

    def test_a_review_in_which_no_reviewer_answered_is_not_a_fast_review(self):
        # Every request of the stage failed: the review returned in 2 s, having reviewed nothing.
        self.record([("reasoning", "a", [(1, "refused"), (1, "timed_out")])])
        self.record([("reasoning", "a", [(40, "answered")])])
        conn = rep.open_database()
        row = rep.review_times(conn)[0]
        under = rep.under_target(conn, minutes=0.5)
        outcomes = sorted(r["outcome"] for r in conn.execute("SELECT outcome FROM runs"))
        conn.close()
        self.assertEqual((row["reviews"], row["median_wall_s"]), (1, 40.0))
        self.assertEqual(under, [])
        self.assertEqual(outcomes, ["finished", "incomplete"])

    def test_a_run_that_errored_says_nothing_about_review_time(self):
        self.record([("reasoning", "a", [(500, "answered")])], outcome="errored")
        self.record([("reasoning", "a", [(10, "answered")])])
        conn = rep.open_database()
        row = rep.review_times(conn)[0]
        self.assertEqual((row["reviews"], row["median_wall_s"]), (1, 10.0))
        conn.close()

    def test_a_stage_that_is_not_built_yet_is_split_out_like_any_other(self):
        self.record([("reasoning", "a", [(10, "answered")]), ("verifier", "b", [(5, "answered")]), ("classifier", "c", [(7, "answered")])])
        conn = rep.open_database()
        row = rep.review_times(conn)[0]
        self.assertEqual(row["combination"], "reasoning:a|verifier:b|classifier:c")
        self.assertEqual(row["middle_review"]["stage_s"]["classifier"], 7.0)
        self.assertIn("classifier 7.0s", rep.format_review_times([row]))
        conn.close()

    def test_this_machine_limits_the_reviews(self):
        self.record([("reasoning", "a", [(10, "answered")])])
        self.record([("reasoning", "a", [(90, "answered")])], machine=OTHER_MACHINE)
        conn = rep.open_database()
        self.assertEqual(rep.review_times(conn)[0]["reviews"], 2)
        self.assertEqual(rep.review_times(conn, this_machine=True)[0]["median_wall_s"], 10.0)
        conn.close()

    def test_this_machine_leaves_out_reviews_run_on_a_remote_lm_studio(self):
        self.record([("reasoning", "a", [(10, "answered")])])
        clock = Clock()
        remote = db.RunRecorder(source="test", review_type="verifier", clock=clock, lmstudio_url="http://gpu-box.local:1234/v1")
        with remote.stage("reasoning", models=("a",)):
            with remote.request(reviewer="rv", role="reasoning", model="a") as req:
                clock.advance(3)
                req.finish("x")
        remote.close()
        conn = rep.open_database()
        self.assertEqual(rep.review_times(conn)[0]["reviews"], 2)
        self.assertEqual([(r["reviews"], r["median_wall_s"]) for r in rep.review_times(conn, this_machine=True)], [(1, 10.0)])
        conn.close()

    def test_this_machine_with_no_runs_here_is_empty_not_an_error(self):
        self.record([("reasoning", "a", [(10, "answered")])], machine=OTHER_MACHINE)
        conn = rep.open_database()
        self.assertEqual(rep.review_times(conn, this_machine=True), [])
        self.assertIn("No finished reviews recorded on this machine", rep.format_review_times([], scope=" on this machine"))
        conn.close()


class UnderTarget(ReportTestCase):
    def test_lists_combinations_with_a_review_inside_the_target_and_how_many(self):
        for seconds in (120, 300, 900):  # 2, 5 and 15 minutes
            self.record([("reasoning", "mixed", [(seconds, "answered")])])
        for seconds in (1200, 1500):
            self.record([("reasoning", "slow", [(seconds, "answered")])])
        self.record([("reasoning", "quick", [(60, "answered")])])
        conn = rep.open_database()
        rows = rep.under_target(conn, minutes=10)
        self.assertEqual([(r["combination"], r["under"], r["reviews"]) for r in rows], [("reasoning:quick", 1, 1), ("reasoning:mixed", 2, 3)])
        text = rep.format_under_target(rows, 10)
        self.assertIn("reasoning:mixed: 2 of 3 review(s) under the target", text)
        self.assertNotIn("reasoning:slow", text)
        conn.close()

    def test_nothing_under_the_target_says_so(self):
        self.record([("reasoning", "a", [(600, "answered")])])
        conn = rep.open_database()
        self.assertEqual(rep.under_target(conn, minutes=1), [])
        self.assertIn("finished within 1 minute", rep.format_under_target([], 1))
        conn.close()

    def test_a_bad_target_is_an_error(self):
        self.record([("reasoning", "a", [(1, "answered")])])
        conn = rep.open_database()
        for bad in (0, -3, float("nan"), float("inf")):
            with self.subTest(minutes=bad), self.assertRaises(rep.ReportError):
                rep.under_target(conn, minutes=bad)
        conn.close()


class Report(ReportTestCase):
    def test_a_missing_database_is_an_error_that_names_it_and_creates_nothing(self):
        with self.assertRaises(rep.ReportError) as ctx:
            rep.report("review_times")
        self.assertIn(str(self.db_path), str(ctx.exception))
        self.assertFalse(self.db_path.exists())

    def test_an_unknown_view_and_a_missing_target_are_errors(self):
        self.record([("reasoning", "a", [(1, "answered")])])
        with self.assertRaises(rep.ReportError):
            rep.report("nope")
        with self.assertRaises(rep.ReportError) as ctx:
            rep.report("under_target")
        self.assertIn("target_minutes", str(ctx.exception))

    def test_each_view_renders_text_and_json(self):
        self.record([("reasoning", "a", [(2, "answered")]), ("verifier", "b", [(3, "answered")])])
        self.assertIn("model a [reasoning]".replace("model a", "- a"), rep.report("response_times"))
        self.assertIn("reasoning:a|verifier:b", rep.report("review_times"))
        self.assertIn("1 of 1 review(s)", rep.report("under_target", target_minutes=5))
        self.assertEqual(json.loads(rep.report("review_times", as_json=True))[0]["reviews"], 1)

    def test_a_file_that_is_not_a_database_is_an_error(self):
        self.db_path.write_text("not sqlite", encoding="utf-8")
        with self.assertRaises(rep.ReportError):
            rep.report("review_times")

    def test_a_path_with_uri_characters_is_opened_as_that_file_and_read_only(self):
        # Unescaped, `#` would end the path at "review" (creating that file) and drop mode=ro.
        # `?` is also a URI character but cannot be in a Windows file name; there `%20` stands in,
        # which an unescaped URI would read as a space.
        path = self.dir / ("review#579%20.db" if sys.platform == "win32" else "review#579?.db")
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB": str(path)}):
            self.record([("reasoning", "a", [(1, "answered")])])
            self.assertIn("reasoning:a", rep.report("review_times"))
            conn = rep.open_database()
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM runs")
            conn.close()
        # Only the database and its own WAL side files: no file named after the path cut at the `#`.
        self.assertEqual([p.name for p in self.dir.iterdir() if p.name.startswith("review") and not p.name.startswith(path.name)], [])

    def test_a_database_without_the_tables_is_an_error_and_its_connection_is_closed(self):
        sqlite3.connect(str(self.db_path)).close()  # a SQLite file, but not this database
        opened = []
        real = sqlite3.connect

        def tracking(*args, **kwargs):
            conn = real(*args, **kwargs)
            opened.append(conn)
            return conn

        with mock.patch.object(rep.sqlite3, "connect", side_effect=tracking):
            with self.assertRaises(rep.ReportError):
                rep.open_database()
        self.assertEqual(len(opened), 1)
        with self.assertRaises(sqlite3.ProgrammingError):  # closed: using it raises
            opened[0].execute("SELECT 1")


class Cli(ReportTestCase):
    def _run(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_prints_a_view(self):
        self.record([("reasoning", "a", [(2, "answered")])])
        code, out, _ = self._run("review_times")
        self.assertEqual(code, 0)
        self.assertIn("reasoning:a", out)

    def test_no_data_exits_1_with_a_message(self):
        code, out, err = self._run("review_times")
        self.assertEqual((code, out), (1, ""))
        self.assertIn("nothing has been recorded yet", err)

    def test_under_target_takes_minutes(self):
        self.record([("reasoning", "a", [(30, "answered")])])
        code, out, _ = self._run("under_target", "--minutes", "1")
        self.assertEqual(code, 0)
        self.assertIn("1 of 1", out)
        self.assertEqual(self._run("under_target")[0], 1)  # no --minutes


if __name__ == "__main__":
    unittest.main()
