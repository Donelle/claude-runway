"""Tests for libs/pr_round_report_lib.py and tools/pr_round_report.py: the read-only report over
the `pr-review-rounds` events. Synthetic rows written to a temporary metrics.db (never the real
one), no network, no live model."""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "libs"))
sys.path.insert(0, str(REPO / "tools"))

import metrics_lib  # noqa: E402
import pr_round_report as CLI  # noqa: E402
import pr_round_report_lib as R  # noqa: E402

REMOTE_A = "Acme/alpha"
REMOTE_B = "Acme/beta"


def round_row(store, pr, review, *, repo=REMOTE_A, source="remote", used=True, kind="copilot", session="s1",
              out=10, cc=100, cr=1000, inp=1, open_=False, kind_number=1, github_id="same", **extra):
    """One `pr_round` row. `used=None` leaves the flag out (a row written before it existed);
    `repo=None` writes no repository at all."""
    meta = {"schema": 2, "pr": pr, "kind": kind, "round": 1, "kind_number": kind_number,
            "review_id": str(review) if kind == "copilot" else None,
            "github_review_id": (str(review) if github_id == "same" else github_id) if kind == "copilot" else None,
            "repo": repo, "repo_source": source if repo else None, "window_open": open_,
            "input_tokens": inp, "cache_creation_tokens": cc, "cache_read_tokens": cr, "output_tokens": out, **extra}
    if used is not None and kind == "copilot":
        meta["local_reviewer_used"] = used
    store.record(R.METRIC_ID, R.EVENT_ROUND, 1.0, meta, session, "2026-11-08T10:00:00Z")


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / "metrics.db"
        self.store = metrics_lib.MetricsStore(self.db)

    def report(self, **kw):
        return R.build_report(self.db, **kw)

    def repo(self, report, name, source="remote"):
        found = [r for r in report["repos"] if (r["repo"], r["repo_source"]) == (name, source)]
        self.assertEqual(len(found), 1, report["repos"])
        return found[0]


class Grouping(_Base):
    def test_both_groups_are_reported_per_round_with_each_token_field_on_its_own(self):
        round_row(self.store, 5, 101, used=True, out=10, cc=100, cr=1000, inp=1)
        round_row(self.store, 5, 102, used=False, out=40, cc=400, cr=4000, inp=4)
        round_row(self.store, 6, 103, used=False, out=60, cc=600, cr=6000, inp=6)
        a = self.repo(self.report(), REMOTE_A)
        self.assertEqual(a["copilot_rounds"], 3)
        self.assertEqual(a["pull_requests"], 2)
        self.assertEqual((a["groups"]["used"]["rounds"], a["groups"]["not_used"]["rounds"]), (1, 2))
        used, not_used = a["groups"]["used"], a["groups"]["not_used"]
        self.assertEqual(used["output_tokens"]["median"], 10)
        self.assertEqual((not_used["output_tokens"]["median"], not_used["output_tokens"]["min"],
                          not_used["output_tokens"]["max"], not_used["output_tokens"]["total"]), (50, 40, 60, 100))
        # Never summed across fields: each stands alone.
        self.assertEqual(not_used["cache_read_tokens"]["total"], 10000)
        self.assertEqual(not_used["input_tokens"]["total"], 10)

    def test_two_repositories_sharing_a_pr_number_are_two_groups_not_one_round(self):
        round_row(self.store, 7, 201, repo=REMOTE_A, used=True)
        round_row(self.store, 7, 201, repo=REMOTE_B, used=False)   # same PR number and same review id text
        report = self.report()
        self.assertEqual(self.repo(report, REMOTE_A)["groups"]["used"]["rounds"], 1)
        self.assertEqual(self.repo(report, REMOTE_A)["groups"]["not_used"]["rounds"], 0)
        self.assertEqual(self.repo(report, REMOTE_B)["groups"]["not_used"]["rounds"], 1)

    def test_a_null_repo_is_its_own_group_and_never_merged_into_a_repository(self):
        round_row(self.store, 7, 301, repo=None, used=False)
        round_row(self.store, 7, 302, repo=REMOTE_A, used=False)
        report = self.report()
        none = self.repo(report, None, None)
        self.assertEqual(none["copilot_rounds"], 1)
        self.assertEqual(self.repo(report, REMOTE_A)["copilot_rounds"], 1)
        self.assertIn("no repository", none["label"])

    def test_a_project_name_is_not_a_repository(self):
        round_row(self.store, 7, 401, repo="my-folder", source="project", used=True)
        round_row(self.store, 7, 402, repo="my-folder", source="remote", used=True)
        report = self.report()
        project = self.repo(report, "my-folder", "project")
        self.assertIn("not a repository", project["label"])
        self.assertEqual(self.repo(report, "my-folder", "remote")["copilot_rounds"], 1)
        self.assertEqual(len(report["repos"]), 2)

    def test_a_row_without_the_flag_is_unknown_never_not_used(self):
        round_row(self.store, 8, 501, used=None)
        a = self.repo(self.report(), REMOTE_A)
        self.assertEqual((a["groups"]["unknown"]["rounds"], a["groups"]["not_used"]["rounds"], a["groups"]["used"]["rounds"]),
                         (1, 0, 0))

    def test_a_pr_with_a_true_round_and_a_false_round_lands_in_both_groups(self):
        round_row(self.store, 9, 601, used=True)
        round_row(self.store, 9, 602, used=True)
        round_row(self.store, 9, 603, used=False)
        a = self.repo(self.report(), REMOTE_A)
        self.assertEqual((a["pull_requests"], a["groups"]["used"]["rounds"], a["groups"]["not_used"]["rounds"]), (1, 2, 1))

    def test_a_round_two_sessions_saw_is_one_round_with_each_field_summed_and_the_flag_or_ed(self):
        round_row(self.store, 10, 701, used=False, session="s1", out=10, cc=100)
        round_row(self.store, 10, 701, used=True, session="s2", out=5, cc=50)
        a = self.repo(self.report(), REMOTE_A)
        self.assertEqual(a["copilot_rounds"], 1)
        self.assertEqual(a["groups"]["used"]["rounds"], 1)
        self.assertEqual(a["groups"]["used"]["output_tokens"]["total"], 15)
        self.assertEqual(a["groups"]["used"]["cache_creation_tokens"]["total"], 150)

    def test_a_round_with_no_review_id_is_not_merged_with_another(self):
        # Two sessions, no id on either row: nothing says they are one round.
        rows = []
        for session in ("s1", "s2"):
            meta = {"kind": "copilot", "pr": 11, "round": 1, "repo": REMOTE_A, "repo_source": "remote",
                    "review_id": None, "github_review_id": None, "output_tokens": 5}
            rows.append((json.dumps(meta), session))
        self.assertEqual(R.aggregate(rows)["repos"][0]["copilot_rounds"], 2)

    def test_local_rounds_are_summed_apart_and_are_not_copilot_rounds(self):
        round_row(self.store, 12, 0, kind="local", kind_number=1, out=7, cc=70, review_pr=12)
        round_row(self.store, 12, 0, kind="local", kind_number=2, out=3, cc=30, review_pr=12)
        round_row(self.store, 12, 901, used=True)
        a = self.repo(self.report(), REMOTE_A)
        self.assertEqual(a["copilot_rounds"], 1)
        self.assertEqual(a["local_rounds"]["rounds"], 2)
        self.assertEqual(a["local_rounds"]["output_tokens"]["total"], 10)
        self.assertEqual(a["local_rounds_unnamed"]["rounds"], 0)

    def test_one_review_call_that_names_no_pr_is_not_counted_once_per_pr_in_the_cost(self):
        # The reader attaches a `review` call without `params.pr` to every PR the session fetched
        # feedback for, so one call is a row under each of them (here two PRs, 10 output tokens
        # each). Summing them would report two rounds and 20 tokens for one call.
        round_row(self.store, 12, 0, kind="local", kind_number=1, out=10, review_pr=None)
        round_row(self.store, 13, 0, kind="local", kind_number=1, out=10, review_pr=None)
        round_row(self.store, 12, 0, kind="local", kind_number=2, out=4, review_pr=12)   # a named call is exact
        a = self.repo(self.report(), REMOTE_A)
        self.assertEqual((a["local_rounds"]["rounds"], a["local_rounds"]["output_tokens"]["total"]), (1, 4))
        unnamed = a["local_rounds_unnamed"]
        self.assertEqual((unnamed["rounds"], unnamed["output_tokens"]["total"]), (2, 20))
        text = R.format_report(self.report())
        self.assertIn("not added to the cost", text)
        self.assertIn("name no pull request", text)

    def test_open_windows_are_counted_and_can_be_left_out(self):
        round_row(self.store, 13, 1001, used=False, out=10)
        round_row(self.store, 13, 1002, used=False, out=20, open_=True)
        a = self.repo(self.report(), REMOTE_A)
        self.assertEqual((a["groups"]["not_used"]["rounds"], a["groups"]["not_used"]["open_windows"]), (2, 1))
        b = self.repo(self.report(exclude_open=True), REMOTE_A)
        self.assertEqual((b["groups"]["not_used"]["rounds"], b["groups"]["not_used"]["output_tokens"]["total"]), (1, 10))

    def test_the_repo_filter_keeps_only_that_value(self):
        round_row(self.store, 14, 1101, repo=REMOTE_A)
        round_row(self.store, 14, 1102, repo=REMOTE_B)
        report = self.report(repo=REMOTE_B)
        self.assertEqual([r["repo"] for r in report["repos"]], [REMOTE_B])

    def test_a_missing_token_field_is_left_out_of_the_stats_not_counted_as_zero(self):
        round_row(self.store, 15, 1201, used=True, out=10)
        round_row(self.store, 15, 1202, used=True, out=None)
        out = self.repo(self.report(), REMOTE_A)["groups"]["used"]["output_tokens"]
        self.assertEqual((out["n"], out["total"], out["median"]), (1, 10, 10))

    def test_a_non_finite_or_boolean_figure_is_left_out_like_a_missing_one(self):
        rows = []
        for out in (float("nan"), True, 8):
            meta = {"kind": "copilot", "pr": 1, "repo": REMOTE_A, "repo_source": "remote", "review_id": str(out),
                    "local_reviewer_used": True, "output_tokens": out}
            rows.append((json.dumps(meta), "s"))
        stats = R.aggregate(rows)["repos"][0]["groups"]["used"]["output_tokens"]
        self.assertEqual((stats["n"], stats["total"]), (1, 8))
        json.loads(json.dumps(R.aggregate(rows), allow_nan=False))   # strict JSON: no NaN anywhere

    def test_other_metric_ids_and_fetch_rows_are_not_read(self):
        round_row(self.store, 16, 1301, used=True)
        self.store.record(R.METRIC_ID, "pr_feedback_fetch", 1.0, {"pr": 16, "repo": REMOTE_A, "output_tokens": 999}, "s1")
        self.store.record("autowork", R.EVENT_ROUND, 1.0, {"kind": "copilot", "pr": 16, "repo": REMOTE_A}, "s1")
        a = self.repo(self.report(), REMOTE_A)
        self.assertEqual((a["copilot_rounds"], a["groups"]["used"]["output_tokens"]["total"]), (1, 10))

    def test_a_row_that_is_not_json_is_counted_not_fatal(self):
        out = R.aggregate([("not json", "s"), (None, "s"), ("[1]", "s")])
        self.assertEqual((out["repos"], out["unreadable_rows"]), ([], 3))


class ReadOnly(_Base):
    def test_a_missing_database_is_reported_and_not_created(self):
        report = self.report()
        self.assertFalse(report["available"])
        self.assertFalse(self.db.exists())
        self.assertIn("No data", R.format_report(report))

    def test_a_file_that_is_not_a_database_is_reported_not_a_traceback(self):
        self.db.write_text("this is not a sqlite file", encoding="utf-8")
        report = self.report()
        self.assertFalse(report["available"])
        self.assertIn("could not be", report["reason"])
        self.assertIn("No data", R.format_report(report))

    def test_a_store_with_no_rounds_says_so(self):
        self.store.record("autowork", "x", 1.0, {})
        self.assertFalse(self.report()["available"])

    def test_the_report_does_not_change_the_database(self):
        round_row(self.store, 20, 1401)
        before = self.db.read_bytes()
        self.report()
        R.format_report(self.report())
        self.assertEqual(self.db.read_bytes(), before)

    def test_it_cannot_write_even_if_asked_to(self):
        round_row(self.store, 20, 1401)
        conn = R.open_read_only(self.db)
        self.assertIsNotNone(conn)
        assert conn is not None
        try:
            with self.assertRaises(sqlite3.OperationalError):   # "attempt to write a readonly database"
                conn.execute("DELETE FROM metrics")
        finally:
            conn.close()


class Text(_Base):
    def test_the_text_names_both_groups_and_every_token_field_and_the_not_a_repository_label(self):
        round_row(self.store, 30, 1501, used=True)
        round_row(self.store, 30, 1502, used=False)
        round_row(self.store, 31, 1503, repo="folder", source="project", used=False)
        round_row(self.store, 32, 1504, repo=None, used=False)
        text = R.format_report(self.report())
        for needle in ("reviewer used: 1 round(s)", "reviewer not used: 1 round(s)", "input_tokens", "cache_creation_tokens",
                       "cache_read_tokens", "output_tokens", "Acme/alpha (repository)", "folder (project folder, not a repository)",
                       "(no repository or project recorded)"):
            self.assertIn(needle, text)


class Cli(_Base):
    def run_cli(self, *argv, expect=0):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = CLI.main(list(argv))
        self.assertEqual(code, expect, err.getvalue())
        return out.getvalue()

    def test_text_and_json(self):
        round_row(self.store, 40, 1601, used=True)
        text = self.run_cli("--db", str(self.db))
        self.assertIn("reviewer used", text)
        data = json.loads(self.run_cli("--db", str(self.db), "--json"))
        self.assertTrue(data["available"])
        self.assertEqual(data["repos"][0]["groups"]["used"]["rounds"], 1)

    def test_a_missing_db_exits_zero_and_creates_nothing(self):
        out = self.run_cli("--db", str(self.db))
        self.assertIn("No data", out)
        self.assertFalse(self.db.exists())

    def test_the_default_database_follows_the_env_override(self):
        round_row(self.store, 41, 1701, used=False)
        old = os.environ.get("CLAUDE_RUNWAY_METRICS_DB")
        os.environ["CLAUDE_RUNWAY_METRICS_DB"] = str(self.db)
        try:
            self.assertIn("reviewer not used", self.run_cli())
        finally:
            if old is None:
                del os.environ["CLAUDE_RUNWAY_METRICS_DB"]
            else:
                os.environ["CLAUDE_RUNWAY_METRICS_DB"] = old

    def test_repo_and_exclude_open_flags(self):
        round_row(self.store, 42, 1801, repo=REMOTE_A, open_=True)
        round_row(self.store, 42, 1802, repo=REMOTE_B)
        data = json.loads(self.run_cli("--db", str(self.db), "--repo", REMOTE_A, "--exclude-open", "--json"))
        self.assertEqual(data["repos"][0]["copilot_rounds"], 0)
        self.assertEqual(len(data["repos"]), 1)

    def test_an_empty_repo_is_a_bad_argument(self):
        self.run_cli("--db", str(self.db), "--repo", " ", expect=2)


if __name__ == "__main__":
    unittest.main()
