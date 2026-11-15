"""Tests for libs/pr_round_cost_lib.py: the review rounds of one PR, and what
each cost, read out of a session transcript. Synthetic fixture transcripts in a temp
dir only -- never the real ~/.claude/projects, and no real transcript is committed."""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "libs"))

import pr_round_cost_lib as R  # noqa: E402

PR = 603
COPILOT = "copilot-pull-request-reviewer[bot]"
ME = "someone-else"


def iso(s):
    return f"2026-11-06T{s}Z"


def epoch(s):
    return datetime.fromisoformat(iso(s).replace("Z", "+00:00")).timestamp()


def usage(i, cc, cr, o):
    return {"input_tokens": i, "cache_creation_input_tokens": cc, "cache_read_input_tokens": cr, "output_tokens": o}


def rest_review(rid, login, ts, commit="abc1234", state="COMMENTED"):
    return {"id": rid, "user": {"login": login}, "state": state, "submitted_at": iso(ts), "commit_id": commit}


def view_review(rid, login, ts, commit="abc1234", state="COMMENTED"):
    return {"id": rid, "author": {"login": login.replace("[bot]", "")}, "state": state,
            "submittedAt": iso(ts), "commit": {"oid": commit}}


class Transcript:
    """Builds a transcript the way Claude Code writes one: an assistant entry per content
    block (each repeating the message's usage) and a user entry per tool result."""

    def __init__(self):
        self.lines = []
        self.n = 0

    def assistant(self, ts, u=None, mid=None, blocks=None, model="claude-sonnet-5-5"):
        self.n += 1
        mid = mid or f"msg_{self.n}"
        blocks = blocks or [{"type": "text", "text": "x"}]
        for b in blocks:
            self.lines.append({"type": "assistant", "timestamp": iso(ts), "sessionId": "sess-1",
                               "message": {"id": mid, "model": model, "usage": u or usage(1, 10, 100, 5), "content": [b]}})
        return mid

    def call(self, ts, name, tool_input, result=None, u=None, result_blocks=None):
        self.n += 1
        tid = f"toolu_{self.n}"
        self.assistant(ts, u=u, blocks=[{"type": "tool_use", "id": tid, "name": name, "input": tool_input}])
        if result is not None or result_blocks is not None:
            content = result_blocks if result_blocks is not None else result
            self.lines.append({"type": "user", "timestamp": iso(ts),
                               "message": {"content": [{"type": "tool_result", "tool_use_id": tid, "content": content}]}})
        return tid

    def bash(self, ts, command, result=None, u=None):
        return self.call(ts, "Bash", {"command": command}, result, u)

    def reviews_api(self, ts, reviews, pr=PR, u=None):
        return self.bash(ts, f"gh api repos/Donelle/claude-runway/pulls/{pr}/reviews --paginate",
                         json.dumps(reviews), u)

    def reviews_view(self, ts, reviews, latest=None, pr=PR, u=None):
        body = {"reviews": reviews, "latestReviews": latest if latest is not None else []}
        return self.bash(ts, f"gh pr view {pr} -R Donelle/claude-runway --json number,reviews,latestReviews",
                         json.dumps(body), u)

    def review_call(self, ts, params=None, result="findings", rtype="verifier"):
        tin = {"type": rtype}
        if params is not None:
            tin["params"] = params
        return self.call(ts, R.REVIEW_TOOL, tin, result)

    def write(self, directory, name="t.jsonl", extra_raw=()):
        p = Path(directory) / name
        with open(p, "w", encoding="utf-8") as f:
            for line in self.lines:
                f.write(json.dumps(line) + "\n")
            for raw in extra_raw:
                f.write(raw + "\n")
        return p


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def rounds(self, t, merge="04:20:28", pr=PR, **kw):
        return R.find_rounds(t.write(self.dir), pr, merge_time=iso(merge) if merge else None, **kw)


class ThreeCopilotRounds(_Base):
    def make(self):
        t = Transcript()
        t.assistant("03:59:00")                      # work before any fetch: in no round
        # Round 1: Copilot's first review, with the user's own COMMENTED replies listed too.
        t.reviews_api("04:01:15", [rest_review(11, COPILOT, "03:27:21")], u=usage(1, 100, 1000, 10))
        t.assistant("04:02:00", u=usage(2, 200, 2000, 20))
        # Round 2: the user's replies and a second Copilot review.
        t.reviews_api("04:11:05", [rest_review(11, COPILOT, "03:27:21"), rest_review(12, ME, "04:07:21"),
                                   rest_review(13, COPILOT, "04:10:15")], u=usage(3, 300, 3000, 30))
        t.assistant("04:12:00", u=usage(4, 400, 4000, 40))
        # Round 3: the approval, read through `gh pr view --json reviews` (a PRR_ node id),
        # with the quirky empty id `latestReviews` returned for the Copilot entry.
        t.reviews_view("04:19:59", [view_review("PRR_a", COPILOT, "03:27:21"), view_review("PRR_b", ME, "04:17:28"),
                                    view_review("PRR_c", COPILOT, "04:10:15"),
                                    view_review("PRR_d", COPILOT, "04:19:20", state="APPROVED")],
                       latest=[view_review("", COPILOT, "04:19:20")], u=usage(5, 500, 5000, 50))
        t.assistant("04:20:28", u=usage(6, 600, 6000, 60))
        t.assistant("04:25:00", u=usage(9, 900, 9000, 90))  # after the merge
        return t

    def test_three_rounds_with_the_users_own_reviews_ignored(self):
        out = self.rounds(self.make())
        self.assertTrue(out["ok"])
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11", "13", "PRR_d"])
        self.assertEqual([r["kind"] for r in out["rounds"]], ["copilot"] * 3)
        self.assertEqual([r["number"] for r in out["rounds"]], [1, 2, 3])
        self.assertEqual(out["rounds"][2]["review_state"], "APPROVED")
        self.assertEqual(out["rounds"][0]["commit_id"], "abc1234")
        self.assertEqual(out["rounds"][0]["submitted_at"], iso("03:27:21"))
        self.assertEqual(out["session_id"], "sess-1")
        self.assertEqual(out["skipped_fetches"], 0)
        self.assertEqual(out["fetch_calls"], 3)

    def test_windows_run_from_marker_to_next_marker_and_the_last_to_merge(self):
        out = self.rounds(self.make())
        r1, r2, r3 = out["rounds"]
        self.assertEqual(r1["start_ts"], epoch("04:01:15"))
        self.assertEqual(r1["end_ts"], epoch("04:11:05"))
        self.assertEqual(r3["end_ts"], epoch("04:20:28"))
        self.assertEqual(r1["wall_clock_s"], 590)
        # The marker's own message is in its round; the next marker's is not.
        self.assertEqual((r1["messages"], r2["messages"], r3["messages"]), (2, 2, 2))
        self.assertEqual(r1["input_tokens"], 1 + 2)
        self.assertEqual(r2["cache_creation_tokens"], 300 + 400)
        # The closing message AT the merge time is in the last window; the one after is not.
        self.assertEqual(r3["output_tokens"], 50 + 60)
        self.assertEqual(r3["cache_read_tokens"], 5000 + 6000)

    def test_without_a_merge_time_the_last_window_ends_at_the_transcript_end(self):
        out = self.rounds(self.make(), merge=None)
        self.assertEqual(out["rounds"][2]["end_ts"], epoch("04:25:00"))
        self.assertEqual(out["rounds"][2]["output_tokens"], 50 + 60 + 90)

    def test_a_fetch_after_the_merge_is_dropped(self):
        t = self.make()
        t.reviews_api("04:30:00", [rest_review(99, COPILOT, "04:29:00")])
        out = self.rounds(t)
        self.assertEqual(len(out["rounds"]), 3)
        self.assertEqual(out["fetch_calls"], 3)

    def test_the_four_token_fields_stay_separate(self):
        out = self.rounds(self.make())
        r = out["rounds"][0]
        for k in ("input_tokens", "cache_creation_tokens", "cache_read_tokens", "output_tokens"):
            self.assertIn(k, r)
        self.assertEqual(r["by_model"]["claude-sonnet-5-5"]["requests"], 2)
        self.assertEqual(r["model"], "claude-sonnet-5-5")


class FetchCalls(_Base):
    def test_repeated_fetches_with_the_same_newest_id_stay_in_one_round(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")])
        t.reviews_api("04:02:00", [rest_review(11, COPILOT, "03:27:21"), rest_review(12, ME, "04:01:30")])
        t.bash("04:03:00", f"gh api repos/o/r/issues/{PR}/comments --paginate", json.dumps([{"id": 1, "body": "x"}]))
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual(len(out["rounds"]), 1)
        self.assertEqual(len(out["rounds"][0]["fetch_calls"]), 3)
        self.assertEqual(out["skipped_fetches"], 0)

    def test_the_users_own_new_review_alone_does_not_start_a_round(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")])
        t.reviews_api("04:02:00", [rest_review(11, COPILOT, "03:27:21"), rest_review(12, ME, "04:01:30")])
        self.assertEqual(len(self.rounds(t, merge="04:10:00")["rounds"]), 1)

    def test_fetch_windows_tile_their_round(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")], u=usage(1, 1, 1, 1))
        t.assistant("04:01:30", u=usage(2, 2, 2, 2))
        t.reviews_api("04:02:00", [rest_review(11, COPILOT, "03:27:21")], u=usage(4, 4, 4, 4))
        t.assistant("04:03:00", u=usage(8, 8, 8, 8))
        out = self.rounds(t, merge="04:10:00")
        rnd = out["rounds"][0]
        a, b = rnd["fetch_calls"]
        self.assertEqual((a["output_tokens"], b["output_tokens"]), (3, 12))
        self.assertEqual(a["output_tokens"] + b["output_tokens"], rnd["output_tokens"])
        self.assertEqual(a["end_ts"], b["ts"])
        self.assertEqual(b["end_ts"], rnd["end_ts"])
        self.assertEqual(a["round"], 1)
        self.assertEqual(a["review_id"], "11")

    def test_a_fetch_before_any_round_is_an_orphan_with_its_own_window(self):
        t = Transcript()
        t.reviews_api("04:00:00", [], u=usage(1, 1, 1, 1))     # no Copilot review yet
        t.assistant("04:00:30", u=usage(2, 2, 2, 2))
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "04:00:50")])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual(len(out["rounds"]), 1)
        orphan = out["orphan_fetch_calls"][0]
        self.assertIsNone(orphan["round"])
        self.assertEqual(orphan["end_ts"], epoch("04:01:00"))
        self.assertEqual(orphan["output_tokens"], 3)
        self.assertTrue(orphan["recognised"])

    def test_one_review_seen_through_both_shapes_is_one_round(self):
        # The numeric REST id and the PRR_ node id differ; the submission time does not.
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(5795366119, COPILOT, "03:27:21")])
        t.reviews_view("04:02:00", [view_review("PRR_kwDOxyz", COPILOT, "03:27:21")])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual(len(out["rounds"]), 1)
        self.assertEqual(out["rounds"][0]["review_ids"], ["5795366119", "PRR_kwDOxyz"])

    def test_a_known_review_id_is_matched_when_a_later_fetch_omits_its_time(self):
        # The same review, first with a submission time (JSON), then as a line with none,
        # and the other way round: a known id is the same round whatever else is shown.
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")])
        t.bash("04:02:00", f"gh api repos/o/r/pulls/{PR}/reviews --jq '.[]'",
               "#11 [copilot-pull-request-reviewer[bot]] COMMENTED commit=abc1234\n")
        t.bash("04:03:00", f"gh api repos/o/r/pulls/{PR}/reviews --jq '.[]'",
               "#13 [copilot-pull-request-reviewer[bot]] COMMENTED commit=def5678\n")
        t.reviews_api("04:04:00", [rest_review(13, COPILOT, "04:02:50")])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11", "13"])
        self.assertEqual(len(out["rounds"][0]["fetch_calls"]), 2)
        self.assertEqual(len(out["rounds"][1]["fetch_calls"]), 2)

    def test_a_time_learned_later_is_an_alias_of_the_retained_identity(self):
        # A line with no time, then REST JSON with it, then the PRR_ form with that same
        # time: one review, one round.
        t = Transcript()
        t.bash("04:01:00", f"gh api repos/o/r/pulls/{PR}/reviews --jq '.[]'",
               "#11 [copilot-pull-request-reviewer[bot]] COMMENTED commit=abc1234\n")
        t.reviews_api("04:02:00", [rest_review(11, COPILOT, "03:27:21")])
        t.reviews_view("04:03:00", [view_review("PRR_kwDOxyz", COPILOT, "03:27:21")])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual(len(out["rounds"]), 1)
        self.assertEqual(out["rounds"][0]["review_ids"], ["11", "PRR_kwDOxyz"])

    def test_another_prs_payload_is_not_this_prs_round(self):
        # The command names this PR, but the output says it is about another one.
        other = {"number": 604, "reviews": [view_review("PRR_z", COPILOT, "03:30:00")]}
        t = Transcript()
        t.bash("04:01:00", f"gh pr view {PR} --json number,reviews", json.dumps(other))
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual(out["rounds"], [])
        self.assertEqual((out["fetch_calls"], out["skipped_fetches"]), (1, 0))

    def test_rest_reviews_of_another_pr_are_filtered_by_their_pull_request_url(self):
        def rest(rid, ts, pr):
            return {**rest_review(rid, COPILOT, ts), "pull_request_url": f"https://api.github.com/repos/o/r/pulls/{pr}"}
        t = Transcript()
        t.bash("04:01:00", f"gh api repos/o/r/pulls/{PR}/reviews",
               json.dumps([rest(11, "03:00:00", PR)]) + "\n" + json.dumps([rest(99, "03:30:00", 604)]))
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])

    def test_a_chained_fetch_of_two_prs_is_unrecognised_not_split(self):
        # Two PRs in one command is no whitelisted shape: it is counted, never read, so
        # neither PR's round can be built from the other's output.
        t = Transcript()
        t.bash("04:01:00", f"gh pr view {PR} --json number,reviews && gh pr view 604 --json number,reviews",
               json.dumps({"number": PR, "reviews": [view_review("PRR_a", COPILOT, "03:00:00")]}) + "\n"
               + json.dumps({"number": 604, "reviews": [view_review("PRR_z", COPILOT, "03:30:00")]}))
        for pr in (PR, 604):
            out = self.rounds(t, merge="04:10:00", pr=pr)
            self.assertEqual((out["rounds"], out["fetch_calls"], out["skipped_fetches"]), ([], 1, 1), pr)

    def test_two_fetches_at_one_instant_keep_transcript_order_for_their_windows(self):
        # An orphan comments fetch, then the fetch that starts the round, at the same
        # second: the round's own fetch must carry the round's cost, not zero.
        t = Transcript()
        t.bash("04:01:00", f"gh api repos/o/r/issues/{PR}/comments", json.dumps([{"id": 1, "body": "x"}]), u=usage(1, 1, 1, 3))
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")], u=usage(1, 1, 1, 4))
        t.assistant("04:02:00", u=usage(1, 1, 1, 5))
        out = self.rounds(t, merge="04:10:00")
        rnd = out["rounds"][0]
        # Messages are placed by time, so the orphan call's own message (same instant) is
        # in the round's window and the orphan's window is empty: counted once.
        self.assertEqual(rnd["output_tokens"], 3 + 4 + 5)
        self.assertEqual(rnd["fetch_calls"][0]["output_tokens"], rnd["output_tokens"])
        self.assertEqual(out["orphan_fetch_calls"][0]["messages"], 0)

    def test_one_line_per_review_output_is_read(self):
        # What a `--jq` printing one review per line produces (the form real sessions used).
        t = Transcript()
        t.bash("04:01:15", f"gh api repos/o/r/pulls/{PR}/reviews --paginate --jq '.[] | \"#\\(.id)\"'",
               "#5795366119 [copilot-pull-request-reviewer[bot]] COMMENTED commit=683c25d 2026-11-06T03:27:21Z\n"
               "#5795611231 [someone-else] COMMENTED commit=af903e1 2026-11-06T04:07:21Z\n")
        t.bash("04:11:05", f"gh api repos/o/r/pulls/{PR}/reviews --jq '.[]'",
               "#5795366119 [copilot-pull-request-reviewer[bot]] COMMENTED commit=683c25d 2026-11-06T03:27:21Z\n"
               "#5795628397 [copilot-pull-request-reviewer[bot]] COMMENTED commit=af903e1 2026-11-06T04:10:15Z\n")
        out = self.rounds(t, merge="04:20:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["5795366119", "5795628397"])
        self.assertEqual(out["rounds"][1]["commit_id"], "af903e1")

    def test_unrecognised_output_shape_is_skipped_not_guessed_at(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")])
        t.bash("04:02:00", f"gh api repos/o/r/pulls/{PR}/reviews --jq '.[].state'", "COMMENTED\nCOMMENTED\n")
        t.bash("04:03:00", f"gh api repos/o/r/pulls/{PR}/reviews", "[compressed 9000 -> 300 chars across 1 chunk(s)]\nsummary")
        t.bash("04:04:00", f"gh api repos/o/r/pulls/{PR}/comments")  # interrupted: no result at all
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual(len(out["rounds"]), 1)
        self.assertEqual(out["skipped_fetches"], 3)
        self.assertEqual(out["fetch_calls"], 4)
        calls = out["rounds"][0]["fetch_calls"]
        self.assertEqual([c["recognised"] for c in calls], [True, False, False, False])

    def test_a_transcript_with_no_feedback_fetch_gives_no_rounds(self):
        t = Transcript()
        t.assistant("04:01:00")
        t.bash("04:02:00", "git status", "clean")
        t.bash("04:03:00", f"gh pr view {PR} --json state,mergeable", "{}")
        out = self.rounds(t)
        self.assertTrue(out["ok"])
        self.assertEqual((out["rounds"], out["orphan_fetch_calls"], out["fetch_calls"]), ([], [], 0))

    def test_a_call_with_no_timestamp_is_counted_skipped(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")])
        t.lines.append({"type": "assistant", "message": {"id": "m", "model": "x", "usage": usage(1, 1, 1, 1), "content": [
            {"type": "tool_use", "id": "toolu_x", "name": "Bash",
             "input": {"command": f"gh api repos/o/r/pulls/{PR}/reviews"}}]}})
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual((out["fetch_calls"], out["skipped_fetches"], len(out["rounds"])), (2, 1, 1))

    def test_result_content_given_as_text_blocks_is_read(self):
        t = Transcript()
        t.call("04:01:00", "Bash", {"command": f"gh api repos/o/r/pulls/{PR}/reviews"},
               result_blocks=[{"type": "text", "text": json.dumps([rest_review(11, COPILOT, "03:27:21")])}])
        self.assertEqual(self.rounds(t)["rounds"][0]["review_id"], "11")

    def test_text_among_the_records_makes_the_output_ambiguous_not_scanned(self):
        # A reviews fetch whose output has free text between the records (a body printed by a
        # jq program) cannot be told from a body that quotes a record: not recognised.
        t = Transcript()
        t.bash("04:01:00", f"gh api repos/o/r/pulls/{PR}/reviews",
               "=== REVIEWS\n" + json.dumps([rest_review(11, COPILOT, "03:27:21")]) + "\n=== done\n")
        out = self.rounds(t)
        self.assertEqual((out["rounds"], out["fetch_calls"], out["skipped_fetches"]), ([], 1, 1))

    def test_inline_comments_are_not_mistaken_for_reviews(self):
        # An inline comment has a `commit_id` and a Copilot author too, but its id is a
        # comment id and it has no `state`.
        t = Transcript()
        t.bash("04:01:00", f"gh api repos/o/r/pulls/{PR}/comments --paginate", json.dumps([
            {"id": 4488718479, "user": {"login": "Copilot"}, "path": "a.py", "commit_id": "abc", "body": "x"}]))
        out = self.rounds(t)
        self.assertEqual(out["rounds"], [])
        self.assertEqual(out["skipped_fetches"], 0)
        self.assertEqual(len(out["orphan_fetch_calls"]), 1)


class TokenAccounting(_Base):
    def test_usage_is_taken_once_per_message_id_as_the_field_wise_max(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")], u=usage(1, 10, 100, 5))
        # One response written as three lines, first in progress: counted once.
        t.assistant("04:02:00", u=usage(1, 50, 500, 3), mid="msg_same",
                    blocks=[{"type": "thinking", "thinking": "t"}, {"type": "text", "text": "x"},
                            {"type": "tool_use", "id": "toolu_a", "name": "Read", "input": {}}])
        t.assistant("04:02:01", u=usage(1, 50, 500, 40), mid="msg_same")
        out = self.rounds(t, merge="04:10:00")
        r = out["rounds"][0]
        self.assertEqual(r["messages"], 2)
        self.assertEqual(r["output_tokens"], 5 + 40)
        self.assertEqual(r["cache_read_tokens"], 100 + 500)
        self.assertEqual(r["tool_calls"], 2)   # the fetch itself and the Read

    def test_a_streamed_message_straddling_a_boundary_is_counted_once_where_it_started(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")], u=usage(1, 1, 1, 1))
        # One response whose first line is before round 2's marker and whose last is after.
        t.assistant("04:01:50", u=usage(1, 1, 1, 20), mid="msg_straddle")
        t.reviews_api("04:02:00", [rest_review(13, COPILOT, "04:01:55")], u=usage(1, 1, 1, 2))
        t.assistant("04:02:05", u=usage(1, 1, 1, 20), mid="msg_straddle")
        t.assistant("04:03:00", u=usage(1, 1, 1, 4))
        r1, r2 = self.rounds(t, merge="04:10:00")["rounds"]
        self.assertEqual((r1["messages"], r1["output_tokens"]), (2, 1 + 20))
        self.assertEqual((r2["messages"], r2["output_tokens"]), (2, 2 + 4))

    def test_synthetic_messages_and_entries_without_usage_are_ignored(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")])
        t.lines.append({"type": "assistant", "timestamp": iso("04:02:00"),
                        "message": {"id": "s", "model": "<synthetic>", "usage": usage(0, 0, 0, 0), "content": []}})
        t.lines.append({"type": "assistant", "timestamp": iso("04:02:30"),
                        "message": {"id": "n", "model": "x", "content": [{"type": "text", "text": "x"}]}})
        self.assertEqual(self.rounds(t, merge="04:10:00")["rounds"][0]["messages"], 1)

    def test_the_model_is_the_one_with_the_most_output(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")], u=usage(1, 1, 1, 5))
        t.assistant("04:02:00", u=usage(1, 1, 1, 500), model="claude-opus-x")
        r = self.rounds(t, merge="04:10:00")["rounds"][0]
        self.assertEqual(r["model"], "claude-opus-x")
        self.assertEqual(set(r["by_model"]), {"claude-sonnet-5-5", "claude-opus-x"})


class LocalRounds(_Base):
    def test_windows_end_at_next_review_call_a_copilot_round_or_the_merge(self):
        t = Transcript()
        t.review_call("04:00:00", {"repo_path": "/r", "base_ref": "origin/main", "pr": PR}, result="x" * 40,
                      rtype="reasoning")
        t.assistant("04:01:00", u=usage(1, 1, 1, 7))
        t.review_call("04:05:00", {"repo_path": "/r", "base_ref": "origin/main", "head_ref": "feat"}, result="y" * 10)
        t.assistant("04:06:00", u=usage(1, 1, 1, 9))
        t.reviews_api("04:10:00", [rest_review(11, COPILOT, "04:09:00")])      # ends the second local round
        t.review_call("04:15:00", {"pr": PR})                                    # ends the Copilot round
        t.assistant("04:16:00", u=usage(1, 1, 1, 11))
        out = self.rounds(t, merge="04:20:00")
        kinds = [r["kind"] for r in out["rounds"]]
        self.assertEqual(kinds, ["local", "local", "copilot", "local"])
        l1, l2, c, l3 = out["rounds"]
        self.assertEqual((l1["end_ts"], l2["end_ts"], c["end_ts"], l3["end_ts"]),
                         (epoch("04:05:00"), epoch("04:10:00"), epoch("04:15:00"), epoch("04:20:00")))
        self.assertEqual((l1["kind_number"], l2["kind_number"], c["kind_number"], l3["kind_number"]), (1, 2, 1, 3))
        self.assertEqual((l1["number"], l3["number"]), (1, 4))
        self.assertEqual(l1["output_tokens"], 5 + 7)
        self.assertEqual(l2["output_tokens"], 5 + 9)
        self.assertEqual(l3["output_tokens"], 5 + 11)
        self.assertEqual((l1["review_type"], l1["pr"], l1["range"], l1["result_chars"]),
                         ("reasoning", PR, "origin/main...HEAD", 40))
        self.assertEqual((l2["pr"], l2["range"], l2["result_chars"]), (None, "origin/main...feat", 10))

    def test_a_review_call_naming_no_pr_is_read_with_the_pr_left_empty(self):
        t = Transcript()
        t.review_call("04:00:00", None, result="r")
        out = self.rounds(t, merge="04:05:00")
        self.assertEqual(len(out["rounds"]), 1)
        self.assertIsNone(out["rounds"][0]["pr"])
        self.assertIsNone(out["rounds"][0]["range"])

    def test_a_review_call_naming_another_pr_is_not_read(self):
        t = Transcript()
        t.review_call("04:00:00", {"pr": 999})
        self.assertEqual(self.rounds(t, merge="04:05:00")["rounds"], [])

    def test_a_pr_with_only_copilot_rounds_has_no_local_round(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")])
        self.assertEqual({r["kind"] for r in self.rounds(t)["rounds"]}, {"copilot"})

    def test_a_same_review_poll_after_a_local_round_stays_inside_it(self):
        # Copilot has not posted yet: this fetch shows none, so it starts no round.
        t = Transcript()
        t.review_call("04:00:00", {"pr": PR})
        t.reviews_api("04:02:00", [])
        out = self.rounds(t, merge="04:05:00")
        self.assertEqual(len(out["rounds"]), 1)
        self.assertEqual(len(out["rounds"][0]["fetch_calls"]), 1)

    def test_an_orphan_fetch_at_the_same_instant_as_a_round_start_does_not_overlap_it(self):
        t = Transcript()
        t.bash("04:00:00", f"gh api repos/o/r/issues/{PR}/comments", json.dumps([{"id": 1, "body": "x"}]), u=usage(1, 1, 1, 3))
        t.review_call("04:00:00", {"pr": PR})
        t.assistant("04:01:00", u=usage(1, 1, 1, 4))
        out = self.rounds(t, merge="04:05:00")
        orphan = out["orphan_fetch_calls"][0]
        self.assertEqual(orphan["end_ts"], epoch("04:00:00"))
        self.assertEqual(orphan["messages"], 0)
        # Every message is counted in at most one of the orphan window and the rounds.
        self.assertEqual(orphan["output_tokens"] + out["rounds"][0]["output_tokens"], 5 + 3 + 4)

    def test_a_review_call_with_no_result_has_no_size(self):
        t = Transcript()
        t.call("04:00:00", R.REVIEW_TOOL, {"type": "verifier", "params": {"pr": PR}})
        self.assertIsNone(self.rounds(t, merge="04:05:00")["rounds"][0]["result_chars"])


class FeedbackMarker(_Base):
    """The marker `/cr-pr-feedback` prints (`pr-feedback-fetch: pr=<N> copilot-review=<id>`),
    read from a Bash call's tool_result, taken as written."""

    def marker(self, pr, review):
        return f"some output\npr-feedback-fetch: pr={pr} copilot-review={review}\n"

    def test_the_marker_alone_yields_a_round(self):
        # A command no whitelisted shape matches, and output that shows no review at all:
        # the marker names the PR and the newest Copilot review, and that is the round.
        t = Transcript()
        t.bash("04:01:00", "bash ./fetch-everything.sh 603", self.marker(PR, 5795366119), u=usage(1, 1, 1, 7))
        t.assistant("04:02:00", u=usage(1, 1, 1, 9))
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["5795366119"])
        rnd = out["rounds"][0]
        self.assertEqual(rnd["fetch_calls"][0]["sources"], [R.SRC_MARKER])
        self.assertTrue(rnd["fetch_calls"][0]["recognised"])
        self.assertEqual((rnd["messages"], rnd["output_tokens"]), (2, 7 + 9))
        self.assertEqual((out["fetch_calls"], out["skipped_fetches"]), (1, 0))

    def test_a_marker_beats_a_wrapped_command_that_would_otherwise_be_counted_unrecognised(self):
        t = Transcript()
        t.bash("04:01:00", f"timeout -s KILL 30 gh api repos/o/r/pulls/{PR}/reviews", self.marker(PR, "PRR_abc"))
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["PRR_abc"])
        self.assertEqual((out["fetch_calls"], out["skipped_fetches"]), (1, 0))

    def test_copilot_review_none_is_a_plain_fetch_call_and_starts_no_round(self):
        t = Transcript()
        t.bash("04:01:00", "./fetch.sh", self.marker(PR, "none"))
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual(out["rounds"], [])
        self.assertEqual(len(out["orphan_fetch_calls"]), 1)
        self.assertTrue(out["orphan_fetch_calls"][0]["recognised"])
        self.assertEqual(out["skipped_fetches"], 0)

    def test_a_marker_for_another_pr_is_not_this_prs_fetch_and_is_not_counted(self):
        t = Transcript()
        t.bash("04:01:00", "./fetch.sh", self.marker(604, 11))
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual((out["rounds"], out["fetch_calls"], out["skipped_fetches"]), ([], 0, 0))

    def test_two_markers_in_one_result_are_attributed_per_pr(self):
        t = Transcript()
        t.bash("04:01:00", "./fetch-both.sh", self.marker(PR, 11) + self.marker(604, 99))
        self.assertEqual([r["review_id"] for r in self.rounds(t, merge="04:10:00")["rounds"]], ["11"])
        self.assertEqual([r["review_id"] for r in self.rounds(t, merge="04:10:00", pr=604)["rounds"]], ["99"])

    def test_only_a_whole_well_formed_line_is_a_marker(self):
        for text in (f"echo pr-feedback-fetch: pr={PR} copilot-review=11",
                     f"pr-feedback-fetch: pr={PR} copilot-review=11 extra",
                     f"pr-feedback-fetch: pr={PR} copilot-review=",
                     f"pr-feedback-fetch: pr={PR} copilot-review=abc",
                     f"pr-feedback-fetch: pr={PR}",
                     f"Pr-Feedback-Fetch: pr={PR} copilot-review=11"):
            self.assertIsNone(R.marker_review(text, PR), text)
        self.assertEqual(R.marker_review(f"x\r\npr-feedback-fetch: pr={PR} copilot-review=11\r\n", PR), "11")
        self.assertIsNone(R.marker_review(None, PR))

    def test_a_marker_and_the_whitelist_name_one_review_once(self):
        # Round 1 by marker (id only), then the same review through a whitelisted fetch that
        # also shows its submission time: the known id keeps its identity, no second round.
        t = Transcript()
        t.bash("04:01:00", "./fetch.sh", self.marker(PR, 11))
        t.reviews_api("04:02:00", [rest_review(11, COPILOT, "03:27:21")])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual(len(out["rounds"]), 1)
        self.assertEqual(len(out["rounds"][0]["fetch_calls"]), 2)

    def test_the_skills_echo_call_is_read_from_its_tool_result(self):
        # What /cr-pr-feedback really produces: the four fetches, then the marker as its own
        # `echo` Bash call whose whole output is the line. One round, five fetch calls, and the
        # echo call is a recognised marker fetch.
        t = Transcript()
        t.bash("04:01:00", f"gh pr view {PR} --json number,title,body,state,mergeable,reviewDecision,reviews",
               json.dumps({"number": PR, "reviews": [view_review("PRR_kwabc", COPILOT, "03:27:21")]}))
        t.bash("04:01:05", f'gh api "repos/{{owner}}/{{repo}}/pulls/{PR}/reviews" --paginate',
               json.dumps([rest_review(5795366119, COPILOT, "03:27:21")]))
        t.bash("04:01:10", f'gh api "repos/{{owner}}/{{repo}}/pulls/{PR}/comments" --paginate', "[]")
        t.bash("04:01:15", f'gh api "repos/{{owner}}/{{repo}}/issues/{PR}/comments" --paginate', "[]")
        echo = t.bash("04:01:20", f'echo "pr-feedback-fetch: pr={PR} copilot-review=5795366119"',
                      f"pr-feedback-fetch: pr={PR} copilot-review=5795366119\n")
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["5795366119"])
        calls = out["rounds"][0]["fetch_calls"]
        self.assertEqual(len(calls), 5)
        marker_call = next(c for c in calls if c["tool_use_id"] == echo)
        self.assertEqual(marker_call["sources"], [R.SRC_MARKER])
        self.assertTrue(marker_call["recognised"])
        self.assertEqual((out["fetch_calls"], out["skipped_fetches"]), (5, 0))

    def test_the_echo_call_alone_is_a_round_when_no_fetch_shape_was_recognised(self):
        # The marker is the path meant to survive a change to how the fetches are issued.
        t = Transcript()
        t.bash("04:01:00", "bash ./some-new-fetch-wrapper.sh", "output the whitelist cannot read")
        t.bash("04:01:05", f'echo "pr-feedback-fetch: pr={PR} copilot-review=5795366119"',
               f"pr-feedback-fetch: pr={PR} copilot-review=5795366119\n")
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["5795366119"])
        self.assertEqual((out["fetch_calls"], out["skipped_fetches"]), (1, 0))

    def test_the_marker_in_assistant_text_is_not_read(self):
        # The old skill wording printed the line in the reply. That is an assistant message, not
        # a tool result, and the reader must not take a quoted marker from it (a reply, a doc
        # or an example could contain one): no round, no fetch call.
        t = Transcript()
        t.assistant("04:01:00", blocks=[{"type": "text", "text": f"pr-feedback-fetch: pr={PR} copilot-review=5795366119"}])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual((out["rounds"], out["fetch_calls"], out["skipped_fetches"]), ([], 0, 0))

    def test_a_marker_after_the_merge_is_dropped(self):
        t = Transcript()
        t.bash("04:30:00", "./fetch.sh", self.marker(PR, 11))
        out = self.rounds(t)
        self.assertEqual((out["rounds"], out["fetch_calls"]), ([], 0))


class UnrecognisedFetches(_Base):
    """A command that mentions a fetch of this PR but is no whitelisted shape is counted,
    never read, whatever its output holds."""

    def test_a_wrapped_fetch_is_counted_not_read(self):
        t = Transcript()
        t.bash("04:01:00", f"timeout -s KILL 30 gh api repos/o/r/pulls/{PR}/reviews",
               json.dumps([rest_review(11, COPILOT, "03:27:21")]))
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual(out["rounds"], [])
        self.assertEqual(out["orphan_fetch_calls"], [])
        self.assertEqual((out["fetch_calls"], out["skipped_fetches"]), (1, 1))

    def test_a_whitelisted_fetch_alone_yields_a_round_beside_unrecognised_ones(self):
        t = Transcript()
        t.bash("04:00:30", f"env GH_PAGER=cat gh pr view {PR} --json reviews", "{}")
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")])
        t.bash("04:02:00", f"if gh api repos/o/r/pulls/{PR}/reviews; then echo y; fi",
               json.dumps([rest_review(13, COPILOT, "04:01:30")]))
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])
        self.assertEqual((out["fetch_calls"], out["skipped_fetches"]), (3, 2))

    def test_an_unrecognised_fetch_after_the_merge_is_not_counted(self):
        t = Transcript()
        t.bash("04:30:00", f"timeout 30 gh api repos/o/r/pulls/{PR}/reviews", "[]")
        self.assertEqual(self.rounds(t)["fetch_calls"], 0)

    def test_a_command_that_does_not_name_this_prs_feedback_is_not_counted(self):
        t = Transcript()
        for cmd in (f"timeout 30 gh pr view {PR} --json state", "timeout 30 gh api repos/o/r/pulls/604/reviews",
                    f"echo reviews {PR}", f"git log --grep=pulls/{PR}/reviews"):
            t.bash("04:01:00", cmd, "[]")
        self.assertEqual(self.rounds(t)["fetch_calls"], 0)

    def test_a_whitelisted_fetch_of_another_pr_is_not_counted(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")], pr=604)
        out = self.rounds(t)
        self.assertEqual((out["rounds"], out["fetch_calls"], out["skipped_fetches"]), ([], 0, 0))


# One table for the whole recogniser: (label, command, want). `want` is None when the
# command is NOT a fetch (not whitelisted: it is counted as unrecognised, never parsed), or
# the list of sources of the first fetch the whitelist reads it as. A new shape is added
# here and to `_SHAPES` together. Nothing in the library reads quoting, so a row that
# "looks like" a fetch inside quotes, a substitution or a wrapper is simply not on the list.
_V, _AR, _AP, _AI = R.SRC_PR_VIEW, R.SRC_API_REVIEWS, R.SRC_API_PULL_COMMENTS, R.SRC_API_ISSUE_COMMENTS
_VC = R.SRC_PR_VIEW_COMMENTS
_JQ = r"""--jq '.[] | "#\(.id) [\(.user.login)] \(.state) commit=\(.commit_id[0:7])"'"""
_REV = "gh api repos/o/r/pulls/603/reviews"
_PCOM = "gh api repos/o/r/pulls/603/comments"
_ICOM = "gh api repos/o/r/issues/603/comments"
_VIEW = "gh pr view 603 --json reviews"
_REV4 = "gh api repos/o/r/pulls/604/reviews"
_VIEW4 = "gh pr view 604 --json reviews"
_BS = "\\"
# The three commands of the session that worked one pull request, whole, as that transcript has them.
_LEGACY_1 = r"""R=repos/Donelle/claude-runway; echo "=== INLINE"; gh api $R/pulls/603/comments --paginate --jq '.[] | "#\(.id) [\(.user.login)] \(.path):\(.line // .original_line) reply_to=\(.in_reply_to_id // "-") commit=\(.commit_id[0:7])\n\(.body)\n"'; echo "=== REVIEWS"; gh api $R/pulls/603/reviews --paginate --jq '.[] | "#\(.id) [\(.user.login)] \(.state) commit=\(.commit_id[0:7])\n\(.body)\n"'; echo "=== ISSUE COMMENTS"; gh api $R/issues/603/comments --paginate --jq '.[] | "#\(.id) [\(.user.login)] \(.created_at)\n\(.body[0:300])\n"'"""
_LEGACY_2 = r"""git remote get-url origin; R=repos/Donelle/claude-runway; gh pr view 603 -R Donelle/claude-runway --json title,state,mergeable,mergeStateStatus,reviewDecision,headRefOid --jq '"\(.state) head=\(.headRefOid[0:7]) \(.mergeStateStatus) mergeable=\(.mergeable) decision=\(.reviewDecision)"'; echo "=== CHECKS"; gh pr checks 603 -R Donelle/claude-runway 2>&1 | cut -c1-70; echo "=== REVIEWS"; gh api $R/pulls/603/reviews --paginate --jq '.[] | "#\(.id) [\(.user.login)] \(.state) commit=\(.commit_id[0:7]) \(.submitted_at)"'; echo "=== INLINE (all)"; gh api $R/pulls/603/comments --paginate --jq '.[] | "#\(.id) [\(.user.login)] \(.path):\(.line // .original_line) reply_to=\(.in_reply_to_id // "-") commit=\(.commit_id[0:7])"'; echo "=== ISSUE COMMENTS"; gh api $R/issues/603/comments --paginate --jq '.[] | "#\(.id) [\(.user.login)] \(.created_at)"'; echo "=== THREADS"; gh api graphql -f query='query { repository(owner:"Donelle", name:"claude-runway") { pullRequest(number:603) { reviewThreads(first:30) { nodes { id isResolved isOutdated comments(first:1) { nodes { databaseId path } } } } } } }' --jq '.data.repository.pullRequest.reviewThreads.nodes[] | "\(.id) resolved=\(.isResolved) outdated=\(.isOutdated) \(.comments.nodes[0].path)"'"""
_LEGACY_3 = r"""R=repos/Donelle/claude-runway; gh pr view 603 -R Donelle/claude-runway --json state,mergeStateStatus,mergeable,headRefOid --jq '"\(.state) head=\(.headRefOid[0:7]) \(.mergeStateStatus) mergeable=\(.mergeable)"'; gh pr checks 603 -R Donelle/claude-runway 2>&1 | cut -c1-60; echo "=== reviews"; gh api $R/pulls/603/reviews --paginate --jq '.[] | "#\(.id) [\(.user.login)] \(.state) commit=\(.commit_id[0:7]) \(.submitted_at)"'; echo "=== threads"; gh api graphql -f query='query { repository(owner:"Donelle", name:"claude-runway") { pullRequest(number:603) { reviewThreads(first:30) { nodes { isResolved } } } } }' --jq '[.data.repository.pullRequest.reviewThreads.nodes[].isResolved] | "\(length) threads, \(map(select(.)) | length) resolved"'; echo "=== .ai-plans"; ls -la .ai-plans | cut -c1-120"""
# What a skill's step issues (the autowork three-endpoint fetch, `/cr-pr-feedback`'s four).
_SKILL_FOUR = "\n".join([
    "gh pr view 603 --json number,title,body,state,mergeable,reviewDecision,reviews",
    'gh api "repos/{owner}/{repo}/pulls/603/reviews" --paginate',
    'gh api "repos/{owner}/{repo}/pulls/603/comments" --paginate',
    'gh api "repos/{owner}/{repo}/issues/603/comments" --paginate'])

COMMAND_TABLE = [
    # --- Fetches: the whitelisted shapes, with and without --paginate / --jq.
    ("api reviews", _REV, [_AR]),
    ("api reviews, paginate", f"{_REV} --paginate", [_AR]),
    ("api reviews, jq", f"{_REV} {_JQ}", [_AR]),
    ("api reviews, paginate and jq", f"{_REV} --paginate {_JQ}", [_AR]),
    ("api reviews, a bare jq word", f"{_REV} --jq length", [_AR]),
    ("api pull comments", _PCOM, [_AP]),
    ("api pull comments, paginate and jq", f"{_PCOM} --paginate {_JQ}", [_AP]),
    ("api issue comments", _ICOM, [_AI]),
    ("api issue comments, paginate and jq", f"{_ICOM} --paginate {_JQ}", [_AI]),
    ("api, the skill's quoted placeholder endpoint", 'gh api "repos/{owner}/{repo}/pulls/603/reviews" --paginate', [_AR]),
    ("api, owner and repo with dots, dashes and digits", "gh api repos/Donelle/claude-runway.v2/issues/603/comments", [_AI]),
    ("pr view reviews", _VIEW, [_V]),
    ("pr view, -R and several fields", "gh pr view 603 -R o/r --json number,title,reviews,state", [_V]),
    # A `comments` field alone is a comments fetch (no review id is read from it); with `reviews` it is a reviews fetch.
    ("pr view comments", "gh pr view 603 --json comments", [_VC]),
    ("pr view comments, jq printing the bodies", "gh pr view 603 --json comments --jq '.comments[].body'", [_VC]),
    ("pr view comments, -R and several fields", "gh pr view 603 -R o/r --json number,comments,state", [_VC]),
    ("pr view latestReviews and comments", "gh pr view 603 --json latestReviews,comments", [_VC]),
    ("pr view reviews and comments", "gh pr view 603 --json comments,reviews", [_V, _VC]),
    ("pr view reviews and comments, jq printing the bodies", "gh pr view 603 --json comments,reviews --jq '.comments[].body'", [_V, _VC]),
    ("pr view, the skill's field list", "gh pr view 603 --json number,title,body,state,mergeable,reviewDecision,reviews", [_V]),
    ("pr view, jq", f"{_VIEW} --jq .reviews", [_V]),
    ("pr view, quoted jq", f"{_VIEW} {_JQ}", [_V]),
    # --- Fetches: the one chain the transcripts hold (autowork's three endpoints), and the skill's block.
    ("three endpoints, ;", f"{_REV}; {_PCOM}; {_ICOM}", [_AR, _AP, _AI]),
    ("three endpoints, &&", f"{_REV} && {_PCOM} && {_ICOM}", [_AR, _AP, _AI]),
    ("three endpoints, paginate and jq", f"{_REV} --paginate {_JQ}; {_PCOM} --paginate {_JQ}; {_ICOM} --paginate {_JQ}", [_AR, _AP, _AI]),
    ("the skill's four commands", _SKILL_FOUR, [_V, _AR, _AP, _AI]),
    ("the skill's four commands with ;", _SKILL_FOUR.replace("\n", "; "), [_V, _AR, _AP, _AI]),
    # --- Fetches: that session's own compound commands, whole.
    ("session 603, comments reviews issue-comments with labels", _LEGACY_1, [_AP, _AR, _AI]),
    ("session 603, status checks reviews comments threads", _LEGACY_2, [_AR, _AP, _AI]),
    ("session 603, status checks reviews threads and a listing", _LEGACY_3, [_AR]),
    # --- NOT fetches: a wrapper in front of a whitelisted fetch. The open review finding is first.
    ("timeout with a signal and a value", "timeout -s KILL 30 gh api repos/o/r/pulls/603/reviews", None),
    ("timeout with a long signal option", f"timeout --signal KILL 30 {_REV}", None),
    ("timeout with -k", f"timeout -k 5 30 {_REV}", None),
    ("timeout with a unit", f"timeout 30s {_REV}", None),
    ("timeout", f"timeout 30 {_REV}", None),
    ("timeout around echo", f"timeout 30 echo {_REV}", None),
    ("env with an assignment", f"env GH_PAGER=cat {_VIEW}", None),
    ("env with a flag", f"env -i {_VIEW}", None),
    ("env with a flag that takes a value", f"env -u GH_TOKEN {_REV}", None),
    ("command", f"command {_REV}", None),
    ("command with a flag", f"command -p {_REV}", None),
    ("nohup", f"nohup {_REV}", None),
    ("an assignment prefix", f"GH_PAGER=cat {_REV}", None),
    ("a cd prefix", f"cd /x && {_REV}", None),
    ("a bare word before the command", f"time {_REV}", None),
    # --- NOT fetches: guards.
    ("if", f"if {_REV}; then echo y; fi", None),
    ("while", f"while ! {_VIEW}; do sleep 1; done", None),
    ("until", f"until {_ICOM}; do sleep 1; done", None),
    ("a for loop", "for n in 603; do gh api repos/o/r/pulls/$n/reviews; done", None),
    ("a negation", f"! {_REV}", None),
    # --- NOT fetches: quoted and escaped literals.
    ("single-quoted whole command", f"echo '{_VIEW}'", None),
    ("double-quoted whole command", f'echo "{_REV}"', None),
    ("printf of a quoted string", f"printf '%s' '{_REV}'", None),
    ("echo of the words", f"echo {_VIEW}", None),
    ("an escaped $( )", f"echo {_BS}$({_VIEW})", None),
    ("an escaped command word", f"{_BS}{_VIEW}", None),
    ("a quoted ; before the command", f'echo ";" {_VIEW}', None),
    ("a quoted && before the command", f"echo '&&' {_VIEW}", None),
    ("an escaped ; before the command", f"echo {_BS}; {_VIEW}", None),
    ("a comment holding it", f"# {_REV}", None),
    ("a trailing comment", f"{_REV} # note", None),
    # --- NOT fetches: substitutions and the skill's own captured-ids form.
    ("assignment from $( )", f"X=$({_REV})", None),
    ("argument from $( )", f"echo $({_VIEW})", None),
    ("double-quoted $( )", f'echo "$({_VIEW})"', None),
    ("assignment from backticks", f"X=`{_REV}`", None),
    ("double-quoted backticks", f'echo "`{_REV}`"', None),
    ("nested $( )", f'echo "$(echo $({_REV}))"', None),
    ("an unclosed $(", f'echo "$({_REV}"', None),
    ("the autowork step's captured ids", f'COMMENT_IDS=$({_PCOM} --paginate --jq ".[] | select(.user.login != \\"$ME\\") | \\"comment:\\" + (.id|tostring)") || FETCH_FAILED=1', None),
    ("a shell variable for the repository", "R=repos/o/r; gh api $R/pulls/603/reviews", None),
    # --- NOT fetches: pipelines and redirects.
    ("pipe to jq", f"{_REV} | jq .", None),
    ("pipe to head", f"{_VIEW} | head", None),
    ("pipe into it", f"echo x | {_REV}", None),
    ("paginate then a pipe", f"{_REV} --paginate | jq length", None),
    ("a redirect", f"{_REV} > /tmp/out.json", None),
    ("stderr to stdout", f"{_REV} 2>&1", None),
    ("a process substitution", f"diff <({_REV}) <({_REV4})", None),
    ("a subshell", f"({_REV})", None),
    # --- NOT fetches: chains other than the listed ones.
    ("&& chain with another command", f"cd /x && {_REV} && {_ICOM}", None),
    ("; chain of two reviews fetches", f"{_VIEW}; {_REV}", None),
    ("|| chain", f"{_REV} || {_ICOM}", None),
    ("two PRs chained", f"{_REV}; {_REV4}", None),
    ("two PRs chained, pr view", f"{_VIEW} && {_VIEW4}", None),
    ("three endpoints in another order", f"{_PCOM}; {_REV}; {_ICOM}", None),
    ("three endpoints of two PRs", f"{_REV}; gh api repos/o/r/pulls/604/comments; {_ICOM}", None),
    ("three endpoints with a fourth", f"{_REV}; {_PCOM}; {_ICOM}; git status", None),
    ("a second line", f"git fetch\n{_REV}", None),
    ("a trailing newline", f"{_REV}\n", None),
    ("a trailing space", f"{_REV} ", None),
    # --- NOT fetches: heredocs and here-strings.
    ("heredoc, bare delimiter", f"cat <<EOF\n{_VIEW}\nEOF", None),
    ("heredoc, quoted delimiter", f"cat <<'EOF'\n{_REV}\nEOF", None),
    ("heredoc fed to gh", f"gh api graphql --input - <<EOF\n{{}}\nEOF\n{_REV}", None),
    ("here-string", f"cat <<< '{_VIEW}'", None),
    # --- NOT fetches: a shape nobody lists.
    ("api with -X GET and a field", "gh api -X GET repos/o/r/pulls/603/reviews -f per_page=100", None),
    ("api with a query string", "gh api repos/o/r/issues/603/comments?per_page=100", None),
    ("api with --jq before --paginate", f"{_REV} {_JQ} --paginate", None),
    ("api, a multi-line jq program", f"{_REV} --jq '.[] |\n  select(.x)'", None),
    ("api, a jq program with an embedded single quote", f"""{_REV} --jq '.[] | "it'"'"'s"'""", None),
    ("api, a reply (a write that names the comments)", "gh api repos/o/r/pulls/603/comments/123/replies --method POST -f body=x", None),
    ("api, a single review", "gh api repos/o/r/pulls/603/reviews/5795628397 --jq .body", None),
    ("api, a single review's comments", "gh api repos/o/r/pulls/603/reviews/5795628397/comments", None),
    ("pr view, the number after the flags", "gh pr view --json reviews 603", None),
    ("pr view, a URL selector", "gh pr view https://github.com/o/r/pull/603 --json reviews", None),
    ("pr view, --comments", "gh pr view 603 --comments", None),
    ("pr view, the equals form", "gh pr view 603 --json=reviews", None),
    ("pr view, an unknown flag", "gh pr view 603 --web --json reviews", None),
    ("gh pr comment, a write", "gh pr comment 603 --body 'reviews look fine'", None),
]

# Mentions no feedback fetch of the pull request at all: not whitelisted and not counted either.
NOT_MENTIONED = [
    ("latestReviews only", "gh pr view 603 --json latestReviews"),
    ("a pr view that asks for neither", "gh pr view 603 --json state,mergeable,reviewDecision"),
    ("another PR's fetch", _REV4),
    ("another PR's pr view", _VIEW4),
    ("no gh", "git log --grep=pulls/603/reviews"),
    ("a graphql query", "gh api graphql -f query='query { pullRequest(number: 603) { id } }'"),
    ("an empty command", ""),
    ("no command", None),
]


_H11 = f"#11 [{COPILOT}] COMMENTED commit=abc1234 2026-11-06T03:27:21Z"
# What those commands print: each `echo` label, then that part's output.
ROW_OUTPUT = {
    "session 603, comments reviews issue-comments with labels":
        f"=== INLINE\n=== REVIEWS\n{_H11}\nthe body\n=== ISSUE COMMENTS\n",
    "session 603, status checks reviews comments threads":
        f"OPEN head=abc\n=== CHECKS\nok\n=== REVIEWS\n{_H11}\n=== INLINE (all)\n=== ISSUE COMMENTS\n=== THREADS\n",
    "session 603, status checks reviews threads and a listing":
        f"OPEN head=abc\nok\n=== reviews\n{_H11}\n=== threads\n1 threads, 1 resolved\n=== .ai-plans\nx\n",
}


class FetchCommandTable(_Base):
    def test_every_row_against_the_whitelist(self):
        for label, cmd, want in COMMAND_TABLE:
            with self.subTest(label):
                got = R.whitelisted_fetch(cmd)
                if want is None:
                    self.assertIsNone(got, cmd)
                    # Not parsed, not guessed at, but counted.
                    self.assertTrue(R.mentions_unread_fetch(cmd, PR), cmd)
                else:
                    self.assertEqual(got, (PR, want), cmd)

    def test_every_row_end_to_end(self):
        # A fetch row yields a round from its output when it fetches reviews (a comments-only
        # fetch carries no review record, so it yields none); a not-a-fetch row yields none,
        # however good its output, and is counted once in the unrecognised-fetch count.
        review = json.dumps([rest_review(11, COPILOT, "03:27:21")])
        for label, cmd, want in COMMAND_TABLE:
            with self.subTest(label):
                t = Transcript()
                t.bash("04:01:00", cmd, ROW_OUTPUT.get(label, review))
                out = self.rounds(t, merge="04:10:00")
                if want is None:
                    self.assertEqual((out["rounds"], out["fetch_calls"], out["skipped_fetches"]), ([], 1, 1), cmd)
                elif R.SRC_API_REVIEWS in want or R.SRC_PR_VIEW in want:
                    self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"], cmd)
                    self.assertEqual(out["rounds"][0]["fetch_calls"][0]["sources"], want)
                    self.assertEqual((out["fetch_calls"], out["skipped_fetches"]), (1, 0), cmd)
                else:
                    self.assertEqual(out["rounds"], [], cmd)
                    self.assertEqual(out["orphan_fetch_calls"][0]["sources"], want)
                    self.assertEqual((out["fetch_calls"], out["skipped_fetches"]), (1, 0), cmd)

    def test_rows_that_mention_no_fetch_of_this_pr_are_not_counted(self):
        for label, cmd in NOT_MENTIONED:
            with self.subTest(label):
                self.assertFalse(R.mentions_unread_fetch(cmd, PR), cmd)
                t = Transcript()
                t.bash("04:01:00", cmd, json.dumps([rest_review(11, COPILOT, "03:27:21")]))
                out = self.rounds(t, merge="04:10:00")
                self.assertEqual((out["rounds"], out["fetch_calls"], out["skipped_fetches"]), ([], 0, 0), cmd)

    def test_the_open_review_finding_is_a_row(self):
        # The command review 5808935880 reported: `-s KILL`'s value was read as the command.
        # Nothing is parsed now, so it is simply not a fetch, and it is counted.
        cmd = "timeout -s KILL 30 gh api repos/o/r/pulls/603/reviews"
        self.assertIn(("timeout with a signal and a value", cmd, None), COMMAND_TABLE)
        self.assertIsNone(R.whitelisted_fetch(cmd))
        self.assertTrue(R.mentions_unread_fetch(cmd, PR))

    def test_the_whitelist_binds_one_pr_per_shape(self):
        # The PR number is bound by its first mention in a shape; a later one must repeat it.
        self.assertEqual(R.whitelisted_fetch(f"{_REV}; {_PCOM}; {_ICOM}"), (PR, [_AR, _AP, _AI]))
        self.assertIsNone(R.whitelisted_fetch(f"{_REV}; gh api repos/o/r/pulls/60/comments; {_ICOM}"))
        self.assertEqual(R.whitelisted_fetch("gh api repos/o/r/pulls/6030/reviews"), (6030, [_AR]))

    def test_labels_are_unique(self):
        labels = [label for label, _, _ in COMMAND_TABLE]
        self.assertEqual(len(labels), len(set(labels)))

    def test_an_absurd_pr_number_is_no_fetch_and_raises_nothing(self):
        # int() of a 19000-digit string raises; the number is bounded before it is converted.
        digits = "9" * 19000
        self.assertIsNone(R.whitelisted_fetch(f"gh api repos/o/r/pulls/{digits}/reviews"))
        self.assertIsNone(R.marker_review(f"pr-feedback-fetch: pr={digits} copilot-review=11", PR))
        t = Transcript()
        t.bash("04:01:00", f"gh api repos/o/r/pulls/{digits}/reviews", "[]")
        t.reviews_api("04:02:00", [rest_review(11, COPILOT, "03:27:21")])
        out = self.rounds(t)
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["rounds"]), 1)

    def test_a_very_long_command_is_not_matched_and_costs_nothing(self):
        cmd = f"{_REV} --jq '" + "x" * 30000 + "'"
        self.assertIsNone(R.whitelisted_fetch(cmd))
        self.assertFalse(R.mentions_unread_fetch(cmd, PR))

    def test_non_text_commands_are_not_fetches(self):
        for bad in (None, 5, ["gh"], b"gh api"):
            self.assertIsNone(R.whitelisted_fetch(bad))
            self.assertFalse(R.mentions_unread_fetch(bad, PR))


class BodiesAreNotRecords(_Base):
    """A comment or review body can quote a review header or a JSON example. Review records
    are read only from the part of an output the reviews fetch produced, and an output that
    cannot be told apart is not recognised (never scanned)."""

    REAL = f"#11 [{COPILOT}] COMMENTED commit=abc1234 2026-11-06T03:27:21Z"
    QUOTED = f"#99 [{COPILOT}] COMMENTED commit=fff9999 2026-11-06T09:00:00Z"
    JQ = """--jq '.[] | "#\\(.id) [\\(.user.login)] \\(.state) commit=\\(.commit_id[0:7])\\n\\(.body)\\n"'"""

    def one(self, cmd, result, pr=PR):
        t = Transcript()
        t.bash("04:01:00", cmd, result)
        return self.rounds(t, merge="04:10:00", pr=pr)

    def test_a_quoted_header_in_a_printed_body_is_not_a_phantom_round(self):
        out = self.one(f"gh api repos/o/r/pulls/{PR}/reviews --paginate {self.JQ}",
                       self.REAL + "\nsee this one:\n" + self.QUOTED + "\n")
        self.assertEqual((out["rounds"], out["fetch_calls"], out["skipped_fetches"]), ([], 1, 1))

    def test_a_quoted_json_example_in_a_printed_body_is_not_a_phantom_round(self):
        quoted = json.dumps([rest_review(77, COPILOT, "09:00:00")])
        out = self.one(f"gh api repos/o/r/pulls/{PR}/reviews --paginate {self.JQ}",
                       self.REAL + "\nexample payload:\n" + quoted + "\n")
        self.assertEqual((out["rounds"], out["skipped_fetches"]), ([], 1))

    def test_raw_json_keeps_a_quoted_header_inside_its_string(self):
        body = f"see this one:\n{self.QUOTED}\n"
        review = {**rest_review(11, COPILOT, "03:27:21"), "body": body}
        out = self.one(f"gh api repos/o/r/pulls/{PR}/reviews --paginate", json.dumps([review]))
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])

    def test_the_three_endpoints_as_text_are_ambiguous(self):
        chain = "; ".join(f"gh api repos/o/r/{k}/{PR}/{leaf} --jq '.[]'"
                          for k, leaf in (("pulls", "reviews"), ("pulls", "comments"), ("issues", "comments")))
        out = self.one(chain, self.REAL + "\nsome inline text\n" + self.QUOTED + "\n")
        self.assertEqual((out["rounds"], out["skipped_fetches"]), ([], 1))

    def test_the_three_endpoints_as_raw_json_are_read(self):
        chain = "; ".join(f"gh api repos/o/r/{k}/{PR}/{leaf} --paginate"
                          for k, leaf in (("pulls", "reviews"), ("pulls", "comments"), ("issues", "comments")))
        comment = {"id": 5, "user": {"login": "x"}, "body": self.QUOTED, "path": "a.py", "commit_id": "abc"}
        out = self.one(chain, "\n".join([json.dumps([rest_review(11, COPILOT, "03:27:21")]),
                                         json.dumps([comment]), json.dumps([{"id": 6, "body": self.QUOTED}])]))
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])

    def test_a_comments_only_fetch_never_yields_a_review(self):
        out = self.one(f"gh api repos/o/r/issues/{PR}/comments --paginate --jq '.[]'",
                       "a comment that pastes:\n" + self.QUOTED + "\n")
        self.assertEqual(out["rounds"], [])
        self.assertEqual((out["fetch_calls"], out["skipped_fetches"]), (1, 0))
        self.assertEqual(len(out["orphan_fetch_calls"]), 1)

    LEGACY = ('R=repos/o/r; echo "=== INLINE"; gh api $R/pulls/603/comments --paginate --jq \'.[]\'; '
              'echo "=== REVIEWS"; gh api $R/pulls/603/reviews --paginate --jq \'.[]\'; '
              'echo "=== ISSUE COMMENTS"; gh api $R/issues/603/comments --paginate --jq \'.[]\'')

    def test_a_labelled_command_reads_only_the_reviews_section(self):
        out = self.one(self.LEGACY, "=== INLINE\ninline text\n" + self.QUOTED + "\n=== REVIEWS\n" + self.REAL
                       + "\n=== ISSUE COMMENTS\npasted:\n" + self.QUOTED + "\n")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])

    RAW = ('R=repos/o/r; echo "=== INLINE"; gh api $R/pulls/603/comments --paginate; '
           'echo "=== REVIEWS"; gh api $R/pulls/603/reviews --paginate; '
           'echo "=== ISSUE COMMENTS"; gh api $R/issues/603/comments --paginate')

    def raw_output(self, reviews_section):
        comment = json.dumps([{"id": 5, "body": self.QUOTED, "user": {"login": "u"}}])
        return f"=== INLINE\n{comment}\n=== REVIEWS\n{reviews_section}\n=== ISSUE COMMENTS\n{comment}\n"

    def test_a_labelled_command_without_jq_reads_its_reviews_section_as_raw_json(self):
        out = self.one(self.RAW, self.raw_output(json.dumps([rest_review(11, COPILOT, "03:27:21")])))
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])
        self.assertEqual((out["fetch_calls"], out["skipped_fetches"]), (1, 0))

    def test_raw_json_in_the_reviews_section_may_carry_a_quoted_header_in_a_string(self):
        review = {**rest_review(11, COPILOT, "03:27:21"), "body": f"look:\n{self.QUOTED}\n"}
        out = self.one(self.RAW, self.raw_output(json.dumps([review])))
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])

    def test_text_in_a_raw_json_reviews_section_is_ambiguous(self):
        out = self.one(self.RAW, self.raw_output(json.dumps([rest_review(11, COPILOT, "03:27:21")]) + "\n" + self.QUOTED))
        self.assertEqual((out["rounds"], out["skipped_fetches"]), ([], 1))

    def test_a_labelled_command_with_jq_is_still_read_as_lines(self):
        # The same commands with a jq program: the section holds lines, not JSON.
        out = self.one(self.LEGACY, "=== INLINE\n=== REVIEWS\n" + self.REAL + "\n=== ISSUE COMMENTS\n")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])

    def test_a_label_that_appears_twice_makes_the_output_ambiguous(self):
        out = self.one(self.LEGACY, "=== INLINE\n=== REVIEWS\n" + self.REAL
                       + "\n=== ISSUE COMMENTS\npasted:\n=== REVIEWS\n" + self.QUOTED + "\n")
        self.assertEqual((out["rounds"], out["skipped_fetches"]), ([], 1))

    def test_a_missing_label_makes_the_output_ambiguous(self):
        out = self.one(self.LEGACY, "=== INLINE\n" + self.REAL + "\n")
        self.assertEqual((out["rounds"], out["skipped_fetches"]), ([], 1))

    def test_an_empty_reviews_section_is_recognised_with_no_review(self):
        out = self.one(self.LEGACY, "=== INLINE\n=== REVIEWS\n=== ISSUE COMMENTS\n")
        self.assertEqual((out["rounds"], out["skipped_fetches"]), ([], 0))
        self.assertEqual(len(out["orphan_fetch_calls"]), 1)


class CommentsAreNotReviewSources(_Base):
    """Only a `reviews` fetch (or the marker) can define a round or supply a review id. A
    comments fetch is a plain fetch call inside the current round and its output is never read
    for review ids, whitelisted or not."""

    HEADER = f"#99 [{COPILOT}] COMMENTED commit=fff9999 2026-11-06T09:00:00Z"

    def one(self, cmd, result):
        t = Transcript()
        t.bash("04:01:00", cmd, result)
        return self.rounds(t, merge="04:10:00")

    def test_a_pr_view_comments_fetch_printing_a_pasted_header_is_no_round(self):
        # The finding of review 5809350361: a printed comment body that is one review-header line.
        out = self.one(f"gh pr view {PR} --json comments --jq '.comments[].body'", self.HEADER)
        self.assertEqual((out["rounds"], out["fetch_calls"], out["skipped_fetches"]), ([], 1, 0))
        self.assertEqual(out["orphan_fetch_calls"][0]["sources"], [R.SRC_PR_VIEW_COMMENTS])

    def test_a_pr_view_comments_fetch_of_raw_json_is_no_round_either(self):
        review_shaped = json.dumps({"comments": [{"body": "x"}], "reviews": [view_review("PRR_z", COPILOT, "09:00:00")]})
        out = self.one(f"gh pr view {PR} --json comments", review_shaped)
        self.assertEqual(out["rounds"], [])
        self.assertEqual(out["skipped_fetches"], 0)

    def test_issue_and_inline_comment_fetches_are_no_round(self):
        for leaf in ("issues", "pulls"):
            out = self.one(f"gh api repos/o/r/{leaf}/{PR}/comments --jq '.[].body'", self.HEADER)
            self.assertEqual((out["rounds"], out["fetch_calls"], out["skipped_fetches"]), ([], 1, 0), leaf)

    def test_a_comments_fetch_is_kept_inside_the_current_round(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")])
        t.bash("04:02:00", f"gh pr view {PR} --json comments --jq '.comments[].body'", self.HEADER)
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])
        self.assertEqual([c["sources"] for c in out["rounds"][0]["fetch_calls"]], [[R.SRC_API_REVIEWS], [R.SRC_PR_VIEW_COMMENTS]])

    def test_a_comments_fetch_with_no_result_is_still_skipped(self):
        t = Transcript()
        t.bash("04:01:00", f"gh pr view {PR} --json comments")
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual((out["rounds"], out["fetch_calls"], out["skipped_fetches"]), ([], 1, 1))

    def test_an_unlabelled_chain_with_comments_reads_raw_json_only(self):
        # Reviews and comments in one unlabelled output: a printed line may be a comment that
        # pastes a header, so a line is not read as a review there.
        chain = "; ".join(f"gh api repos/o/r/{k}/{PR}/{leaf} --jq '.[]'"
                          for k, leaf in (("pulls", "reviews"), ("pulls", "comments"), ("issues", "comments")))
        out = self.one(chain, BodiesAreNotRecords.REAL + "\n" + self.HEADER + "\n")
        self.assertEqual((out["rounds"], out["fetch_calls"], out["skipped_fetches"]), ([], 1, 1))

    def test_a_reviews_only_fetch_still_reads_lines(self):
        out = self.one(f"gh api repos/o/r/pulls/{PR}/reviews --jq '.[]'", BodiesAreNotRecords.REAL + "\n")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])

    def test_pr_view_with_reviews_and_comments_is_a_reviews_fetch_for_raw_json(self):
        body = {"reviews": [view_review("PRR_a", COPILOT, "03:27:21")], "comments": [{"body": self.HEADER}]}
        out = self.one(f"gh pr view {PR} --json comments,reviews", json.dumps(body))
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["PRR_a"])

    def test_pr_view_with_reviews_and_comments_and_a_jq_printing_bodies_is_not_read(self):
        # The finding of review 5809442797: the comment bodies are printed beside the reviews,
        # and a pasted header among them cannot be told from a review record.
        out = self.one(f"gh pr view {PR} --json comments,reviews --jq '.comments[].body'", self.HEADER)
        self.assertEqual((out["rounds"], out["fetch_calls"], out["skipped_fetches"]), ([], 1, 1))
        # With the reviews field alone the same printed line is a review line, as before.
        out = self.one(f"gh pr view {PR} --json reviews --jq '.reviews[]'", BodiesAreNotRecords.REAL)
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])

    def test_a_comments_object_is_not_a_recognised_review_shape(self):
        self.assertIsNone(R.read_reviews(json.dumps({"comments": [{"body": "x"}]})))


class ReviewIdentity(_Base):
    """One Copilot review is one round whichever id form a fetch shows: the numeric REST id is
    the identity once any fetch has shown it, and a `PRR_` id (the marker's, `gh pr view`'s)
    is an alias that stands alone only when no numeric form was ever seen."""

    def marker(self, review):
        return f"pr-feedback-fetch: pr={PR} copilot-review={review}\n"

    def skill_fetch(self, t, ts, view_id, rest_id, marker_id, when="03:27:21"):
        """What `/cr-pr-feedback` prints: its four commands' outputs, then the marker."""
        parts = [json.dumps({"number": PR, "reviews": [view_review(view_id, COPILOT, when)]}),
                 json.dumps([rest_review(rest_id, COPILOT, when)] if rest_id else []), "[]", "[]"]
        return t.bash(ts, _SKILL_FOUR, "\n".join(parts) + "\n" + self.marker(marker_id))

    def test_a_marker_prr_id_and_a_later_rest_numeric_id_are_one_round(self):
        # The finding of review 5809350361: the marker named PRR_..., the output it came with
        # showed the review by both ids, and a later REST fetch shows only the numeric one.
        t = Transcript()
        self.skill_fetch(t, "04:01:00", "PRR_abc", 5795366119, "PRR_abc")
        t.reviews_api("04:02:00", [rest_review(5795366119, COPILOT, "03:27:21")])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual(len(out["rounds"]), 1)
        rnd = out["rounds"][0]
        self.assertEqual(rnd["review_id"], "5795366119")
        self.assertEqual(sorted(rnd["review_ids"]), ["5795366119", "PRR_abc"])
        self.assertEqual(len(rnd["fetch_calls"]), 2)

    def test_the_marker_keeps_the_submission_time_its_output_shows(self):
        t = Transcript()
        self.skill_fetch(t, "04:01:00", "PRR_abc", None, "PRR_abc")
        rnd = self.rounds(t, merge="04:10:00")["rounds"][0]
        self.assertEqual((rnd["review_id"], rnd["submitted_at"], rnd["commit_id"]), ("PRR_abc", iso("03:27:21"), "abc1234"))
        # And the time is what ties it to a later REST fetch of the same review.
        t.reviews_api("04:02:00", [rest_review(11, COPILOT, "03:27:21")])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])
        self.assertEqual(len(out["rounds"]), 1)

    def test_a_marker_with_an_unlisted_command_still_ties_raw_json_output(self):
        t = Transcript()
        view = json.dumps({"number": PR, "reviews": [view_review("PRR_abc", COPILOT, "03:27:21")]})
        t.bash("04:01:00", "./fetch.sh", view + "\n" + self.marker("PRR_abc"))
        t.reviews_api("04:02:00", [rest_review(11, COPILOT, "03:27:21")])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])

    def test_a_marker_prr_id_alone_stands_alone(self):
        # No fetch ever showed a numeric form (or a time): the PRR_ id is the identity.
        t = Transcript()
        t.bash("04:01:00", "./fetch.sh", self.marker("PRR_abc"))
        t.reviews_view("04:02:00", [view_review("PRR_abc", COPILOT, "03:27:21")])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["PRR_abc"])

    def test_a_round_started_by_a_prr_id_is_keyed_by_the_numeric_id_once_shown(self):
        t = Transcript()
        t.reviews_view("04:01:00", [view_review("PRR_abc", COPILOT, "03:27:21")])
        t.reviews_api("04:02:00", [rest_review(11, COPILOT, "03:27:21")])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([(r["review_id"], r["review_ids"]) for r in out["rounds"]], [("11", ["PRR_abc", "11"])])

    def test_a_numeric_id_stays_the_round_id_when_the_prr_form_comes_later(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")])
        t.bash("04:02:00", "./fetch.sh", self.marker("PRR_abc") + json.dumps(
            {"number": PR, "reviews": [view_review("PRR_abc", COPILOT, "03:27:21")]}))
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])

    def test_two_different_reviews_are_not_tied_by_a_marker(self):
        # The output shows a newer review than the one the marker names: the marker wins, and
        # the output cannot start a round of its own or rename the review.
        t = Transcript()
        both = json.dumps([rest_review(11, COPILOT, "03:27:21"), rest_review(13, COPILOT, "04:00:00")])
        t.bash("04:01:00", f"gh api repos/o/r/pulls/{PR}/reviews", both + "\n" + self.marker(11))
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([(r["review_id"], r["submitted_at"]) for r in out["rounds"]], [("11", iso("03:27:21"))])

    def test_a_different_commit_at_the_same_second_is_not_the_same_review(self):
        reviews = R.read_reviews(json.dumps([rest_review(11, COPILOT, "03:27:21", commit="aaaaaaa1"),
                                             view_review("PRR_x", COPILOT, "03:27:21", commit="bbbbbbb2")]))
        newest = R.newest_copilot_review(reviews or [])
        self.assertEqual(newest["aliases"] if newest else None, [])

    def test_two_numeric_ids_in_one_second_with_different_commits_are_two_rounds(self):
        # The finding of review 5809442797: the time alone must not merge two distinct reviews.
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21", commit="aaaaaaa1")])
        t.reviews_api("04:02:00", [rest_review(13, COPILOT, "03:27:21", commit="bbbbbbb2")])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([(r["review_id"], r["review_ids"], r["commit_id"]) for r in out["rounds"]],
                         [("11", ["11"], "aaaaaaa1"), ("13", ["13"], "bbbbbbb2")])

    def test_two_numeric_ids_in_one_second_are_two_rounds_even_with_no_commit(self):
        # Two numeric ids are never aliases of each other.
        t = Transcript()
        t.bash("04:01:00", f"gh api repos/o/r/pulls/{PR}/reviews --jq '.[]'",
               f"#11 [{COPILOT}] COMMENTED 2026-11-06T03:27:21Z\n")
        t.bash("04:02:00", f"gh api repos/o/r/pulls/{PR}/reviews --jq '.[]'",
               f"#13 [{COPILOT}] COMMENTED 2026-11-06T03:27:21Z\n")
        self.assertEqual([r["review_id"] for r in self.rounds(t, merge="04:10:00")["rounds"]], ["11", "13"])

    def test_two_ids_of_one_form_in_one_output_are_never_aliases(self):
        # The claim in review 5809496118's headline: distinct reviews sharing a second AND a
        # commit. Two numeric ids (or two node ids) are two reviews, in one output or across fetches.
        both = R.read_reviews(json.dumps([rest_review(11, COPILOT, "03:27:21"), rest_review(13, COPILOT, "03:27:21")]))
        newest = R.newest_copilot_review(both or [])
        self.assertEqual((newest["id"], newest["aliases"]) if newest else None, ("13", []))
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21"), rest_review(13, COPILOT, "03:27:21")])
        t.reviews_view("04:02:00", [view_review("PRR_a", COPILOT, "03:27:21")])
        t.reviews_view("04:03:00", [view_review("PRR_b", COPILOT, "03:27:21")])
        out = self.rounds(t, merge="04:10:00")
        # 13 is shown first; PRR_a is its alias (same second and commit), PRR_b is another review.
        self.assertEqual([(r["review_id"], r["review_ids"]) for r in out["rounds"]],
                         [("13", ["13", "PRR_a"]), ("PRR_b", ["PRR_b"])])

    def test_a_prr_id_with_another_commit_than_the_numeric_review_is_not_its_alias(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21", commit="aaaaaaa1")])
        t.reviews_view("04:02:00", [view_review("PRR_x", COPILOT, "03:27:21", commit="bbbbbbb2")])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11", "PRR_x"])

    def test_a_prr_id_with_the_same_commit_in_short_form_is_the_numeric_reviews_alias(self):
        # A review line carries the 7-character commit, `gh pr view` the full one.
        t = Transcript()
        t.bash("04:01:00", f"gh api repos/o/r/pulls/{PR}/reviews --jq '.[]'",
               f"#11 [{COPILOT}] COMMENTED commit=abc1234 2026-11-06T03:27:21Z\n")
        t.reviews_view("04:02:00", [view_review("PRR_x", COPILOT, "03:27:21", commit="abc1234def5678")])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([(r["review_id"], r["review_ids"]) for r in out["rounds"]], [("11", ["11", "PRR_x"])])

    def test_the_numeric_id_reaches_the_copilot_round_when_a_local_round_has_started_since(self):
        # The finding of review 5809442797: PRR_abc, a local review, then a poll showing it as
        # numeric 11. The Copilot round (not the local window the poll lands in) is re-keyed.
        t = Transcript()
        t.reviews_view("04:01:00", [view_review("PRR_abc", COPILOT, "03:27:21")])
        t.review_call("04:02:00", {"pr": PR})
        t.reviews_api("04:03:00", [rest_review(11, COPILOT, "03:27:21")])
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["kind"] for r in out["rounds"]], ["copilot", "local"])
        copilot, local = out["rounds"]
        self.assertEqual((copilot["review_id"], copilot["review_ids"]), ("11", ["PRR_abc", "11"]))
        self.assertEqual(len(copilot["fetch_calls"]), 1)
        self.assertEqual(len(local["fetch_calls"]), 1)  # the poll stays in the window it landed in

    def test_a_pasted_marker_output_cannot_start_a_second_round(self):
        # A body that pastes another review header, beside the marked one, in an unlisted command.
        t = Transcript()
        t.bash("04:01:00", "./fetch.sh", self.marker(11) + BodiesAreNotRecords.QUOTED + "\n")
        out = self.rounds(t, merge="04:10:00")
        self.assertEqual([r["review_id"] for r in out["rounds"]], ["11"])


class ReviewReading(unittest.TestCase):
    def test_empty_id_and_latest_reviews_are_never_used(self):
        text = json.dumps({"reviews": [], "latestReviews": [view_review("", COPILOT, "04:00:00"),
                                                            view_review("PRR_x", COPILOT, "04:00:00")]})
        self.assertEqual(R.read_reviews(text), [])

    def test_pending_reviews_and_missing_authors_are_not_rounds(self):
        text = json.dumps([rest_review(1, COPILOT, "04:00:00", state="PENDING"),
                           {"id": 2, "state": "COMMENTED", "submitted_at": iso("04:00:00")}])
        self.assertEqual(R.read_reviews(text), [])

    def test_the_newest_copilot_review_is_chosen_by_submission_time(self):
        reviews = R.read_reviews(json.dumps([rest_review(3, COPILOT, "05:00:00"), rest_review(2, ME, "06:00:00"),
                                             rest_review(1, COPILOT, "04:00:00")]))
        newest = R.newest_copilot_review(reviews or [])
        self.assertIsNotNone(newest)
        self.assertEqual(newest["id"] if newest else None, "3")
        self.assertIsNone(R.newest_copilot_review([r for r in reviews or [] if r["login"] == ME]))

    def test_an_unrecognised_shape_is_none_and_an_empty_list_is_not(self):
        self.assertIsNone(R.read_reviews("not json at all"))
        self.assertIsNone(R.read_reviews(json.dumps(["a", "b"])))
        self.assertIsNone(R.read_reviews(json.dumps({"state": "OPEN"})))
        self.assertEqual(R.read_reviews("[]"), [])

    def test_a_flood_of_unparseable_brace_lines_is_bounded(self):
        self.assertIsNone(R.read_reviews("{\n" * 50000))


class FailOpen(_Base):
    def test_a_malformed_transcript_gives_an_empty_result(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")])
        p = t.write(self.dir, extra_raw=["{not json", "[1, 2", "null", '{"message": 5}', '{"type": "assistant", "message": {"content": 7}}'])
        out = R.find_rounds(p, PR, merge_time=iso("04:10:00"))
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["rounds"]), 1)

    def test_garbage_only_file(self):
        p = Path(self.dir) / "g.jsonl"
        p.write_text("\x00\x01 garbage\n{{{{\n", encoding="utf-8")
        out = R.find_rounds(p, PR)
        self.assertEqual((out["ok"], out["rounds"], out["fetch_calls"]), (True, [], 0))

    @unittest.skipIf(sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                     "file permissions do not stop this process from reading")
    def test_a_transcript_that_cannot_be_opened_is_not_ok(self):
        p = Transcript().write(self.dir, name="locked.jsonl")
        p.chmod(0)
        self.addCleanup(p.chmod, 0o600)
        self.assertFalse(R.find_rounds(p, PR)["ok"])

    def test_a_directory_in_place_of_the_file_is_not_ok(self):
        self.assertFalse(R.find_rounds(Path(self.dir), PR)["ok"])

    def test_missing_file_and_bad_arguments(self):
        self.assertFalse(R.find_rounds(Path(self.dir) / "nope.jsonl", PR)["ok"])
        t = Transcript().write(self.dir)
        for bad in (0, -1, "603", None, True, 6.5):
            self.assertFalse(R.find_rounds(t, bad)["ok"], bad)
        # A bad merge time is ignored, not fatal.
        self.assertTrue(R.find_rounds(t, PR, merge_time="garbage")["ok"])
        self.assertTrue(R.find_rounds(t, PR, merge_time=True)["ok"])

    def test_an_unexpected_error_inside_the_reader_fails_open(self):
        original = R._read_transcript
        R._read_transcript = lambda path: 1 / 0
        try:
            out = R.find_rounds(Transcript().write(self.dir), PR)
        finally:
            R._read_transcript = original
        self.assertFalse(out["ok"])
        self.assertEqual(out["rounds"], [])

    def test_a_merge_time_as_epoch_seconds_is_accepted(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")])
        t.assistant("04:02:00", u=usage(1, 1, 1, 8))
        t.assistant("04:09:00", u=usage(1, 1, 1, 100))
        out = R.find_rounds(t.write(self.dir), PR, merge_time=epoch("04:05:00"))
        self.assertEqual(out["rounds"][0]["end_ts"], epoch("04:05:00"))
        self.assertEqual(out["rounds"][0]["output_tokens"], 5 + 8)

    def test_a_timestamp_with_no_offset_is_read_as_utc(self):
        self.assertEqual(R._parse_ts("2026-11-06T04:00:00"), epoch("04:00:00"))


if __name__ == "__main__":
    unittest.main()
