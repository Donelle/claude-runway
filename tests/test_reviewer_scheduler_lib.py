#!/usr/bin/env python3
"""Tests for libs/reviewer_scheduler_lib.py: stage order, the MAX_REVIEWERS
limit, model grouping, and when on-demand models are loaded and unloaded.

No network and no LM Studio: the model loader gets a fake REST API and the
reviewer is a recording callable.

    .venv/bin/python -m unittest discover -s tests
"""

import os
import sys
import threading
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from codebase_reviewer_config_lib import Reviewer, ReviewersConfig  # noqa: E402
from fake_lmstudio import FakeLmStudio  # noqa: E402
from lmstudio_model_loader_lib import ModelLoader, ModelNotLoadedError  # noqa: E402
from reviewer_scheduler_lib import (  # noqa: E402
    describe_plan,
    order_for_carry_over,
    plan_stages,
    run_stages,
)

BASE = "http://localhost:1234/v1"


def rv(name, model, roles=("reasoning",), *, on_demand=False, active=True, context=None) -> Reviewer:
    return Reviewer(
        name=name,
        model=model,
        prompt="p",
        active=active,
        roles=tuple(r for r in ("reasoning", "verifier", "classifier") if r in roles),
        load_on_demand=on_demand,
        context_length=context,
    )


def cfg(*reviewers) -> ReviewersConfig:
    return ReviewersConfig(rubric=(), reviewers=tuple(reviewers))


class PlanStages(unittest.TestCase):
    def test_stages_run_reasoning_then_verifier_then_classifier(self):
        config = cfg(
            rv("c", "m3", ["classifier"]),
            rv("v", "m2", ["verifier"]),
            rv("r", "m1", ["reasoning"]),
        )
        self.assertEqual([s.role for s in plan_stages(config)], ["reasoning", "verifier", "classifier"])

    def test_a_reviewer_with_several_roles_takes_part_in_each_stage(self):
        config = cfg(rv("both", "m1", ["reasoning", "verifier"]), rv("c", "m2", ["classifier"]))
        stages = plan_stages(config)
        self.assertEqual([s.role for s in stages], ["reasoning", "verifier", "classifier"])
        self.assertEqual([r.name for r in stages[0].reviewers()], ["both"])
        self.assertEqual([r.name for r in stages[1].reviewers()], ["both"])

    def test_inactive_reviewers_are_left_out_and_empty_stages_are_skipped(self):
        config = cfg(rv("off", "m1", ["reasoning"], active=False), rv("v", "m2", ["verifier"]))
        self.assertEqual([s.role for s in plan_stages(config)], ["verifier"])

    def test_no_active_reviewers_means_no_stages(self):
        self.assertEqual(plan_stages(cfg()), [])
        self.assertEqual(plan_stages(cfg(rv("off", "m", active=False))), [])

    def test_reviewers_sharing_a_model_are_grouped_together(self):
        config = cfg(rv("a1", "A"), rv("b1", "B"), rv("a2", "A"), rv("b2", "B"))
        stage = plan_stages(config)[0]
        self.assertEqual([g.model for g in stage.resident], ["A", "B"])
        self.assertEqual([r.name for r in stage.resident[0].reviewers], ["a1", "a2"])
        self.assertEqual([r.name for r in stage.resident[1].reviewers], ["b1", "b2"])
        # Each group is contiguous in the flat list, so same-model reviewers run together.
        self.assertEqual([r.name for r in stage.reviewers()], ["a1", "a2", "b1", "b2"])

    def test_on_demand_and_resident_groups_are_kept_apart(self):
        config = cfg(rv("res", "R"), rv("od", "O", on_demand=True, context=8192))
        stage = plan_stages(config)[0]
        self.assertEqual([g.model for g in stage.resident], ["R"])
        self.assertEqual([g.model for g in stage.on_demand], ["O"])
        self.assertEqual(stage.on_demand[0].context_length, 8192)
        self.assertTrue(stage.on_demand[0].load_on_demand)

    def test_roles_argument_limits_the_stages(self):
        config = cfg(rv("both", "m", ["reasoning", "verifier"]))
        self.assertEqual([s.role for s in plan_stages(config, roles=["verifier"])], ["verifier"])

    def test_carry_over_puts_held_models_first_and_is_otherwise_stable(self):
        config = cfg(rv("a", "A", on_demand=True), rv("b", "B", on_demand=True), rv("c", "C", on_demand=True))
        groups = plan_stages(config)[0].on_demand
        self.assertEqual([g.model for g in order_for_carry_over(groups, ["C"])], ["C", "A", "B"])
        self.assertEqual([g.model for g in order_for_carry_over(groups, [])], ["A", "B", "C"])

    def test_a_model_the_next_stage_needs_goes_last_among_the_rest(self):
        config = cfg(rv("a", "A", on_demand=True), rv("b", "B", on_demand=True), rv("c", "C", on_demand=True))
        groups = plan_stages(config)[0].on_demand
        self.assertEqual([g.model for g in order_for_carry_over(groups, [], ["A"])], ["B", "C", "A"])
        # Held first, then models not needed next, then models needed next.
        self.assertEqual([g.model for g in order_for_carry_over(groups, ["C"], ["A", "C"])], ["C", "B", "A"])

    def test_describe_plan_shows_stages_models_and_how_they_load(self):
        text = describe_plan(plan_stages(cfg(rv("r", "A"), rv("c", "B", ["classifier"], on_demand=True))))
        self.assertIn("1. reasoning", text)
        self.assertIn("model A (already loaded): r", text)
        self.assertIn("2. classifier", text)
        self.assertIn("model B (loaded on demand): c", text)
        self.assertEqual(describe_plan([]), "Stage plan: no active reviewers.")


class Recorder:
    """A run_reviewer that records the order reviewers start and the peak number running at once."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.started: list = []
        self._lock = threading.Lock()
        self.running = 0
        self.peak = 0

    def __call__(self, reviewer, role):
        with self._lock:
            self.started.append((role, reviewer.name))
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            if self.delay:
                time.sleep(self.delay)
            return f"{role}:{reviewer.name}"
        finally:
            with self._lock:
                self.running -= 1


class RunStages(unittest.TestCase):
    def _run(self, config, fake=None, max_reviewers=3, recorder=None):
        fake = fake or FakeLmStudio(loaded=["A", "B", "C", "m1", "m2", "m3"])
        recorder = recorder or Recorder()
        report = run_stages(config, ModelLoader(BASE, http=fake), recorder, max_reviewers)
        return report, fake, recorder

    def test_stages_run_in_order_and_a_stage_finishes_before_the_next_starts(self):
        config = cfg(
            rv("c", "m3", ["classifier"]),
            rv("v1", "m2", ["verifier"]),
            rv("v2", "m2", ["verifier"]),
            rv("r1", "m1", ["reasoning"]),
            rv("r2", "m1", ["reasoning"]),
        )
        report, _, rec = self._run(config, recorder=Recorder(delay=0.02))
        roles = [role for role, _ in rec.started]
        self.assertEqual(roles, ["reasoning"] * 2 + ["verifier"] * 2 + ["classifier"])
        self.assertEqual([o.role for o in report.outcomes], roles)

    def test_outcomes_carry_each_reviewers_value_in_order(self):
        report, _, _ = self._run(cfg(rv("a", "A"), rv("b", "B")))
        self.assertEqual([o.value for o in report.outcomes], ["reasoning:a", "reasoning:b"])
        self.assertTrue(all(o.error is None for o in report.outcomes))

    def test_a_reviewer_with_several_roles_runs_in_each_stage(self):
        report, _, rec = self._run(cfg(rv("both", "A", ["reasoning", "verifier"])))
        self.assertEqual(rec.started, [("reasoning", "both"), ("verifier", "both")])

    def test_max_reviewers_one_runs_them_one_at_a_time(self):
        config = cfg(*(rv(f"r{i}", f"m{i % 3 + 1}") for i in range(6)))
        _, _, rec = self._run(config, max_reviewers=1, recorder=Recorder(delay=0.02))
        self.assertEqual(rec.peak, 1)

    def test_max_reviewers_caps_parallelism_and_actually_runs_in_parallel(self):
        config = cfg(*(rv(f"r{i}", "A") for i in range(6)))
        _, _, rec = self._run(config, max_reviewers=2, recorder=Recorder(delay=0.05))
        self.assertEqual(rec.peak, 2)

    def test_reviewers_really_overlap_up_to_the_limit(self):
        # Two reviewers must both be inside run_reviewer at once to pass the
        # barrier; with a limit of 1 this would time out instead.
        barrier = threading.Barrier(2, timeout=5)

        def run(reviewer, role):
            barrier.wait()
            return "ok"

        config = cfg(rv("a", "A"), rv("b", "B"))
        report = run_stages(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["A", "B"])), run, 2)
        self.assertTrue(all(o.error is None for o in report.outcomes))

    def test_resident_reviewers_of_different_models_share_one_pool(self):
        config = cfg(rv("a", "A"), rv("b", "B"), rv("c", "C"))
        _, _, rec = self._run(config, max_reviewers=3, recorder=Recorder(delay=0.05))
        self.assertEqual(rec.peak, 3)

    def test_a_reviewer_that_raises_becomes_an_error_outcome_and_the_rest_continue(self):
        def run(reviewer, role):
            if reviewer.name == "bad":
                raise RuntimeError("model fell over")
            return "fine"

        config = cfg(rv("good1", "A"), rv("bad", "A"), rv("good2", "A"))
        report = run_stages(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["A"])), run, 2)
        self.assertEqual([o.error for o in report.outcomes], [None, "RuntimeError: model fell over", None])
        self.assertEqual(report.outcomes[2].value, "fine")


class ResidentModelsMustBeLoaded(unittest.TestCase):
    def test_a_resident_model_that_is_not_loaded_stops_the_run_before_anything_runs(self):
        config = cfg(rv("a", "A"), rv("b", "B", ["verifier"]))
        fake = FakeLmStudio(loaded=["A"])
        rec = Recorder()
        with self.assertRaises(ModelNotLoadedError) as ctx:
            run_stages(config, ModelLoader(BASE, http=fake), rec, 3)
        self.assertIn("'B'", str(ctx.exception))
        self.assertEqual(rec.started, [])
        self.assertEqual(fake.posts("/load"), [])

    def test_every_missing_model_is_named_at_once(self):
        config = cfg(rv("a", "A"), rv("b", "B"), rv("c", "C"))
        with self.assertRaises(ModelNotLoadedError) as ctx:
            run_stages(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["B"])), Recorder(), 3)
        self.assertIn("'A'", str(ctx.exception))
        self.assertIn("'C'", str(ctx.exception))
        self.assertNotIn("'B'", str(ctx.exception))

    def test_an_on_demand_model_need_not_be_loaded_beforehand(self):
        config = cfg(rv("od", "X", on_demand=True))
        fake = FakeLmStudio()
        report = run_stages(config, ModelLoader(BASE, http=fake), Recorder(), 3)
        self.assertEqual(report.outcomes[0].error, None)


class LoadAndUnload(unittest.TestCase):
    """Load/unload happen only for loadOnDemand reviewers, and unload only for
    instances the run loaded itself."""

    def _events(self, fake):
        """Load/unload calls as ('load'|'unload', model-or-instance), in order."""
        out = []
        for method, url, body in fake.calls:
            if method == "POST" and url.endswith("/load"):
                out.append(("load", body["model"]))
            elif method == "POST" and url.endswith("/unload"):
                out.append(("unload", body["instance_id"].split(":")[0]))
        return out

    def test_no_load_or_unload_when_no_reviewer_is_load_on_demand(self):
        config = cfg(rv("a", "A"), rv("v", "B", ["verifier"]))
        fake = FakeLmStudio(loaded=["A", "B"])
        run_stages(config, ModelLoader(BASE, http=fake), Recorder(), 3)
        self.assertEqual(self._events(fake), [])

    def test_an_on_demand_model_is_loaded_before_use_and_unloaded_after(self):
        fake = FakeLmStudio()
        order: list = []

        def run(reviewer, role):
            order.append(("run", reviewer.name, sorted(fake.loaded)))
            return None

        config = cfg(rv("od", "X", on_demand=True, context=16384))
        run_stages(config, ModelLoader(BASE, http=fake), run, 3)
        self.assertEqual(self._events(fake), [("load", "X"), ("unload", "X")])
        self.assertEqual(order, [("run", "od", ["X"])])
        self.assertEqual(fake.loaded, {})
        self.assertEqual(fake.posts("/load")[0][1], {"model": "X", "context_length": 16384})

    def test_a_model_used_in_two_stages_loads_once_per_stage_when_not_used_by_the_next(self):
        # Reasoning uses X, verifier uses Y, classifier uses X again: X must
        # be unloaded in between (the next stage does not use it).
        config = cfg(
            rv("r", "X", ["reasoning", "classifier"], on_demand=True),
            rv("v", "Y", ["verifier"], on_demand=True),
        )
        fake = FakeLmStudio()
        run_stages(config, ModelLoader(BASE, http=fake), Recorder(), 3)
        self.assertEqual(
            self._events(fake),
            [("load", "X"), ("unload", "X"), ("load", "Y"), ("unload", "Y"), ("load", "X"), ("unload", "X")],
        )

    def test_a_model_the_next_stage_also_uses_stays_loaded(self):
        config = cfg(rv("both", "X", ["reasoning", "verifier"], on_demand=True))
        fake = FakeLmStudio()
        run_stages(config, ModelLoader(BASE, http=fake), Recorder(), 3)
        self.assertEqual(self._events(fake), [("load", "X"), ("unload", "X")])

    def test_it_stays_loaded_across_all_three_stages_and_is_unloaded_last(self):
        config = cfg(rv("all", "X", ["reasoning", "verifier", "classifier"], on_demand=True))
        fake = FakeLmStudio()
        loaded_during: list = []

        def run(reviewer, role):
            loaded_during.append((role, sorted(fake.loaded)))

        run_stages(config, ModelLoader(BASE, http=fake), run, 3)
        self.assertEqual(self._events(fake), [("load", "X"), ("unload", "X")])
        self.assertEqual([m for _, m in loaded_during], [["X"], ["X"], ["X"]])

    def test_reviewers_sharing_an_on_demand_model_load_it_once_per_stage(self):
        config = cfg(rv("a", "X", on_demand=True), rv("b", "X", on_demand=True), rv("c", "X", on_demand=True))
        fake = FakeLmStudio()
        run_stages(config, ModelLoader(BASE, http=fake), Recorder(), 3)
        self.assertEqual(self._events(fake), [("load", "X"), ("unload", "X")])

    def test_two_on_demand_models_are_never_loaded_at_the_same_time(self):
        config = cfg(rv("a", "X", on_demand=True), rv("b", "Y", on_demand=True))
        fake = FakeLmStudio()
        peak = []

        def run(reviewer, role):
            peak.append(len(fake.loaded))

        run_stages(config, ModelLoader(BASE, http=fake), run, 3)
        self.assertEqual(peak, [1, 1])
        self.assertEqual(self._events(fake), [("load", "X"), ("unload", "X"), ("load", "Y"), ("unload", "Y")])

    def test_a_held_model_runs_first_in_the_next_stage_so_two_are_not_loaded_at_once(self):
        # Stage 1 holds Y (the next stage uses it); stage 2 lists X before Y in
        # file order, but Y must go first or X and Y would both be loaded.
        config = cfg(
            rv("first", "X", ["verifier"], on_demand=True),
            rv("second", "Y", ["reasoning", "verifier"], on_demand=True),
        )
        fake = FakeLmStudio()
        peak = []

        def run(reviewer, role):
            peak.append((role, reviewer.name, len(fake.loaded)))

        run_stages(config, ModelLoader(BASE, http=fake), run, 3)
        self.assertEqual(peak, [("reasoning", "second", 1), ("verifier", "second", 1), ("verifier", "first", 1)])
        self.assertEqual(
            self._events(fake),
            [("load", "Y"), ("unload", "Y"), ("load", "X"), ("unload", "X")],
        )

    def test_a_model_kept_for_the_next_stage_is_released_before_another_loads(self):
        # X has reasoning+verifier, Y only reasoning. X is wanted again by the
        # next stage, but Y has to load first, and the machine holds one model:
        # X must not still be loaded then (or Y's reviewer fails for memory).
        config = cfg(
            rv("x", "X", ["reasoning", "verifier"], on_demand=True),
            rv("y", "Y", ["reasoning"], on_demand=True),
        )
        fake = FakeLmStudio(capacity=1)
        report = run_stages(config, ModelLoader(BASE, http=fake), Recorder(), 1)
        self.assertEqual([o.error for o in report.outcomes], [None, None, None])
        self.assertEqual(report.warnings, [])

    def test_the_model_the_next_stage_needs_goes_last_so_it_is_not_loaded_twice(self):
        # Same shape: running Y first and X last lets X stay loaded straight
        # into the verifier stage, one load of X instead of two.
        config = cfg(
            rv("x", "X", ["reasoning", "verifier"], on_demand=True),
            rv("y", "Y", ["reasoning"], on_demand=True),
        )
        fake = FakeLmStudio(capacity=1)
        run_stages(config, ModelLoader(BASE, http=fake), Recorder(), 1)
        self.assertEqual(self._events(fake), [("load", "Y"), ("unload", "Y"), ("load", "X"), ("unload", "X")])

    def test_a_group_is_skipped_when_a_model_it_must_displace_will_not_unload(self):
        # X is used by both stages, so it is still held when the verifier
        # stage's Y has to load. Displacing X fails, so Y is skipped with an
        # error instead of being loaded on top of it.
        config = cfg(
            rv("x", "X", ["reasoning", "verifier"], on_demand=True),
            rv("y", "Y", ["verifier"], on_demand=True),
        )
        fake = FakeLmStudio(capacity=1)

        def flaky(method, url, body, timeout):
            if url.endswith("/unload") and body["instance_id"].startswith("X"):
                raise OSError("stuck")
            return fake(method, url, body, timeout)

        report = run_stages(config, ModelLoader(BASE, http=flaky), Recorder(), 1)
        by_key = {(o.role, o.reviewer.name): o for o in report.outcomes}
        self.assertIsNone(by_key[("reasoning", "x")].error)
        self.assertIsNone(by_key[("verifier", "x")].error)
        self.assertIn("could not unload model 'X'", by_key[("verifier", "y")].error)
        self.assertEqual(fake.posts("/load")[-1][1]["model"], "X")  # Y was never loaded

    def test_a_model_the_user_loaded_is_never_unloaded_even_when_a_reviewer_is_on_demand(self):
        fake = FakeLmStudio(loaded=["compression-model"])
        config = cfg(rv("od", "compression-model", on_demand=True), rv("other", "X", ["verifier"], on_demand=True))
        run_stages(config, ModelLoader(BASE, http=fake), Recorder(), 3)
        self.assertEqual(self._events(fake), [("load", "X"), ("unload", "X")])
        self.assertIn("compression-model", fake.loaded)

    def test_unloads_only_instances_this_run_loaded_even_if_the_reviewer_raises(self):
        fake = FakeLmStudio(loaded=["resident"])

        def run(reviewer, role):
            raise RuntimeError("boom")

        config = cfg(rv("res", "resident"), rv("od", "X", on_demand=True))
        report = run_stages(config, ModelLoader(BASE, http=fake), run, 3)
        self.assertEqual(self._events(fake), [("load", "X"), ("unload", "X")])
        self.assertEqual(set(fake.loaded), {"resident"})
        self.assertTrue(all(o.error for o in report.outcomes))

    def test_everything_loaded_is_unloaded_when_a_later_step_raises(self):
        fake = FakeLmStudio()
        loader = ModelLoader(BASE, http=fake)
        config = cfg(rv("a", "X", ["reasoning", "verifier"], on_demand=True))
        calls = {"n": 0}

        def run(reviewer, role):
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt  # not an Exception: escapes the per-reviewer catch
            return None

        with self.assertRaises(KeyboardInterrupt):
            run_stages(config, loader, run, 1)
        self.assertEqual(fake.loaded, {})
        self.assertEqual(loader.owned_models, [])

    def test_a_failed_load_fails_only_that_groups_reviewers(self):
        fake = FakeLmStudio(fail_on={"/models/load": OSError("no memory")})
        config = cfg(rv("res", "R"), rv("od1", "X", on_demand=True), rv("od2", "X", on_demand=True))
        fake.loaded["R"] = "R"
        rec = Recorder()
        report = run_stages(config, ModelLoader(BASE, http=fake), rec, 3)
        by_name = {o.reviewer.name: o for o in report.outcomes}
        self.assertIsNone(by_name["res"].error)
        self.assertIn("could not load model 'X'", by_name["od1"].error)
        self.assertIn("could not load model 'X'", by_name["od2"].error)
        self.assertEqual(rec.started, [("reasoning", "res")])

    def test_an_unload_failure_is_a_warning_not_a_failure(self):
        fake = FakeLmStudio(fail_on={"/models/unload": OSError("stuck")})
        config = cfg(rv("od", "X", on_demand=True))
        report = run_stages(config, ModelLoader(BASE, http=fake), Recorder(), 3)
        self.assertIsNone(report.outcomes[0].error)
        self.assertTrue(any("could not unload model 'X'" in w for w in report.warnings))


if __name__ == "__main__":
    unittest.main()
