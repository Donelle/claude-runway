#!/usr/bin/env python3
"""Tests for .claude/skills/my-gh-autowork/SKILL.md -- the documentation of
the server-side auto mode classifier that can block the orchestrator's own `Agent`
call, independent of `.claude/settings.json`.

The skill is a markdown document of prose instructions, not a Python module, so this
mirrors tests/test_my_resume_skill.py's content-pinning strategy: assert the key
strings introduced for the classifier caveat remain present, so a future edit that removes or
waters down the caveat/preflight check gets an explicit failure here instead of a
silent regression. The Step 0 preflight check's actual `jq` regex is also exercised
directly against representative settings.json shapes, so the check's logic is
verified, not just its presence in the text.

Stdlib-only (unittest, no pytest), no network, no LM Studio, no Qdrant, no `jq`
subprocess needed for the logic check (re-implemented in Python for portability).

    .venv/bin/python -m unittest discover -s tests
"""

import os
import re
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL_PATH = os.path.join(REPO_ROOT, ".claude", "skills", "my-gh-autowork", "SKILL.md")
SETTINGS_TEMPLATE_PATH = os.path.join(REPO_ROOT, "templates", "settings.json.template")

# Mirrors the regex embedded in SKILL.md's Step 0 preflight check:
#   jq -e '(.permissions.allow // []) | any(test("^Bash\\((git|gh|uv)[ :)]|^Bash\\(\\.venv/bin/"))'
# The `[ :)]` delimiter right after git/gh/uv is required (Copilot review round 2 on
# PR #196): an earlier version with no delimiter (`^Bash\((git|gh|uv|\.venv/bin/)`)
# false-positived on unrelated tool names sharing the same leading letters, e.g.
# `Bash(github-cli:*)`, `Bash(ghastly:*)`, `Bash(uvicorn:*)` -- confirmed live by
# running this exact regex against those three strings before fixing it.
_BASELINE_PERMISSION_RE = re.compile(r"^Bash\((git|gh|uv)[ :)]|^Bash\(\.venv/bin/")


def has_baseline_permission(allow_list):
    """Reference implementation of the Step 0 preflight check's jq test()."""
    return any(_BASELINE_PERMISSION_RE.match(entry) for entry in allow_list)


class SkillFileExists(unittest.TestCase):
    """Guard against the file being renamed or deleted accidentally."""

    def test_skill_file_is_present(self):
        self.assertTrue(
            os.path.isfile(SKILL_PATH),
            f".claude/skills/my-gh-autowork/SKILL.md not found at {SKILL_PATH}",
        )


class SkillContentRequirements(unittest.TestCase):
    """Pin the key content added for the auto mode classifier in SKILL.md's text."""

    def setUp(self):
        with open(SKILL_PATH, encoding="utf-8") as f:
            self.content = f.read()

    def test_mentions_auto_mode_classifier(self):
        self.assertIn("auto mode classifier", self.content.lower())

    def test_caveat_qualifies_zero_touchpoints_claim(self):
        # The caveat must sit near the original claim, not be buried elsewhere,
        # so a reader of the opening claim actually sees the qualification.
        idx_claim = self.content.find("Zero required human touchpoints")
        idx_caveat = self.content.find("Caveat: this claim")
        self.assertNotEqual(idx_claim, -1, "original claim text must still exist")
        self.assertNotEqual(idx_caveat, -1, "caveat must exist")
        self.assertLess(
            idx_claim,
            idx_caveat,
            "caveat should immediately follow the original claim, not precede it",
        )

    def test_step_0_has_baseline_permission_preflight(self):
        self.assertIn("Baseline Bash permission preflight", self.content)

    def test_preflight_checks_both_settings_scopes(self):
        self.assertIn(".claude/settings.json", self.content)
        self.assertIn("~/.claude/settings.json", self.content)

    def test_preflight_stops_on_missing_permissions(self):
        # Must instruct a hard stop, matching the existing dogfood-preflight pattern.
        section = self.content[self.content.find("Baseline Bash permission preflight"):]
        self.assertIn("STOP", section[:2000])

    def test_does_not_overclaim_certainty(self):
        # The causal relationship between the permission fix and the classifier's
        # behavior was explicitly NOT confirmed in the issue -- the skill must not
        # overstate it as a guaranteed fix.
        self.assertIn("not confirm", self.content.lower())

    def test_does_not_recommend_wildcard_permissions(self):
        # Copilot review on PR #196 (round 1): an earlier draft of this fix
        # recommended broadening the confirmed narrow rules into a blanket
        # 'Bash(git:*)'/'Bash(gh:*)'/'Bash(uv:*)' wildcard. Verified live that this
        # is a real arbitrary-command-execution bypass (git alias mechanism,
        # `uv run`), not just a broader convenience -- the skill must only ever cite
        # these wildcard forms as an explicit warning, never as something to add.
        self.assertIn("git -c alias", self.content)
        self.assertIn("wildcard", self.content.lower())
        self.assertIn("arbitrary-command-execution bypass", self.content)

    def test_confirms_only_the_specific_narrow_rules(self):
        # The actually-confirmed workaround -- these exact four
        # rules, not a generalization of them.
        for rule in ["git fetch *", "git push *", "gh pr *", "gh api *"]:
            self.assertIn(rule, self.content)


class SettingsTemplateContentRequirements(unittest.TestCase):
    """Pin the new _permissions_note documenting suggested baseline rules."""

    def setUp(self):
        with open(SETTINGS_TEMPLATE_PATH, encoding="utf-8") as f:
            self.content = f.read()

    def test_has_permissions_note(self):
        self.assertIn("_permissions_note", self.content)

    def test_permissions_note_warns_against_wildcards(self):
        self.assertIn("arbitrary-command-execution bypass", self.content)
        self.assertIn("git -c alias", self.content)

    def test_permissions_note_prefers_project_scope(self):
        self.assertIn("PREFERRED", self.content)

    def test_permissions_note_does_not_present_narrow_rules_as_safe(self):
        # Even the narrow per-subcommand rules still permit code execution through
        # legitimate flags (git fetch --upload-pack, python -c); the note must not
        # imply per-subcommand scoping is a security boundary.
        self.assertIn("means SAFE in an absolute sense", self.content)

    def test_permissions_note_quotes_confirmed_rules_exactly(self):
        # An earlier review round found this note restating the confirmed rules in
        # colon syntax ('Bash(git fetch:*)') instead of quoting them exactly as the
        # original report did (space before the wildcard). The "confirmed-to-work"
        # claim must use the original report's own literal quote.
        for rule in ["git fetch *", "git push *", "gh pr *", "gh api *"]:
            self.assertIn(rule, self.content)

    def test_permissions_note_does_not_overclaim_narrow_safety(self):
        # The template note is where the fullest account of the code-execution
        # caveat lives, so it must carry it itself: even narrow per-subcommand
        # rules still permit arbitrary commands (e.g. git fetch --upload-pack), and
        # autonomous development inherently needs code-execution authority.
        self.assertIn("upload-pack", self.content)
        self.assertIn("code-execution authority", self.content.lower())

    def test_still_valid_json(self):
        import json

        # Must remain parseable JSON after the new key was added -- a stray
        # trailing comma or unescaped quote in the note would break every
        # consumer of this template (setup_project_lib.py's load_json()).
        json.loads(self.content)


class BaselinePermissionRegexLogic(unittest.TestCase):
    """Exercise the Step 0 preflight check's matching logic directly."""

    def test_matches_git_rule(self):
        self.assertTrue(has_baseline_permission(["Bash(git push:*)"]))

    def test_matches_gh_rule(self):
        self.assertTrue(has_baseline_permission(["Bash(gh pr:*)"]))

    def test_matches_uv_rule(self):
        self.assertTrue(has_baseline_permission(["Bash(uv pip install:*)"]))

    def test_matches_venv_rule(self):
        self.assertTrue(has_baseline_permission(["Bash(.venv/bin/python:*)"]))

    def test_no_match_on_unrelated_rules(self):
        self.assertFalse(has_baseline_permission(["Bash(dotnet build:*)"]))

    def test_no_match_on_empty_list(self):
        self.assertFalse(has_baseline_permission([]))

    def test_matches_among_multiple_unrelated_entries(self):
        self.assertTrue(
            has_baseline_permission(["Bash(dotnet build:*)", "Bash(git fetch:*)"])
        )

    def test_matches_bare_git_with_no_args(self):
        # Closing paren immediately after the command name is a valid delimiter too.
        self.assertTrue(has_baseline_permission(["Bash(git)"]))

    def test_no_false_positive_on_github_cli(self):
        # Copilot review round 2 on PR #196: 'github-cli' starts with the same three
        # letters as 'git' ('g','i','t'), so a delimiter-less prefix check wrongly
        # matched it. Confirmed live before fixing.
        self.assertFalse(has_baseline_permission(["Bash(github-cli:*)"]))

    def test_no_false_positive_on_ghastly(self):
        # 'ghastly' starts with 'gh' -- same false-positive class as above.
        self.assertFalse(has_baseline_permission(["Bash(ghastly:*)"]))

    def test_no_false_positive_on_uvicorn(self):
        # 'uvicorn' starts with 'uv' -- same false-positive class as above.
        self.assertFalse(has_baseline_permission(["Bash(uvicorn:*)"]))

    def test_no_false_positive_does_not_mask_real_match_in_same_list(self):
        # A false-positive-prone entry alongside a genuine one must not change the
        # (correct) overall result -- the real rule should still be what's detected.
        self.assertTrue(
            has_baseline_permission(["Bash(uvicorn:*)", "Bash(git push:*)"])
        )


if __name__ == "__main__":
    unittest.main()
