#!/usr/bin/env python3
"""Regression test for issue #355: `gh issue list` / `gh pr list` return at most 30
results unless given `--limit`, so a skill that tells the model to run one without it
silently drops everything older than the 30 newest items (it once produced a work order
covering 30 of 73 open issues, and a duplicate ticket because the duplicate-check search
came back truncated).

Invariant: every `gh issue list` / `gh pr list` command written in any
`.claude/skills/*/SKILL.md` carries `--limit`, unless it is scoped by `--head <branch>`
(which returns a handful of results by construction).

Commands are extracted whole, not line by line: a fenced-block command continues across
`\\` line continuations (my-gh-autowork's auto-pick command puts `--limit 200` on its
last line), and an inline-code command runs to its closing backtick even when the prose
wraps the span onto the next line. A bare mention with no arguments (e.g. "`gh issue list`
silently truncates...") is not a command.

Stdlib-only, no network.

    .venv/bin/python -m unittest discover -s tests
"""

import glob
import os
import re
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_GLOB = os.path.join(REPO_ROOT, ".claude", "skills", "*", "SKILL.md")

_LIST_CMD_RE = re.compile(r"gh\s+(?:issue|pr)\s+list\b")

# Commands deliberately shown as what NOT to do. Keyed by (skill dir name, normalized
# command text); keep this list as short as possible.
_NEGATIVE_EXAMPLES = {
    ("my-gh-autowork", 'gh pr list --search "<N> in:body"'),
}


def extract_list_commands(text):
    """Return every full `gh issue list`/`gh pr list` command in markdown text,
    whitespace-normalized, continuations joined."""
    commands = []
    for match in _LIST_CMD_RE.finditer(text):
        start = match.start()
        line_start = text.rfind("\n", 0, start) + 1
        inline = "`" in text[line_start:start] and text[line_start:start].count("`") % 2 == 1
        if inline:
            end = text.find("`", start)
            end = len(text) if end == -1 else end
            raw = text[start:end]
        else:
            # Fenced/plain line: follow trailing-backslash continuations. A plain-prose
            # mention ends at its closing backtick or end of line, whichever is first.
            pos = start
            while True:
                eol = text.find("\n", pos)
                if eol == -1:
                    eol = len(text)
                if text[pos:eol].rstrip().endswith("\\"):
                    pos = eol + 1
                    continue
                break
            raw = text[start:eol]
            tick = raw.find("`")
            if tick != -1:
                raw = raw[:tick]
        commands.append(" ".join(raw.replace("\\\n", " ").split()))
    return commands


def find_unbounded(skill_name, commands):
    """Commands that have arguments but neither --limit nor --head."""
    bad = []
    for cmd in commands:
        args = _LIST_CMD_RE.sub("", cmd, count=1).strip()
        if not args:  # bare mention, not a command
            continue
        if re.search(r"(^|\s)--(limit|head)\b", cmd):
            continue
        if (skill_name, cmd) in _NEGATIVE_EXAMPLES:
            continue
        bad.append(cmd)
    return bad


class ExtractorBehavior(unittest.TestCase):
    def test_fenced_continuation_lines_are_joined(self):
        text = (
            "  gh issue list -R X --state open \\\n"
            '    --search "a b" \\\n'
            "    --limit 200 --json number\n"
        )
        (cmd,) = extract_list_commands(text)
        self.assertIn("--limit 200", cmd)
        self.assertEqual(find_unbounded("s", [cmd]), [])

    def test_inline_span_wrapped_across_lines(self):
        text = 'e.g. `gh pr list --search\n    "<N> in:body"` done'
        (cmd,) = extract_list_commands(text)
        self.assertEqual(cmd, 'gh pr list --search "<N> in:body"')

    def test_bare_mention_is_not_a_command(self):
        text = "IMPORTANT: `gh issue list` silently truncates to 30 results."
        self.assertEqual(find_unbounded("s", extract_list_commands(text)), [])

    def test_unbounded_command_is_flagged(self):
        text = "Check `gh issue list --state open --json number` now."
        self.assertEqual(len(find_unbounded("s", extract_list_commands(text))), 1)

    def test_head_scoped_command_is_allowed(self):
        text = "`gh pr list -R X --head <branch> --state all`"
        self.assertEqual(find_unbounded("s", extract_list_commands(text)), [])


class SkillsBoundListCommands(unittest.TestCase):
    def test_skills_found(self):
        self.assertTrue(glob.glob(SKILLS_GLOB), "no .claude/skills/*/SKILL.md found")

    def test_every_list_command_has_limit_or_head(self):
        offenders = []
        for path in sorted(glob.glob(SKILLS_GLOB)):
            skill = os.path.basename(os.path.dirname(path))
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            for cmd in find_unbounded(skill, extract_list_commands(text)):
                offenders.append(f"{skill}: {cmd}")
        self.assertEqual(
            offenders,
            [],
            "gh issue list / gh pr list without --limit (or --head) silently caps at 30 results",
        )


if __name__ == "__main__":
    unittest.main()
