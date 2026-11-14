"""
Guard for the my-* -> cr-* skill rename: no old skill name may come back in a tracked file, and every skill's
directory name must equal its `name:` frontmatter.

The old names legitimately survive in a few places, because each one has to name the OLD skill to do its job: the
migration table and its tests (whole files, in ALLOWED), and, in the two docs that explain the rename to users,
only the passage that does the explaining (ALLOWED_PASSAGES).
"""
import re
import subprocess
import unittest
from pathlib import Path

from libs.setup_project_lib import RENAMED_PRODUCT_SKILLS

REPO = Path(__file__).resolve().parent.parent

# The contributor skills live in `.claude/skills/` and are renamed in place, so they are not in
# RENAMED_PRODUCT_SKILLS; their old names are listed here so the guard still covers them.
OLD_CONTRIBUTOR_SKILLS = tuple(
    "my-gh-" + n for n in (
        "autowork", "code-it", "code-review", "create-issue", "load-context",
        "pr", "pr-feedback", "run-harness", "triage",
    )
)
OLD_NAMES = tuple(RENAMED_PRODUCT_SKILLS) + OLD_CONTRIBUTOR_SKILLS

ALLOWED = {
    "libs/setup_project_lib.py",           # RENAMED_PRODUCT_SKILLS, the old -> new table
    "tests/test_skill_rename_migration.py",  # its fixtures install skills under the old names
    "tests/test_skill_old_names_gone.py",  # this file
}

# Docs that may name an old skill only inside one passage, so an old name put back anywhere else in them (the
# uninstall lists, the product-skill table) still fails. Each entry: (first line of the passage, and the pattern
# that ends it); the passage runs from the matching start line up to, not including, the first later end match.
ALLOWED_PASSAGES = {
    "docs/skills-and-hooks.md": (re.compile(r"^## Renamed\s*$"), re.compile(r"^## ")),
    "docs/installation.md": (re.compile(r"^\*\*`upgrade` also moves product skills"), re.compile(r"^\s*$")),
}


def _allowed_line_numbers(rel, lines):
    """1-based numbers of the lines of `rel` that sit inside its allowed passage, if it has one."""
    if rel not in ALLOWED_PASSAGES:
        return set()
    start_re, end_re = ALLOWED_PASSAGES[rel]
    allowed, inside = set(), False
    for n, line in enumerate(lines, 1):
        if not inside and start_re.search(line):
            inside = True
        elif inside and end_re.search(line):
            inside = False
        if inside:
            allowed.add(n)
    return allowed

# A name is a whole token: `my-gh-pr` must not match inside `my-gh-pr-feedback`, nor `my-compact` inside `my-compacts`.
_PATTERN = re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, OLD_NAMES)) + r")(?![\w-])")


def _tracked_files():
    out = subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True)
    return [p for p in out.splitlines() if p]


def _tracked_skill_mds(base):
    """The tracked `<base>/<name>/SKILL.md` files. A contributor's own untracked skill (`.claude/` is gitignored,
    so a local one never ships) is not this repo's and is left out."""
    prefix = base + "/"
    return sorted(
        REPO / rel for rel in _tracked_files()
        if rel.startswith(prefix) and rel.endswith("/SKILL.md") and rel.count("/") == base.count("/") + 2
    )


class OldSkillNamesGoneTest(unittest.TestCase):
    def test_no_old_skill_name_outside_the_allow_list(self):
        found = []
        for rel in _tracked_files():
            if rel in ALLOWED or rel.startswith(".plans/"):
                continue
            try:
                text = (REPO / rel).read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # binary or removed-but-listed file
            lines = text.splitlines()
            skip = _allowed_line_numbers(rel, lines)
            for lineno, line in enumerate(lines, 1):
                m = None if lineno in skip else _PATTERN.search(line)
                if m:
                    found.append(f"{rel}:{lineno}: {m.group(1)}")
        self.assertEqual(found, [], "old skill names found:\n" + "\n".join(found))

    def test_every_skill_directory_name_equals_its_declared_name(self):
        # Contributor skills carry `name:` frontmatter; product skills ship without it (the installer adds it to
        # the installed copy) and are named by their `# Skill: <name>` heading, so check whichever one exists.
        checked = 0
        for base in ("skills", ".claude/skills"):
            for skill_md in _tracked_skill_mds(base):
                text = skill_md.read_text(encoding="utf-8")
                m = re.search(r"^name:\s*(\S+)\s*$", text, re.MULTILINE) or re.search(
                    r"^# Skill:\s*(\S+)\s*$", text, re.MULTILINE
                )
                self.assertIsNotNone(m, f"{skill_md} declares no name (frontmatter or `# Skill:` heading)")
                assert m is not None
                self.assertEqual(m.group(1).strip("\"'"), skill_md.parent.name, str(skill_md))
                checked += 1
        self.assertGreater(checked, 0)

    def test_the_allowed_doc_passages_exist(self):
        # If a doc is reworded so its passage no longer matches, the allowance would silently cover nothing.
        for rel in ALLOWED_PASSAGES:
            lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
            self.assertTrue(_allowed_line_numbers(rel, lines), f"no allowed passage found in {rel}")

    def test_every_skill_directory_uses_the_cr_prefix(self):
        for base in ("skills", ".claude/skills"):
            for skill_md in _tracked_skill_mds(base):
                self.assertTrue(skill_md.parent.name.startswith("cr-"), str(skill_md.parent))

    def test_every_renamed_product_skill_ships_under_its_new_name(self):
        for old, new in RENAMED_PRODUCT_SKILLS.items():
            self.assertTrue((REPO / "skills" / new / "SKILL.md").is_file(), new)
            self.assertFalse((REPO / "skills" / old).exists(), old)


if __name__ == "__main__":
    unittest.main()
