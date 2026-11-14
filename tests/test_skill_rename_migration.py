"""
Tests for `upgrade`'s my-* -> cr-* product-skill rename migration
(`libs/setup_project_lib.py`'s `plan_skill_renames`, `tools/setup_project.py`'s
`migrate_renamed_skills`/`warn_old_skill_names`), plus the guard that a skill's
directory name matches its frontmatter `name:`.

Everything runs against temp directories; the real ~/.claude is never touched.
The shipped `skills/` is a fixture here, because the migration is inert until
the real `skills/` holds the `cr-*` directories.
"""

import importlib.util
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))

from setup_project_lib import (  # noqa: E402
    RENAMED_PRODUCT_SKILLS,
    plan_skill_renames,
    set_skill_frontmatter_name,
    split_skill_frontmatter,
)


def _load_setup_project():
    spec = importlib.util.spec_from_file_location("setup_project", REPO_ROOT / "tools" / "setup_project.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fm(name: str, description: str = "the original description") -> str:
    return f'---\nname: {name}\ndescription: "{description}"\n---\n\n'


class SkillRenameBase(unittest.TestCase):
    # Two of the six mapped skills are enough; each test says which ones it installs.
    SHIPPED = ("cr-compact", "cr-resume")

    def setUp(self):
        self.mod = _load_setup_project()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.tools = root / "tools"
        self.skills = root / "skills-dest"
        for name in self.SHIPPED:
            self._ship(name)

    def _ship(self, name):
        d = self.tools / "skills" / name
        d.mkdir(parents=True, exist_ok=True)
        # Bytes, not write_text: write_text would turn newlines into CRLF on Windows.
        (d / "SKILL.md").write_bytes(f"# Skill: {name}\nnew body\n".encode())

    def _install(self, name, text):
        (self.skills / name).mkdir(parents=True, exist_ok=True)
        (self.skills / name / "SKILL.md").write_bytes(text.encode())

    def _read(self, name):
        return (self.skills / name / "SKILL.md").read_bytes().decode()

    def _migrate(self, **kw):
        kw.setdefault("auto_yes", True)
        kw.setdefault("dry_run", False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            done = self.mod.migrate_renamed_skills(self.skills, tools_repo_dir=self.tools, **kw)
        return done, buf.getvalue()

    def _install_old(self):
        self._install("my-compact", _fm("my-compact") + "# Skill: my-compact\nold body\n")
        self._install("my-resume", _fm("my-resume", "customized") + "# Skill: my-resume\nold body\n")


class Migration(SkillRenameBase):
    def test_dry_run_writes_nothing_and_never_prompts(self):
        self._install_old()
        done, out = self._migrate(dry_run=True, auto_yes=False, prompt_fn=lambda p: self.fail("prompted"))
        self.assertEqual(done, ["my-compact", "my-resume"])
        self.assertIn("Would install cr-compact", out)
        self.assertTrue((self.skills / "my-compact" / "SKILL.md").is_file())
        self.assertFalse((self.skills / "cr-compact").exists())

    def test_auto_yes_installs_new_names_with_new_frontmatter_and_removes_old(self):
        self._install_old()
        done, out = self._migrate()
        self.assertEqual(done, ["my-compact", "my-resume"])
        self.assertEqual(self._read("cr-compact"), _fm("cr-compact") + "# Skill: cr-compact\nnew body\n")
        # A customized description is carried over; only name: changes.
        self.assertEqual(self._read("cr-resume"), _fm("cr-resume", "customized") + "# Skill: cr-resume\nnew body\n")
        self.assertFalse((self.skills / "my-compact").exists())
        self.assertFalse((self.skills / "my-resume").exists())
        # The path of what is removed is shown.
        self.assertIn(str(self.skills / "my-compact"), out)

    def test_declining_one_leaves_it_untouched(self):
        self._install_old()
        before = self._read("my-resume")
        done, _ = self._migrate(auto_yes=False, prompt_fn=lambda p: p.old_name == "my-compact")
        self.assertEqual(done, ["my-compact"])
        self.assertFalse((self.skills / "cr-resume").exists())
        self.assertEqual(self._read("my-resume"), before)
        self.assertTrue((self.skills / "cr-compact" / "SKILL.md").is_file())

    def test_rerun_is_a_no_op(self):
        self._install_old()
        self._migrate()
        snapshot = sorted(str(p) for p in self.skills.rglob("*"))
        done, out = self._migrate()
        self.assertEqual(done, [])
        self.assertEqual(out, "")
        self.assertEqual(sorted(str(p) for p in self.skills.rglob("*")), snapshot)

    def test_non_claude_runway_my_skill_is_never_touched(self):
        self._install_old()
        self._install("my-code-it", _fm("my-code-it") + "mine\n")
        # Even a user skill sharing a prefix with a mapped name stays put.
        self._install("my-compact-extra", _fm("my-compact-extra") + "mine\n")
        self._migrate()
        self.assertEqual(self._read("my-code-it"), _fm("my-code-it") + "mine\n")
        self.assertEqual(self._read("my-compact-extra"), _fm("my-compact-extra") + "mine\n")

    def test_old_copy_without_frontmatter_gets_the_shipped_file_whole(self):
        self._install("my-compact", "# Skill: my-compact\nold body\n")
        self._migrate()
        self.assertEqual(self._read("cr-compact"), "# Skill: cr-compact\nnew body\n")

    def test_existing_new_copy_is_kept_and_only_the_old_directory_goes(self):
        self._install_old()
        mine = _fm("cr-compact", "hand edited") + "mine\n"
        self._install("cr-compact", mine)
        self._migrate()
        self.assertEqual(self._read("cr-compact"), mine)
        self.assertFalse((self.skills / "my-compact").exists())

    def test_existing_new_copy_with_a_stale_name_line_gets_the_new_name(self):
        self._install("my-compact", _fm("my-compact") + "old\n")
        self._install("cr-compact", _fm("my-compact", "hand edited") + "kept\n")
        self._migrate()
        self.assertEqual(self._read("cr-compact"), _fm("cr-compact", "hand edited") + "kept\n")

    def test_existing_frontmatterless_new_copy_gains_the_old_frontmatter_before_removal(self):
        # What `init --install-skills` leaves behind when the shipped cr-* has no frontmatter.
        self._install("my-resume", _fm("my-resume", "customized") + "# Skill: my-resume\nold body\n")
        self._install("cr-resume", "# Skill: cr-resume\nbody on disk\n")
        self._migrate()
        self.assertEqual(self._read("cr-resume"), _fm("cr-resume", "customized") + "# Skill: cr-resume\nbody on disk\n")
        self.assertFalse((self.skills / "my-resume").exists())

    def test_symlinked_new_destination_is_skipped_and_the_old_copy_kept(self):
        self._install("my-compact", _fm("my-compact") + "old\n")
        (self.skills / "cr-compact").mkdir()
        try:
            os.symlink(self.skills / "my-compact" / "SKILL.md", self.skills / "cr-compact" / "SKILL.md")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        done, _ = self._migrate()
        self.assertEqual(done, [])
        self.assertTrue((self.skills / "my-compact" / "SKILL.md").is_file())

    def test_directory_named_skill_md_is_not_a_valid_replacement(self):
        self._install("my-compact", _fm("my-compact") + "old\n")
        (self.skills / "cr-compact" / "SKILL.md").mkdir(parents=True)
        done, _ = self._migrate()
        self.assertEqual(done, [])
        self.assertTrue((self.skills / "my-compact").is_dir())

    def test_replacement_removed_while_the_prompt_is_open_is_recreated(self):
        self._install("my-compact", _fm("my-compact") + "old\n")
        self._install("cr-compact", _fm("cr-compact") + "kept\n")

        def ask(plan):
            plan.new_dest.unlink()
            return True

        done, _ = self._migrate(auto_yes=False, prompt_fn=ask)
        self.assertEqual(done, ["my-compact"])
        self.assertEqual(self._read("cr-compact"), _fm("cr-compact") + "kept\n")
        self.assertFalse((self.skills / "my-compact").exists())

    def test_replacement_appearing_while_the_prompt_is_open_is_not_overwritten(self):
        self._install("my-compact", _fm("my-compact") + "old\n")
        mine = _fm("cr-compact", "customized") + "mine\n"

        def ask(plan):
            self._install("cr-compact", mine)
            return True

        done, _ = self._migrate(auto_yes=False, prompt_fn=ask)
        self.assertEqual(done, [])
        self.assertEqual(self._read("cr-compact"), mine)
        self.assertTrue((self.skills / "my-compact").is_dir())

    def test_symlinked_old_directory_is_never_followed_or_removed(self):
        real = self.skills.parent / "elsewhere"
        real.mkdir()
        (real / "SKILL.md").write_bytes(b"# Skill: my-compact\n")
        self.skills.mkdir()
        try:
            os.symlink(real, self.skills / "my-compact", target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        done, _ = self._migrate()
        self.assertEqual(done, [])
        self.assertTrue((real / "SKILL.md").is_file())
        self.assertFalse((self.skills / "cr-compact").exists())


class InertWithoutShippedNewNames(SkillRenameBase):
    SHIPPED = ()

    def test_with_no_cr_directory_shipped_nothing_is_offered(self):
        self._install_old()
        self.assertEqual(plan_skill_renames(self.tools, self.skills), [])
        done, out = self._migrate(dry_run=True)
        self.assertEqual((done, out), ([], ""))
        self.assertTrue((self.skills / "my-compact" / "SKILL.md").is_file())

    def test_with_no_skills_directory_at_all_it_does_not_raise(self):
        self.assertEqual(plan_skill_renames(self.tools / "missing", self.skills), [])

    def test_this_repos_real_skills_offer_every_old_name_a_move(self):
        # Against the real `skills/` directory, an install holding all six old names is offered
        # each move, so the table and the shipped directories cannot drift apart.
        for old in RENAMED_PRODUCT_SKILLS:
            self._install(old, _fm(old) + f"# Skill: {old}\nold body\n")
        plans = plan_skill_renames(REPO_ROOT, self.skills)
        self.assertEqual({p.old_name: p.new_name for p in plans}, RENAMED_PRODUCT_SKILLS)


class InitWarning(SkillRenameBase):
    def _warn(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.warn_old_skill_names(self.skills, tools_repo_dir=self.tools)
        return buf.getvalue()

    def test_warns_about_an_old_name_still_installed_and_points_at_upgrade(self):
        self._install_old()
        out = self._warn()
        self.assertIn("my-compact", out)
        self.assertIn("my-resume", out)
        self.assertIn("upgrade", out)
        # A skills-only user must be told `upgrade` works without a project path.
        self.assertIn("no project path", out)

    def test_silent_when_no_old_name_is_installed(self):
        self._install("cr-compact", "# Skill: cr-compact\n")
        self.assertEqual(self._warn(), "")

    def test_silent_for_an_old_name_whose_new_name_has_not_shipped(self):
        self._install("my-savings", "# Skill: my-savings\n")  # cr-savings is not shipped here
        self.assertEqual(self._warn(), "")


class FrontmatterRewrite(unittest.TestCase):
    def test_only_the_name_line_changes(self):
        prefix = split_skill_frontmatter(_fm("my-compact", "keep me").encode())[0]
        self.assertEqual(set_skill_frontmatter_name(prefix, "cr-compact"), _fm("cr-compact", "keep me").encode())

    def test_a_description_that_mentions_name_is_not_touched(self):
        text = '---\ndescription: "name: x"\nname: my-compact\n---\n\n'
        out = set_skill_frontmatter_name(text.encode(), "cr-compact").decode()
        self.assertEqual(out, '---\ndescription: "name: x"\nname: cr-compact\n---\n\n')

    def test_missing_name_line_is_added_and_crlf_is_kept(self):
        text = '---\r\ndescription: "d"\r\n---\r\n\r\n'
        out = set_skill_frontmatter_name(text.encode(), "cr-compact")
        self.assertEqual(out, b'---\r\nname: cr-compact\r\ndescription: "d"\r\n---\r\n\r\n')

    def test_bom_survives(self):
        text = b"\xef\xbb\xbf" + _fm("my-compact").encode()
        out = set_skill_frontmatter_name(text, "cr-compact")
        self.assertEqual(out, b"\xef\xbb\xbf" + _fm("cr-compact").encode())


class DirectoryMatchesFrontmatterName(unittest.TestCase):
    """A skill's directory name is the name it is installed and listed under; a
    frontmatter `name:` that disagrees makes the two diverge (the failure a
    rename is most likely to leave behind). Shipped `skills/` files may carry no
    frontmatter at all (the installer's copy supplies it); only a present `name:`
    is checked."""

    def test_every_skill_directory_matches_its_frontmatter_name(self):
        checked = 0
        for root in (REPO_ROOT / "skills", REPO_ROOT / ".claude" / "skills"):
            for skill_md in sorted(root.glob("*/SKILL.md")):
                prefix = split_skill_frontmatter(skill_md.read_bytes())[0].decode("utf-8")
                name = None
                for line in prefix.splitlines():
                    if line.startswith("name:"):
                        name = line[len("name:"):].strip().strip("\"'")
                        break
                if name is None:
                    continue
                checked += 1
                self.assertEqual(name, skill_md.parent.name, f"{skill_md}: name: {name!r} != directory")
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
