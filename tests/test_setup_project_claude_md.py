#!/usr/bin/env python3
"""Tests for `init`'s CLAUDE.md step (libs/setup_project_lib.py's
`extract_claude_md_sections`/`plan_claude_md`/`run_setup(claude_md=...)` and
tools/setup_project.py's `cmd_init`).

`init` used to write `.mcp.json` and `.claude/settings.json` only, so a project
configured from the CLI (and every pipx/`uv tool install` user, who cannot run
the /my-setup-clauderunway skill) got working servers with no usage guidance.
These pin the rules the skill's step 6 followed: verbatim template sections, a
missing section appended, the section of a server that is not configured
removed, nothing else touched, and a re-run that changes nothing.

Stdlib-only, no network, real templates:

    .venv/bin/python -m unittest discover -s tests
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from setup_project_lib import (  # noqa: E402
    CLAUDE_MD_ANALYTICS_HEADING,
    CLAUDE_MD_COMPRESS_HEADING,
    CLAUDE_MD_MEMORY_BANK_HEADING,
    CLAUDE_MD_QDRANT_HEADING,
    CLAUDE_MD_REVIEWER_HEADING,
    CLAUDE_MD_SECTION_SERVERS,
    MalformedConfigError,
    extract_claude_md_sections,
    plan_claude_md,
    run_setup,
    wanted_claude_md_headings,
)
from test_setup_project_cli import _init_args, _load_setup_project  # noqa: E402

TEMPLATE_TEXT = (REPO_ROOT / "templates" / "CLAUDE.md.template").read_text(encoding="utf-8")
SECTIONS = extract_claude_md_sections(TEMPLATE_TEXT)
FULL = [CLAUDE_MD_QDRANT_HEADING, CLAUDE_MD_MEMORY_BANK_HEADING, CLAUDE_MD_COMPRESS_HEADING]
QDRANT_ONLY = [CLAUDE_MD_QDRANT_HEADING, CLAUDE_MD_MEMORY_BANK_HEADING]
HOME = Path("/home/user")


def _headings(text):
    return [line.rstrip() for line in text.splitlines() if line.startswith("## ")]


class ExtractSections(unittest.TestCase):
    def test_every_managed_section_is_cut_verbatim_from_the_template(self):
        for heading, _server in CLAUDE_MD_SECTION_SERVERS:
            with self.subTest(heading=heading):
                section = SECTIONS[heading]
                self.assertTrue(section.startswith(heading + "\n"))
                self.assertIn(section, TEMPLATE_TEXT)
                self.assertFalse(section.endswith("\n"))
                # Exactly one heading: the section stops before the next one.
                self.assertEqual(_headings(section), [heading])

    def test_the_template_note_is_never_part_of_a_section(self):
        for section in SECTIONS.values():
            self.assertNotIn("<!--", section)
            self.assertNotIn("Copy the sections below", section)

    def test_a_section_runs_to_the_next_heading(self):
        # The section's last line is the template's last non-blank line before the next heading.
        lines = TEMPLATE_TEXT.splitlines()
        start = lines.index(CLAUDE_MD_QDRANT_HEADING)
        nxt = lines.index(CLAUDE_MD_MEMORY_BANK_HEADING)
        expected = "\n".join(lines[start:nxt]).rstrip()
        self.assertEqual(SECTIONS[CLAUDE_MD_QDRANT_HEADING], expected)

    def test_a_missing_managed_heading_is_a_loud_error(self):
        broken = TEMPLATE_TEXT.replace(CLAUDE_MD_ANALYTICS_HEADING, "## Analytics renamed")
        with self.assertRaises(ValueError) as ctx:
            extract_claude_md_sections(broken)
        self.assertIn(CLAUDE_MD_ANALYTICS_HEADING, str(ctx.exception))


class WantedHeadings(unittest.TestCase):
    def test_each_section_follows_its_server(self):
        servers = {"qdrant": {}, "codebase-indexer": {}, "memory-bank": {}}
        self.assertEqual(wanted_claude_md_headings(servers), QDRANT_ONLY)
        servers.update({"local-compress": {}, "analytics": {}, "codebase-reviewer": {}})
        self.assertEqual(
            wanted_claude_md_headings(servers),
            FULL + [CLAUDE_MD_ANALYTICS_HEADING, CLAUDE_MD_REVIEWER_HEADING],
        )


class PlanNewFile(unittest.TestCase):
    def test_no_file_creates_the_wanted_sections_a_blank_line_apart(self):
        content, added, removed = plan_claude_md(None, SECTIONS, FULL)
        self.assertEqual(content, "\n\n".join(SECTIONS[h] for h in FULL) + "\n")
        self.assertEqual(added, FULL)
        self.assertEqual(removed, [])

    def test_qdrant_only_leaves_out_local_compression(self):
        content, _, _ = plan_claude_md(None, SECTIONS, QDRANT_ONLY)
        self.assertEqual(_headings(content), QDRANT_ONLY)


class PlanExistingFile(unittest.TestCase):
    def test_appends_only_what_is_missing_after_one_blank_line(self):
        existing = "# My project\n\nOwn notes.\n\n" + SECTIONS[CLAUDE_MD_QDRANT_HEADING] + "\n"
        content, added, removed = plan_claude_md(existing, SECTIONS, FULL)
        self.assertEqual(added, [CLAUDE_MD_MEMORY_BANK_HEADING, CLAUDE_MD_COMPRESS_HEADING])
        self.assertEqual(removed, [])
        self.assertTrue(content.startswith(existing))
        self.assertEqual(
            content[len(existing):],
            "\n" + SECTIONS[CLAUDE_MD_MEMORY_BANK_HEADING] + "\n\n" + SECTIONS[CLAUDE_MD_COMPRESS_HEADING] + "\n",
        )

    def test_a_file_without_a_trailing_newline_still_gets_a_blank_line(self):
        content, _, _ = plan_claude_md("Own notes.", SECTIONS, [CLAUDE_MD_QDRANT_HEADING])
        self.assertEqual(content, "Own notes.\n\n" + SECTIONS[CLAUDE_MD_QDRANT_HEADING] + "\n")

    def test_an_empty_file_gets_no_leading_blank_line(self):
        content, _, _ = plan_claude_md("", SECTIONS, [CLAUDE_MD_QDRANT_HEADING])
        self.assertEqual(content, SECTIONS[CLAUDE_MD_QDRANT_HEADING] + "\n")

    def test_rerun_is_a_no_op(self):
        first, _, _ = plan_claude_md(None, SECTIONS, FULL)
        self.assertEqual(plan_claude_md(first, SECTIONS, FULL), (None, [], []))

    def test_an_edited_section_is_left_alone(self):
        existing = CLAUDE_MD_QDRANT_HEADING + "\n\nMy own shorter version.\n\n" + SECTIONS[CLAUDE_MD_MEMORY_BANK_HEADING] + "\n"
        self.assertEqual(plan_claude_md(existing, SECTIONS, QDRANT_ONLY), (None, [], []))

    def test_qdrant_only_removes_a_trailing_local_compression_section(self):
        full, _, _ = plan_claude_md(None, SECTIONS, FULL)
        content, added, removed = plan_claude_md(full, SECTIONS, QDRANT_ONLY)
        self.assertEqual(added, [])
        self.assertEqual(removed, [CLAUDE_MD_COMPRESS_HEADING])
        # Exactly what a qdrant-only run would have created: no gap left at the end.
        self.assertEqual(content, plan_claude_md(None, SECTIONS, QDRANT_ONLY)[0])

    def test_removing_a_middle_section_keeps_the_surrounding_content(self):
        existing = (
            "# Project\n\nIntro.\n\n"
            + SECTIONS[CLAUDE_MD_COMPRESS_HEADING]
            + "\n\n## Team conventions\n\nUse tabs.\n\n### Sub-heading stays\n"
        )
        content, _, removed = plan_claude_md(existing, SECTIONS, [])
        self.assertEqual(removed, [CLAUDE_MD_COMPRESS_HEADING])
        self.assertEqual(content, "# Project\n\nIntro.\n\n## Team conventions\n\nUse tabs.\n\n### Sub-heading stays\n")

    def test_a_section_ends_at_a_level_one_heading_too(self):
        existing = SECTIONS[CLAUDE_MD_ANALYTICS_HEADING] + "\n\n# Appendix\n\nKeep me.\n"
        content, _, _ = plan_claude_md(existing, SECTIONS, [])
        self.assertEqual(content, "# Appendix\n\nKeep me.\n")

    def test_a_level_three_heading_inside_a_removed_section_goes_with_it(self):
        existing = "Intro.\n\n" + CLAUDE_MD_COMPRESS_HEADING + "\n\nText.\n\n### Detail\n\nMore.\n"
        content, _, _ = plan_claude_md(existing, SECTIONS, [])
        self.assertEqual(content, "Intro.\n")

    def test_every_duplicate_of_an_unwanted_section_is_removed(self):
        compress = SECTIONS[CLAUDE_MD_COMPRESS_HEADING]
        existing = "Intro.\n\n" + compress + "\n\n## Mine\n\nKeep.\n\n" + compress + "\n"
        content, _, removed = plan_claude_md(existing, SECTIONS, [])
        self.assertEqual(removed, [CLAUDE_MD_COMPRESS_HEADING])
        self.assertEqual(content, "Intro.\n\n## Mine\n\nKeep.\n")

    def test_analytics_and_reviewer_sections_are_added_and_removed_with_their_servers(self):
        wanted = FULL + [CLAUDE_MD_ANALYTICS_HEADING, CLAUDE_MD_REVIEWER_HEADING]
        with_both, _, _ = plan_claude_md(None, SECTIONS, wanted)
        self.assertEqual(_headings(with_both), wanted)
        content, _, removed = plan_claude_md(with_both, SECTIONS, FULL)
        self.assertEqual(removed, [CLAUDE_MD_ANALYTICS_HEADING, CLAUDE_MD_REVIEWER_HEADING])
        self.assertEqual(content, plan_claude_md(None, SECTIONS, FULL)[0])

    def test_a_heading_inside_a_code_fence_is_not_a_section(self):
        existing = "Example:\n\n```markdown\n" + CLAUDE_MD_COMPRESS_HEADING + "\n```\n\n~~~\n" + CLAUDE_MD_QDRANT_HEADING + "\n~~~\n"
        # Not removed (it is an example, not a section) ...
        content, added, removed = plan_claude_md(existing, SECTIONS, [CLAUDE_MD_QDRANT_HEADING])
        self.assertEqual(removed, [])
        # ... and not counted as present either.
        self.assertEqual(added, [CLAUDE_MD_QDRANT_HEADING])
        self.assertTrue(content.startswith(existing))

    def test_a_heading_after_a_closed_fence_is_seen(self):
        existing = "```\ncode\n```\n\n" + SECTIONS[CLAUDE_MD_QDRANT_HEADING] + "\n"
        self.assertEqual(plan_claude_md(existing, SECTIONS, [CLAUDE_MD_QDRANT_HEADING]), (None, [], []))

    def test_crlf_files_stay_crlf(self):
        existing = "Own notes.\r\n"
        content, _, _ = plan_claude_md(existing, SECTIONS, QDRANT_ONLY)
        self.assertNotIn("\n", content.replace("\r\n", ""))
        self.assertEqual(plan_claude_md(content, SECTIONS, QDRANT_ONLY), (None, [], []))
        full, _, _ = plan_claude_md(content, SECTIONS, FULL)
        self.assertEqual(plan_claude_md(full, SECTIONS, QDRANT_ONLY)[0], content)

    def test_a_byte_order_mark_is_kept_and_does_not_hide_the_first_heading(self):
        existing = "﻿" + SECTIONS[CLAUDE_MD_QDRANT_HEADING] + "\n"
        self.assertEqual(plan_claude_md(existing, SECTIONS, [CLAUDE_MD_QDRANT_HEADING]), (None, [], []))
        content, added, _ = plan_claude_md(existing, SECTIONS, QDRANT_ONLY)
        self.assertEqual(added, [CLAUDE_MD_MEMORY_BANK_HEADING])
        self.assertTrue(content.startswith("﻿" + CLAUDE_MD_QDRANT_HEADING))


class RunSetupClaudeMd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.target = Path(self._tmp.name) / "proj"
        self.target.mkdir()

    def test_plans_without_writing(self):
        result = run_setup(self.target, REPO_ROOT, home_dir=HOME)
        self.assertFalse((self.target / "CLAUDE.md").exists())
        self.assertTrue(result.claude_md.created)
        self.assertEqual(_headings(result.claude_md.content), FULL)
        self.assertTrue(any(c.startswith("CLAUDE.md (new file with Vector memory") for c in result.changes))

    def test_sections_follow_the_written_servers(self):
        result = run_setup(self.target, REPO_ROOT, home_dir=HOME, include_compress=False, track_savings=True)
        self.assertEqual(_headings(result.claude_md.content), QDRANT_ONLY)
        result = run_setup(self.target, REPO_ROOT, home_dir=HOME, track_savings=True, codebase_reviewer=True)
        self.assertEqual(
            _headings(result.claude_md.content), FULL + [CLAUDE_MD_ANALYTICS_HEADING, CLAUDE_MD_REVIEWER_HEADING]
        )

    def test_claude_md_false_plans_nothing(self):
        result = run_setup(self.target, REPO_ROOT, home_dir=HOME, claude_md=False)
        self.assertIsNone(result.claude_md)
        self.assertFalse(any("CLAUDE.md" in c for c in result.changes))

    def test_up_to_date_file_adds_no_change_entry(self):
        content = run_setup(self.target, REPO_ROOT, home_dir=HOME).claude_md.content
        (self.target / "CLAUDE.md").write_text(content, encoding="utf-8")
        result = run_setup(self.target, REPO_ROOT, home_dir=HOME)
        self.assertIsNone(result.claude_md.content)
        self.assertFalse(any("CLAUDE.md" in c for c in result.changes))

    def test_template_comes_from_templates_dir(self):
        # A tool install reads <env>/src/templates; run_setup must use the dir
        # it is given, not a path baked in relative to this checkout.
        templates = Path(self._tmp.name) / "templates"
        templates.mkdir()
        for name in ("mcp.json.template", "settings.json.template"):
            (templates / name).write_bytes((REPO_ROOT / "templates" / name).read_bytes())
        (templates / "CLAUDE.md.template").write_text(
            TEMPLATE_TEXT.replace("Prefer `qdrant-find` over Grep", "MARKER from templates_dir"), encoding="utf-8"
        )
        result = run_setup(self.target, REPO_ROOT, home_dir=HOME, templates_dir=templates)
        self.assertIn("MARKER from templates_dir", result.claude_md.content)

    def test_a_directory_named_claude_md_is_a_clean_error(self):
        (self.target / "CLAUDE.md").mkdir()
        with self.assertRaises(MalformedConfigError):
            run_setup(self.target, REPO_ROOT, home_dir=HOME)

    def test_non_utf8_bytes_survive(self):
        (self.target / "CLAUDE.md").write_bytes(b"caf\xe9 notes\n")
        result = run_setup(self.target, REPO_ROOT, home_dir=HOME)
        encoded = result.claude_md.content.encode("utf-8", errors="surrogateescape")
        self.assertTrue(encoded.startswith(b"caf\xe9 notes\n\n## Vector memory"))


class InitCliClaudeMd(unittest.TestCase):
    def setUp(self):
        self.mod = _load_setup_project()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.target = Path(self._tmp.name) / "proj"
        self.target.mkdir()
        self.claude_md = self.target / "CLAUDE.md"

    def _init(self, **over):
        over.setdefault("dry_run", False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(_init_args(self.target, **over))
        return buf.getvalue()

    def test_init_creates_claude_md(self):
        out = self._init()
        self.assertEqual(_headings(self.claude_md.read_text(encoding="utf-8")), FULL)
        self.assertIn("Writing CLAUDE.md (new file with Vector memory (Qdrant MCP)", out)

    def test_rerun_leaves_the_file_byte_identical_and_says_so(self):
        self._init()
        before = self.claude_md.read_bytes()
        mtime = self.claude_md.stat().st_mtime_ns
        out = self._init()
        self.assertEqual(self.claude_md.read_bytes(), before)
        self.assertEqual(self.claude_md.stat().st_mtime_ns, mtime)
        self.assertIn("CLAUDE.md already up to date", out)

    def test_dry_run_writes_nothing_and_names_the_sections(self):
        out = self._init(dry_run=True)
        self.assertFalse(self.claude_md.exists())
        self.assertIn("Would write CLAUDE.md (new file with Vector memory (Qdrant MCP)", out)

    def test_qdrant_only_rerun_removes_local_compression_and_keeps_own_text(self):
        self.claude_md.write_text("# Mine\n\nKeep this.\n", encoding="utf-8")
        self._init()
        out = self._init(qdrant_only=True)
        text = self.claude_md.read_text(encoding="utf-8")
        self.assertEqual(_headings(text), QDRANT_ONLY)
        self.assertTrue(text.startswith("# Mine\n\nKeep this.\n\n"))
        self.assertIn("remove Local compression (LM Studio MCP)", out)

    def test_skip_claude_md_leaves_it_alone(self):
        self.claude_md.write_text("Mine.\n", encoding="utf-8")
        out = self._init(skip_claude_md=True)
        self.assertEqual(self.claude_md.read_text(encoding="utf-8"), "Mine.\n")
        self.assertNotIn("CLAUDE.md", out)
        # The config is still written.
        self.assertIn("qdrant", json.loads((self.target / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"])

    def test_skip_claude_md_creates_nothing(self):
        self._init(skip_claude_md=True)
        self.assertFalse(self.claude_md.exists())

    def test_a_symlinked_claude_md_keeps_its_link(self):
        shared = self.target / "AGENTS.md"
        shared.write_text("Shared.\n", encoding="utf-8")
        try:
            self.claude_md.symlink_to("AGENTS.md")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not available")
        self._init()
        self.assertTrue(self.claude_md.is_symlink())
        self.assertEqual(_headings(shared.read_text(encoding="utf-8")), FULL)

    def test_parse_args_accepts_skip_claude_md(self):
        argv = ["setup_project.py", "init", str(self.target), "--skip-claude-md"]
        old = sys.argv
        sys.argv = argv
        try:
            args = self.mod.parse_args()
        finally:
            sys.argv = old
        self.assertTrue(args.skip_claude_md)


if __name__ == "__main__":
    unittest.main()
