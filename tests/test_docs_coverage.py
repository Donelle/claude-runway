"""Drift tests: everything the toolkit ships must be named where users look for it.

Each assertion fails when something is added without its docs line:
- every `@mcp.tool()` on every `mcpservers/*_mcp_server.py` is named in
  `templates/CLAUDE.md.template` (a model in an installed project only learns
  a tool exists from that template);
- every hook in `hooks/` and every directory in `skills/` is named in
  `docs/skills-and-hooks.md`, and every skill is also named in the template;
- every environment variable the code (or a shipped skill) reads is named in
  `docs/environment-variables.md`, apart from a short allow-list of
  intentionally internal ones;
- the `qdrant-find`/`qdrant-store` descriptions, and the template outside its
  memory-bank section, never assign the lessons/decisions purpose that belongs
  to memory-bank.

Stdlib only, no network.
"""

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Variables the code reads that are deliberately not user-facing. Keep this
# short and give each entry a reason.
INTERNAL_ENV_VARS = {
    # Standard OS variables the tools read for paths, not toolkit settings.
    "HOME",
    "USERPROFILE",
    "PATH",
    # The HTML-comment marker (`<!--CLAUDE_RUNWAY_SAVINGS:...-->`) the savings
    # metrics used to travel in, still named in docstrings and tests that explain
    # why it was replaced; not a setting.
    "CLAUDE_RUNWAY_SAVINGS",
}

_TOOL_DECORATOR = re.compile(
    r"@mcp\.tool\(\s*(?:name\s*=\s*[\"']([\w-]+)[\"'])?\s*\)\s*(?:async\s+)?def\s+(\w+)"
)
_ENV_READ = re.compile(
    r"(?:\benviron(?:\.get)?|\bgetenv|\benv\w*\.get)\s*[\(\[]\s*[\"']([A-Z][A-Z0-9_]+)[\"']"
)
# `[A-Z0-9]` last, so a prose wildcard such as `CLAUDE_RUNWAY_LMSTUDIO_*` is not a name.
_RUNWAY_VAR = re.compile(r"\b(CLAUDE_RUNWAY_[A-Z0-9_]*[A-Z0-9])\b(?!\*)")


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _mentioned(name: str, text: str) -> bool:
    """True when `name` appears as a whole token (not inside a longer identifier)."""
    return re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", text) is not None


def registered_tool_names() -> dict[str, list[str]]:
    """Server file -> the names its tools are registered under."""
    out: dict[str, list[str]] = {}
    for path in sorted((ROOT / "mcpservers").glob("*_mcp_server.py")):
        found = [explicit or func for explicit, func in _TOOL_DECORATOR.findall(path.read_text(encoding="utf-8"))]
        out[path.name] = found
    return out


def env_vars_read() -> set[str]:
    names: set[str] = set()
    for directory in ("libs", "tools", "mcpservers", "hooks"):
        for path in (ROOT / directory).glob("*.py"):
            text = path.read_text(encoding="utf-8")
            names.update(_ENV_READ.findall(text))
            names.update(_RUNWAY_VAR.findall(text))
    # Shipped skills read variables through the shell, not Python.
    for path in (ROOT / "skills").glob("*/SKILL.md"):
        names.update(_RUNWAY_VAR.findall(path.read_text(encoding="utf-8")))
    return names - INTERNAL_ENV_VARS


class McpToolsInTemplate(unittest.TestCase):
    def test_every_server_registers_tools(self):
        # Guards the regex: a silent zero would make the next test vacuous.
        for server, names in registered_tool_names().items():
            self.assertTrue(names, f"{server}: found no @mcp.tool() registrations")

    def test_every_registered_tool_is_in_the_template(self):
        template = _read("templates/CLAUDE.md.template")
        missing = [
            f"{server}: {name}"
            for server, names in registered_tool_names().items()
            for name in names
            if not _mentioned(name, template)
        ]
        self.assertEqual(missing, [], "tools absent from templates/CLAUDE.md.template")


class QdrantToolsStayContentAgnostic(unittest.TestCase):
    """`qdrant-store`/`qdrant-find` take and rank whatever text they are given;
    the server has no notion of what a stored point is "for". Lessons,
    corrections and design decisions belong to memory-bank, and the template's
    memory-bank section is the one place that says so. If the qdrant tools'
    descriptions (or the rest of the template) also asked for them, two
    instructions would compete after every resolved root cause, and the notes
    written to the project collection would sit beside the indexer's chunks
    where a per-file delete can remove them."""

    # Wording that assigns a purpose of the memory-bank kind.
    _MEMORY_BANK_PURPOSE = re.compile(
        r"\blessons?\b|\bdecisions?\b|\bdecide[ds]?\b|\broot[- ]causes?\b|\brationale\b|\bcorrections?\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _qdrant_tool_descriptions() -> dict[str, str]:
        tree = ast.parse(_read("mcpservers/qdrant_mcp_server.py"))
        wanted = {"qdrant_find": "qdrant-find", "qdrant_store": "qdrant-store"}
        out = {
            wanted[node.name]: ast.get_docstring(node) or ""
            for node in ast.walk(tree)
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name in wanted
        }
        return out

    @staticmethod
    def _split_template() -> tuple[str, str]:
        """(memory-bank section, everything else) of the template."""
        template = _read("templates/CLAUDE.md.template")
        start = template.index("## Durable memory (memory-bank MCP)")
        nxt = template.find("\n## ", start + 1)
        end = len(template) if nxt == -1 else nxt
        return template[start:end], template[:start] + template[end:]

    def test_descriptions_are_found(self):
        # Guards the AST lookup: a silent miss would make the next test vacuous.
        found = self._qdrant_tool_descriptions()
        self.assertEqual(sorted(found), ["qdrant-find", "qdrant-store"])
        for name, text in found.items():
            self.assertTrue(text.strip(), f"{name}: empty description")

    def test_descriptions_do_not_assign_a_memory_bank_purpose(self):
        for name, text in self._qdrant_tool_descriptions().items():
            self.assertIsNone(
                self._MEMORY_BANK_PURPOSE.search(text),
                f"{name}'s description names lessons/decisions, which memory-bank is for",
            )

    def test_only_the_memory_bank_section_of_the_template_names_that_purpose(self):
        memory_bank, rest = self._split_template()
        self.assertIsNotNone(
            self._MEMORY_BANK_PURPOSE.search(memory_bank),
            "the memory-bank section should be where lessons/decisions are named",
        )
        hit = self._MEMORY_BANK_PURPOSE.search(rest)
        self.assertIsNone(hit, f"template outside its memory-bank section mentions {hit.group(0)!r}" if hit else "")

    def test_copy_memories_is_documented_where_it_is_used(self):
        # test_every_registered_tool_is_in_the_template only needs the name
        # anywhere in the template; copy_memories writes memories, so it must be
        # in the memory-bank section (where the dry-run-first rule lives), and
        # the memory-bank page must describe it.
        memory_bank, _ = self._split_template()
        self.assertTrue(_mentioned("copy_memories", memory_bank), "copy_memories missing from the memory-bank section")
        self.assertIn("dry_run", memory_bank.split("copy_memories", 1)[1])
        self.assertTrue(_mentioned("copy_memories", _read("docs/memory-bank.md")))

    def test_template_does_not_ask_for_notes_after_solving_a_problem(self):
        template = _read("templates/CLAUDE.md.template")
        self.assertNotIn("distilled finding", template.lower())


class HooksAndSkillsInDocs(unittest.TestCase):
    def test_every_hook_is_in_skills_and_hooks(self):
        page = _read("docs/skills-and-hooks.md")
        hooks = [p.name for p in (ROOT / "hooks").glob("*.py") if p.name != "__init__.py"]
        self.assertTrue(hooks)
        self.assertEqual([h for h in hooks if h not in page], [])

    def test_every_skill_is_in_skills_and_hooks(self):
        page = _read("docs/skills-and-hooks.md")
        skills = [p.name for p in (ROOT / "skills").iterdir() if p.is_dir()]
        self.assertTrue(skills)
        self.assertEqual([s for s in skills if not _mentioned(s, page)], [])

    def test_every_skill_is_in_the_template(self):
        template = _read("templates/CLAUDE.md.template")
        skills = [p.name for p in (ROOT / "skills").iterdir() if p.is_dir()]
        # The skill directory is `my-compact`; the template spells the slash command.
        self.assertEqual([s for s in skills if "/" + s not in template], [])


class EnvVarsInDocs(unittest.TestCase):
    def test_every_env_var_is_in_environment_variables_page(self):
        page = _read("docs/environment-variables.md")
        names = env_vars_read()
        self.assertIn("MEMORY_BANK_ID", names, "the scan found nothing it should: regex broken?")
        missing = sorted(n for n in names if not _mentioned(n, page))
        self.assertEqual(missing, [], "variables absent from docs/environment-variables.md")


if __name__ == "__main__":
    unittest.main()
