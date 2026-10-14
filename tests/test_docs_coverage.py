"""Drift tests: everything the toolkit ships must be named where users look for it.

Each assertion fails when something is added without its docs line:
- every `@mcp.tool()` on every `tools/*_mcp_server.py` is named in
  `templates/CLAUDE.md.template` (a model in an installed project only learns
  a tool exists from that template);
- every hook in `hooks/` and every directory in `skills/` is named in
  `docs/skills-and-hooks.md`, and every skill is also named in the template;
- every environment variable the code (or a shipped skill) reads is named in
  `docs/environment-variables.md`, apart from a short allow-list of
  intentionally internal ones.

Stdlib only, no network.
"""

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
    # The savings footer's HTML-comment marker (`<!--CLAUDE_RUNWAY_SAVINGS:...-->`),
    # not a setting.
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
    for path in sorted((ROOT / "tools").glob("*_mcp_server.py")):
        found = [explicit or func for explicit, func in _TOOL_DECORATOR.findall(path.read_text(encoding="utf-8"))]
        out[path.name] = found
    return out


def env_vars_read() -> set[str]:
    names: set[str] = set()
    for directory in ("libs", "tools", "hooks"):
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
