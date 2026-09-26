#!/usr/bin/env python3
"""
Computes the wheel's `data_files` list at BUILD time (issue #206) --
supplements pyproject.toml's static `[project]`/`[tool.setuptools]` tables
with the one thing PEP 621 has no declarative equivalent for: shipping
libs/, tools/, hooks/, templates/, and skills/ as wheel DATA files under
<env>/src/{libs,tools,hooks,templates,skills} instead of as
`[tool.setuptools] packages` (which previously put them directly in
site-packages -- the generic top-level-namespace layout this issue moves
away from; see pyproject.toml's own `[tool.setuptools]` comment block).

A GLOB, not a hand-typed file list, for exactly the reason the issue asks
for: a new module dropped into any of these directories ships automatically
next time a wheel is built, with no pyproject.toml/setup.py edit required --
a hand-typed list could silently miss one and nobody would notice until a
runtime ImportError/FileNotFoundError in an installed environment. Failing
loudly (`RuntimeError`) if a glob matches nothing catches the opposite
mistake -- a typo'd pattern -- at build time instead of shipping an empty
directory silently.

skills/ needs one `data_files` tuple PER skills/<name>/ subdirectory, not one
shared "src/skills" target covering all of them -- setuptools' `data_files`
copies every matched source file into the SAME target directory by BASENAME
ALONE, so a single shared target would collapse every skill's identically
named SKILL.md into just one, silently dropping every skill but the last
(confirmed live during this issue's own development: a shared
`"src/skills": ["skills/*/SKILL.md"]` entry kept only the alphabetically
last skill's SKILL.md in the built wheel). One tuple per subdirectory
preserves each skill's own name as part of the install path instead.

Every script under tools/ and hooks/ already computes its own repo root
from `__file__` and resolves libs/'s and templates/'s content relative to
that (e.g. tools/setup_project.py's `TOOLS_REPO_DIR`, each hook's `_root`/
`_candidates` lookup) -- that arithmetic keeps resolving correctly here
because these five directories are still installed as SIBLINGS of each
other, just under <env>/src/ instead of directly under site-packages.
"""

import glob
import os

from setuptools import setup

# setuptools REQUIRES data_files sources to be relative, "/"-separated paths
# relative to this file's own directory -- it errors outright on an absolute
# path (confirmed live: "setup script specifies an absolute path"). PEP517
# build frontends invoke setup.py with the repo root as cwd, so a bare
# relative glob (no os.path.join against an absolute root) is both correct
# and required here.


def _to_posix(path: str) -> str:
    """setuptools also requires "/"-separated paths even on Windows --
    glob.glob returns OS-native separators, so normalize explicitly rather
    than relying on this only ever running on POSIX."""
    return path.replace(os.sep, "/")


def _flat_data_files(directory: str, pattern: str) -> tuple[str, list[str]]:
    """One data_files tuple for a flat (non-nested) source directory, e.g.
    libs/*.py -> <env>/src/libs/*.py."""
    matches = sorted(_to_posix(p) for p in glob.glob(os.path.join(directory, pattern)))
    return (f"src/{directory}", matches)


def _skill_data_files() -> list[tuple[str, list[str]]]:
    """One data_files tuple PER skills/<name>/ subdirectory -- see this
    module's own docstring for why a single shared target silently drops
    every skill but one."""
    entries = []
    for skill_dir in sorted(glob.glob(os.path.join("skills", "*"))):
        if not os.path.isdir(skill_dir):
            continue
        name = os.path.basename(skill_dir)
        matches = sorted(_to_posix(p) for p in glob.glob(os.path.join(skill_dir, "*.md")))
        if matches:
            entries.append((f"src/skills/{name}", matches))
    return entries


def data_files() -> list[tuple[str, list[str]]]:
    entries = [
        _flat_data_files("libs", "*.py"),
        _flat_data_files("tools", "*.py"),
        _flat_data_files("hooks", "*.py"),
        _flat_data_files("templates", "*.template"),
    ]
    entries.extend(_skill_data_files())
    # A glob that matches nothing almost always means a typo'd pattern or a
    # directory that moved -- fail the build loudly instead of silently
    # shipping an empty/missing data_files target.
    for target, matches in entries:
        if not matches:
            raise RuntimeError(
                f"setup.py's data_files() matched no files for target {target!r} -- "
                "check the glob pattern against the directories actually on disk."
            )
    return entries


setup(data_files=data_files())
