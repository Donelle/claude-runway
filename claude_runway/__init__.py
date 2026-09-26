# Not a rogue folder. This exists so 'claude-runway-setup'/'claude-runway-ingest'/
# 'claude-runway-doctor' work as installed commands in the pipx/uv install path.
"""
Tiny importable shim package backing claude-runway's 3 console-script entry
points (issue #206). Exists purely to bridge a gap `[project.scripts]`
otherwise can't close: pyproject.toml ships libs/tools/hooks/templates/
skills as wheel DATA files under <env>/src/{libs,tools,hooks,templates,
skills} (see setup.py), not as `[tool.setuptools] packages` -- so, unlike
before this issue, `tools` is no longer importable on sys.path just because
the package installed successfully. Python's own console-script mechanism
still needs SOME real, importable site-packages module for an entry point
to resolve at all, so this package is that module: it puts <env>/src on
sys.path, then imports and calls the real tools/*.py `main()` the exact
same way `python tools/setup_project.py` (the clone workflow) already does.

Locate <env>/src via `sys.prefix`, NEVER a hardcoded `Lib`/`Scripts`
(Windows) or `lib/pythonX.Y`/`bin` (macOS/Linux) segment -- pipx/`uv tool
install` both create a dedicated venv per tool, and `sys.prefix` while
running as that venv's own console script is already that venv's root,
identically across all three platforms (verified live for `uv tool install`
specifically during issue #206's own research; the wheel `data` category's
"goes to sys.prefix" rule is a wheel-spec guarantee, not OS-specific, so
pipx behaves the same way).
"""

import importlib.util
import os
import sys
from typing import Any, Callable


def _add_src_to_sys_path() -> None:
    src_dir = os.path.join(sys.prefix, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _load_tool_main(name: str) -> "Callable[[], Any]":
    """
    Load <env>/src/tools/<name>.py by exact file path, under a unique module
    name, and return its `main`. NOT `from tools.<name> import main`:
    <env>/src/tools has no __init__.py, so it's only a namespace-package
    portion, and Python prefers ANY regular `tools` package anywhere on
    sys.path (another dependency's, or one on PYTHONPATH) over a namespace
    portion, even one earlier on sys.path -- the command would then fail with
    ModuleNotFoundError or run unrelated code. Each tools/*.py script puts
    its own sibling libs/ on sys.path relative to its __file__, so loading
    it by path needs nothing else.
    """
    path = os.path.join(sys.prefix, "src", "tools", f"{name}.py")
    module_name = f"_claude_runway_tool_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"claude-runway: can't load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.main  # type: ignore[no-any-return]


def ingest_main() -> None:
    """Entry point for the `claude-runway-ingest` console script."""
    _add_src_to_sys_path()
    main = _load_tool_main("ingest_to_qdrant")

    main()


def setup_main() -> None:
    """Entry point for the `claude-runway-setup` console script."""
    _add_src_to_sys_path()
    main = _load_tool_main("setup_project")

    main()


def doctor_main() -> int:
    """
    Entry point for the `claude-runway-doctor` console script. Returns
    (rather than just calls) `tools.doctor.main()`'s int exit code --
    setuptools' generated console-script wrapper does `sys.exit(doctor_main())`,
    so the return value here IS the process exit code CI/pre-flight callers
    rely on (issue #49); swallowing it here would silently turn every
    doctor-detected mismatch into a false "exit 0".
    """
    _add_src_to_sys_path()
    main = _load_tool_main("doctor")

    return int(main())
