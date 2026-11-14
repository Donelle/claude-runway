"""Shared pieces of the live review harnesses (`harness/live_*.py`).

These scripts run the codebase-reviewer against a REAL LM Studio, which the unit
tests deliberately never do. They exist because the unit tests prove the
plumbing and nothing about review quality: whether a given model finds the
bugs, how many false alarms it raises, and what the loader and chunker actually
did on the wire.

What lives here, none of it needing LM Studio itself:
- a thread-safe event `Recorder`, and wrappers that log every LM Studio REST
  call and every completion request into it;
- `build_repo`, which turns a `cases.Case` into a throwaway git repository;
- `score`, which matches findings against a case's known bugs and decoys;
- command-line options shared by the scripts, including `--config`, a JSON file naming the
  reviewers (models, prompts) to test and the stages to test them in;
- `run_reasoning`, the one place that drives the reasoning stage end to end.
"""

import argparse
from contextlib import contextmanager
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per request buries the report
REPO_ROOT = Path(__file__).resolve().parent.parent
for _directory in (REPO_ROOT / "libs", Path(__file__).resolve().parent):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from cases import Case, bug_line, line_of  # noqa: E402

# The development model for reviewers and verifiers (see README.md, "Code Reviewer Models"): the -Instruct build,
# since the base build never ends its turn. Classifiers use Qwen2.5-Coder-3B-Instruct-8bit.
DEFAULT_MODEL = "qwen2.5-coder-7b-instruct-mlx"
DEFAULT_CONTEXT = 8192
DEFAULT_LENS = (
    "You review Python changes for defects that would crash, lose data, or return a wrong result, "
    "and for security problems such as shell, SQL or path injection. Be specific and skip style remarks."
)
# Reported lines are the model's reading of a diff, and drift by a few lines.
LINE_TOLERANCE = 3
EXIT_NOT_BUILT = 3


# ---------------------------------------------------------------------------
# Recording what happened on the wire
# ---------------------------------------------------------------------------


class Recorder:
    """Timestamped events from any thread: REST calls (`rest`) and completions (`req`)."""

    def __init__(self) -> None:
        self._start = time.time()
        self._lock = threading.Lock()
        self._events: list = []

    def log(self, kind: str, text: str) -> None:
        with self._lock:
            self._events.append((time.time() - self._start, kind, text))

    def elapsed(self) -> float:
        return time.time() - self._start

    def events(self) -> list:
        with self._lock:
            # Sort on the time alone: a stable sort keeps logging order for events with equal timestamps.
            return sorted(self._events, key=lambda event: event[0])

    def timeline(self) -> str:
        return "\n".join(f"{at:7.1f}s {kind:4} {text}" for at, kind, text in self.events())

    def count(self, kind: str) -> int:
        return sum(1 for _, k, _ in self.events() if k == kind)


def logged_http(recorder: Recorder, real_http):
    """Wrap the model loader's HTTP function so each load/unload/list call is recorded."""

    def call(method, url, body, timeout):
        recorder.log("rest", f"{method} {url.split('/', 3)[-1] if '//' in url else url} {json.dumps(body) if body else ''}".rstrip())
        return real_http(method, url, body, timeout)

    return call


def timed_complete(recorder: Recorder, real_complete):
    """Wrap `local_compress_lib.complete` so each completion's thread, model, size and reply length are recorded."""

    def call(client, model, system, content):
        started = recorder.elapsed()
        reply = real_complete(client, model, system, content)
        recorder.log(
            "req",
            f"{threading.current_thread().name} {model} {started:6.1f}s -> {recorder.elapsed():6.1f}s "
            f"prompt={len(system)}+{len(content)} chars reply={len(reply) if reply is not None else None}",
        )
        return reply

    return call


def lmstudio_root(base_url: str) -> str:
    """The server root (no /v1) of an OpenAI-compatible LM Studio URL."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def loaded_models(base_url: str) -> list:
    """Ids of the models LM Studio currently has loaded; [] when it cannot be reached."""
    try:
        with urllib.request.urlopen(f"{lmstudio_root(base_url)}/api/v0/models", timeout=10) as response:
            return sorted(m["id"] for m in json.load(response)["data"] if m.get("state") == "loaded")
    except (OSError, ValueError, KeyError):
        return []


def model_known(base_url: str, model: str) -> Optional[bool]:
    """True/False when LM Studio lists/does not list `model`; None when it cannot be asked."""
    try:
        with urllib.request.urlopen(f"{lmstudio_root(base_url)}/api/v0/models", timeout=10) as response:
            return model in {m["id"] for m in json.load(response)["data"]}
    except (OSError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Turning a case into a repository
# ---------------------------------------------------------------------------


def git(repo: Path, *args: str) -> None:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
           "GIT_AUTHOR_NAME": "harness", "GIT_AUTHOR_EMAIL": "h@h", "GIT_COMMITTER_NAME": "harness", "GIT_COMMITTER_EMAIL": "h@h"}
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def _write_tree(repo: Path, files: dict) -> None:
    for relative, text in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def build_repo(parent: Path, case: Case) -> tuple:
    """A git repository holding `case.base` on `main` and `case.head` on `feature`.

    Returns (repo, base_ref, head_ref). Files in base but not head are deleted on feature."""
    repo = parent / f"repo-{case.name}"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    _write_tree(repo, case.base)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    git(repo, "checkout", "-q", "-b", "feature")
    for relative in set(case.base) - set(case.head):
        (repo / relative).unlink()
    _write_tree(repo, case.head)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "change")
    return repo, "main", "feature"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class Score:
    """How one run's findings line up with a case's known answers."""

    found: list = field(default_factory=list)  # bug ids reported
    missed: list = field(default_factory=list)  # bug ids not reported
    on_decoys: list = field(default_factory=list)  # decoy ids reported (false positives the verifier should reject)
    other: list = field(default_factory=list)  # findings matching neither: false positives, or real extras worth a look
    total: int = 0


def _distance(finding, file: str, line: int) -> Optional[int]:
    """Lines between the finding's range and `line` (0 when inside it); None when another file, beyond LINE_TOLERANCE,
    or the finding has no known start line (the parser records a missing or invalid line number as 0, which is not a place)."""
    same_file = finding.file == file or finding.file.endswith("/" + file) or file.endswith("/" + finding.file)
    if not same_file or finding.line_start <= 0:
        return None
    gap = max(finding.line_start - line, line - max(finding.line_end, finding.line_start), 0)
    return gap if gap <= LINE_TOLERANCE else None


def _nearest(finding, anchors: dict, taken: frozenset = frozenset()) -> Optional[tuple]:
    """`(anchor id, distance)` of the anchor `finding` sits nearest to, or None. Two defects a few lines apart must
    not be credited to whichever happens to come first, or an exact hit on the second reads as a miss.

    When several anchors are equally near, the first listed wins, EXCEPT among anchors at the very same file and
    line as that first one (two Copilot comments on one line): there, one not yet credited in `taken` is preferred,
    so two claims at that line are matched one to one. Anchors merely at the same distance from a wide range are
    different places, and a claim there must not be credited with one it did not report."""
    gaps = {anchor_id: _distance(finding, file, line) for anchor_id, (file, line) in anchors.items()}
    near = {anchor_id: gap for anchor_id, gap in gaps.items() if gap is not None}
    if not near:
        return None
    lead = min(near, key=lambda anchor_id: near[anchor_id])  # min() keeps the first listed of equals
    same_place = [anchor_id for anchor_id in near if near[anchor_id] == near[lead] and anchors[anchor_id] == anchors[lead]]
    return next((anchor_id for anchor_id in same_place if anchor_id not in taken), lead), near[lead]


@dataclass(frozen=True)
class Claim:
    """One reported claim to score, with its own lines and the reviewers who made it."""

    file: str
    line_start: int
    line_end: int
    cls: str
    claim: str
    reviewers: tuple


def distinct_claims(reported: list) -> list:
    """Every finding the reviewers reported, as scoring claims with the bounds and authors of the report itself.

    The library's merged `Finding` widens its range to cover every claim folded into it, so two defects a few
    lines apart would both read as the wider span and be scored as one. These are the findings as reported,
    before merging; only a repeat of the same claim over an overlapping range (the same reviewer on two
    chunks, or two reviewers agreeing) is folded together, adding the reviewer and keeping the first bounds."""
    claims: list = []
    for f in reported:
        key = " ".join(f.claim.lower().split()).rstrip(".")
        for index, c in enumerate(claims):
            same = c.file == f.file and c.cls == f.cls and " ".join(c.claim.lower().split()).rstrip(".") == key
            if same and c.line_start <= f.line_end and f.line_start <= c.line_end:
                claims[index] = Claim(c.file, c.line_start, c.line_end, c.cls, c.claim, c.reviewers + tuple(r for r in f.reviewers if r not in c.reviewers))
                break
        else:
            claims.append(Claim(f.file, f.line_start, f.line_end, f.cls, f.claim, tuple(f.reviewers)))
    return claims


@contextmanager
def capture_reported(sink: list):
    """While active, copy the findings handed to the library's `collect_findings` (before it merges them) into `sink`."""
    import reasoning_reviewer_lib

    original = reasoning_reviewer_lib.collect_findings

    def spy(findings):
        findings = list(findings)
        sink.extend(findings)
        return original(findings)

    reasoning_reviewer_lib.collect_findings = spy
    try:
        yield
    finally:
        reasoning_reviewer_lib.collect_findings = original


def score(findings: list, case: Case) -> Score:
    """Match each finding to the nearest bug or decoy (by file and line proximity), else call it `other`.

    The class the model chose is not matched on: models name the same defect many ways, and
    line proximity is what says the model pointed at the right code."""
    bug_lines = {bug.id: (bug.file, bug_line(case, bug)) for bug in case.bugs}
    decoy_lines = {decoy.id: (decoy.file, line_of(case.head[decoy.file], decoy.marker)) for decoy in case.decoys}
    result = Score(total=len(findings))
    found: set = set()
    on_decoys: set = set()
    for finding in findings:
        # The nearer of the best bug and the best decoy: a false alarm that sits exactly on a decoy must not be
        # credited as a bug three lines away. A tie goes to the bug.
        bug = _nearest(finding, bug_lines, frozenset(found))
        decoy = _nearest(finding, decoy_lines)
        if bug is not None and (decoy is None or bug[1] <= decoy[1]):
            found.add(bug[0])
            continue
        if decoy is not None:
            on_decoys.add(decoy[0])
            continue
        result.other.append(f"{finding.file}:{finding.line_start}-{finding.line_end} [{finding.cls}] {finding.claim[:90]}")
    result.found = [bug.id for bug in case.bugs if bug.id in found]
    result.missed = [bug.id for bug in case.bugs if bug.id not in found]
    result.on_decoys = [decoy.id for decoy in case.decoys if decoy.id in on_decoys]
    return result


def format_score(case: Case, scores: list) -> str:
    """One block for a case across `len(scores)` runs: per-bug hit rate, then false alarms."""
    runs = len(scores)
    lines = [f"case {case.name}: {len(case.bugs)} bug(s), {len(case.decoys)} decoy(s), {runs} run(s)"]
    for bug in case.bugs:
        hits = sum(1 for s in scores if bug.id in s.found)
        lines.append(f"  {'FOUND ' if hits == runs else 'partly' if hits else 'MISSED'} {bug.id:<18} {hits}/{runs}  {bug.claim[:70]}")
    for decoy in case.decoys:
        hits = sum(1 for s in scores if decoy.id in s.on_decoys)
        lines.append(f"  decoy  {decoy.id:<18} raised {hits}/{runs}   (should be 0; the verifier stage is meant to reject these)")
    other = [text for s in scores for text in s.other]
    # Only a seeded case with known contents is "clean". A replayed Copilot round with no scored inline comment is
    # not: Copilot may have missed defects or put them only in its review body, so those findings are unjudged.
    note = (" (false positives on a clean case)" if case.head else " (unjudged: Copilot raised no scored inline finding)") if not case.bugs else ""
    lines.append(f"  other findings: {len(other)} over {runs} run(s)" + note)
    for text in dict.fromkeys(other):
        lines.append(f"    - {text}")
    recall = sum(len(s.found) for s in scores) / (len(case.bugs) * runs) if case.bugs else None
    if recall is not None:
        lines.append(f"  recall {recall:.0%}")
    return "\n".join(lines)


def history_case(snapshot_path: str, round_index: int) -> tuple:
    """`(Case, base_sha, head_sha)` replaying one Copilot review from a recorded snapshot.

    Exits with a plain message when the reviewed commits are not in this clone (collect_copilot_history.py
    fetches them). Reads git only; nothing here contacts GitHub or Copilot."""
    import history_lib

    snapshot = history_lib.load_snapshot(snapshot_path)
    if snapshot.get("base_changes"):
        raise SystemExit(f"PR #{snapshot['pr']} was retargeted {snapshot['base_changes']} time(s): the base Copilot reviewed is not "
                         "recorded, so an exact replay is not possible. Pick a PR that was never retargeted.")
    case, base, head = history_lib.snapshot_case(snapshot, round_index)
    for sha in (base, head):
        found = subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True, check=False)
        if found.returncode != 0:
            raise SystemExit(f"commit {sha} is not in this clone; re-run collect_copilot_history.py for that PR (it fetches every reviewed commit by SHA) and retry.")
    return case, base, head


def seeded_line(case: Case, scores: list) -> str:
    """One line for a seeded case: bugs found, decoys alarmed, and findings that matched neither (per run)."""
    runs = len(scores)
    found = sum(len(s.found) for s in scores) / runs if runs else 0
    decoys = sum(len(s.on_decoys) for s in scores) / runs if runs else 0
    other = sum(len(s.other) for s in scores) / runs if runs else 0
    return f"{found:g}/{len(case.bugs)} bug(s) found; {decoys:g}/{len(case.decoys)} decoy alarm(s) (should be 0); {other:g} other finding(s) per run"


def parity_line(case: Case, scores: list) -> str:
    """One line: how many of Copilot's findings were also raised, and how many findings Copilot did not raise.

    The extras are not all wrong: Copilot misses things. They are what a verifier stage should sort into real and false."""
    runs = len(scores)
    matched = sum(len(s.found) for s in scores) / runs if runs else 0
    extra = sum(len(s.other) + len(s.on_decoys) for s in scores) / runs if runs else 0
    return f"parity: {matched:g}/{len(case.bugs)} of Copilot's findings raised; {extra:g} further finding(s) per run that Copilot did not raise (unjudged)"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def add_model_options(parser: argparse.ArgumentParser, *, config: bool = False) -> None:
    """The options every script shares.

    `config=True` also adds `--config` and `--show-config`, for the scripts that can use them. Such a script must
    call `finish_args` on the parsed result: `--model` and `--context` then default to None, so a flag you passed
    can be told from one you did not, and `finish_args` fills in the defaults afterwards. Without `config`, every
    default is the plain value and nothing needs finishing."""
    if config:
        parser.add_argument("--config", default=None, metavar="FILE",
                            help="a JSON file naming the reviewers (models, prompts) and the stages to test; see README.md. "
                                 "Replaces --model, --prompt, --prompt-file, --context and --resident, which it refuses alongside it")
        parser.add_argument("--show-config", action="store_true",
                            help="print the reviewers and stages this run resolves to, then exit; needs no LM Studio")
    parser.add_argument("--model", default=None if config else DEFAULT_MODEL, help=f"LM Studio model id (default {DEFAULT_MODEL})")
    parser.add_argument("--context", type=int, default=None if config else DEFAULT_CONTEXT,
                        help=f"contextLength to load it with (default {DEFAULT_CONTEXT})")
    parser.add_argument("--prompt", default=None, help="the reviewer's lens (default: a general defect lens)")
    parser.add_argument("--prompt-file", default=None, help="use this promptFile instead of --prompt (a path)")
    parser.add_argument("--resident", action="store_true", help="the model is already loaded: do not load/unload it (loadOnDemand false)")
    parser.add_argument("--runs", type=int, default=1, help="repeat each case this many times (one run proves little)")
    parser.add_argument("--chunk-chars", type=int, default=None, help="override the chunk size in characters")
    parser.add_argument("--context-lines", type=int, default=10, help="lines of code shown around each change")
    parser.add_argument("--timeline", action="store_true", help="print every REST call and request, in order")
    parser.add_argument("--no-probe", action="store_true", help="skip the up-front check that the model ends its turn")
    parser.add_argument("--probe-tokens", type=int, default=PROBE_MAX_TOKENS,
                        help=f"token cap for that check (default {PROBE_MAX_TOKENS}; a reasoning model such as DeepSeek-R1 needs ~2000)")


def absolute_path(raw: str, base: Optional[Path] = None) -> str:
    """`raw` as an absolute path (`~` expanded, relative to `base`, default the current directory). The reviewers
    file is written into a scratch directory, where a relative path would be looked up and not found."""
    path = Path(raw).expanduser()
    return str(path if path.is_absolute() else (base or Path.cwd()) / path)


def reviewer_entry(name: str, model: str, *, prompt: Optional[str], prompt_file: Optional[str], roles: tuple = ("reasoning",),
                   on_demand: bool = True, context: int = DEFAULT_CONTEXT) -> dict:
    """One `reviewers[]` entry in the reviewers-file format."""
    entry: dict = {"name": name, "model": model, "active": True, "type": list(roles),
                   "loadOnDemand": on_demand, "contextLength": context}
    if prompt_file:
        entry["promptFile"] = absolute_path(prompt_file)
        if prompt:
            entry["prompt"] = prompt
    else:
        entry["prompt"] = prompt or DEFAULT_LENS
    return entry


def single_reviewer(args: argparse.Namespace, name: str = "general") -> dict:
    return reviewer_entry(name, args.model, prompt=args.prompt, prompt_file=args.prompt_file,
                          on_demand=not args.resident, context=args.context)


def preflight(model: str) -> None:
    """Exit with a plain message when LM Studio is unreachable or does not list the model."""
    server = load_server()
    known = model_known(server.lmstudio_base_url(), model)
    if known is None:
        raise SystemExit(f"LM Studio is not reachable at {server.lmstudio_base_url()}; start it and its server first.")
    if not known:
        raise SystemExit(f"LM Studio does not list a model named {model!r}; download it, or pass --model with an id from `lms ls`.")


PROBE_MAX_TOKENS = 40
PROBE_TIMEOUT_SECONDS = 90


def probe_ends_turn(model: str, *, on_demand: bool, context: int, max_tokens: int = PROBE_MAX_TOKENS) -> Optional[str]:
    """Ask `model` for one word with a small token limit; None when it ends its turn, else a plain explanation.

    A model that never emits its end-of-turn token (usually a base model rather than the -Instruct
    build, or a build without its chat template) keeps generating until the context is full, so every
    review request times out. Found live: `qwen2.5-coder-7b` from mlx-community echoed the system
    prompt and rambled, and a four-case run produced nothing but four 60 s timeouts. This takes a
    few seconds and says so up front. Loads the model the way the scheduler would and unloads it
    again only if it loaded it."""
    from lmstudio_model_loader_lib import ModelLoader, ModelLoadError

    url = load_server().lmstudio_base_url()
    loader = ModelLoader(url)
    problem: Optional[str] = None
    try:
        loader.acquire(model, load_on_demand=on_demand, context_length=context)
        body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                           "messages": [{"role": "system", "content": "You review code."},
                                        {"role": "user", "content": "Reply with the single word: ok"}]}).encode()
        request = urllib.request.Request(f"{url.rstrip('/')}/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_SECONDS) as response:
            choice = json.load(response)["choices"][0]
        if choice.get("finish_reason") != "stop":
            problem = (f"{model} did not end its turn: asked to reply with one word and capped at {max_tokens} tokens, "
                       f"it stopped for {choice.get('finish_reason')!r} after {str(choice.get('message', {}).get('content', ''))[:80]!r}. "
                       "Such a model runs to the context limit on every request and times out. Use its -Instruct build, "
                       "or check its chat template in LM Studio. A reasoning model thinks before it answers: raise --probe-tokens. (--no-probe skips this check.)")
    except (ModelLoadError, OSError, ValueError, KeyError, IndexError) as exc:
        problem = f"could not probe {model}: {type(exc).__name__}: {exc}"
    finally:
        # Always unload what we loaded, whether or not the probe found a problem.
        unload_errors = loader.release_all()
    return problem or ("; ".join(unload_errors) or None)


def require_chat_model(model: str, args: argparse.Namespace, *, on_demand: Optional[bool] = None, context: Optional[int] = None) -> None:
    """Exit with the probe's explanation unless `model` ends its turn (or --no-probe was given)."""
    if getattr(args, "no_probe", False):
        return
    problem = probe_ends_turn(model, on_demand=(not args.resident) if on_demand is None else on_demand,
                              context=args.context if context is None else context,
                              max_tokens=getattr(args, "probe_tokens", PROBE_MAX_TOKENS))
    if problem:
        raise SystemExit(problem)


# ---------------------------------------------------------------------------
# The --config file
# ---------------------------------------------------------------------------

STAGES = ("reasoning", "verifier", "classifier")
_CONFIG_KEYS = ("stages", "reviewers", "rubric")


@dataclass(frozen=True)
class HarnessConfig:
    """A parsed `--config` file: the stages to test, and the reviewers and rubric to test them with.

    `reviewers` and `rubric` are in the reviewers-file format, with the values the validator normalized (model ids
    trimmed, `promptFile` and `definitionFile` resolved to absolute paths)."""

    path: str
    stages: tuple
    reviewers: list = field(default_factory=list)
    rubric: list = field(default_factory=list)


def entries_from_config(config) -> tuple:
    """`(reviewers, rubric)` as reviewers-file entries, built from a validated `ReviewersConfig` and not from the raw JSON.

    The validator trims model ids and resolves `promptFile` and `definitionFile` (trimmed, `~` expanded, relative to the
    file); rebuilding them from the raw text would disagree with it, so a value like `" lens.md "` would pass the
    validator and then fail when the run rewrites the reviewers file. Every key a reviewers file allows is carried over."""
    reviewers = []
    for reviewer in config.reviewers:
        entry: dict = {"name": reviewer.name, "model": reviewer.model, "active": reviewer.active,
                       "type": list(reviewer.roles), "loadOnDemand": reviewer.load_on_demand}
        if reviewer.context_length is not None:
            entry["contextLength"] = reviewer.context_length
        if reviewer.prompt:
            entry["prompt"] = reviewer.prompt
        if reviewer.prompt_file:
            entry["promptFile"] = reviewer.prompt_file
        reviewers.append(entry)
    rubric = []
    for item in config.rubric:
        rubric_entry: dict = {"name": item.name, "level": item.level, "definition": item.definition, "score": item.score}
        if item.definition_file:
            rubric_entry["definitionFile"] = item.definition_file
        rubric.append(rubric_entry)
    return reviewers, rubric


def load_config(path: str) -> HarnessConfig:
    """Read and validate a `--config` file; raises `ConfigError` listing every problem at once.

    The file is JSON with three optional keys:
      reviewers  the reviewers to test, exactly as in a reviewers file (name, model, prompt/promptFile, active,
                 type, loadOnDemand, contextLength), and validated by the same code the server uses, so a file
                 that passes here is one `REVIEWERS_FILE` accepts;
      rubric     the rubric the classifier matches against, also as in a reviewers file;
      stages     which of reasoning, verifier, classifier to test. Left out, it is every stage an active reviewer
                 has a role in (every stage when there are no reviewers).
    A reviewers file is therefore a valid config as it stands."""
    from codebase_reviewer_config_lib import ConfigError, validate_reviewers

    resolved = Path(path).expanduser().resolve()
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Harness config {resolved} cannot be read ({getattr(exc, 'strerror', None) or exc})") from exc
    except ValueError as exc:
        raise ConfigError(f"Harness config {resolved} is not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Harness config {resolved} must be a JSON object (got {type(data).__name__})")
    problems: list = []
    for key in sorted(k for k in data if k not in _CONFIG_KEYS):
        problems.append(f"unknown top-level key {key!r} (allowed: {', '.join(_CONFIG_KEYS)})")
    reviewers: list = []
    rubric: list = []
    if "reviewers" in data and data["reviewers"] == []:
        # Valid for the server's validator, but here an empty list would read as "no reviewers" and silently
        # fall back to the command line flags: say which of the two is meant.
        problems.append("'reviewers' is empty: name at least one reviewer, or leave the key out to use the command line flags")
    elif "reviewers" in data:
        try:
            validated = validate_reviewers({"rubric": data.get("rubric", []), "reviewers": data["reviewers"]},
                                           base_dir=resolved.parent, path=str(resolved))
            reviewers, rubric = entries_from_config(validated)
        except ConfigError as exc:
            problems.extend(exc.problems)
    elif "rubric" in data:
        problems.append("'rubric' needs 'reviewers': no stage uses a rubric on its own")
    stages = _config_stages(data, reviewers, problems)
    if problems:
        raise ConfigError(f"Harness config {resolved} is invalid ({len(problems)} problem{'s' if len(problems) != 1 else ''}):\n"
                          + "\n".join(f"  - {problem}" for problem in problems), problems)
    return HarnessConfig(path=str(resolved), stages=stages, reviewers=reviewers, rubric=rubric)


def _config_stages(data: dict, reviewers: list, problems: list) -> tuple:
    """The stages a config tests, in pipeline order; problems found are appended to `problems`."""
    def has_role(stage: str) -> bool:
        return any(entry["active"] and stage in entry["type"] for entry in reviewers)

    if "stages" not in data:
        return tuple(stage for stage in STAGES if has_role(stage)) if reviewers else STAGES
    raw = data["stages"]
    if not isinstance(raw, list) or not raw or not all(isinstance(stage, str) for stage in raw):
        problems.append(f"'stages' must be a non-empty list of stage names from {list(STAGES)} (got {raw!r})")
        return ()
    unknown = [stage for stage in raw if stage not in STAGES]
    if unknown:
        problems.append(f"'stages' names unknown stage(s) {unknown}; the stages are {list(STAGES)}")
    if len(set(raw)) != len(raw):
        problems.append(f"'stages' lists a stage more than once ({raw!r})")
    stages = tuple(stage for stage in STAGES if stage in raw)
    for stage in stages:
        if reviewers and not has_role(stage):
            problems.append(f"stage {stage!r} is selected but no active reviewer has the type {stage!r}")
    return stages


def finish_args(args: argparse.Namespace) -> argparse.Namespace:
    """Complete the parsed command line: read `--config` into `args.config_data` (None without it) and fill in
    the `--model` and `--context` defaults.

    A config that names reviewers owns the model, prompt and loading of every reviewer, so a flag that says the
    same thing is refused instead of being silently ignored or silently winning."""
    from codebase_reviewer_config_lib import ConfigError

    args.config_data = None
    if args.config:
        try:
            args.config_data = load_config(args.config)
        except ConfigError as exc:
            raise SystemExit(str(exc)) from exc
    given = [flag for flag, value in (("--model", args.model), ("--prompt", args.prompt),
                                      ("--prompt-file", args.prompt_file), ("--context", args.context)) if value is not None]
    if args.resident:
        given.append("--resident")
    if args.config_data is not None and args.config_data.reviewers and given:
        raise SystemExit(f"--config already names the reviewers (model, prompt and loading), so {', '.join(given)} would be "
                         "ignored: drop the flag(s), or put the setting in the config file.")
    if args.model is None:
        args.model = DEFAULT_MODEL
    if args.context is None:
        args.context = DEFAULT_CONTEXT
    return args


def selected_stages(args: argparse.Namespace) -> tuple:
    """The stages this run is asked to test: the config's, else all of them."""
    return args.config_data.stages if args.config_data is not None else STAGES


def stage_selected(args: argparse.Namespace, stage: str) -> bool:
    """False only when a config was given and its `stages` leave `stage` out."""
    return stage in selected_stages(args)


def stage_reviewers(args: argparse.Namespace, stage: str) -> list:
    """The reviewers to run for `stage`: the config's active ones with that role, else one built from the flags.

    Each is a copy restricted to `stage`. A reviewer that also has another role carries that role's requirements
    with it (an active classifier needs the rubric), and a run of one stage rebuilds a reviewers file without them,
    which then fails to load: found by running a reviewer typed reasoning and classifier through `live_reasoning.py`."""
    config = args.config_data
    if config is not None and config.reviewers:
        return [{**entry, "type": [stage]} for entry in config.reviewers if entry["active"] and stage in entry["type"]]
    return [reviewer_entry("general", args.model, prompt=args.prompt, prompt_file=args.prompt_file, roles=(stage,),
                           on_demand=not args.resident, context=args.context)]


def _loading(entry: dict) -> str:
    if not entry.get("loadOnDemand", True):
        return "resident"
    return f"ctx {entry['contextLength']}" if entry.get("contextLength") else "on demand"


def describe_reviewers(entries: list) -> str:
    """One line naming each reviewer, its model and how that model is loaded."""
    return "; ".join(f"{entry['name']} ({entry['model']}, {_loading(entry)})" for entry in entries)


def describe_config(args: argparse.Namespace) -> str:
    """What `--show-config` prints: the stages, the reviewers behind them and which stages this checkout can run."""
    config = args.config_data
    if config is None:
        return ("no --config: each script builds one reviewer from --model, --prompt/--prompt-file, --context and --resident.\n"
                f"stages: the script's own. built in this checkout: {', '.join(available_review_types())}")
    lines = [f"config {config.path}",
             f"stages: {', '.join(config.stages) or 'none'}   (built in this checkout: {', '.join(available_review_types())})"]
    if not config.reviewers:
        lines.append("reviewers: none in the file; each stage uses one reviewer built from the command line flags")
    for entry in config.reviewers:
        prompt = " + ".join((["inline text"] if "prompt" in entry else []) + ([entry["promptFile"]] if "promptFile" in entry else []))
        lines.append(f"  {entry['name']:<18} {entry['model']:<34} {'active' if entry['active'] else 'inactive':<8} "
                     f"roles={'/'.join(entry['type'])}  {_loading(entry)}  prompt: {prompt}")
    if config.rubric:
        lines.append("rubric: " + ", ".join(f"{e['name']} ({e['level']}, score {e['score']})" for e in config.rubric))
    return "\n".join(lines)


def preflight_models(entries: list) -> None:
    """`preflight` for each distinct model the reviewers name."""
    for model in sorted({entry["model"] for entry in entries}):
        preflight(model)


def preflight_reviewers(entries: list, args: argparse.Namespace) -> None:
    """Check each distinct model is listed and ends its turn, loading it the way the reviewer asks."""
    preflight_models(entries)
    for model in sorted({entry["model"] for entry in entries}):
        first = next(entry for entry in entries if entry["model"] == model)
        require_chat_model(model, args, on_demand=first.get("loadOnDemand", True), context=first.get("contextLength") or args.context)


# ---------------------------------------------------------------------------
# Driving the stages
# ---------------------------------------------------------------------------

_server = None


def load_server():
    """The codebase-reviewer server module, loaded once (for its git-diff helper and URL setting)."""
    global _server
    if _server is None:
        spec = importlib.util.spec_from_file_location("codebase_reviewer_server", REPO_ROOT / "mcpservers" / "codebase_reviewer_mcp_server.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _server = module
    return _server


def available_review_types() -> list:
    """The `type`s the `review` tool currently accepts. Grows as the verifier and classifier land."""
    return list(load_server().REVIEW_TYPES)


def write_reviewers_file(directory: Path, reviewers: list, rubric: Optional[list] = None) -> Path:
    path = directory / "reviewers.json"
    path.write_text(json.dumps({"rubric": rubric or [], "reviewers": reviewers}), encoding="utf-8")
    return path


def run_reasoning(repo: Path, base: str, head: str, reviewers: list, args: argparse.Namespace, recorder: Recorder,
                  *, max_reviewers: Optional[int] = None, rubric: Optional[list] = None) -> tuple:
    """Run the reasoning stage once over base...head. Returns (claims, ReasoningReview): the findings as
    reported, not the library's merged ones (see `distinct_claims`).

    Calls the library directly (not the `review` tool) so findings arrive as objects to score,
    through the same `_git_diff`, config loader, scheduler and model loader the tool uses."""
    import local_compress_lib
    from codebase_reviewer_config_lib import load_reviewers_file
    from lmstudio_client import build_client
    from lmstudio_model_loader_lib import ModelLoader, _urllib_http
    from reasoning_reviewer_lib import run_reasoning_review

    server = load_server()
    with tempfile.TemporaryDirectory(prefix="live-reviewers-") as scratch:
        config = load_reviewers_file(str(write_reviewers_file(Path(scratch), reviewers, rubric)))
    url = server.lmstudio_base_url()
    diff = server._git_diff(str(repo), base, head, args.context_lines)
    reported: list = []
    with capture_reported(reported):
        review = run_reasoning_review(
            config,
            ModelLoader(url, http=logged_http(recorder, _urllib_http)),
            build_client(url),
            diff,
            repo,
            max_reviewers or len(reviewers),
            chunk_chars=args.chunk_chars,
            complete=timed_complete(recorder, local_compress_lib.complete),
        )
    return distinct_claims(reported), review


def report_run(args: argparse.Namespace, recorder: Recorder, base_url: str, before: list) -> str:
    """The closing lines every script prints: loaded models before/after, request counts, optional timeline."""
    lines = [f"loaded before: {before}", f"loaded after:  {loaded_models(base_url)}",
             f"{recorder.count('req')} completion request(s), {recorder.count('rest')} REST call(s), {recorder.elapsed():.1f}s total"]
    if args.timeline:
        lines += ["----- timeline", recorder.timeline()]
    return "\n".join(lines)


def refuse_unbuilt(stage: str, needs: str) -> None:
    """Exit with EXIT_NOT_BUILT and a message when the stage a harness exercises does not exist yet,
    or exists but the harness has no runner for it."""
    if stage not in available_review_types():
        print(f"The {stage} stage is not built yet: `review` accepts only {available_review_types()}.\n"
              f"Nothing to run. `--check` validates this harness's fixtures without it.")
    else:
        print(f"The `review` tool now accepts type {stage!r}, but this harness has no runner for it yet.\n"
              f"Wire one in the `run_stage` function of this script: {needs}")
    raise SystemExit(EXIT_NOT_BUILT)
