#!/usr/bin/env python3
"""Reply-route reliability: which way of getting a model's reply into the server's format works for it.

    .venv/bin/python harness/live_routes.py --model <lm-studio-id>                    # lines and json, 10 runs
    .venv/bin/python harness/live_routes.py --model a --model b --runs 5              # each model in turn
    .venv/bin/python harness/live_routes.py --model <id> \\
        --route "native=lines,promptFile=my-prompt.md,translator=my-translator.py"     # also try a custom route
    .venv/bin/python harness/live_routes.py --model <id> --show-config                # what it would run; no LM Studio

Every reply must reach the server as `lines` or `json`, and a model has three ways to get there: it replies in
`lines`, it replies in `json`, or it replies in its own format and a translator converts that (a prompt written
for the model is part of any route). For each model this runs each route over the same cases, counts the replies
that could be read, and names the first route that is reliable: 95% or more of the replies readable.

The routes are `lines` and `json` (the server's own reply instructions), then every `--route` in the order given.
A route is `NAME=FORMAT` followed by optional `promptFile=PATH` and `translator=PATH`; FORMAT is how the reply, or
the translator's output, is read. A prompt file replaces the built-in reply instructions, so it has to ask for
FORMAT itself (see harness/prompts/README.md). `--no-builtin` leaves `lines` and `json` out.

What counts. A reply is unreadable when the reader could not parse it as FORMAT, or the translator failed on it.
A request LM Studio refused or timed out is not a reply and is not counted against the format: it is listed
apart. Bugs found and false alarms are printed for information; they never decide the verdict.

Needs LM Studio running with its server enabled and each model downloaded.
"""

import argparse
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_lib as h  # noqa: E402
from cases import ALL_CASES  # noqa: E402

# A route is reliable when at least this share of the replies it got could be read.
RELIABLE_PERCENT = 95
DEFAULT_RUNS = 10
DEFAULT_CASES = ("seeded-three", "clean")
FORMATS = ("lines", "json")
_ROUTE_KEYS = ("promptFile", "translator")
# How many distinct other problems a route's notes list (the same few usually repeat).
OTHER_PROBLEMS_SHOWN = 3


@dataclass(frozen=True)
class Route:
    name: str
    reply_format: str
    prompt_file: str = ""
    translator: str = ""

    def describe(self) -> str:
        parts = [f"reads {self.reply_format}"]
        parts.append(f"prompt file {self.prompt_file}" if self.prompt_file else "the server's own reply instructions")
        if self.translator:
            parts.append(f"translator {self.translator}")
        return ", ".join(parts)


BUILTIN_ROUTES = (Route("lines", "lines"), Route("json", "json"))


def parse_route(text: str, problems: list) -> Optional[Route]:
    """One `--route` value as a `Route`; every problem found is appended to `problems`, and None is returned if any."""
    fields = [field_text.strip() for field_text in text.split(",")]
    name, _, reply_format = fields[0].partition("=")
    name, reply_format = name.strip(), reply_format.strip()
    found = len(problems)
    if not name or not reply_format:
        problems.append(f"--route {text!r}: it must start with NAME=FORMAT, for example `native=lines`")
        return None
    if reply_format not in FORMATS:
        problems.append(f"--route {text!r}: the format must be one of {', '.join(FORMATS)} (got {reply_format!r})")
    values: dict = {}
    for item in fields[1:]:
        key, sep, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or key not in _ROUTE_KEYS:
            problems.append(f"--route {text!r}: unknown setting {item!r} (allowed: {', '.join(_ROUTE_KEYS)})")
        elif key in values:
            problems.append(f"--route {text!r}: {key} is given twice")
        elif not value:
            problems.append(f"--route {text!r}: {key} needs a path")
        else:
            path = Path(value).expanduser()
            if not path.is_file():
                problems.append(f"--route {text!r}: {key} {str(path)!r} is not a file")
            values[key] = h.absolute_path(value)
    if len(problems) > found:
        return None
    return Route(name, reply_format, values.get("promptFile", ""), values.get("translator", ""))


def build_routes(route_args: list, *, builtin: bool = True) -> list:
    """The routes to run, in order: `lines`, `json`, then each `--route`. Raises SystemExit naming every problem."""
    problems: list = []
    routes = list(BUILTIN_ROUTES) if builtin else []
    for text in route_args:
        route = parse_route(text, problems)
        if route is not None:
            routes.append(route)
    seen: set = set()
    for route in routes:
        if route.name in seen:
            problems.append(f"route name {route.name!r} is used twice (`lines` and `json` are the built-in routes)")
        seen.add(route.name)
    if not routes:
        problems.append("no routes to run: drop --no-builtin, or add a --route")
    if not problems:
        problems.extend(_server_problems(routes))
    if problems:
        raise SystemExit("\n".join(["The routes are not usable:", *(f"  - {problem}" for problem in problems)]))
    return routes


def _server_problems(routes: list) -> list:
    """What the server's own reviewers-file validator says about each route, so a route it would refuse (an empty
    prompt file, a translator that is not executable) is reported up front, together with every other problem, instead
    of aborting the run after the earlier routes have already been measured. The check is the same one a
    reviewers file gets; the model id here is a placeholder, since the validator does not look at it."""
    from codebase_reviewer_config_lib import ConfigError, validate_reviewers

    problems: list = []
    for route in routes:
        entry = h.reviewer_entry("general", "model", prompt=None, prompt_file=route.prompt_file or None,
                                 reply_format=route.reply_format, translator=route.translator or None)
        try:
            validate_reviewers({"rubric": [], "reviewers": [entry]})
        except ConfigError as exc:
            problems.extend(f"route {route.name!r}: {problem.split(': ', 1)[-1]}" for problem in exc.problems)
    return problems


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

FORMAT_PROBLEM = "format"
TRANSLATOR_PROBLEM = "translator"
REQUEST_PROBLEM = "request"
SKIPPED_PROBLEM = "skipped"
OTHER_PROBLEM = "other"
# What the reply readers say when a reply cannot be read, mapped to a short reason. The tests build these from the
# readers' own output, so a changed message is caught there instead of silently counting as readable.
_FORMAT_REASONS = (
    ("the reply was not JSON", "not JSON"),
    ("the reply has no 'findings' list", "no 'findings' list"),
    ("the reply had no FINDING blocks and was not NONE", "no FINDING blocks"),
)


def classify_problem(text: str) -> tuple:
    """`(kind, reason)` for one problem line a review reported.

    `kind` is FORMAT_PROBLEM (the reply could not be read as the format), TRANSLATOR_PROBLEM (the translator failed
    on the reply), REQUEST_PROBLEM (LM Studio gave no reply), SKIPPED_PROBLEM (one item left out of a reply that
    was read: the reply itself was readable) or OTHER_PROBLEM (anything else the review reported, such as a model
    that would not load: no reply was read, and the text is kept so the cause is shown)."""
    if "the translator " in text:
        detail = text.split("the translator ", 1)[1].strip()
        return TRANSLATOR_PROBLEM, f"translator {detail}"
    for marker, reason in _FORMAT_REASONS:
        if marker in text:
            return FORMAT_PROBLEM, reason
    if "LM Studio request failed" in text or "every request failed" in text:
        return REQUEST_PROBLEM, "request failed"
    if " skipped (" in text:
        return SKIPPED_PROBLEM, text
    return OTHER_PROBLEM, text


@dataclass
class RouteResult:
    """What one route did for one model, over every case and run."""

    route: Route
    sent: int = 0
    failed: int = 0
    unreadable: Counter = field(default_factory=Counter)
    skipped_items: int = 0
    other_problems: Counter = field(default_factory=Counter)
    found: int = 0
    possible: int = 0
    false_alarms: int = 0
    runs: int = 0

    @property
    def received(self) -> int:
        return max(self.sent - self.failed, 0)

    @property
    def unreadable_count(self) -> int:
        return sum(self.unreadable.values())

    @property
    def readable(self) -> int:
        return max(self.received - self.unreadable_count, 0)

    def reliable(self) -> Optional[bool]:
        """True at RELIABLE_PERCENT or more readable; None when no reply came back at all (there is nothing to judge)."""
        if self.received == 0:
            return None
        # Whole numbers, so 19 of 20 is exactly on the line and not at the mercy of a rounded float.
        return self.readable * 100 >= RELIABLE_PERCENT * self.received


def verdict(results: list) -> str:
    """The first reliable route in `results` (already in route order), or a plain statement that none was."""
    for result in results:
        if result.reliable():
            return f"{result.route.name} is reliable ({_percent(result)})"
    unanswered = [result.route.name for result in results if result.reliable() is None]
    note = f" ({', '.join(unanswered)}: no reply came back to judge)" if unanswered else ""
    return f"no route was reliable (need {RELIABLE_PERCENT}% or more readable){note}"


def _percent(result: RouteResult) -> str:
    return f"{100 * result.readable / result.received:.1f}%" if result.received else "n/a"


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

RunFn = Callable[..., tuple]


def measure_route(route: Route, model: str, cases: list, runs: int, args: argparse.Namespace, recorder: h.Recorder,
                  repos: dict, run_fn: RunFn) -> RouteResult:
    """Run `route` on `model` over every case `runs` times and count what came back."""
    result = RouteResult(route)
    entry = h.reviewer_entry("general", model, prompt=None, prompt_file=route.prompt_file or None,
                             on_demand=not args.resident, context=args.context,
                             reply_format=route.reply_format, translator=route.translator or None)
    for case in cases:
        repo, base, head = repos[case.name]
        for _run in range(runs):
            sent_before, failed_before = recorder.count("req"), recorder.count("fail")
            claims, review = run_fn(repo, base, head, [entry], args, recorder)
            result.sent += recorder.count("req") - sent_before
            result.failed += recorder.count("fail") - failed_before
            for problem in review.errors:
                kind, reason = classify_problem(problem)
                if kind in (FORMAT_PROBLEM, TRANSLATOR_PROBLEM):
                    result.unreadable[reason] += 1
                elif kind == SKIPPED_PROBLEM:
                    result.skipped_items += 1
                elif kind == OTHER_PROBLEM:
                    result.other_problems[reason] += 1
            score = h.score(claims, case)
            result.found += len(score.found)
            result.possible += len(case.bugs)
            result.false_alarms += len(score.on_decoys) + len(score.other)
    result.runs = runs * len(cases)
    return result


def format_model(model: str, cases: list, runs: int, results: list) -> str:
    """One table for one model, then the verdict."""
    lines = [f"{model}   {len(cases)} case(s) x {runs} run(s): {', '.join(case.name for case in cases)}"]
    width = max([len("route"), *(len(result.route.name) for result in results)])
    lines.append(f"  {'route':<{width}}  {'sent':>5}  {'failed':>6}  {'readable':>16}  not readable")
    for result in results:
        if result.received:
            readable = f"{result.readable}/{result.received} {_percent(result):>6}"
        else:
            readable = "no reply"
        why = ", ".join(f"{count} {reason}" for reason, count in result.unreadable.most_common()) or "-"
        lines.append(f"  {result.route.name:<{width}}  {result.sent:>5}  {result.failed:>6}  {readable:>16}  {why}")
    lines.append(f"  verdict: {verdict(results)}")
    notes = [f"{result.route.name}: {result.failed} request(s) got no reply from LM Studio (not counted against the format)"
             for result in results if result.failed]
    notes += [f"{result.route.name}: {result.skipped_items} item(s) skipped from replies that were read"
              for result in results if result.skipped_items]
    for result in results:
        for reason, count in result.other_problems.most_common(OTHER_PROBLEMS_SHOWN):
            notes.append(f"{result.route.name}: {count} x {reason}")
    lines.extend(f"  note: {note}" for note in notes)
    info = [f"{result.route.name} found {result.found / max(runs, 1):g}/{result.possible / max(runs, 1):g} bug(s) and raised "
            f"{result.false_alarms / max(runs, 1):g} false alarm(s) per run" for result in results]
    lines.append("  for information only (does not decide the verdict): " + "; ".join(info))
    return "\n".join(lines)


def describe_run(models: list, routes: list, case_names: list, runs: int) -> str:
    """What `--show-config` prints: the models, routes and cases, and the least number of requests the run sends."""
    least = len(models) * len(routes) * len(case_names) * runs
    lines = [f"models: {', '.join(models)}", f"cases: {', '.join(case_names)}   runs: {runs}", "routes:"]
    lines.extend(f"  {route.name:<12} {route.describe()}" for route in routes)
    lines.append(f"verdict: the first route with {RELIABLE_PERCENT}% or more of its replies readable")
    lines.append(f"requests: at least {least} (one per diff chunk for each model, route, case and run)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", action="append", required=True, metavar="ID",
                        help="LM Studio model id as `lms ls` shows it (repeatable: each model is measured in turn)")
    parser.add_argument("--route", action="append", default=[], metavar="NAME=FORMAT[,promptFile=PATH][,translator=PATH]",
                        help="a custom route to try after `lines` and `json` (repeatable)")
    parser.add_argument("--no-builtin", action="store_true", help="run only the --route routes, not `lines` and `json`")
    parser.add_argument("--case", action="append", choices=list(ALL_CASES), default=None,
                        help=f"case to run (repeatable; default {' and '.join(DEFAULT_CASES)})")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help=f"runs of each case per route (default {DEFAULT_RUNS})")
    parser.add_argument("--context", type=int, default=h.DEFAULT_CONTEXT,
                        help=f"contextLength to load each model with (default {h.DEFAULT_CONTEXT})")
    parser.add_argument("--resident", action="store_true", help="the models are already loaded: do not load/unload them")
    parser.add_argument("--chunk-chars", type=int, default=None, help="override the chunk size in characters")
    parser.add_argument("--context-lines", type=int, default=10, help="lines of code shown around each change")
    parser.add_argument("--timeline", action="store_true", help="print every REST call and request, in order")
    parser.add_argument("--no-probe", action="store_true", help="skip the up-front check that each model ends its turn")
    parser.add_argument("--probe-tokens", type=int, default=h.PROBE_MAX_TOKENS, help="token cap for that check")
    parser.add_argument("--show-config", action="store_true", help="print what a run would do, then exit; needs no LM Studio")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be 1 or more")
    return args


def main(argv=None, *, run_fn: RunFn = h.run_reasoning) -> int:
    args = parse_args(argv)
    routes = build_routes(args.route, builtin=not args.no_builtin)
    # A case named twice is run once: its repository is built in a directory named for it.
    case_names = list(dict.fromkeys(args.case or DEFAULT_CASES))
    if args.show_config:
        print(describe_run(args.model, routes, case_names, args.runs))
        return 0
    probe_entries = [h.reviewer_entry("general", model, prompt=None, prompt_file=None, on_demand=not args.resident,
                                      context=args.context) for model in args.model]
    h.preflight_reviewers(probe_entries, args)
    cases = [ALL_CASES[name] for name in case_names]
    base_url = h.load_server().lmstudio_base_url()
    before = h.loaded_models(base_url)
    recorder = h.Recorder()
    with tempfile.TemporaryDirectory(prefix="live-routes-") as tmp:
        repos = {case.name: h.build_repo(Path(tmp), case) for case in cases}
        for model in args.model:
            results = []
            for route in routes:
                print(f"--- {model}: route {route.name} ({route.describe()})", flush=True)
                results.append(measure_route(route, model, cases, args.runs, args, recorder, repos, run_fn))
            print(format_model(model, cases, args.runs, results))
    print(h.report_run(args, recorder, base_url, before))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
