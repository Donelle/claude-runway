#!/usr/bin/env python3
"""
Measures, from real Claude Code session transcripts, what this toolkit's MCP
servers add to a session's STARTING context -- instead of estimating it
(replaces the old chars / 3.5 "tool overhead per turn" figure).

Two modes:

    python tools/measure_startup_context.py --with A.jsonl [B.jsonl ...] \
                                            --without C.jsonl [D.jsonl ...]
        The controlled comparison EVALUATION.md's Steps A4/B4/E4 describe: sessions
        with the toolkit's servers connected vs. removed, reported per model as the
        difference in starting context (the first assistant turn's input +
        cache_read + cache_creation tokens, as the API itself counted them).

    python tools/measure_startup_context.py --survey [PROJECTS_DIR]
        Reads every transcript under ~/.claude/projects (or PROJECTS_DIR) and
        reports which sessions had the toolkit's tools listed as DEFERRED (not sent
        up front) at their first turn, per server; then compares those sessions with
        the ones that listed none, per model. This is OBSERVATIONAL: the two groups
        are different sessions (different repos, CLAUDE.md files, other MCP servers),
        and "none listed" does not prove the toolkit was absent, so the difference is
        context, not a controlled measurement. Only --with/--without is.

What this can and cannot show, deliberately stated in the output:

- A transcript records one starting-context number for the WHOLE prefix, so a
  single session cannot attribute a share of it to this toolkit.
- It does record which tools Claude Code deferred (loads on demand instead of
  sending up front). A toolkit tool named there is not paid every turn.
- A connected toolkit whose schemas were sent up front and never called leaves no
  trace, so "no toolkit tools listed" is not proof the servers were absent.
- A toolkit tool that was called but not listed as deferred at the first turn is
  flagged, not classified: either its schema was sent up front or its server
  connected after the first turn, and the transcript does not say which.

No data is fabricated: a transcript with no assistant turn is reported as skipped,
and a model seen on only one side gets no difference.

Stdlib-only; reads transcripts, writes nothing.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))

import transcript_context_lib as T  # noqa: E402


def _fmt_num(n) -> str:
    return "n/a" if n is None else f"{n:,.0f}"


def _print_comparison(result: dict, with_label: str = "with toolkit", without_label: str = "without toolkit") -> None:
    if not result["models"]:
        print("No transcript with an assistant turn was found on either side; nothing to compare.")
    for model, m in result["models"].items():
        print(f"\nModel {model}")
        print(f"  {with_label:<15} : {m['with_count']} session(s), mean starting context {_fmt_num(m['with_mean'])} tokens")
        print(f"  {without_label:<15} : {m['without_count']} session(s), mean starting context {_fmt_num(m['without_mean'])} tokens")
        if m["difference"] is None:
            print("  difference      : n/a (this model is not present on both sides)")
        else:
            print(f"  difference      : {m['difference']:+,.0f} tokens (with minus without)")
        print(f"  toolkit tools listed as deferred, per with-session   : {m['with_toolkit_deferred']}")
        print(f"  toolkit tools listed as deferred, per without-session: {m['without_toolkit_deferred']}")
    for path in result["skipped"]:
        print(f"\nSkipped (no assistant turn with usage data): {path}")


def _run_compare(with_paths, without_paths) -> int:
    # A --without transcript that lists toolkit tools as deferred contradicts its own
    # label: printing "with minus without" for it would be a number that is not the
    # comparison it claims to be, and this is the figure EVALUATION.md says to cite.
    # (The reverse -- a --with transcript that lists none -- is not rejected: a toolkit
    # sent up front and never called leaves no trace, so it can't be called wrong.)
    # A toolkit tool CALLED in a --without session also contaminates it: a call on the
    # first turn proves the schema was available up front, and a server that connected
    # later still does not satisfy "servers removed".
    contaminated = []
    for p in without_paths:
        s = T.summarize(p)
        if s and (s["toolkit_deferred_count"] or s["toolkit_used_not_deferred"]):
            contaminated.append(p)
    if contaminated:
        for p in contaminated:
            print(f"Error: {p} is listed under --without but names toolkit tools as deferred at its "
                  "first turn or called a toolkit tool, so the toolkit was connected in that session.", file=sys.stderr)
        print("No measurement printed. Re-run with sessions that really had the toolkit's MCP servers removed.",
              file=sys.stderr)
        return 2
    rows = [T.summarize(p) for p in list(with_paths) + list(without_paths)]
    flagged = [r for r in rows if r and r["toolkit_used_not_deferred"]]
    _print_comparison(T.compare(with_paths, without_paths))
    for r in flagged:
        print(f"\nNote: {r['path']} called toolkit tools that were not listed as deferred at its first turn "
              f"({', '.join(r['toolkit_used_not_deferred'])}). Either their schemas were sent up front or "
              "their server connected after the first turn; the transcript does not say which.")
    print("\nA difference is only as clean as the sessions are alike: same model, same repo, "
          "same CLAUDE.md, same other MCP servers. Anything else changing between the two "
          "sides is in the number.")
    return 0


def _run_survey(root: str) -> int:
    paths = T.find_transcripts(root)
    # "listed" = at least one toolkit tool was named in the deferred-tools list at
    # the first assistant turn. "none listed" is NOT proof the servers were absent:
    # a connected toolkit whose schemas were sent up front and never called leaves no
    # trace in a transcript, so that group can contain sessions that had the toolkit.
    # A session that called a toolkit tool without ever listing one as deferred at the
    # first turn is unclassified (its server may simply have connected later), so it
    # is left out of both groups rather than guessed into one.
    listed, none_listed, unclassified, no_deferral_data, no_data = [], [], [], 0, 0
    summarized = []
    for p in paths:
        s = T.summarize(p)
        if s is None:
            no_data += 1
            continue
        summarized.append(s)
        if s["deferred_tool_count"] == 0:
            # No deferred-tools record at all: a Claude Code build (or session) that
            # does not defer tools, so it is not comparable with one that does.
            no_deferral_data += 1
        elif s["toolkit_deferred_count"] > 0:
            listed.append(s)
        elif s["toolkit_used_not_deferred"]:
            unclassified.append(s)
        else:
            none_listed.append(s)
    print(f"Scanned {len(paths)} transcript(s) under {root}")
    print(f"  toolkit tools listed as deferred at the first turn : {len(listed)} session(s)")
    print(f"  no toolkit tool listed (absence NOT verified)      : {len(none_listed)} session(s)")
    print(f"  unclassified (called a toolkit tool, none listed)  : {len(unclassified)} session(s)")
    print(f"  left out: no assistant turn {no_data}, no deferred-tools data {no_deferral_data}")

    print(f"\nPer server, sessions (of {len(listed)}) whose first-turn deferred list named that server's tools:")
    for server in T.TOOLKIT_SERVERS:
        count = sum(1 for s in listed if server in s["toolkit_servers_deferred"])
        print(f"  {server}: {count}")
    # "All" means every always-on (core) server. Optional servers are left out of it:
    # one that exists only in projects that opted in would otherwise drag this toward 0.
    all_core = sum(1 for s in listed if set(T.TOOLKIT_SERVERS) <= set(s["toolkit_servers_deferred"]))
    print(f"  all {len(T.TOOLKIT_SERVERS)} servers: {all_core}")
    # Optional servers get their own line and count, outside the all-servers figure.
    for server in T.OPTIONAL_TOOLKIT_SERVERS:
        count = sum(1 for s in listed if server in s["toolkit_servers_deferred"])
        print(f"  {server} (optional, not in the all-servers count): {count}")

    late_or_upfront = [s for s in listed + unclassified if s["toolkit_used_not_deferred"]]
    if late_or_upfront:
        print(f"\n{len(late_or_upfront)} session(s) called a toolkit tool that was not in the first-turn deferred "
              "list. Either that tool's schema was sent up front, or its server connected after the first turn; "
              "the transcript does not say which.")

    # What the toolkit's schemas really weigh when a session DOES load them on
    # demand: the deferred_tools_record entries carry the full schema. Union
    # across sessions (a tool's size is the same wherever it was loaded; the
    # largest observed wins in case a server's docstrings changed between runs).
    # Scanned over EVERY session that parsed, not just the `listed` cohort: a
    # deferred_tools_record is direct evidence of a load whatever the first-turn
    # classification was. The cohorts only matter for the
    # observational comparison below.
    sizes: dict = {}
    sessions_that_loaded = 0
    for s in summarized:
        loaded = T.loaded_schema_chars(s["path"])
        if loaded:
            sessions_that_loaded += 1
        for name, chars in loaded.items():
            sizes[name] = max(chars, sizes.get(name, 0))
    print(f"\n{sessions_that_loaded} of {len(summarized)} session(s) that parsed recorded at least one toolkit tool's schema as loaded on demand.")
    if sizes:
        largest = max(sizes, key=lambda n: sizes[n])
        print(f"Schemas recorded as loaded, union across sessions: {len(sizes)} tool(s), {sum(sizes.values()):,} characters "
              f"in total, largest {largest} at {sizes[largest]:,} characters. Characters, not tokens: "
              "a token count depends on the model's vocabulary, which is not public.")

    # Observational comparison: paths are re-summarized by compare(); pass the
    # grouped paths back in.
    result = T.compare([s["path"] for s in listed], [s["path"] for s in none_listed])
    _print_comparison(result, with_label="toolkit listed", without_label="none listed")
    print("\nThis is an OBSERVATIONAL comparison of different sessions, not a controlled "
          "with/without measurement: the sides differ in repo, CLAUDE.md, other MCP servers "
          "and conversation length, and the 'none listed' side may include sessions that had "
          "the toolkit connected (see above). Treat the difference as context, not as the "
          "toolkit's cost. Use --with/--without for the controlled version.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().split("\n\n")[0])
    parser.add_argument("--with", dest="with_paths", nargs="+", metavar="JSONL",
                        help="transcripts of sessions with the toolkit's MCP servers connected")
    parser.add_argument("--without", dest="without_paths", nargs="+", metavar="JSONL",
                        help="transcripts of sessions with the toolkit's MCP servers removed")
    parser.add_argument("--survey", nargs="?", const=os.path.expanduser("~/.claude/projects"), metavar="PROJECTS_DIR",
                        help="scan a transcripts directory (default ~/.claude/projects)")
    args = parser.parse_args(argv)
    if args.survey is not None:
        return _run_survey(args.survey)
    if args.with_paths and args.without_paths:
        return _run_compare(args.with_paths, args.without_paths)
    parser.error("give --with and --without, or --survey")
    return 2


if __name__ == "__main__":
    sys.exit(main())
