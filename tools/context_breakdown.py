#!/usr/bin/env python3
"""
CLI for itemizing a subagent transcript's context (see libs/context_breakdown_lib.py
for what is measured and why each figure is a fit or a difference, not a count).
Stdlib only. Reads transcripts only; writes nothing. Output is counts, sizes and
token figures, never transcript text.

    python tools/context_breakdown.py start TRANSCRIPT [TRANSCRIPT ...]
        Request 0's token count and what it is made of: each item's characters and
        tokens, the tool definitions as the remainder, each tool's apportioned share.

    python tools/context_breakdown.py calibrate [--projects-root DIR] [--model-contains sonnet]
        Tokens per character for each class of text that enters the context after
        request 0, fitted over every subagent transcript under the root.

    python tools/context_breakdown.py growth TRANSCRIPT [--projects-root DIR]
        One transcript's growth after request 0 split by class using that fit, the
        remainder, and the requests the fit explains worst.

    python tools/context_breakdown.py reads [--projects-root DIR] [--min-chars N]
            [--since-agent ID] [--autowork-only]
        Read calls over every transcript: how many were whole-file (no offset or
        limit) with a result of at least N characters, the biggest ones by base
        name, how often a README read was a transcript's largest result, and the
        tokens those results were carried for (result size times the requests
        after it) as a share of the transcripts' total token volume.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))

import autowork_cost_lib  # noqa: E402
import context_breakdown_lib as C  # noqa: E402
import transcript_context_lib  # noqa: E402


def _model_of(path) -> str:
    for obj in transcript_context_lib.iter_entries(path):
        message = obj.get("message")
        if obj.get("type") == "assistant" and isinstance(message, dict) and isinstance(message.get("model"), str):
            return message["model"]
    return ""


def _files(root, model_contains) -> list:
    out = []
    for f in autowork_cost_lib.subagent_files(root):
        if not model_contains or model_contains in _model_of(f):
            out.append(f)
    return out


def cmd_start(args) -> int:
    for p in args.transcripts:
        s = C.start_components(p)
        if s is None:
            print(f"{p}: no request found")
            continue
        it = C.itemize_start(s)
        print(f"{os.path.basename(p)}: request 0 = {s['context0']:,} tokens")
        for name, chars in sorted(s["components"].items(), key=lambda kv: -it["items"].get(kv[0], 0)):
            print(f"  {name:<20} {chars:>8,} chars  {it['items'][name]:>8,} tokens")
        if s["tools"]:
            print("  tool definitions, apportioned by recorded characters:")
            for name, tok in sorted(it["tool_definitions_by_tool"].items(), key=lambda kv: -kv[1]):
                print(f"    {name:<18} {s['tools'][name]:>8,} chars  {tok:>8,} tokens")
        else:
            print("  (no tool list recorded: tool_definitions is everything not itemized)")
    return 0


def cmd_calibrate(args) -> int:
    files = _files(args.projects_root or autowork_cost_lib.default_projects_root(), args.model_contains)
    cal = C.calibrate_growth(files)
    print(f"{len(files)} transcripts, {cal['rows']} requests, rmse {cal['rmse']:.0f} tokens per request")
    for k, v in cal["per_char"].items():
        print(f"  {k:<20} {v:.4f} tokens/char" + (f"  ({1 / v:.2f} chars/token)" if v > 0.001 else ""))
    print(f"  per-request constant {cal['per_request']:.0f} tokens")
    return 0


def cmd_growth(args) -> int:
    files = _files(args.projects_root or autowork_cost_lib.default_projects_root(), args.model_contains)
    cal = C.calibrate_growth(files)
    r = C.reconcile(args.transcript, cal)
    if r is None:
        print("fewer than two requests: nothing to reconcile")
        return 0
    print(f"context {r['context0']:,} -> {r['final']:,}  (growth {r['growth']:,})")
    for k, v in sorted(r["by_class"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20} {r['chars'].get(k, 0):>9,} chars  {v:>8,} tokens")
    print(f"  per-request constant {r['per_request']:>8,} tokens")
    print(f"  explained {r['explained']:,}, remainder {r['remainder']:,}")
    print("  requests the fit explains worst (index, actual growth, predicted):", r["worst"])
    return 0


def cmd_reads(args) -> int:
    root = args.projects_root or autowork_cost_lib.default_projects_root()
    files = _files(root, args.model_contains)
    if args.since_agent:
        marker = next((f for f in files if args.since_agent in os.path.basename(f)), None)
        if marker is None:
            print("agent id not found")
            return 1
        marker_attempt = autowork_cost_lib.parse_attempt(marker)
        first_ts = marker_attempt["first_ts"] if marker_attempt else None
        if first_ts is None:
            print("agent has no timestamps")
            return 1
        files = [f for f in files
                 if (a := autowork_cost_lib.parse_attempt(f)) and a["first_ts"] is not None and a["first_ts"] >= first_ts]
    if args.autowork_only:
        # An autowork run names a ticket in its prompt; a probe or an ad-hoc subagent does not.
        files = [f for f in files if (a := autowork_cost_lib.parse_attempt(f)) and a["issue"] is not None]
    whole = ranged = 0
    big = []
    readme_largest = with_readme = 0
    carried_big = carried_readme = volume = 0.0
    for f in files:
        attempt = autowork_cost_lib.parse_attempt(f)
        if attempt:
            volume += autowork_cost_lib.totals(attempt["by_model"])["total_tokens"]
        calls = C.read_calls(f)
        if not calls:
            continue
        for c in calls:
            if c["ranged"]:
                ranged += 1
            else:
                whole += 1
                if c["chars"] >= args.min_chars:
                    big.append((c["chars"], os.path.basename(c["file"])))
                    carried_big += c["chars"] * C.READ_TOKENS_PER_CHAR * c["carried_requests"]
        readme = [c for c in calls if os.path.basename(c["file"]) == "README.md" and not c["ranged"]]
        if readme:
            with_readme += 1
            carried_readme += sum(c["chars"] * C.READ_TOKENS_PER_CHAR * c["carried_requests"] for c in readme)
            if max(c["chars"] for c in readme) >= max(c["chars"] for c in calls):
                readme_largest += 1
    big.sort(reverse=True)
    print(f"{len(files)} transcripts: {whole} whole-file Reads, {ranged} ranged Reads; "
          f"{len(big)} whole-file Reads of >= {args.min_chars:,} chars ({sum(b[0] for b in big):,} chars)")
    for chars, name in big[:15]:
        print(f"  {chars:>8,} chars  {name}")
    print(f"README read whole in {with_readme} transcripts, and it was that transcript's largest Read result in {readme_largest}")
    if volume:
        print(f"Tokens carried (result tokens x later requests): big whole-file Reads {carried_big:,.0f} "
              f"({100 * carried_big / volume:.1f}% of the {volume:,.0f}-token volume of these transcripts), "
              f"README reads {carried_readme:,.0f} ({100 * carried_readme / volume:.1f}%)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start")
    p.add_argument("transcripts", nargs="+")
    p.set_defaults(fn=cmd_start)
    for name, fn in (("calibrate", cmd_calibrate), ("growth", cmd_growth), ("reads", cmd_reads)):
        p = sub.add_parser(name)
        p.add_argument("--projects-root")
        p.add_argument("--model-contains", default="sonnet")
        if name == "growth":
            p.add_argument("transcript")
        if name == "reads":
            p.add_argument("--min-chars", type=int, default=12000)
            p.add_argument("--since-agent")
            p.add_argument("--autowork-only", action="store_true")
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
