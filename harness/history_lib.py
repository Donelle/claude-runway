"""Past Copilot reviews as the answer key for the live harnesses.

The goal is parity with Copilot, so its recorded findings are the ground truth
for the diff it reviewed: each inline comment becomes a `Bug` on the file and
line it was made on. A snapshot is a plain JSON file written once by
`collect_copilot_history.py` from the PR's existing reviews and comments.

Nothing here, or anywhere under `harness/`, asks GitHub to start a review.
Collecting history reads what is already there; replaying it is offline.

Stdlib only. `build_snapshot` takes already-fetched API data, so it is tested
with recorded responses and no network.
"""

import json
from pathlib import Path

from cases import Bug, Case

SNAPSHOT_VERSION = 1


def is_copilot(login: object) -> bool:
    return isinstance(login, str) and "copilot" in login.lower()


def _first_sentence(body: str, limit: int = 160) -> str:
    text = " ".join(body.split())
    return (text.split(". ")[0] if ". " in text else text)[:limit]


def build_snapshot(pr_meta: dict, reviews: list, comments: list, repo: str, base_changes: int = 0) -> dict:
    """One round per Copilot review, oldest first, each with the commit it reviewed and its inline findings.

    `reviews` and `comments` are the GitHub REST lists for the PR (`/pulls/N/reviews`, `/pulls/N/comments`).
    A comment belongs to a round by `pull_request_review_id`; replies (`in_reply_to_id`) are not findings.
    The review body is kept verbatim: it can list findings that have no inline comment, which is the
    separate intake problem of the Copilot-feedback work and is not parsed here."""
    rounds = []
    copilot_reviews = [r for r in reviews if is_copilot((r.get("user") or {}).get("login"))]
    for review in sorted(copilot_reviews, key=lambda r: r.get("submitted_at") or ""):
        findings = [
            {
                "id": c["id"],
                "path": c.get("path") or "",
                "line": c.get("original_line") or c.get("line") or 0,
                "start_line": c.get("start_line") or 0,
                # LEFT = the comment is on the OLD file (a removed line), so its line number is not a line of the
                # code the reviewers read. GitHub omits nothing here; an older snapshot without it is all RIGHT.
                "side": c.get("side") or "RIGHT",
                "commit": c.get("original_commit_id") or c.get("commit_id") or review.get("commit_id"),
                "body": c.get("body") or "",
            }
            for c in comments
            if c.get("pull_request_review_id") == review.get("id") and not c.get("in_reply_to_id") and c.get("path")
        ]
        rounds.append({"review_id": review["id"], "commit": review.get("commit_id"), "submitted_at": review.get("submitted_at"),
                       "state": review.get("state"), "body": review.get("body") or "", "findings": findings})
    return {"version": SNAPSHOT_VERSION, "repo": repo, "pr": pr_meta["number"], "title": pr_meta.get("title", ""),
            "base_sha": pr_meta.get("baseRefOid", ""), "head_sha": pr_meta.get("headRefOid", ""),
            # How many times the PR was retargeted. GitHub records the base only as it is NOW, so a replay of a
            # retargeted PR would diff against a base Copilot never saw; `history_case` refuses those.
            "base_changes": base_changes, "rounds": rounds}


# ---------------------------------------------------------------------------
# Choosing which PRs to record
# ---------------------------------------------------------------------------
#
# A pile of the PRs with the most findings would all look alike (big Python changes argued over for many rounds). A
# balanced set covers every value of four traits: what kind of change it was, how big, how many Copilot rounds and how many
# inline findings. `choose_balanced` picks it by a fixed rule so the same PRs give the same set.

SHELL_SUFFIXES = (".sh", ".ps1", ".bat", ".cmd")
DOC_SUFFIXES = (".md", ".txt", ".rst")


def kind_of(paths: list) -> str:
    """`docs` when every changed file is prose, `shell` when any is a shell or batch script, `python` when any is Python,
    else `other`."""
    names = [str(p).lower() for p in paths]
    if names and all(n.endswith(DOC_SUFFIXES) for n in names):
        return "docs"
    if any(n.endswith(SHELL_SUFFIXES) for n in names):
        return "shell"
    if any(n.endswith(".py") for n in names):
        return "python"
    return "other"


def size_of(lines_changed: int) -> str:
    return "small" if lines_changed < 150 else "medium" if lines_changed < 1000 else "large"


def rounds_of(rounds: int) -> str:
    return "few rounds" if rounds <= 2 else "some rounds" if rounds <= 5 else "many rounds"


def findings_of(inline: int) -> str:
    return "few findings" if inline <= 2 else "some findings" if inline <= 5 else "many findings"


def profile(pr: dict, rounds: int, inline: int) -> dict:
    """What `choose_balanced` needs to know about one PR: `pr` is a `gh pr list` item (number, title, additions, deletions,
    files), `rounds` the number of Copilot reviews it had and `inline` the inline findings across them."""
    paths = [f.get("path", "") for f in pr.get("files") or []]
    return {"pr": pr["number"], "title": pr.get("title", ""), "rounds": rounds, "inline": inline,
            "traits": (kind_of(paths), size_of(int(pr.get("additions") or 0) + int(pr.get("deletions") or 0)),
                       rounds_of(rounds), findings_of(inline))}


def choose_balanced(profiles: list, count: int, exclude: "set | frozenset" = frozenset()) -> list:
    """The PR numbers of a balanced set of up to `count`, in the order chosen.

    Only a PR with at least one Copilot review and one inline finding can be scored. Each pick is the PR that adds the
    most trait values not yet covered; ties go to the PR with more findings, then the lower number, so the result never
    depends on the order the PRs arrive in. Once every value is covered the rest are the PRs whose combination of traits has
    not been chosen yet, then the ones with the most findings."""
    pool = [p for p in profiles if p["rounds"] >= 1 and p["inline"] >= 1 and p["pr"] not in exclude]
    chosen: list = []
    covered: set = set()
    combos: set = set()
    while pool and len(chosen) < count:
        def rank(p):
            new = sum(1 for dimension, value in enumerate(p["traits"]) if (dimension, value) not in covered)
            return (new, p["traits"] not in combos, p["inline"], -p["pr"])
        best = max(pool, key=rank)
        chosen.append(best["pr"])
        covered.update(enumerate(best["traits"]))
        combos.add(best["traits"])
        pool.remove(best)
    return chosen


def load_snapshot(path: "str | Path") -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != SNAPSHOT_VERSION or "rounds" not in data:
        raise ValueError(f"{path} is not a version {SNAPSHOT_VERSION} Copilot history snapshot")
    return data


def snapshot_case(snapshot: dict, round_index: int = 0) -> tuple:
    """`(Case, base_sha, head_sha)` for one round: the diff Copilot saw, and its findings as the answers.

    The case carries no file contents (the diff comes from git at those two commits); its bugs hold
    literal lines. Round indexes count from 0 = Copilot's first review."""
    rounds = snapshot["rounds"]
    if not 0 <= round_index < len(rounds):
        raise ValueError(f"round {round_index} does not exist: PR #{snapshot['pr']} has {len(rounds)} Copilot review(s)")
    review = rounds[round_index]
    # A comment on a removed line (side LEFT) is in old-file coordinates; the reviewers read the new file and are
    # shown a removed line only as a marker. It cannot be matched by line number, so it is left out of the scored
    # answers (and so out of the parity denominator) and reported, rather than scored against the wrong line.
    scored = [f for f in review["findings"] if f["line"] and f.get("side", "RIGHT") != "LEFT"]
    unsupported = sum(1 for f in review["findings"] if f["line"] and f.get("side", "RIGHT") == "LEFT")
    bugs = tuple(
        Bug(id=f"copilot-{f['id']}", file=f["path"], marker="", cls="copilot", claim=_first_sentence(f["body"]),
            failing_input="", line=int(f["line"]))
        for f in scored
    )
    left_out = f"; {unsupported} on removed line(s) not scored" if unsupported else ""
    case = Case(name=f"pr{snapshot['pr']}-round{round_index + 1}",
                description=f"Copilot's review {review['review_id']} of PR #{snapshot['pr']} at {str(review['commit'])[:9]}: {len(bugs)} inline finding(s) scored{left_out}.",
                base={}, head={}, bugs=bugs)
    return case, snapshot["base_sha"], review["commit"]
