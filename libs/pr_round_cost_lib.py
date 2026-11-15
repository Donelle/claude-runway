#!/usr/bin/env python3
"""
Shared, dependency-free reading of the REVIEW ROUNDS of one pull request out of a
Claude Code session transcript, and of what Claude spent in each (part
of the "Claude's cost per Copilot review round" work). A pure reader: it returns
plain data and writes nothing, calls no GitHub API and opens no database; what is
recorded, and where, is a later piece.

WHAT A ROUND IS
  - A COPILOT round is one review by the Copilot reviewer bot, identified by the
    newest Copilot review a feedback fetch shows. A fetch whose newest Copilot review
    differs from the newest one the previous Copilot-showing fetch showed STARTS a
    round; a fetch that shows the same one, or none, is a plain fetch call kept inside
    the current round. Only a REVIEWS fetch (`gh api .../reviews`, `gh pr view --json
    ...reviews...`) or the marker can start a round or supply a review id; every
    `comments` or issue-comments fetch, whitelisted or not, is a plain fetch call and its
    output is never read for review ids.
  - ONE REVIEW, ONE ROUND, WHATEVER ITS ID FORM. A review has a numeric id in `gh api`
    output and a `PRR_...` node id in `gh pr view --json` and in the marker, so ids alone
    would split one review into two rounds whenever a session used both. The round is keyed
    by the numeric id once any fetch has shown it for that review; a `PRR_` id is an alias
    that resolves to it when one output shows both (same submission second, and the same
    commit when both show one) or when a later fetch shows the numeric id with the same
    submission time and a compatible commit, and stands alone only when no numeric form was
    ever seen. Two different numeric ids, two different node ids, or two reviews whose
    commits differ, are never the same review, whatever second they share, in one output or
    across fetches. The numeric id reaches the Copilot round itself,
    not the window the fetch lands in (a local round may have started since). The marker
    keeps the submission time (and the other id form) its own call's output shows for the
    review it names, or a later REST fetch could not be tied to it; that output only
    enriches the named review, it can never start a round or name another review. A marker
    whose call shows nothing about its review (an unlisted command with no raw JSON) carries
    the id alone, and a later numeric form of that review cannot be tied to it.
  - A LOCAL round is one call to the codebase-reviewer's `review` tool
    (`mcp__codebase-reviewer__review`) on this PR. It is a marker of its own kind.
    The call names the PR in `params.pr` (optional): a call naming ANOTHER PR is not
    read, one naming none is read with `pr` None.
  - Window rules, one for both kinds: a round runs from its marker to the START of
    the next round (of either kind), or, for the last one, to the merge time the
    caller passes in, else to the last timestamp in the transcript. A local round
    therefore ends at the next `review` call, at the first fetch that starts a
    Copilot round, or at the merge. (A fetch that starts no round, because it shows
    the same Copilot review as before or none, does NOT end a local round: the first
    one after a local review is usually a poll for a Copilot review that has not
    arrived yet, and it is part of the same stretch of work.) Messages are placed by
    their request's start time (the earliest line of a streamed response), so a response
    that straddles a boundary is counted once, in the window it started in; the one
    that issues a marker call is in the window BEFORE it when its thinking or text lines
    are stamped earlier than the call. The interval is half-open, [start, end), except
    that the last window also includes its end.

WHAT IS A FEEDBACK FETCH: THE MARKER, AND A CLOSED LIST, NOTHING ELSE
  A Bash call counts as a feedback fetch in exactly two ways. Nothing in a command is
  parsed: no tokenising, no quoting rules, no looking through `timeout`, `env`, `if`,
  a substitution, a pipeline or a here-document. That is deliberate. Every attempt to
  read free-form shell found one more edge (an escaped quote, a heredoc, an option's
  value read as the command), because shell is open-ended; a closed list has no such
  class.
  1. THE MARKER. A `tool_result` carrying the line
         pr-feedback-fetch: pr=<N> copilot-review=<id>
     (`<id>` numeric, a `PRR_...` node id, or `none`), which the `/cr-pr-feedback`
     skill prints after its fetch by running `echo` as its own Bash call (the line is
     that call's output, so it is a `tool_result`; the same line in an assistant message
     is never read). The skill names the numeric id. It is taken as written: the PR and the newest
     Copilot review it names are the round's identity, whatever the command was and
     whatever the output looked like. A marker naming another PR is not this PR's.
  2. THE WHITELIST (`_SHAPES`), for transcripts that predate the marker or never had
     it. A command counts only if the WHOLE command text matches one entry, which is a
     fixed template with slots for the PR number, the repository and a `--jq` program
     (a slot is a character class that cannot hold the quote around it, so no rule
     needs to know what quoting is). The entries are the shapes that really occur in
     the transcripts that exist, as the skills issue them: `gh api
     repos/<owner>/<repo>/pulls/<N>/reviews`, `.../pulls/<N>/comments`,
     `.../issues/<N>/comments` and `gh pr view <N> --json ...reviews...` (or `...comments...`
     without `reviews`, a comments-only fetch), each with the
     optional `--paginate` and `--jq` forms; the three-endpoint sequence the autowork
     skill's fetch step issues and the four-command block `/cr-pr-feedback` issues;
     and the three ad hoc compound commands of the session that anchors the epic's
     table, listed whole. A new shape is added to `_SHAPES` with a fixture row in
     `COMMAND_TABLE`, never by making the recogniser understand more shell.
  Any other Bash command that mentions `gh pr` or `gh api` together with this PR's number
  and `reviews` or `comments` is NOT read and NOT guessed at: it is counted in
  `fetch_calls` and `skipped_fetches` (the unrecognised-fetch count) and nothing else.
  (This count is a plain text test, not a parse: it also counts a reply to a review
  comment, which names the same PR and path. It says "there was a call here the reader
  did not read", never what the call did.)

WHICH OUTPUT SHAPES ARE READ (never skill names: the reader keys on command and
output, so a session that fetched by hand, or with the older /cr-gh-pr-feedback,
reads the same way)
  - raw JSON from `gh api .../pulls/<N>/reviews` (numeric `id`, `user.login`,
    `submitted_at`, `commit_id`, `state`) and from `gh pr view --json reviews`
    (`id` like `PRR_...`, `author.login`, `submittedAt`, `commit.oid`, `state`);
  - one line per review in the form `#<id> [<login>] <STATE> commit=<sha> <time>`,
    which is what a `--jq` that prints reviews commonly produces and what real
    sessions use;
  - a comments fetch (inline, issue, or `gh pr view --json comments` without `reviews`)
    is not read at all: it is a plain fetch call, `recognised` whenever it has a result.
    A `gh pr view` with both fields, or a chain of a reviews and a comments fetch, holds
    comment text beside the reviews, so only raw JSON is read from it.
  WHERE IN THE OUTPUT A REVIEW IS READ. A comment or review body printed as text can quote
  a review header or a JSON example, and that would read as a review (a phantom round). So
  records are read only from the part of an output the reviews fetch produced, and an
  output that cannot be told apart is not recognised rather than scanned: a fetch with
  no reviews source (comments only) is not read at all and never yields a review; one that
  chains a reviews and a comments fetch with no label between them (or a `gh pr view` of
  both fields) is read as raw JSON only (a printed line may be a comment that pastes a header); a command that prints an `echo`
  label before its reviews and another after (one session's) is read between those two
  lines only, each of which must appear exactly once, as raw JSON when its reviews command
  has no `--jq` and as review lines when it has; every other fetch must print nothing but
  JSON values, or nothing but review lines (raw `gh` JSON qualifies, its bodies stay inside
  strings; a `--jq` that prints body text among the records does not, and is counted
  unrecognised). The one residual case: a review body printed inside the reviews section of
  a labelled command with a `--jq`, which itself quotes a review header.
  `latestReviews` is never read (it returned an empty `id` for the Copilot entry), and
  the author is filtered to the Copilot reviewer, so the user's own `COMMENTED` replies
  never look like a new round. A review counts only if it is this PR's, when the
  output says which PR it is (`number` on a `gh pr view` object, `pull_request_url` on a
  REST review). A whitelisted fetch whose output is none of the above (a `--jq`
  that prints something else, an error, output persisted to a file, output the
  compression hook summarised) is NOT a marker: it is kept as a plain fetch call,
  flagged `recognised: False`, and counted in `skipped_fetches`, never guessed at.

WHAT IS MEASURED
  Per round and per fetch call, over its window: messages (distinct API responses),
  tool calls, wall clock, the model with the most output tokens, and the four token
  fields summed, never merged into one number (cache reads dominate by orders of
  magnitude). Usage is taken once per `message.id` (field-wise max, since a streamed
  message can be written in progress first), the same rule as libs/autowork_cost_lib.py,
  which this module deliberately does not import or edit. A fetch call's window runs
  from the call to the next fetch call, or to the end of its round, so a round's fetch
  windows tile the round from its first fetch.

KNOWN LIMITS
  A window includes anything else Claude did in it (work on another PR, an unrelated
  question); a Copilot round's first messages, before its first fetch call, are not in
  it; nothing a subagent did is here (its transcript is a separate file this module is
  never given), so a fetch made only inside a subagent starts no round and a subagent's
  tokens are in no round (whether they can be read is an open spike); a fetch that names the PR
  by branch or by nothing (`gh pr view`) is not matched, since only the number is
  known here; the repository in an `gh api` path is not checked, so a PR with the same
  number in another repository fetched in the same session would be read as this one;
  a fetch the whitelist does not list (wrapped, piped, captured in a variable, issued
  inside a loop, with an unlisted flag) is counted unrecognised and starts no round,
  so a session that fetches only like that has no rounds here until it prints the
  marker or the shape is added; a `review` call whose result is missing (interrupted)
  has `result_chars` None; a call with no timestamp cannot be placed and counts in
  `skipped_fetches` if it was a fetch. The transcript format is internal to Claude Code
  and undocumented; it may change with any release, so every reader here fails open
  (an empty result with `ok` False) rather than raising.
"""

import itertools
import json
import re
from datetime import datetime, timezone
from typing import Optional

import transcript_context_lib

_TOKEN_KEYS = (
    ("input_tokens", "input_tokens"),
    ("cache_creation_input_tokens", "cache_creation_tokens"),
    ("cache_read_input_tokens", "cache_read_tokens"),
    ("output_tokens", "output_tokens"),
)
_FIELDS = tuple(k for _, k in _TOKEN_KEYS)

KIND_COPILOT = "copilot"
KIND_LOCAL = "local"

# The tool name a transcript records for the codebase-reviewer server's `review` tool.
REVIEW_TOOL = "mcp__codebase-reviewer__review"

# Fetch sources, one per shape of command (a whitelisted command can carry several), and
# the one a marker gives when no whitelisted shape matched.
SRC_PR_VIEW = "pr_view"                      # `gh pr view --json ...reviews...`: a reviews source
SRC_PR_VIEW_COMMENTS = "pr_view_comments"    # `gh pr view --json ...comments...` with no `reviews`: comments only
SRC_API_REVIEWS = "api_reviews"
SRC_API_PULL_COMMENTS = "api_pull_comments"
SRC_API_ISSUE_COMMENTS = "api_issue_comments"
SRC_MARKER = "marker"

_COPILOT_LOGINS = ("copilot-pull-request-reviewer", "copilot")

# One review per line, as an ad hoc `--jq` prints it: `5795366119 [login[bot]] COMMENTED
# commit=683c25d 2026-11-06T03:27:21Z`, the id optionally after a `#`. The id is numeric (REST) or a PRR_ node id.
_REVIEW_LINE_RE = re.compile(
    r"^[ \t]*#?(\d+|PRR_[A-Za-z0-9_-]+)[ \t]+\[([^\s\]]+(?:\[bot\])?)\][ \t]+"
    r"(APPROVED|COMMENTED|CHANGES_REQUESTED|DISMISSED)\b(.*)$", re.MULTILINE)
_ISO_RE = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)?")
_COMMIT_RE = re.compile(r"\bcommit=([0-9a-fA-F]+)")

# The marker `/cr-pr-feedback` prints after its fetch: one whole line.
_MARKER_RE = re.compile(
    r"^pr-feedback-fetch: pr=(\d{1,9}) copilot-review=(\d+|PRR_[A-Za-z0-9_-]+|none)[ \t\r]*$", re.MULTILINE)

# Bound on how many candidate JSON starts one output is scanned for, so an output made of
# thousands of lines that begin with a brace cannot make one call quadratic.
_MAX_JSON_STARTS = 200

# No whitelisted command is anywhere near this long (the longest is under 2,000 characters); a longer one is not matched, which
# also bounds the work the patterns below can do.
_MAX_COMMAND_CHARS = 8000


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _count(v) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0


def _parse_ts(s) -> Optional[float]:
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    # A time with no offset is read as UTC, the zone every transcript timestamp is in;
    # without this it would be read in the machine's local zone and shift by hours.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content
                         if isinstance(b, dict) and isinstance(b.get("text"), str))
    return ""


def _is_copilot(login) -> bool:
    if not isinstance(login, str):
        return False
    name = login.lower()
    if name.endswith("[bot]"):
        name = name[:-len("[bot]")]
    return name in _COPILOT_LOGINS


# --------------------------------------------------------------------------
# Recognising a feedback fetch: the marker, and a closed list of command shapes
# --------------------------------------------------------------------------
#
# A shape is a list of parts joined by a separator; a part is (pattern, source) where
# source is the feedback source the part fetches, or None for a part that fetches nothing
# (an `echo` label, a status check). The PR number is written `<N>` in a part: the first
# one in a shape binds it and every later one must repeat it, so a shape never mixes PRs.
# Every pattern is matched against the WHOLE command, so a command that adds anything to
# a shape (a wrapper, a pipe, a redirect, a trailing word) is not that shape.

_N = "<N>"
_REPO = r"[\w.-]+/[\w.-]+"
# `--jq '<program>'` (the program cannot hold a single quote or a newline) or a bare word
# such as `.` or `length`.
_JQ = r"(?: --jq (?:'[^'\n]*'|[\w.]+))?"
_PAGINATE = r"(?: --paginate)?"
_QUOTED = r"'[^'\n]*'"
_QUOTE_IDS = itertools.count(1)
_LABEL_IDS = itertools.count(1)
_JQ_IDS = itertools.count(1)


def _api(kind: str, leaf: str, source: str, shell_var: bool = False) -> tuple:
    """`gh api <endpoint> [--paginate] [--jq ...]` for one PR endpoint. The endpoint is
    `repos/<owner>/<repo>/...` or the skill's `repos/{owner}/{repo}/...` (each may be
    double-quoted), or `$R/...` where the same command has just set `R=repos/<owner>/<repo>`."""
    base = r"\$R" if shell_var else rf"repos/(?:{_REPO}|\{{owner\}}/\{{repo\}})"
    path = rf"{base}/{kind}/{_N}/{leaf}"
    if shell_var:
        # The reviews part also captures its `--jq` (empty when absent): with none, its output
        # is raw JSON, which is read differently from the lines a jq program prints.
        jq = rf"(?P<jq{next(_JQ_IDS)}>{_JQ})" if source == SRC_API_REVIEWS else _JQ
        return rf"gh api {path}{_PAGINATE}{jq}", source
    # The same quote (or none) on both ends of the endpoint; one group name per part.
    quote = f"q{next(_QUOTE_IDS)}"
    return rf'gh api (?P<{quote}>"?){path}(?P={quote}){_PAGINATE}{_JQ}', source


# `gh pr view <N> [-R <owner>/<repo>] --json <fields>`: a fetch only when the fields include
# `reviews` (a reviews source, SRC_PR_VIEW), `comments` (comments only, SRC_PR_VIEW_COMMENTS), or
# both (the pair of sources). `latestReviews` is a different word and never counts.
# With both, the source is the pair: the output holds comment text beside the reviews, so it is
# read as raw JSON only (a `--jq` that prints the comment bodies cannot be told from records).
_VIEW_FETCH = (rf"gh pr view {_N}(?: -R {_REPO})? --json (?=[\w,]*\breviews\b)(?![\w,]*\bcomments\b)\w+(?:,\w+)*{_JQ}",
               SRC_PR_VIEW)
_VIEW_COMMENTS = (rf"gh pr view {_N}(?: -R {_REPO})? --json (?![\w,]*\breviews\b)(?=[\w,]*\bcomments\b)\w+(?:,\w+)*{_JQ}",
                  SRC_PR_VIEW_COMMENTS)
_VIEW_BOTH = (rf"gh pr view {_N}(?: -R {_REPO})? --json (?=[\w,]*\breviews\b)(?=[\w,]*\bcomments\b)\w+(?:,\w+)*{_JQ}",
              (SRC_PR_VIEW, SRC_PR_VIEW_COMMENTS))

_API_REVIEWS = _api("pulls", "reviews", SRC_API_REVIEWS)
_API_PULL_COMMENTS = _api("pulls", "comments", SRC_API_PULL_COMMENTS)
_API_ISSUE_COMMENTS = _api("issues", "comments", SRC_API_ISSUE_COMMENTS)
_VAR_REVIEWS = _api("pulls", "reviews", SRC_API_REVIEWS, shell_var=True)
_VAR_PULL_COMMENTS = _api("pulls", "comments", SRC_API_PULL_COMMENTS, shell_var=True)
_VAR_ISSUE_COMMENTS = _api("issues", "comments", SRC_API_ISSUE_COMMENTS, shell_var=True)

# Parts that fetch nothing, used only by the three compound commands of the legacy session.
_SET_R = (rf"R=repos/{_REPO}", None)
_GIT_REMOTE = ("git remote get-url origin", None)


def _label() -> tuple:
    """An `echo "<label>"` line. The label text is captured, because it is also the line the
    command prints into the output: it marks where the next part's output begins."""
    return rf'echo "(?P<lab{next(_LABEL_IDS)}>[^"\\$`\n]*)"', None


_VIEW_STATUS = (rf"gh pr view {_N} -R {_REPO} --json (?!\S*\b(?:reviews|comments)\b)\w+(?:,\w+)* --jq {_QUOTED}", None)
_PR_CHECKS = (rf"gh pr checks {_N} -R {_REPO} 2>&1 \| cut -c1-\d+", None)
_GRAPHQL = (rf"gh api graphql -f query={_QUOTED} --jq {_QUOTED}", None)
_LS_PLANS = (r"ls -la \.ai-plans \| cut -c1-\d+", None)

_ANY_JOIN = r"(?:; | && )"
_SKILL_JOIN = r"(?:\n|; | && )"


def _label_group(pattern: str) -> Optional[str]:
    m = re.search(r"\(\?P<(lab\d+)>", pattern)
    return m.group(1) if m else None


def _shape(parts: list, join: str = "; ") -> tuple:
    """(compiled pattern, sources, section) for one whole-command template. `section` is None,
    or (start label group, end label group or None, jq group) when the part before the reviews
    fetch is an `echo` label: the command then prints that label before the reviews and the
    next label after them, and only the text between is the reviews fetch's output."""
    pattern = join.join(p for p, _ in parts)
    first = pattern.index(_N)
    pattern = pattern[:first] + r"(?P<pr>\d{1,9})" + pattern[first + len(_N):].replace(_N, r"(?P=pr)")
    section = None
    for i, (_, source) in enumerate(parts):
        if source == SRC_API_REVIEWS and i > 0 and _label_group(parts[i - 1][0]):
            end = next((g for g in (_label_group(q) for q, _ in parts[i + 1:]) if g), None)
            jq = re.search(r"\(\?P<(jq\d+)>", parts[i][0])
            section = (_label_group(parts[i - 1][0]), end, jq.group(1) if jq else None)
            break
    sources = tuple(s for _, part in parts if part for s in ((part,) if isinstance(part, str) else part))
    return re.compile(pattern), sources, section


# The closed list. Each entry is one whole-command template; the row that pins it is in
# tests/test_pr_round_cost_lib.py's COMMAND_TABLE.
_SHAPES = (
    # One fetch, as the skills issue it.
    _shape([_VIEW_FETCH]),
    _shape([_VIEW_COMMENTS]),
    _shape([_VIEW_BOTH]),
    _shape([_API_REVIEWS]),
    _shape([_API_PULL_COMMENTS]),
    _shape([_API_ISSUE_COMMENTS]),
    # The three-endpoint fetch of the autowork skill's step, in its order, as a sequence of
    # three commands (the one chain the real transcripts hold, in dozens of calls).
    _shape([_API_REVIEWS, _API_PULL_COMMENTS, _API_ISSUE_COMMENTS], _ANY_JOIN),
    # `/cr-pr-feedback`'s fetch step: its four commands, in its order.
    _shape([_VIEW_FETCH, _API_REVIEWS, _API_PULL_COMMENTS, _API_ISSUE_COMMENTS], _SKILL_JOIN),
    # The three ad hoc compound commands of the session that worked one pull request, the one
    # transcript that anchors the epic's table: listed whole, because that is what they were.
    _shape([_SET_R, _label(), _VAR_PULL_COMMENTS, _label(), _VAR_REVIEWS, _label(), _VAR_ISSUE_COMMENTS]),
    _shape([_GIT_REMOTE, _SET_R, _VIEW_STATUS, _label(), _PR_CHECKS, _label(), _VAR_REVIEWS, _label(),
            _VAR_PULL_COMMENTS, _label(), _VAR_ISSUE_COMMENTS, _label(), _GRAPHQL]),
    _shape([_SET_R, _VIEW_STATUS, _PR_CHECKS, _label(), _VAR_REVIEWS, _label(), _GRAPHQL, _label(), _LS_PLANS]),
)


def _match_shape(command) -> Optional[tuple]:
    """(PR number, sources, reviews section or None) of the whitelisted shape the whole
    `command` is, else None. The section is (start label, end label or None, raw JSON), the
    last saying that the reviews command has no `--jq`, so its output is raw JSON."""
    if not isinstance(command, str) or len(command) > _MAX_COMMAND_CHARS:
        return None
    for pattern, sources, section in _SHAPES:
        m = pattern.fullmatch(command)
        if m:
            labels = None if section is None else (
                m.group(section[0]), m.group(section[1]) if section[1] else None,
                section[2] is not None and m.group(section[2]) == "")
            return int(m.group("pr")), list(sources), labels
    return None


def whitelisted_fetch(command) -> Optional[tuple]:
    """(PR number, sources) when the whole `command` is one of the whitelisted shapes
    (sources in order of appearance), else None. No other command is a fetch here."""
    found = _match_shape(command)
    return None if found is None else found[:2]


def marker_review(text, pr: int) -> Optional[str]:
    """The Copilot review id the marker line for PR `pr` in a result names (`none` when
    that reviewer had not reviewed yet), or None when the result carries no marker for
    this PR. With several such lines the last one wins."""
    if not isinstance(text, str) or "pr-feedback-fetch:" not in text:
        return None
    found = None
    for m in _MARKER_RE.finditer(text):
        if int(m.group(1)) == pr:
            found = m.group(2)
    return found


def mentions_unread_fetch(command, pr: int) -> bool:
    """Whether a command that is not a whitelisted shape still names a feedback fetch of
    PR `pr`: it mentions `gh pr` or `gh api`, the number and `reviews` or `comments`. A plain
    text test: it counts the call, it does not say what the call did or read anything."""
    if not isinstance(command, str) or len(command) > _MAX_COMMAND_CHARS:
        return False
    return bool(re.search(r"\bgh[ \t]+(?:pr|api)\b", command)
                and re.search(rf"(?<!\d){pr}(?!\d)", command)
                and re.search(r"reviews|comments", command))


# --------------------------------------------------------------------------
# Reading a fetch's output
# --------------------------------------------------------------------------

def _review_of(item) -> Optional[dict]:
    """One review object (either JSON shape) -> {id, login, ts, submitted_at, commit,
    state}, or None when it is not a submitted review with an id (an empty id, the
    `latestReviews` quirk, or a pending draft has nothing to identify a round by)."""
    if not isinstance(item, dict) or "state" not in item or "path" in item or "diff_hunk" in item:
        return None
    rid = item.get("id")
    if isinstance(rid, bool) or not isinstance(rid, (int, str)) or rid == "":
        return None
    who = item.get("user") if isinstance(item.get("user"), dict) else item.get("author")
    login = who.get("login") if isinstance(who, dict) else None
    submitted = item.get("submitted_at") or item.get("submittedAt")
    submitted = submitted if isinstance(submitted, str) and submitted else None
    commit = item.get("commit_id")
    if not isinstance(commit, str):
        c = item.get("commit")
        commit = c.get("oid") if isinstance(c, dict) and isinstance(c.get("oid"), str) else None
    state = item.get("state")
    if state == "PENDING" or not isinstance(login, str):
        return None
    return {"id": str(rid), "login": login, "ts": _parse_ts(submitted), "submitted_at": submitted,
            "commit": commit or None, "state": state if isinstance(state, str) else None}


def _json_values(text: str) -> list:
    """Every JSON value that starts a line of `text`, each parsed to its own end, as
    (value, start, end)."""
    out = []
    dec = json.JSONDecoder()
    pos = 0
    tries = 0
    while pos < len(text) and tries < _MAX_JSON_STARTS:
        if text[pos] in "[{":
            tries += 1
            try:
                value, end = dec.raw_decode(text, pos)
            except (ValueError, RecursionError):
                pass
            else:
                out.append((value, pos, end))
                pos = end
                continue
        nl = text.find("\n", pos)
        if nl < 0:
            break
        pos = nl + 1
    return out


def _item_pr(item) -> Optional[int]:
    """The PR a REST review object belongs to, from its `pull_request_url`."""
    url = item.get("pull_request_url") if isinstance(item, dict) else None
    m = re.search(r"/pulls/(\d+)$", url) if isinstance(url, str) else None
    return int(m.group(1)) if m else None


# How `read_reviews` treats text it does not recognise as a review record.
MODE_LOOSE = "loose"      # read every JSON value and review line found, whatever surrounds them
MODE_STRICT = "strict"    # the output must be nothing but review records, else it is not recognised
MODE_LINES = "lines"      # only review lines (`#<id> [<login>] <STATE> ...`), no JSON
MODE_JSON = "json"        # STRICT, and only JSON values: review lines are not read at all


def read_reviews(text: str, pr: Optional[int] = None, mode: str = MODE_LOOSE) -> Optional[list]:
    """The reviews a fetch's output lists, or None when its shape is not recognised.
    A recognised output with no reviews (a list of comments, an empty list) is `[]`.

    With `pr` given, a payload that says which PR it is about (`number` on a
    `gh pr view` object, `pull_request_url` on a REST review) is dropped when it is
    another PR's; one that does not say is taken as this PR's (a whitelisted command
    names exactly one PR).

    `mode` is how much else the output may hold. A comment or review body printed as text
    can contain anything, including a pasted review header or a JSON example, which would
    read as a review. `MODE_STRICT` therefore accepts an output only if it is nothing but
    JSON values, or nothing but review lines (raw `gh` JSON keeps every body inside a
    string, so it qualifies; free text between records, or JSON mixed with lines, does
    not: ambiguous, so not recognised). `MODE_JSON` is `MODE_STRICT` for an output that may
    hold comment text beside the reviews (an unlabelled chain of a reviews and a comments
    fetch): a printed line cannot be told from a comment that pastes one, so only JSON values
    are read there and a review line is never taken as a review.
    `MODE_LINES` reads review lines only and ignores everything else."""
    recognised = False
    reviews: list = []
    spans: list = []

    def take(items: list, owner: Optional[int]) -> None:
        for item in items:
            r = _review_of(item)
            if r is None:
                continue
            item_pr = _item_pr(item) if owner is None else owner
            if pr is not None and item_pr is not None and item_pr != pr:
                continue
            reviews.append(r)

    if mode != MODE_LINES:
        for value, start, end in _json_values(text):
            spans.append((start, end))
            if isinstance(value, list) and all(isinstance(i, dict) for i in value):
                recognised = True
                take(value, None)
            elif isinstance(value, dict):
                number = value.get("number")
                owner = number if isinstance(number, int) and not isinstance(number, bool) else None
                if isinstance(value.get("reviews"), list):
                    recognised = True
                    take(value["reviews"], owner)
    json_spans = len(spans)
    for m in (_REVIEW_LINE_RE.finditer(text) if mode != MODE_JSON else ()):
        recognised = True
        spans.append((m.start(), m.end()))
        rest = m.group(4)
        iso = _ISO_RE.search(rest)
        commit = _COMMIT_RE.search(rest)
        reviews.append({"id": m.group(1), "login": m.group(2), "ts": _parse_ts(iso.group(0)) if iso else None,
                        "submitted_at": iso.group(0) if iso else None,
                        "commit": commit.group(1) if commit else None, "state": m.group(3)})
    if mode in (MODE_STRICT, MODE_JSON) and recognised:
        if json_spans and len(spans) > json_spans:
            return None  # raw JSON and printed lines together: a printed line may be a quoted one
        spans.sort()
        at = 0
        for start, end in spans:
            if text[at:start].strip():
                return None
            at = max(at, end)
        if text[at:].strip():
            return None
    return reviews if recognised else None


def _section(text: str, start: str, end: Optional[str]) -> Optional[str]:
    """The part of `text` between the line `start` and the line `end` (or the end of the
    text), each of which must appear exactly once: a command prints its own label lines, so
    a label that appears twice means a body printed one too, and the section cannot be told."""
    lines = text.replace("\r\n", "\n").split("\n")
    starts = [i for i, line in enumerate(lines) if line == start]
    if len(starts) != 1:
        return None
    stop = len(lines)
    if end is not None:
        ends = [i for i, line in enumerate(lines) if line == end]
        if len(ends) != 1 or ends[0] <= starts[0]:
            return None
        stop = ends[0]
    return "\n".join(lines[starts[0] + 1:stop])


def read_fetch_output(text, pr: int, sources: list, section: Optional[tuple]) -> Optional[list]:
    """The reviews a whitelisted fetch's output lists, read only from the part of the output
    the reviews fetch produced: None when the output is not recognised or is ambiguous.
      - a command that prints a label before its reviews (the `section` labels): only what
        is between that label and the next one, so comment bodies printed in other sections
        are never scanned. With no `--jq` on the reviews command that is raw JSON, read as
        any raw output is (`MODE_STRICT`); with one it is the lines the program prints
        (`MODE_LINES`, which a body-printing program may follow with body text);
      - a command with no reviews source (comments only, `gh api` or `gh pr view`): its
        output is not read at all, whatever it holds, so a comment body that pastes a review
        header cannot become a review (`[]` when there is an output, None when there is none);
      - a command with a reviews source AND a comments source, no labels between them: raw JSON
        only (`MODE_JSON`), since the comment text is mixed into what is read;
      - any other: `MODE_STRICT`, so an output with free text among the records is ambiguous."""
    if not isinstance(text, str):
        return None
    if SRC_API_REVIEWS not in sources and SRC_PR_VIEW not in sources:
        return []
    if section is not None:
        part = _section(text, section[0], section[1])
        if part is None:
            return None
        if not part.strip():
            return []
        return read_reviews(part, pr, MODE_STRICT if section[2] else MODE_LINES)
    mixed = any(s in sources for s in (SRC_API_PULL_COMMENTS, SRC_API_ISSUE_COMMENTS, SRC_PR_VIEW_COMMENTS))
    return read_reviews(text, pr, MODE_JSON if mixed else MODE_STRICT)


def _is_numeric_id(rid) -> bool:
    return isinstance(rid, str) and rid.isdigit()


def _same_review(a: dict, b: dict) -> bool:
    """Whether two Copilot entries of one output are one review shown in both id forms (REST
    number and `PRR_` node id): the same submission second, and the same commit when both show
    one. A review shown by one shape only has no twin. Two ids of the same form are two
    reviews (a review has one numeric id and one node id), whatever second they share."""
    if a is b:
        return True
    if a["id"] != b["id"] and _is_numeric_id(a["id"]) == _is_numeric_id(b["id"]):
        return False
    if a["ts"] is None or b["ts"] is None or int(a["ts"]) != int(b["ts"]):
        return False
    return a["commit"] is None or b["commit"] is None or a["commit"][:7] == b["commit"][:7]


def _identified(reviews: list, review: dict) -> dict:
    """`review` plus `aliases`, the other ids the same output shows for it, and `key`, the
    identity two fetches compare by. The review is reported under its numeric id when the
    output shows one (the numeric id is the identity; a `PRR_` id is only an alias of it and
    stands alone when no numeric form was ever seen)."""
    twins = [r for r in reviews if _is_copilot(r["login"]) and _same_review(review, r)]
    ids = list(dict.fromkeys(r["id"] for r in twins))
    primary = next((r for r in twins if _is_numeric_id(r["id"])), review)
    key = ("t", int(review["ts"])) if review["ts"] is not None else ("id", primary["id"])
    return {**primary, "aliases": [i for i in ids if i != primary["id"]], "key": key}


def newest_copilot_review(reviews: list) -> Optional[dict]:
    """The Copilot review with the latest submission time (the last listed on a tie or
    when none has a time), plus `key`, the identity two fetches compare by, and `aliases`,
    the other ids the same output shows for it (see `_identified`)."""
    best = None
    best_rank = None
    for i, r in enumerate(reviews):
        if not _is_copilot(r["login"]):
            continue
        rank = (r["ts"] if r["ts"] is not None else float("-inf"), i)
        if best_rank is None or rank >= best_rank:
            best, best_rank = r, rank
    return None if best is None else _identified(reviews, best)


def _marked_review(text: str, pr: int, marked: str, shape: Optional[tuple]) -> dict:
    """The review a marker names, taken as written, with whatever the SAME output shows for it:
    its submission time, commit and other id form, so that a later fetch (REST, with a numeric
    id) can be recognised as the same review. The output is read only to enrich the review the
    marker names (it is found by its id): it can neither start a round nor name another review.
    The marker lines are removed first; with no whitelisted shape the output is read as raw JSON
    only (a body printed as text cannot be told from a record)."""
    bare = {"id": marked, "login": None, "ts": None, "submitted_at": None, "commit": None, "state": None,
            "aliases": [], "key": ("id", marked)}
    rest = _MARKER_RE.sub("", text)
    if shape is not None and shape[0] == pr:
        reviews = read_fetch_output(rest, pr, shape[1], shape[2])
    else:
        reviews = read_reviews(rest, pr, MODE_JSON)
    named = next((r for r in reviews or [] if r["id"] == marked and _is_copilot(r["login"])), None)
    return bare if named is None else _identified(reviews or [], named)


# --------------------------------------------------------------------------
# Reading the transcript
# --------------------------------------------------------------------------

def _pr_of_review_call(tool_input) -> tuple:
    """(pr number or None, named: bool) from a `review` call's `params.pr`."""
    params = tool_input.get("params") if isinstance(tool_input, dict) else None
    raw = params.get("pr") if isinstance(params, dict) else None
    if isinstance(raw, bool):
        return None, False
    if isinstance(raw, int):
        return raw, True
    if isinstance(raw, str) and raw.strip().lstrip("#").isdigit():
        return int(raw.strip().lstrip("#")), True
    return None, False


def _range_of_review_call(tool_input) -> Optional[str]:
    params = tool_input.get("params") if isinstance(tool_input, dict) else None
    if not isinstance(params, dict) or not isinstance(params.get("base_ref"), str) or not params["base_ref"]:
        return None
    head = params.get("head_ref")
    return f"{params['base_ref']}...{head if isinstance(head, str) and head else 'HEAD'}"


def _read_transcript(path) -> dict:
    messages: dict = {}     # message.id -> {start, model, tools, **fields}
    calls: list = []        # tool_use blocks worth keeping, in transcript order
    seen_calls: set = set()
    results: dict = {}      # tool_use_id -> result text
    last_ts: Optional[float] = None
    session_id: Optional[str] = None
    anon = 0
    for obj in transcript_context_lib.iter_entries(path):
        ts = _parse_ts(obj.get("timestamp"))
        if ts is not None:
            last_ts = ts if last_ts is None else max(last_ts, ts)
        if session_id is None and isinstance(obj.get("sessionId"), str) and obj["sessionId"]:
            session_id = obj["sessionId"]
        message = obj.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        kind = obj.get("type")
        if kind == "user" and isinstance(content, list):
            for block in content:
                if (isinstance(block, dict) and block.get("type") == "tool_result"
                        and isinstance(block.get("tool_use_id"), str)):
                    results[block["tool_use_id"]] = _result_text(block.get("content"))
            continue
        if kind != "assistant" or message.get("model") == "<synthetic>":
            continue
        mid = message.get("id")
        usage = message.get("usage")
        slot = None
        if isinstance(usage, dict) and usage:
            if not isinstance(mid, str) or not mid:
                anon += 1
                mid = f"\0anon{anon}"
            model = message.get("model") if isinstance(message.get("model"), str) and message.get("model") else "unknown"
            slot = messages.setdefault(mid, {"start": ts, "model": model, "tools": set(), **{k: 0 for k in _FIELDS}})
            for src, dst in _TOKEN_KEYS:
                slot[dst] = max(slot[dst], _count(usage.get(src)))
            if ts is not None:
                slot["start"] = ts if slot["start"] is None else min(slot["start"], ts)
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use" or not isinstance(block.get("id"), str):
                continue
            if slot is not None:
                slot["tools"].add(block["id"])
            if block["id"] in seen_calls:
                continue
            seen_calls.add(block["id"])
            calls.append({"id": block["id"], "name": block.get("name"), "input": block.get("input"),
                          "ts": ts, "time": obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None})
    return {"messages": messages, "calls": calls, "results": results, "last_ts": last_ts, "session_id": session_id}


def _window_stats(messages: list, start: float, end: float, inclusive: bool) -> dict:
    """Totals over the messages whose request started in [start, end) (or [start, end]
    when `inclusive`)."""
    sel = [m for m in messages if m["start"] >= start and (m["start"] <= end if inclusive else m["start"] < end)]
    by_model: dict = {}
    tools: set = set()
    out: dict = {k: 0 for k in _FIELDS}
    for m in sel:
        tools |= m["tools"]
        row = by_model.setdefault(m["model"], {"requests": 0, **{k: 0 for k in _FIELDS}})
        row["requests"] += 1
        for k in _FIELDS:
            row[k] += m[k]
            out[k] += m[k]
    model = max(by_model.items(), key=lambda kv: (kv[1]["output_tokens"], kv[1]["requests"], kv[0]))[0] if by_model else None
    return {"messages": len(sel), "tool_calls": len(tools), "wall_clock_s": max(0, round(end - start)),
            "model": model, "by_model": by_model, **out}


def _empty(pr, ok: bool) -> dict:
    return {"ok": ok, "pr": pr, "session_id": None, "rounds": [], "orphan_fetch_calls": [],
            "fetch_calls": 0, "skipped_fetches": 0, "end_ts": None}


def find_rounds(path, pr, merge_time=None) -> dict:
    """The review rounds of PR `pr` in the session transcript at `path`.

    `merge_time` (epoch seconds or an ISO string) ends the last window and drops any
    call after it; without it the last window ends at the last timestamp in the file.
    Returns
      {"ok", "pr", "session_id", "rounds", "orphan_fetch_calls", "fetch_calls",
       "skipped_fetches", "end_ts"}
    `rounds` is in time order; each is the window figures (`messages`, `tool_calls`,
    `wall_clock_s`, `model`, `by_model`, the four token fields) plus `number`, `kind`
    (`copilot` or `local`), `kind_number`, `start_ts`/`end_ts` (epoch seconds), `start`
    (the marker's own timestamp text) and `fetch_calls`. A Copilot round also carries
    `review_id` (the numeric id once any fetch showed it, else the `PRR_` id), `review_ids`
    (every form of its id the session showed), `submitted_at`,
    `commit_id` and `review_state`; a local round `review_type`, `pr` (None when the
    call named none), `range` and `result_chars`. Each fetch call carries `tool_use_id`,
    `time`, `ts`, `sources`, `recognised`, `round` (the round's number) and its window
    figures plus `end_ts`. `orphan_fetch_calls` are fetch calls before any round.
    `fetch_calls` is how many fetch calls there were in all, `skipped_fetches` how many
    of them were not read: the output shape was not recognised, there was no timestamp, or
    (the unrecognised-fetch count) the command named this PR's reviews or comments but was
    neither a whitelisted shape nor backed by the marker. Those last calls are counted
    only: they have no entry in `rounds` or `orphan_fetch_calls`.

    Fails open: an unreadable file, a bad `pr` or any unexpected error gives the empty
    result with `ok` False."""
    if isinstance(pr, bool) or not isinstance(pr, int) or pr <= 0:
        return _empty(None, False)
    try:
        # The shared line reader swallows OSError (so a transcript that cannot be opened
        # would read as an empty one and look like "no rounds"); open it here first so that
        # case is `ok` False, as documented.
        with open(str(path), "rb"):
            pass
        merge_ts = _parse_ts(merge_time) if isinstance(merge_time, str) else (
            float(merge_time) if isinstance(merge_time, (int, float)) and not isinstance(merge_time, bool) else None)
        return _find_rounds(str(path), pr, merge_ts)
    except Exception:
        # Broad on purpose: the transcript format is internal and undocumented, and this
        # reader is a measurement that must never break the session that asks for it.
        return _empty(pr, False)


def _find_rounds(path: str, pr: int, merge_ts: Optional[float]) -> dict:
    data = _read_transcript(path)
    out = _empty(pr, True)
    out["session_id"] = data["session_id"]
    final_end = merge_ts if merge_ts is not None else data["last_ts"]
    msgs = sorted((m for m in data["messages"].values() if m["start"] is not None), key=lambda m: m["start"])

    # Build the events: fetch calls and `review` calls, in transcript order.
    events: list = []
    skipped = 0
    fetch_total = 0
    for call in data["calls"]:
        name = call["name"]
        if name == "Bash":
            cmd = call["input"].get("command") if isinstance(call["input"], dict) else None
            text = data["results"].get(call["id"])
            marked = marker_review(text, pr)
            shape = _match_shape(cmd)
            sources = shape[1] if shape is not None and shape[0] == pr else []
            if marked is None and not sources:
                # Not a fetch of this PR by either way. A command that still names one is
                # counted, never read: no part of it is parsed.
                if (mentions_unread_fetch(cmd, pr) and shape is None
                        and not (merge_ts is not None and call["ts"] is not None and call["ts"] > merge_ts)):
                    fetch_total += 1
                    skipped += 1
                continue
            if merge_ts is not None and call["ts"] is not None and call["ts"] > merge_ts:
                continue  # after the merge: not part of this PR's review
            fetch_total += 1
            if call["ts"] is None:
                skipped += 1  # cannot be placed in any window
                continue
            if marked is not None:
                # Taken as written: the PR and the newest Copilot review the marker names. The
                # time (and other id form) the same output shows for that review is kept, so a
                # later fetch showing it by its numeric id is the same round.
                reviews = []
                newest = None if marked == "none" else _marked_review(text, pr, marked, shape)
                sources = sources or [SRC_MARKER]
            else:
                reviews = read_fetch_output(text, pr, sources, shape[2] if shape is not None else None)
                newest = newest_copilot_review(reviews) if reviews else None
            if reviews is None:
                skipped += 1
            events.append({"type": "fetch", "ts": call["ts"], "time": call["time"], "id": call["id"],
                           "sources": sources, "recognised": reviews is not None, "newest": newest})
        elif name == REVIEW_TOOL:
            named_pr, named = _pr_of_review_call(call["input"])
            if call["ts"] is None or (named and named_pr != pr) or (merge_ts is not None and call["ts"] > merge_ts):
                continue
            text = data["results"].get(call["id"])
            tin = call["input"] if isinstance(call["input"], dict) else {}
            events.append({"type": "local", "ts": call["ts"], "time": call["time"], "id": call["id"],
                           "review_type": tin.get("type") if isinstance(tin.get("type"), str) else None,
                           "pr": named_pr if named else None, "range": _range_of_review_call(tin),
                           "result_chars": len(text) if isinstance(text, str) else None})
    out["fetch_calls"] = fetch_total
    out["skipped_fetches"] = skipped
    out["end_ts"] = final_end
    events.sort(key=lambda e: e["ts"])  # stable: transcript order breaks a tie

    rounds: list = []
    orphans: list = []
    current: Optional[dict] = None
    last_key = None
    known_ids: dict = {}    # review id (any form) -> the identity key of the round it first showed in
    known_keys: dict = {}   # a key a fetch derived -> the identity keys seen under it (one, but for a clash)
    identity: dict = {}     # identity key -> {"numeric": its numeric id, "node": its PRR_ id, "commit": its commit}, once known
    round_of: dict = {}     # identity key -> the Copilot round it started (a fetch may land in another window)
    for ev in events:
        if ev["type"] == "local":
            current = {"kind": KIND_LOCAL, "start_ts": ev["ts"], "start": ev["time"], "review_type": ev["review_type"],
                       "pr": ev["pr"], "range": ev["range"], "result_chars": ev["result_chars"], "fetch_calls": []}
            rounds.append(current)
            continue
        newest = ev["newest"]
        key = None
        ids = [] if newest is None else [newest["id"], *newest["aliases"]]
        if newest is not None:
            # A review id already seen (in any of its forms) keeps the identity it had then,
            # so the same review shown later without its time (or first without, later with)
            # is not a new one. Failing that, the submission time seen with it is an alias of
            # an identity, so the review's other id form (REST number or PRR_ node id) shown
            # later with the same time lands on the same round: but only a COMPATIBLE one, never
            # one bound to another numeric id, another node id or another commit (two distinct
            # reviews that share a second are two rounds). An output that shows both forms of one review (or
            # a marker call whose output does) ties them directly.
            seen = {"numeric": next((i for i in ids if _is_numeric_id(i)), None),
                    "node": next((i for i in ids if not _is_numeric_id(i)), None),
                    "commit": newest["commit"][:7] if newest["commit"] else None}
            key = next((known_ids[i] for i in ids if i in known_ids), None)
            if key is None:
                key = next((k for k in known_keys.get(newest["key"], [])
                            if all(identity[k][f] is None or v is None or identity[k][f] == v for f, v in seen.items())), None)
            if key is None:
                key = newest["key"] if newest["key"] not in identity else ("id", newest["id"])
            info = identity.setdefault(key, {"numeric": None, "node": None, "commit": None})
            for f, v in seen.items():
                info[f] = info[f] or v
            for i in ids:
                known_ids.setdefault(i, key)
            if key not in known_keys.setdefault(newest["key"], []):
                known_keys[newest["key"]].append(key)
        if newest is not None and key != last_key:
            last_key = key
            current = {"kind": KIND_COPILOT, "start_ts": ev["ts"], "start": ev["time"], "review_id": newest["id"],
                       "review_ids": [], "submitted_at": newest["submitted_at"], "commit_id": newest["commit"],
                       "review_state": newest["state"], "fetch_calls": []}
            rounds.append(current)
            round_of[key] = current
        if newest is not None and key in round_of:
            # The round is keyed by the numeric id once any fetch has shown it: a PRR_ id that
            # started the round is replaced by it, and stays in `review_ids`. It is the review's
            # own round that is updated, not the window the fetch lands in (a local round may
            # have started since).
            rnd = round_of[key]
            if _is_numeric_id(newest["id"]) and not _is_numeric_id(rnd["review_id"]):
                rnd["review_id"] = newest["id"]
            for i in ids:
                if i not in rnd["review_ids"]:
                    rnd["review_ids"].append(i)
        (current["fetch_calls"] if current is not None else orphans).append(ev)

    if final_end is None:
        return out

    # Window figures. A round ends where the next one starts; the last ends at final_end
    # (inclusive: the closing message at the merge belongs to it).
    counters = {KIND_COPILOT: 0, KIND_LOCAL: 0}
    for i, rnd in enumerate(rounds):
        counters[rnd["kind"]] += 1
        nxt = rounds[i + 1]["start_ts"] if i + 1 < len(rounds) else None
        rnd["number"] = i + 1
        rnd["kind_number"] = counters[rnd["kind"]]
        rnd["end"] = (nxt, False) if nxt is not None else (final_end, True)
    # Taken from `events` (already in time order, transcript order on a tie) rather than
    # rebuilt from the rounds and orphans, which would reverse two fetches at one instant.
    all_fetches = [e for e in events if e["type"] == "fetch"]
    position = {id(e): i for i, e in enumerate(all_fetches)}
    next_round_start: dict = {}
    for e in orphans:
        # `>=`: a round that starts at the very instant of the fetch still bounds it.
        later = [r["start_ts"] for r in rounds if r["start_ts"] >= e["ts"]]
        next_round_start[id(e)] = min(later) if later else None

    def _fetch_record(ev: dict, round_number: Optional[int], round_end: tuple) -> dict:
        idx = position[id(ev)]
        cands = [(final_end, True), round_end]
        if idx + 1 < len(all_fetches):
            cands.append((all_fetches[idx + 1]["ts"], False))
        end, inclusive = min(cands)
        rec = {"tool_use_id": ev["id"], "time": ev["time"], "ts": ev["ts"], "sources": ev["sources"],
               "recognised": ev["recognised"], "round": round_number, "end_ts": end,
               **_window_stats(msgs, ev["ts"], end, inclusive)}
        if ev["newest"] is not None:
            rec["review_id"] = ev["newest"]["id"]
            rec["submitted_at"] = ev["newest"]["submitted_at"]
        return rec

    for rnd in rounds:
        end, inclusive = rnd["end"]
        del rnd["end"]
        rnd["end_ts"] = end
        rnd.update(_window_stats(msgs, rnd["start_ts"], end, inclusive))
        rnd["fetch_calls"] = [_fetch_record(e, rnd["number"], (end, inclusive)) for e in rnd["fetch_calls"]]
    out["rounds"] = rounds
    out["orphan_fetch_calls"] = [
        _fetch_record(e, None, (next_round_start[id(e)], False) if next_round_start[id(e)] is not None else (final_end, True))
        for e in orphans]
    return out
