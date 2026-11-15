#!/usr/bin/env bash
# Every fact /cr-gh-load-context checks, gathered in one read-only call.
#
# Usage: bash .claude/skills/cr-gh-load-context/facts.sh --porcelain
#
# The output is exact and meant to be read as it is. `--porcelain` changes
# nothing here; it is in the command line so that claude-runway's compress hook
# treats the output as exactness-critical and never summarizes it (Bash output
# over CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS, default 2000, is otherwise sent to
# LM Studio, which is slow and paraphrases). Output is still kept short: print
# verdicts and line numbers, not file contents.
#
# Plain bash 3.2 (macOS) and Git Bash on Windows; needs git and an authenticated gh.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1

# Point every gh call at origin's repo. Without this, gh picks the repo itself,
# and in a checkout that also has an `upstream` remote (a fork) it can pick the
# wrong one, so every issue and PR figure below would come from another repo.
# gh reads GH_REPO as its default repository for each command, including the
# ones in subshells. Handles https:// and ssh (git@host:owner/repo) remote URLs.
origin_slug=$(git remote get-url origin 2>/dev/null \
  | sed -E 's#^(git@[^:]+:|ssh://[^/]+/|https?://[^/]+/)##; s#\.git$##')
case "$origin_slug" in
  */*) export GH_REPO="$origin_slug" ;;
  *)   origin_slug="" ;;
esac

section() { printf '\n## %s\n' "$1"; }
# Every regular, non-hidden file directly in a folder, one per line.
files_in() { find "$1" -maxdepth 1 -type f ! -name '.*' -exec basename {} \; 2>/dev/null | sort; }
# A .plans/ artifact name: the whole name is {digits}-{plan|research|validate}.md.
TRIAD_RE='^[0-9]+-(plan|research|validate)\.md$'

section "git"
echo "branch: $(git branch --show-current)"
echo "main: $(git rev-parse --short main 2>/dev/null) origin/main: $(git rev-parse --short origin/main 2>/dev/null || echo '?')"
echo "commits on this branch beyond main: $(git rev-list --count main..HEAD 2>/dev/null || echo '?')"
echo "uncommitted paths: $(git status --porcelain | wc -l | tr -d ' ')"

section "github"
if [ -n "$origin_slug" ]; then
  echo "repo: $origin_slug (from origin)"
else
  echo "repo: UNKNOWN (origin has no owner/repo URL); gh will choose one, so treat the numbers below with care"
fi
gh issue list --state open --limit 300 --json labels --jq '
  "open issues: \(length) (" +
  ([.[] | ([.labels[].name | select(startswith("priority-"))][0] // "no-priority")]
   | group_by(.) | map("\(.[0] | sub("priority-"; "")) \(length)") | join(", ")) + ")"' \
  || echo "gh issue list failed"
echo "assigned to you:"
gh issue list --assignee @me --state open --limit 300 --json number,title \
  --jq '.[] | "  #\(.number) \(.title[0:90])"' || echo "  gh failed"
echo "open PRs:"
gh pr list --state open --limit 300 --json number,title,headRefName \
  --jq '.[] | "  #\(.number) [\(.headRefName)] \(.title[0:80])"' || echo "  gh failed"

section "local-only files (gitignored)"
for p in CLAUDE.md .plans .ai-plans; do
  if [ -e "$p" ]; then echo "$p: present"; else echo "$p: ABSENT"; fi
done

if [ -d .plans ]; then
  section ".plans"
  plans=$(files_in .plans)
  [ -z "$plans" ] && echo "(empty)"
  printf '%s\n' "$plans" | grep -vE "$TRIAD_RE" | grep -v '^$' | sed 's/^/MISPLACED (not part of a triad): /'
  for n in $(printf '%s\n' "$plans" | grep -E "$TRIAD_RE" | grep -oE '^[0-9]+' | sort -un); do
    have=""
    for k in research plan validate; do [ -f ".plans/$n-$k.md" ] && have="$have $k"; done
    state=$(gh issue view "$n" --json state,closedByPullRequestsReferences \
      --jq '"\(.state), closing PRs: \([.closedByPullRequestsReferences[].number] | map("#\(.)") | join(",") | if . == "" then "none" else . end)"' 2>/dev/null)
    echo "#$n: has${have:- nothing}; issue ${state:-lookup failed}"
    open_q=$(grep -n '^## Open questions' ".plans/$n-plan.md" 2>/dev/null | cut -d: -f1)
    [ -n "$open_q" ] && echo "  $n-plan.md: open questions at line $open_q"
  done
fi

if [ -d .ai-plans ]; then
  section ".ai-plans"
  files_in .ai-plans | while IFS= read -r f; do
    echo "$f ($(wc -l < ".ai-plans/$f" | tr -d ' ') lines): $(head -n 1 ".ai-plans/$f" | cut -c1-90)"
  done
  wo=.ai-plans/ticket-work-order.md
  if [ -f "$wo" ]; then
    total=$(wc -l < "$wo" | tr -d ' ')
    earlier=$(grep -n '^\*\*Earlier snapshot' "$wo" | head -n 1 | cut -d: -f1)
    left=$(grep -n "^## 1\. What's left" "$wo" | head -n 1 | cut -d: -f1)
    done_h=$(grep -n '^## 2\. Done' "$wo" | head -n 1 | cut -d: -f1)
    sections=$(grep -n '^Sections:' "$wo" | head -n 1 | cut -d: -f1)
    # The latest snapshot ends before the first earlier one; with none, before the
    # "Sections:" line or the first section heading; with neither, at the end.
    end_at=${earlier:-${sections:-${left:-}}}
    if [ -n "$end_at" ]; then latest_end=$(( end_at - 1 )); else latest_end=$total; fi
    if [ -n "$left" ]; then
      left_end=$(( ${done_h:-$(( total + 1 ))} - 1 ))
      left_range="lines $left-$left_end"
    else
      left_range="not found"
    fi
    echo "work order: latest snapshot = lines 1-$latest_end; '1. What's left' = $left_range"
    snap=$(head -n "$latest_end" "$wo")
    wo_count=$(echo "$snap" | grep -oE '[0-9]+ open issues' | head -n 1 | grep -oE '^[0-9]+')
    wo_sha=$(echo "$snap" | grep -oE '`main` is (still )?at `[0-9a-f]+`' | head -n 1 | grep -oE '[0-9a-f]{7,}')
    live_count=$(gh issue list --state open --limit 300 --json number --jq length 2>/dev/null)
    live_sha=$(git rev-parse --short main 2>/dev/null)
    # Compare only when both sides are known; a missing side is UNKNOWN, not STALE or ok.
    if [ -z "$wo_count" ] || [ -z "$live_count" ]; then c=UNKNOWN
    elif [ "$wo_count" = "$live_count" ]; then c=ok; else c=STALE; fi
    if [ -z "$wo_sha" ] || [ -z "$live_sha" ]; then s=UNKNOWN
    else
      # Abbreviations can differ in length: one must be a prefix of the other.
      case "$wo_sha" in "$live_sha"*) s=ok ;; *) case "$live_sha" in "$wo_sha"*) s=ok ;; *) s=STALE ;; esac ;; esac
    fi
    echo "work order says ${wo_count:-?} open, main ${wo_sha:-?}; live ${live_count:-?} open, main ${live_sha:-?} -> count $c, sha $s"
  fi
fi

section "skill inventories"
miss=0
for d in skills/*/; do
  n=$(basename "$d")
  grep -qF "skills/$n/" docs/files.md || { echo "docs/files.md does not list skills/$n"; miss=1; }
done
for p in $(grep -oE 'skills/[A-Za-z0-9_-]+/SKILL\.md' docs/files.md | sort -u); do
  [ -f "$p" ] || { echo "docs/files.md lists $p, which does not exist"; miss=1; }
done
# The skills table in docs/skills-and-hooks.md: one row per shipped skill, which
# is the page a reader checks to learn what a skill does (a new skill was once
# listed in docs/files.md only).
for d in skills/*/; do
  n=$(basename "$d")
  grep -qF "| \`$n\` |" docs/skills-and-hooks.md || { echo "docs/skills-and-hooks.md does not list skills/$n"; miss=1; }
done
# A file shipped beside a SKILL.md (a skill's own script, e.g. facts.sh) ships
# with `init --install-skills`, so docs/files.md must name it too.
for f in skills/*/*; do
  [ -f "$f" ] || continue
  [ "$(basename "$f")" = SKILL.md ] && continue
  grep -qF "\`$f\`" docs/files.md || { echo "docs/files.md does not list $f"; miss=1; }
done
for d in .claude/skills/*/; do
  n=$(basename "$d")
  grep -qF "| \`$n\` |" docs/development-workflow.md || { echo "docs/development-workflow.md does not list .claude/skills/$n"; miss=1; }
done
for n in $(grep -oE '^\| `cr-gh-[A-Za-z0-9_-]+`' docs/development-workflow.md | grep -oE 'cr-gh-[A-Za-z0-9_-]+' | sort -u); do
  [ -d ".claude/skills/$n" ] || { echo "docs/development-workflow.md lists $n, which is not in .claude/skills"; miss=1; }
done
[ "$miss" = 0 ] && echo "ok: skills/ matches docs/files.md and docs/skills-and-hooks.md, .claude/skills/ matches docs/development-workflow.md"

section "EVALUATION.md status lines"
grep -nE '^## Track|^\*\*Status|^\*\*The real run|was NOT made' EVALUATION.md | cut -c1-120

section "docs/ (not read by default)"
# The title is each page's first "# " heading: the README's table of contents lists
# pages by name only, so this is what tells a reader what a page covers.
for f in docs/*.md; do
  title=$(grep -m 1 '^# ' "$f" | sed 's/^# //' | cut -c1-80)
  printf '%s (%s lines): %s\n' "$f" "$(wc -l < "$f" | tr -d ' ')" "${title:-no title}"
done
