#!/usr/bin/env bash
# The facts /cr-load-context needs from the project and its forge, in one
# read-only call.
#
# Usage (from anywhere inside the project):
#   bash "<this skill's folder>/facts.sh" --porcelain
#
# The output is exact and meant to be read as it is. `--porcelain` changes
# nothing here; it is on the command line so that claude-runway's compress hook
# (hooks/compress_output.py) treats the output as exactness-critical and never
# summarizes it: Bash output over CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS is
# otherwise sent to LM Studio, which is slow and paraphrases. The output is
# kept short on purpose (verdicts and one line per item, never file contents).
#
# Core versus optional. git is the only thing this script needs. gh and az are
# optional extras: when one is missing, not signed in, slow or failing, only
# the forge section says "forge facts skipped: <reason>" and every other
# section is unaffected. The script always exits 0.
#
# What it never does: read .mcp.json (an inline QDRANT_API_KEY would reach the
# model context), print a credential (userinfo is stripped from every remote
# URL it shows), or write a file.
#
# Portable bash: macOS's /bin/bash is 3.2 (no associative arrays, mapfile,
# readarray or ${var,,}; BSD sed/grep/find/date), and Git Bash on Windows and
# Linux bash 4/5 must run it too. Only POSIX flags of sed, grep and find are used.
set -u

# Unknown arguments, `--porcelain` included, are ignored on purpose.

# --- helpers ---------------------------------------------------------------

section() { printf '\n## %s\n' "$1"; }

# Runs a command with a time limit. A caller treats status 124 (timeout(1)) and
# 142 (our perl fallback) as "timed out".
#
# Recent macOS ships /usr/bin/timeout and Linux has it; older macOS has neither
# timeout nor gtimeout, and then perl (on macOS and Git for Windows) does the
# job. The perl fallback must NOT be `alarm` + `exec`: that kills only the
# process it became, and Homebrew's az is a bash wrapper that runs python as a
# child without exec, so the alarm would kill the wrapper while python kept
# running and held the output pipe open (the call was never bounded). So the
# fallback forks, puts the command in its own process group, and on the alarm
# kills the whole group. Without perl, run with no limit.
run_limited() {
  _secs=$1; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$_secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$_secs" "$@"
  elif command -v perl >/dev/null 2>&1; then
    perl -e '
      my $secs = shift;
      my $pid = fork();
      exit 127 unless defined $pid;
      if ($pid == 0) { setpgrp(0, 0); exec @ARGV or exit 127; }
      $SIG{ALRM} = sub { kill "KILL", -$pid; kill "KILL", $pid; exit 142; };
      alarm $secs;
      waitpid($pid, 0);
      alarm 0;
      my $st = $?;
      exit(($st & 127) ? 128 + ($st & 127) : ($st >> 8));
    ' "$_secs" "$@"
  else
    "$@"
  fi
}

timed_out() { [ "$1" = 124 ] || [ "$1" = 142 ]; }

# Text cut to N characters, with "..." when something was cut, so a shortened
# title or message is never mistaken for the whole of it.
clip() {
  _text=$1; _n=$2
  if [ "${#_text}" -gt "$_n" ]; then
    printf '%s...' "$(printf '%s' "$_text" | cut -c1-"$_n")"
  else
    printf '%s' "$_text"
  fi
}

# The first line of a command's error output, without the noise.
first_line() { clip "$(printf '%s\n' "$1" | sed -n '/[^[:space:]]/{p;q;}')" 160; }

tab=$(printf '\t')

# --- where we are ----------------------------------------------------------

if top=$(git rev-parse --show-toplevel 2>/dev/null); then
  cd "$top" || exit 0
  in_git=1
else
  in_git=0
fi

section "git"
if [ "$in_git" = 0 ]; then
  echo "not a git repository: $(pwd)"
  branch=""
  remote=""
else
  branch=$(git branch --show-current 2>/dev/null)
  if [ -n "$branch" ]; then
    echo "branch: $branch"
  else
    echo "branch: (detached HEAD at $(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
  fi
  echo "head: $(clip "$(git log -1 --pretty='%h %s' 2>/dev/null)" 100)"
  upstream=$(git rev-parse --abbrev-ref '@{upstream}' 2>/dev/null)
  if [ -n "$upstream" ]; then
    # "<behind>\t<ahead>" for upstream...HEAD.
    counts=$(git rev-list --left-right --count "$upstream...HEAD" 2>/dev/null)
    behind=$(printf '%s' "$counts" | cut -f1)
    ahead=$(printf '%s' "$counts" | cut -f2)
    echo "upstream: $upstream (ahead ${ahead:-?}, behind ${behind:-?})"
  else
    echo "upstream: none"
  fi
  echo "uncommitted paths: $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  # Userinfo (user:token@) is removed so a credentialed remote never reaches
  # the output. It runs to the LAST @ before the first /, so a password that
  # itself contains an @ is removed whole. scp-style remotes (git@host:path)
  # carry no secret and are kept.
  remote=$(git remote get-url origin 2>/dev/null | sed -E 's#^([A-Za-z][A-Za-z0-9+.-]*://)[^/]*@#\1#')
  echo "origin: ${remote:-none}"
  echo "recent commits:"
  git log -5 --pretty='%h %s' 2>/dev/null | while IFS= read -r line; do
    echo "  $(clip "$line" 100)"
  done
fi

# --- the project's own docs ------------------------------------------------

section "entry docs"
readmes=$(find . -maxdepth 1 -type f -iname 'readme*' 2>/dev/null | sed 's#^\./##' | sort)
if [ -n "$readmes" ]; then
  printf '%s\n' "$readmes" | while IFS= read -r f; do
    echo "README: $f ($(wc -l < "$f" | tr -d ' ') lines)"
  done
else
  echo "README: none"
fi
if [ -f CLAUDE.md ]; then
  echo "CLAUDE.md: present ($(wc -l < CLAUDE.md | tr -d ' ') lines)"
else
  echo "CLAUDE.md: absent"
fi
if [ -d docs ]; then
  # Each page's first "# " heading says what it covers, so the report can pick
  # what to read without opening every page. Capped so a docs-heavy repo stays short.
  total=$(find docs -maxdepth 1 -type f -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
  echo "docs/ ($total markdown files):"
  find docs -maxdepth 1 -type f -name '*.md' 2>/dev/null | sort | head -n 40 | while IFS= read -r f; do
    title=$(clip "$(grep -m 1 '^# ' "$f" 2>/dev/null | sed 's/^# //')" 80)
    printf '  %s (%s lines): %s\n' "$f" "$(wc -l < "$f" | tr -d ' ')" "${title:-no title}"
  done
  [ "$total" -gt 40 ] && echo "  ... $((total - 40)) more not listed"
else
  echo "docs/: none"
fi

# --- the index -------------------------------------------------------------

section "index"
# The skill starts sync_repo only when this manifest exists (a project with no
# manifest would get a full rebuild, a long embed that is index_repo's to start
# and the user's to ask for), so ABSENT stays true while this script runs.
if [ -f .qdrant_index_manifest.json ]; then
  echo "sync manifest: present (.qdrant_index_manifest.json)"
else
  echo "sync manifest: ABSENT (this project has not been indexed here; the first embed is long, so sync_repo was not started)"
fi

# --- the forge: pull requests only -----------------------------------------

section "forge"
forge_note=""
forge=none
host=""
org=""; project=""; repo=""; slug=""

if [ -z "$remote" ]; then
  forge_note="no origin remote"
else
  # Split the remote into host and path. Forms handled:
  #   https://github.com/owner/repo.git            git@github.com:owner/repo.git
  #   https://contoso.visualstudio.com/DefaultCollection/<project>/_git/<repo>
  #   https://dev.azure.com/<org>/<project>/_git/<repo>
  #   git@ssh.dev.azure.com:v3/<org>/<project>/<repo>
  case "$remote" in
    *://*)
      rest=${remote#*://}
      host=${rest%%/*}
      path=${rest#*/}
      ;;
    *:*)
      u=${remote#*@}
      host=${u%%:*}
      path=${u#*:}
      ;;
    *)
      host=""; path=""
      ;;
  esac
  host=${host%%:*}          # drop a :port
  path=${path%/}
  case "$host" in
    github.com)
      forge=github
      slug=$(printf '%s' "$path" | sed -E 's#\.git$##')
      ;;
    dev.azure.com|ssh.dev.azure.com|*.visualstudio.com|vs-ssh.visualstudio.com)
      forge=ado
      case "$path" in
        v3/*)                       # ssh form: v3/ORG/PROJECT/REPO
          p=${path#v3/}
          org=${p%%/*}; p=${p#*/}
          project=${p%%/*}; repo=${p#*/}
          ;;
        *)
          case "$host" in
            dev.azure.com) org=${path%%/*}; path=${path#*/} ;;
            *)             org=${host%%.*} ;;
          esac
          path=${path#DefaultCollection/}
          case "$path" in
            _git/*)    repo=${path#_git/}; project=$repo ;;   # no project segment
            */_git/*)  project=${path%%/_git/*}; repo=${path##*/_git/} ;;
            *)         project=""; repo="" ;;
          esac
          ;;
      esac
      # Only %20 is decoded: it is the one escape a project or repo name really carries.
      project=$(printf '%s' "$project" | sed 's/%20/ /g')
      repo=$(printf '%s' "$repo" | sed 's/%20/ /g')
      ;;
    *)
      forge_note="origin host ${host:-unknown} is neither GitHub nor Azure DevOps"
      ;;
  esac
fi

if [ "$forge" = github ]; then
  echo "forge: GitHub ($slug)"
  if ! command -v gh >/dev/null 2>&1; then
    forge_note="gh is not installed"
  elif ! run_limited 20 gh auth status --hostname github.com >/dev/null 2>&1; then
    forge_note="gh is not signed in to github.com (run: gh auth login)"
  else
    out=$(run_limited 25 gh pr list -R "$slug" --state open --limit 30 --json number,title,headRefName,isDraft,isCrossRepository \
      --jq '.[] | "\(.number)\t\(.headRefName)\t\(if .isDraft then "draft" else "open" end)\t\(.isCrossRepository)\t\(.title)"' 2>&1)
    rc=$?
    if timed_out "$rc"; then
      forge_note="gh pr list timed out"
    elif [ "$rc" != 0 ]; then
      forge_note="gh pr list failed: $(first_line "$out")"
    else
      # Only lines that start with a PR number are rows: a stderr line that came
      # with a successful call (2>&1 merges the streams) is not a pull request.
      rows=$(printf '%s\n' "$out" | grep -E "^[0-9]+${tab}" || true)
      n=$(printf '%s' "$rows" | grep -c . || true)
      echo "open pull requests: $n"
      [ "$n" -ge 30 ] && echo "  (capped at 30)"
      printf '%s\n' "$rows" | while IFS="$tab" read -r num head state cross title; do
        [ -z "$num" ] && continue
        mark=""
        # A fork's PR from a branch with the same name (its own main) is not this branch's.
        [ -n "$branch" ] && [ "$head" = "$branch" ] && [ "$cross" != true ] && mark="  <- current branch"
        printf '  #%s [%s] (%s) %s%s\n' "$num" "$head" "$state" "$(clip "$title" 90)" "$mark"
      done
    fi
  fi
elif [ "$forge" = ado ]; then
  echo "forge: Azure DevOps (org $org, project $project, repo $repo)"
  if [ -z "$org" ] || [ -z "$project" ] || [ -z "$repo" ]; then
    forge_note="could not read the organization, project and repository from the origin URL"
  elif ! command -v az >/dev/null 2>&1; then
    forge_note="az is not installed"
  else
    org_url="https://dev.azure.com/$org"
    # --only-show-errors drops az's upgrade and preview warnings, which would
    # otherwise be mixed into the output (stderr is merged so errors can be read).
    if ! run_limited 40 az account show --only-show-errors -o none >/dev/null 2>&1; then
      forge_note="az is not signed in (run: az login)"
    else
      listing=$(run_limited 40 az repos list --org "$org_url" --project "$project" \
        --query '[].[name,isDisabled]' -o tsv --only-show-errors 2>&1)
      rc=$?
      if timed_out "$rc"; then
        forge_note="az repos list timed out"
      elif [ "$rc" != 0 ]; then
        case "$listing" in
          *"not recognized"*|*"misspelled"*)
            forge_note="the azure-devops extension is not installed (run: az extension add --name azure-devops)" ;;
          *) forge_note="az repos list failed: $(first_line "$listing")" ;;
        esac
      else
        disabled=$(printf '%s\n' "$listing" | awk -F '\t' -v r="$repo" 'tolower($1) == tolower(r) { print tolower($2); exit }')
        if [ "$disabled" = true ]; then
          forge_note="repo $repo is disabled (isDisabled: true), so its pull requests are unavailable"
        else
          out=$(run_limited 40 az repos pr list --org "$org_url" --project "$project" --repository "$repo" \
            --status active --query '[].[pullRequestId,sourceRefName,isDraft,title]' -o tsv --only-show-errors 2>&1)
          rc=$?
          if timed_out "$rc"; then
            forge_note="az repos pr list timed out"
          elif [ "$rc" != 0 ]; then
            case "$out" in
              *TF401019*) forge_note="repo $repo is disabled, removed, or you have no access to it (TF401019)" ;;
              *) forge_note="az repos pr list failed: $(first_line "$out")" ;;
            esac
          else
            rows=$(printf '%s\n' "$out" | grep -E "^[0-9]+${tab}" || true)
            n=$(printf '%s' "$rows" | grep -c . || true)
            echo "active pull requests: $n"
            printf '%s\n' "$rows" | while IFS="$tab" read -r num ref draft title; do
              [ -z "$num" ] && continue
              head=${ref#refs/heads/}
              state=active
              [ "$draft" = true ] || [ "$draft" = True ] && state=draft
              mark=""
              [ -n "$branch" ] && [ "$head" = "$branch" ] && mark="  <- current branch"
              printf '  #%s [%s] (%s) %s%s\n' "$num" "$head" "$state" "$(clip "$title" 90)" "$mark"
            done
          fi
        fi
      fi
    fi
  fi
fi

if [ -n "$forge_note" ]; then
  echo "forge facts skipped: $forge_note"
else
  echo "scope: pull requests only"
fi

exit 0
