# Every fact /cr-gh-load-context checks, gathered in one read-only call. This is the
# PowerShell twin of facts.sh and prints the same sections in the same words; it is
# the default on Windows, where bash cannot be assumed.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File .claude\skills\cr-gh-load-context\facts.ps1 --porcelain
#        (pwsh works the same way)
#
# `--porcelain` changes nothing here; it is on the command line so that
# claude-runway's compress hook treats the output as exactness-critical and never
# summarizes it. Output is still kept short: verdicts and line numbers, not file
# contents.
#
# Needs git and an authenticated gh. Works in Windows PowerShell 5.1 and PowerShell
# 7 (no ternary, `??` or `&&`); the file is ASCII only because 5.1 reads a BOM-less
# file as the ANSI codepage.
param([Parameter(ValueFromRemainingArguments = $true)] $Ignored)

$ErrorActionPreference = 'Continue'
# gh prints UTF-8; without this a non-ASCII title is decoded with the Windows locale
# codepage (cp1252) and corrupted.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$OutputEncoding = [System.Text.Encoding]::UTF8

$top = (& git rev-parse --show-toplevel 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $top) { Write-Output 'not a git repository'; exit 1 }
Set-Location -LiteralPath $top

# Point every gh call at origin's repo. Without this, gh picks the repo itself, and in a
# checkout that also has an `upstream` remote (a fork) it can pick the wrong one, so every
# issue and PR figure below would come from another repo. Handles https:// and ssh
# (git@host:owner/repo) remote URLs.
$originSlug = ((& git remote get-url origin 2>$null) | Out-String).Trim()
$originSlug = $originSlug -replace '^(git@[^:]+:|ssh://[^/]+/|https?://[^/]+/)', '' -replace '\.git$', ''
if ($originSlug -match '/') { $env:GH_REPO = $originSlug } else { $originSlug = '' }

function Write-Section([string]$Name) { Write-Output ''; Write-Output "## $Name" }

function Get-Cut([string]$Text, [int]$N) {
    if ($null -eq $Text) { return '' }
    if ($Text.Length -gt $N) { return $Text.Substring(0, $N) }
    return $Text
}

# Runs gh and returns Ok plus the parsed JSON items (an array, also for "[]" and for a
# single object). Windows PowerShell 5.1 returns a JSON array as one object, so it is piped
# through ForEach-Object to unroll it.
function Invoke-GhJson([string[]]$GhArgs) {
    try {
        $out = & gh @GhArgs 2>$null
        if ($LASTEXITCODE -ne 0) { return [pscustomobject]@{ Ok = $false; Items = @() } }
        $text = ($out | Out-String).Trim()
        if (-not $text) { return [pscustomobject]@{ Ok = $true; Items = @() } }
        $items = @($text | ConvertFrom-Json | ForEach-Object { $_ } | Where-Object { $null -ne $_ })
        return [pscustomobject]@{ Ok = $true; Items = $items }
    } catch {
        return [pscustomobject]@{ Ok = $false; Items = @() }
    }
}

function Get-PlainFiles([string]$Dir) {
    return @(Get-ChildItem -LiteralPath $Dir -File -ErrorAction SilentlyContinue |
        Where-Object { -not $_.Name.StartsWith('.') } | Sort-Object Name)
}

function Get-FileLines([string]$Path) { return @(Get-Content -LiteralPath $Path -Encoding UTF8) }

function Get-Short([string]$Ref) {
    $r = ((& git rev-parse --short $Ref 2>$null) | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { return '' }
    return $r
}

# A .plans/ artifact name: the whole name is {digits}-{plan|research|validate}.md.
$TriadRe = '^[0-9]+-(plan|research|validate)\.md$'

Write-Section 'git'
Write-Output "branch: $(((& git branch --show-current 2>$null) | Out-String).Trim())"
$mainSha = Get-Short 'main'
$originMain = Get-Short 'origin/main'
if (-not $originMain) { $originMain = '?' }
Write-Output "main: $mainSha origin/main: $originMain"
$beyond = ((& git rev-list --count main..HEAD 2>$null) | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $beyond) { $beyond = '?' }
Write-Output "commits on this branch beyond main: $beyond"
Write-Output "uncommitted paths: $(@(& git status --porcelain 2>$null | Where-Object { $_ }).Count)"

Write-Section 'github'
if ($originSlug) {
    Write-Output "repo: $originSlug (from origin)"
} else {
    Write-Output 'repo: UNKNOWN (origin has no owner/repo URL); gh will choose one, so treat the numbers below with care'
}
$issues = Invoke-GhJson @('issue', 'list', '--state', 'open', '--limit', '300', '--json', 'labels')
if ($issues.Ok) {
    $keys = foreach ($i in $issues.Items) {
        $p = @($i.labels | ForEach-Object { $_.name } | Where-Object { $_ -like 'priority-*' })
        if ($p.Count -gt 0) { $p[0] } else { 'no-priority' }
    }
    $groups = @($keys | Group-Object | Sort-Object { $_.Name } -CaseSensitive)
    $parts = foreach ($g in $groups) { "$($g.Name -replace '^priority-', '') $($g.Count)" }
    Write-Output "open issues: $($issues.Items.Count) ($($parts -join ', '))"
} else {
    Write-Output 'gh issue list failed'
}
Write-Output 'assigned to you:'
$mine = Invoke-GhJson @('issue', 'list', '--assignee', '@me', '--state', 'open', '--limit', '300', '--json', 'number,title')
if ($mine.Ok) { foreach ($i in $mine.Items) { Write-Output "  #$($i.number) $(Get-Cut $i.title 90)" } } else { Write-Output '  gh failed' }
Write-Output 'open PRs:'
$prs = Invoke-GhJson @('pr', 'list', '--state', 'open', '--limit', '300', '--json', 'number,title,headRefName')
if ($prs.Ok) { foreach ($p in $prs.Items) { Write-Output "  #$($p.number) [$($p.headRefName)] $(Get-Cut $p.title 80)" } } else { Write-Output '  gh failed' }

Write-Section 'local-only files (gitignored)'
foreach ($p in @('CLAUDE.md', '.plans', '.ai-plans')) {
    if (Test-Path -LiteralPath $p) { Write-Output "${p}: present" } else { Write-Output "${p}: ABSENT" }
}

if (Test-Path -LiteralPath '.plans' -PathType Container) {
    Write-Section '.plans'
    $plans = @(Get-PlainFiles '.plans' | ForEach-Object { $_.Name })
    if ($plans.Count -eq 0) { Write-Output '(empty)' }
    foreach ($n in ($plans | Where-Object { $_ -cnotmatch $TriadRe })) { Write-Output "MISPLACED (not part of a triad): $n" }
    $numbers = @($plans | Where-Object { $_ -cmatch $TriadRe } | ForEach-Object { [int64]($_ -replace '-.*$', '') } | Sort-Object -Unique)
    foreach ($n in $numbers) {
        $have = ''
        foreach ($k in @('research', 'plan', 'validate')) { if (Test-Path -LiteralPath ".plans/$n-$k.md" -PathType Leaf) { $have += " $k" } }
        if (-not $have) { $have = ' nothing' }
        $state = 'lookup failed'
        $view = Invoke-GhJson @('issue', 'view', "$n", '--json', 'state,closedByPullRequestsReferences')
        if ($view.Ok -and $view.Items.Count -gt 0) {
            $v = $view.Items[0]
            $closing = @($v.closedByPullRequestsReferences | Where-Object { $null -ne $_ } | ForEach-Object { "#$($_.number)" })
            $c = if ($closing.Count -gt 0) { $closing -join ',' } else { 'none' }
            $state = "$($v.state), closing PRs: $c"
        }
        Write-Output "#${n}: has$have; issue $state"
        if (Test-Path -LiteralPath ".plans/$n-plan.md" -PathType Leaf) {
            $m = Select-String -LiteralPath ".plans/$n-plan.md" -Pattern '^## Open questions' -Encoding UTF8 | Select-Object -First 1
            if ($m) { Write-Output "  $n-plan.md: open questions at line $($m.LineNumber)" }
        }
    }
}

if (Test-Path -LiteralPath '.ai-plans' -PathType Container) {
    Write-Section '.ai-plans'
    foreach ($f in (Get-PlainFiles '.ai-plans')) {
        $lines = Get-FileLines $f.FullName
        $first = ''
        if ($lines.Count -gt 0) { $first = Get-Cut $lines[0] 90 }
        Write-Output "$($f.Name) ($($lines.Count) lines): $first"
    }
    $wo = '.ai-plans/ticket-work-order.md'
    if (Test-Path -LiteralPath $wo -PathType Leaf) {
        $lines = Get-FileLines $wo
        $total = $lines.Count
        # 1-based line number of the first line matching a pattern, or $null.
        function Find-Line([string]$Re) {
            for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -cmatch $Re) { return ($i + 1) } }
            return $null
        }
        $earlier = Find-Line '^\*\*Earlier snapshot'
        $left = Find-Line "^## 1\. What's left"
        $doneH = Find-Line '^## 2\. Done'
        $sections = Find-Line '^Sections:'
        # The latest snapshot ends before the first earlier one; with none, before the
        # "Sections:" line or the first section heading; with neither, at the end.
        $endAt = $null
        foreach ($c in @($earlier, $sections, $left)) { if ($null -ne $c) { $endAt = $c; break } }
        if ($null -ne $endAt) { $latestEnd = $endAt - 1 } else { $latestEnd = $total }
        if ($null -ne $left) {
            if ($null -ne $doneH) { $leftEnd = $doneH - 1 } else { $leftEnd = $total }
            $leftRange = "lines $left-$leftEnd"
        } else {
            $leftRange = 'not found'
        }
        Write-Output "work order: latest snapshot = lines 1-$latestEnd; '1. What's left' = $leftRange"
        $snap = ($lines | Select-Object -First $latestEnd) -join "`n"
        $woCount = ''
        $m = [regex]::Match($snap, '([0-9]+) open issues')
        if ($m.Success) { $woCount = $m.Groups[1].Value }
        $woSha = ''
        $m = [regex]::Match($snap, '`main` is (still )?at `([0-9a-f]+)`')
        if ($m.Success) { $woSha = $m.Groups[2].Value }
        $liveCount = ''
        $lc = Invoke-GhJson @('issue', 'list', '--state', 'open', '--limit', '300', '--json', 'number')
        if ($lc.Ok) { $liveCount = "$($lc.Items.Count)" }
        $liveSha = Get-Short 'main'
        # Compare only when both sides are known; a missing side is UNKNOWN, not STALE or ok.
        if (-not $woCount -or -not $liveCount) { $cv = 'UNKNOWN' } elseif ($woCount -eq $liveCount) { $cv = 'ok' } else { $cv = 'STALE' }
        if (-not $woSha -or -not $liveSha) {
            $sv = 'UNKNOWN'
        } elseif ($woSha.StartsWith($liveSha) -or $liveSha.StartsWith($woSha)) {
            # Abbreviations can differ in length: one must be a prefix of the other.
            $sv = 'ok'
        } else {
            $sv = 'STALE'
        }
        $shownCount = if ($woCount) { $woCount } else { '?' }
        $shownSha = if ($woSha) { $woSha } else { '?' }
        $shownLiveCount = if ($liveCount) { $liveCount } else { '?' }
        $shownLiveSha = if ($liveSha) { $liveSha } else { '?' }
        Write-Output "work order says $shownCount open, main $shownSha; live $shownLiveCount open, main $shownLiveSha -> count $cv, sha $sv"
    }
}

Write-Section 'skill inventories'
function Read-Doc([string]$Path) {
    if (Test-Path -LiteralPath $Path -PathType Leaf) { return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8) }
    return ''
}
$filesDoc = Read-Doc 'docs/files.md'
$hooksDoc = Read-Doc 'docs/skills-and-hooks.md'
$workflowDoc = Read-Doc 'docs/development-workflow.md'
$miss = $false
$skillDirs = @(Get-ChildItem -LiteralPath 'skills' -Directory -ErrorAction SilentlyContinue | Sort-Object Name)
foreach ($d in $skillDirs) {
    if (-not $filesDoc.Contains("skills/$($d.Name)/")) { Write-Output "docs/files.md does not list skills/$($d.Name)"; $miss = $true }
}
$listed = @([regex]::Matches($filesDoc, 'skills/[A-Za-z0-9_-]+/SKILL\.md') | ForEach-Object { $_.Value } | Sort-Object -Unique)
foreach ($p in $listed) {
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { Write-Output "docs/files.md lists $p, which does not exist"; $miss = $true }
}
# The skills table in docs/skills-and-hooks.md: one row per shipped skill, which is the page
# a reader checks to learn what a skill does.
foreach ($d in $skillDirs) {
    if (-not $hooksDoc.Contains("| ``$($d.Name)`` |")) { Write-Output "docs/skills-and-hooks.md does not list skills/$($d.Name)"; $miss = $true }
}
# A file shipped beside a SKILL.md (a skill's own script, e.g. facts.sh or facts.ps1) ships
# with `init --install-skills`, so docs/files.md must name it too.
foreach ($d in $skillDirs) {
    foreach ($f in @(Get-ChildItem -LiteralPath $d.FullName -File | Sort-Object Name)) {
        if ($f.Name -eq 'SKILL.md') { continue }
        $rel = "skills/$($d.Name)/$($f.Name)"
        if (-not $filesDoc.Contains("``$rel``")) { Write-Output "docs/files.md does not list $rel"; $miss = $true }
    }
}
$devDirs = @(Get-ChildItem -LiteralPath '.claude/skills' -Directory -ErrorAction SilentlyContinue | Sort-Object Name)
foreach ($d in $devDirs) {
    if (-not $workflowDoc.Contains("| ``$($d.Name)`` |")) { Write-Output "docs/development-workflow.md does not list .claude/skills/$($d.Name)"; $miss = $true }
}
$wfNames = @([regex]::Matches($workflowDoc, '(?m)^\| `(cr-gh-[A-Za-z0-9_-]+)`') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
foreach ($n in $wfNames) {
    if (-not (Test-Path -LiteralPath ".claude/skills/$n" -PathType Container)) { Write-Output "docs/development-workflow.md lists $n, which is not in .claude/skills"; $miss = $true }
}
if (-not $miss) { Write-Output 'ok: skills/ matches docs/files.md and docs/skills-and-hooks.md, .claude/skills/ matches docs/development-workflow.md' }

Write-Section 'EVALUATION.md status lines'
if (Test-Path -LiteralPath 'EVALUATION.md' -PathType Leaf) {
    $ev = Get-FileLines 'EVALUATION.md'
    for ($i = 0; $i -lt $ev.Count; $i++) {
        if ($ev[$i] -cmatch '^## Track|^\*\*Status|^\*\*The real run|was NOT made') { Write-Output (Get-Cut "$($i + 1):$($ev[$i])" 120) }
    }
}

Write-Section 'docs/ (not read by default)'
# The title is each page's first "# " heading: the README's table of contents lists pages by
# name only, so this is what tells a reader what a page covers.
foreach ($f in @(Get-ChildItem -LiteralPath 'docs' -File -Filter '*.md' -ErrorAction SilentlyContinue | Sort-Object Name)) {
    $lines = Get-FileLines $f.FullName
    $h = $lines | Where-Object { $_ -cmatch '^# ' } | Select-Object -First 1
    $title = ''
    if ($h) { $title = Get-Cut ($h -replace '^# ', '') 80 }
    if (-not $title) { $title = 'no title' }
    Write-Output "docs/$($f.Name) ($($lines.Count) lines): $title"
}
exit 0
