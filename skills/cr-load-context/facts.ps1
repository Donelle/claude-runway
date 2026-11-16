# The facts /cr-load-context needs from the project and its forge, in one
# read-only call. This is the PowerShell twin of facts.sh and prints the same
# sections in the same words, so the skill's report does not depend on which
# one ran. It is the default on Windows, where bash cannot be assumed.
#
# Usage (from anywhere inside the project):
#   powershell -NoProfile -ExecutionPolicy Bypass -File "<this skill's folder>\facts.ps1" --porcelain
#   (pwsh works the same way)
#
# `--porcelain` changes nothing here; it is on the command line so that
# claude-runway's compress hook treats the output as exactness-critical and
# never summarizes it. Unknown arguments are ignored on purpose.
#
# Core versus optional. git is the only thing this script needs. gh and az are
# optional extras: when one is missing, not signed in, slow or failing, only the
# forge section says "forge facts skipped: <reason>". The script always exits 0.
#
# What it never does: read .mcp.json (an inline QDRANT_API_KEY would reach the
# model context), print a credential (userinfo is stripped from every remote URL
# it shows), or write a file.
#
# Works in Windows PowerShell 5.1 and PowerShell 7: no ternary, `??` or `&&`, and
# the file is ASCII only (5.1 reads a BOM-less file as the ANSI codepage).
param([Parameter(ValueFromRemainingArguments = $true)] $Ignored)

$ErrorActionPreference = 'Continue'
# gh and az print UTF-8; without this a non-ASCII title is decoded with the
# Windows locale codepage (cp1252) and corrupted, a bug class this repo has had.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$OutputEncoding = [System.Text.Encoding]::UTF8

# --- helpers ---------------------------------------------------------------

function Write-Section([string]$Name) { Write-Output ''; Write-Output "## $Name" }

# Text cut to N characters, with "..." when something was cut, so a shortened
# title or message is never mistaken for the whole of it.
function Get-Clip([string]$Text, [int]$N) {
    if ($null -eq $Text) { return '' }
    if ($Text.Length -gt $N) { return $Text.Substring(0, $N) + '...' }
    return $Text
}

# The first non-blank line of a command's error output, clipped.
function Get-FirstLine([string]$Text) {
    foreach ($l in ($Text -split "`r?`n")) {
        if ($l.Trim()) { return (Get-Clip $l.Trim() 160) }
    }
    return ''
}

# Runs an external command with a time limit, in a job so a hung az cannot hang
# the script. Returns Out (stdout and stderr merged, as one string) and Rc;
# Rc 124 means timed out, 127 means the command could not be started.
function Invoke-Limited([int]$Secs, [string]$Exe, [string[]]$CmdArgs) {
    $job = Start-Job -ScriptBlock {
        param($e, $a)
        try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
        try {
            $o = & $e @a 2>&1 | ForEach-Object { "$_" }
            $rc = $LASTEXITCODE
            if ($null -eq $rc) { $rc = 0 }
            [pscustomobject]@{ Out = ($o -join "`n"); Rc = $rc }
        } catch {
            [pscustomobject]@{ Out = "$_"; Rc = 127 }
        }
    } -ArgumentList @($Exe, , $CmdArgs)
    if (-not (Wait-Job $job -Timeout $Secs)) {
        Stop-Job $job
        Remove-Job $job -Force
        return [pscustomobject]@{ Out = ''; Rc = 124 }
    }
    $res = Receive-Job $job
    Remove-Job $job -Force
    if ($null -eq $res) { return [pscustomobject]@{ Out = ''; Rc = 127 } }
    return $res
}

function Test-TimedOut($Res) { return ($Res.Rc -eq 124) }

function Test-Command([string]$Name) { return [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

# --- where we are ----------------------------------------------------------

$inGit = $false
$top = $null
if (Test-Command git) {
    $top = (& git rev-parse --show-toplevel 2>$null)
    if ($LASTEXITCODE -eq 0 -and $top) { $inGit = $true }
}
if ($inGit) { Set-Location -LiteralPath $top }

$branch = ''
$remote = ''
Write-Section 'git'
if (-not $inGit) {
    Write-Output "not a git repository: $((Get-Location).Path)"
} else {
    $branch = ((& git branch --show-current 2>$null) | Out-String).Trim()
    if ($branch) {
        Write-Output "branch: $branch"
    } else {
        $short = ((& git rev-parse --short HEAD 2>$null) | Out-String).Trim()
        if (-not $short) { $short = '?' }
        Write-Output "branch: (detached HEAD at $short)"
    }
    $head = ((& git log -1 --pretty='%h %s' 2>$null) | Out-String).Trim()
    Write-Output "head: $(Get-Clip $head 100)"
    $upstream = ((& git rev-parse --abbrev-ref '@{upstream}' 2>$null) | Out-String).Trim()
    if ($LASTEXITCODE -eq 0 -and $upstream) {
        # "<behind>\t<ahead>" for upstream...HEAD.
        $counts = ((& git rev-list --left-right --count "$upstream...HEAD" 2>$null) | Out-String).Trim() -split '\s+'
        $behind = '?'; $ahead = '?'
        if ($counts.Count -ge 2) { $behind = $counts[0]; $ahead = $counts[1] }
        Write-Output "upstream: $upstream (ahead $ahead, behind $behind)"
    } else {
        Write-Output 'upstream: none'
    }
    $porcelain = @(& git status --porcelain 2>$null | Where-Object { $_ })
    Write-Output "uncommitted paths: $($porcelain.Count)"
    # Userinfo (user:token@) is removed so a credentialed remote never reaches the
    # output. It runs to the LAST @ before the first /, so a password that itself
    # contains an @ is removed whole. scp-style remotes (git@host:path) carry no
    # secret and are kept.
    $rawRemote = ((& git remote get-url origin 2>$null) | Out-String).Trim()
    $remote = $rawRemote -replace '^([A-Za-z][A-Za-z0-9+.-]*://)[^/]*@', '$1'
    $rawRemote = $null
    if ($remote) { Write-Output "origin: $remote" } else { Write-Output 'origin: none' }
    Write-Output 'recent commits:'
    foreach ($line in @(& git log -5 --pretty='%h %s' 2>$null)) {
        Write-Output "  $(Get-Clip $line 100)"
    }
}

# --- the project's own docs ------------------------------------------------

function Get-LineCount([string]$Path) { return @(Get-Content -LiteralPath $Path -Encoding UTF8).Count }

Write-Section 'entry docs'
$readmes = @(Get-ChildItem -File -Filter 'readme*' -ErrorAction SilentlyContinue | Sort-Object Name)
if ($readmes.Count -gt 0) {
    foreach ($f in $readmes) { Write-Output "README: $($f.Name) ($(Get-LineCount $f.FullName) lines)" }
} else {
    Write-Output 'README: none'
}
if (Test-Path -LiteralPath 'CLAUDE.md' -PathType Leaf) {
    Write-Output "CLAUDE.md: present ($(Get-LineCount 'CLAUDE.md') lines)"
} else {
    Write-Output 'CLAUDE.md: absent'
}
if (Test-Path -LiteralPath 'docs' -PathType Container) {
    # Each page's first "# " heading says what it covers, so the report can pick
    # what to read without opening every page. Capped so a docs-heavy repo stays short.
    $pages = @(Get-ChildItem -LiteralPath 'docs' -File -Filter '*.md' -ErrorAction SilentlyContinue | Sort-Object Name)
    Write-Output "docs/ ($($pages.Count) markdown files):"
    foreach ($f in ($pages | Select-Object -First 40)) {
        $h = Get-Content -LiteralPath $f.FullName -Encoding UTF8 | Where-Object { $_ -match '^# ' } | Select-Object -First 1
        $title = ''
        if ($h) { $title = Get-Clip ($h -replace '^# ', '') 80 }
        if (-not $title) { $title = 'no title' }
        Write-Output "  docs/$($f.Name) ($(Get-LineCount $f.FullName) lines): $title"
    }
    if ($pages.Count -gt 40) { Write-Output "  ... $($pages.Count - 40) more not listed" }
} else {
    Write-Output 'docs/: none'
}

# --- the index -------------------------------------------------------------

Write-Section 'index'
# The skill starts cb_indexer_sync_repo only when this manifest exists (a project
# with no manifest would get a full rebuild, a long embed that is
# cb_indexer_index_repo's to start and the user's to ask for).
if (Test-Path -LiteralPath '.qdrant_index_manifest.json' -PathType Leaf) {
    Write-Output 'sync manifest: present (.qdrant_index_manifest.json)'
} else {
    Write-Output 'sync manifest: ABSENT (this project has not been indexed here; the first embed is long, so cb_indexer_sync_repo was not started)'
}

# --- the forge: pull requests only -----------------------------------------

Write-Section 'forge'
$forgeNote = ''
$forge = 'none'
$hostName = ''
$org = ''; $project = ''; $repo = ''; $slug = ''

if (-not $remote) {
    $forgeNote = 'no origin remote'
} else {
    # Forms handled:
    #   https://github.com/owner/repo.git            git@github.com:owner/repo.git
    #   https://contoso.visualstudio.com/DefaultCollection/<project>/_git/<repo>
    #   https://dev.azure.com/<org>/<project>/_git/<repo>
    #   git@ssh.dev.azure.com:v3/<org>/<project>/<repo>
    $path = ''
    if ($remote -match '://') {
        $rest = $remote -replace '^[^:]*://', ''
        $parts = $rest -split '/', 2
        $hostName = $parts[0]
        if ($parts.Count -gt 1) { $path = $parts[1] }
    } elseif ($remote -match ':') {
        $u = $remote -replace '^[^@]*@', ''
        $parts = $u -split ':', 2
        $hostName = $parts[0]
        $path = $parts[1]
    }
    $hostName = $hostName -replace ':.*$', ''   # drop a :port
    $path = $path.TrimEnd('/')
    if ($hostName -eq 'github.com') {
        $forge = 'github'
        $slug = $path -replace '\.git$', ''
    } elseif ($hostName -in @('dev.azure.com', 'ssh.dev.azure.com', 'vs-ssh.visualstudio.com') -or $hostName -like '*.visualstudio.com') {
        $forge = 'ado'
        if ($path.StartsWith('v3/')) {                  # ssh form: v3/ORG/PROJECT/REPO
            $p = $path.Substring(3)
            $org = ($p -split '/', 2)[0]
            $p = if ($p.Contains('/')) { ($p -split '/', 2)[1] } else { '' }
            $project = ($p -split '/', 2)[0]
            $repo = if ($p.Contains('/')) { ($p -split '/', 2)[1] } else { '' }
        } else {
            if ($hostName -eq 'dev.azure.com') {
                $org = ($path -split '/', 2)[0]
                $path = if ($path.Contains('/')) { ($path -split '/', 2)[1] } else { '' }
            } else {
                $org = ($hostName -split '\.')[0]
            }
            $path = $path -replace '^DefaultCollection/', ''
            if ($path.StartsWith('_git/')) {            # no project segment
                $repo = $path.Substring(5); $project = $repo
            } elseif ($path.Contains('/_git/')) {
                $project = $path.Substring(0, $path.IndexOf('/_git/'))
                $repo = $path.Substring($path.LastIndexOf('/_git/') + 6)
            }
        }
        # Only %20 is decoded: it is the one escape a project or repo name really carries.
        $project = $project -replace '%20', ' '
        $repo = $repo -replace '%20', ' '
    } else {
        $shown = $hostName
        if (-not $shown) { $shown = 'unknown' }
        $forgeNote = "origin host $shown is neither GitHub nor Azure DevOps"
    }
}

if ($forge -eq 'github') {
    Write-Output "forge: GitHub ($slug)"
    if (-not (Test-Command gh)) {
        $forgeNote = 'gh is not installed'
    } else {
        $auth = Invoke-Limited 20 'gh' @('auth', 'status', '--hostname', 'github.com')
        if ($auth.Rc -ne 0) {
            $forgeNote = 'gh is not signed in to github.com (run: gh auth login)'
        } else {
            $res = Invoke-Limited 25 'gh' @('pr', 'list', '-R', $slug, '--state', 'open', '--limit', '30', '--json', 'number,title,headRefName,isDraft,isCrossRepository')
            if (Test-TimedOut $res) {
                $forgeNote = 'gh pr list timed out'
            } elseif ($res.Rc -ne 0) {
                $forgeNote = "gh pr list failed: $(Get-FirstLine $res.Out)"
            } else {
                $prs = @()
                # Windows PowerShell 5.1 returns a JSON array as ONE object (and "[]" as a
                # single empty one), so it is piped through ForEach-Object to unroll it and
                # nulls are dropped; PowerShell 7 already unrolls.
                try { $prs = @($res.Out | ConvertFrom-Json | ForEach-Object { $_ } | Where-Object { $null -ne $_ }) } catch { $prs = @() }
                Write-Output "open pull requests: $($prs.Count)"
                if ($prs.Count -ge 30) { Write-Output '  (capped at 30)' }
                foreach ($pr in $prs) {
                    $state = if ($pr.isDraft) { 'draft' } else { 'open' }
                    $mark = ''
                    # A fork's PR from a branch with the same name (its own main) is not this branch's.
                    if ($branch -and $pr.headRefName -eq $branch -and -not $pr.isCrossRepository) { $mark = '  <- current branch' }
                    Write-Output "  #$($pr.number) [$($pr.headRefName)] ($state) $(Get-Clip $pr.title 90)$mark"
                }
            }
        }
    }
} elseif ($forge -eq 'ado') {
    Write-Output "forge: Azure DevOps (org $org, project $project, repo $repo)"
    if (-not $org -or -not $project -or -not $repo) {
        $forgeNote = 'could not read the organization, project and repository from the origin URL'
    } elseif (-not (Test-Command az)) {
        $forgeNote = 'az is not installed'
    } else {
        $orgUrl = "https://dev.azure.com/$org"
        # --only-show-errors drops az's upgrade and preview warnings, which would
        # otherwise be mixed into the output (stderr is merged so errors can be read).
        $acct = Invoke-Limited 40 'az' @('account', 'show', '--only-show-errors', '-o', 'none')
        if ($acct.Rc -ne 0) {
            $forgeNote = 'az is not signed in (run: az login)'
        } else {
            $list = Invoke-Limited 40 'az' @('repos', 'list', '--org', $orgUrl, '--project', $project, '--query', '[].[name,isDisabled]', '-o', 'tsv', '--only-show-errors')
            if (Test-TimedOut $list) {
                $forgeNote = 'az repos list timed out'
            } elseif ($list.Rc -ne 0) {
                if ($list.Out -match 'not recognized|misspelled') {
                    $forgeNote = 'the azure-devops extension is not installed (run: az extension add --name azure-devops)'
                } else {
                    $forgeNote = "az repos list failed: $(Get-FirstLine $list.Out)"
                }
            } else {
                $disabled = ''
                foreach ($row in ($list.Out -split "`n")) {
                    $c = $row.TrimEnd("`r") -split "`t"
                    if ($c.Count -ge 2 -and $c[0].ToLower() -eq $repo.ToLower()) { $disabled = $c[1].ToLower(); break }
                }
                if ($disabled -eq 'true') {
                    $forgeNote = "repo $repo is disabled (isDisabled: true), so its pull requests are unavailable"
                } else {
                    $res = Invoke-Limited 40 'az' @('repos', 'pr', 'list', '--org', $orgUrl, '--project', $project, '--repository', $repo, '--status', 'active', '--query', '[].[pullRequestId,sourceRefName,isDraft,title]', '-o', 'tsv', '--only-show-errors')
                    if (Test-TimedOut $res) {
                        $forgeNote = 'az repos pr list timed out'
                    } elseif ($res.Rc -ne 0) {
                        if ($res.Out -match 'TF401019') {
                            $forgeNote = "repo $repo is disabled, removed, or you have no access to it (TF401019)"
                        } else {
                            $forgeNote = "az repos pr list failed: $(Get-FirstLine $res.Out)"
                        }
                    } else {
                        # Only lines that start with a PR number are rows: a stderr line that
                        # came with a successful call (stderr is merged) is not a pull request.
                        $rows = @($res.Out -split "`n" | ForEach-Object { $_.TrimEnd("`r") } | Where-Object { $_ -match '^[0-9]+\t' })
                        Write-Output "active pull requests: $($rows.Count)"
                        foreach ($row in $rows) {
                            $c = $row -split "`t", 4
                            $headRef = $c[1] -replace '^refs/heads/', ''
                            $state = 'active'
                            if ($c[2] -eq 'true' -or $c[2] -eq 'True') { $state = 'draft' }
                            $mark = ''
                            if ($branch -and $headRef -eq $branch) { $mark = '  <- current branch' }
                            Write-Output "  #$($c[0]) [$headRef] ($state) $(Get-Clip $c[3] 90)$mark"
                        }
                    }
                }
            }
        }
    }
}

if ($forgeNote) {
    Write-Output "forge facts skipped: $forgeNote"
} else {
    Write-Output 'scope: pull requests only'
}

exit 0
