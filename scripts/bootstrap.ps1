[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

function Stop-Bootstrap {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,
        [int]$ExitCode = 1
    )

    [Console]::Error.WriteLine("bootstrap: $Message")
    exit $ExitCode
}

function Invoke-BootstrapCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Description,
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host "> $Description"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        Stop-Bootstrap "$Description failed (exit code $LASTEXITCODE)."
    }
}

# PowerShell can reject a script before its contents run; the docs also show
# this recovery command so it is available even when this check cannot run.
$executionPolicy = Get-ExecutionPolicy
if ($executionPolicy -eq 'Restricted' -or $executionPolicy -eq 'AllSigned') {
    Stop-Bootstrap "The current execution policy ($executionPolicy) blocks this local script. Run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
}

$gitCommand = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCommand) {
    Stop-Bootstrap 'Git was not found. Install Git for Windows and rerun this script.'
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repoRoot
$null = & $gitCommand.Source rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -ne 0) {
    Stop-Bootstrap 'Run this script from a Git clone of ClaudeRunway.'
}

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
$pyCommand = Get-Command py -ErrorAction SilentlyContinue
$pythonPath = $null

if ($uvCommand) {
    $candidate = & $uvCommand.Source python find 3.12 2>$null
    if ($LASTEXITCODE -eq 0 -and $candidate) {
        $pythonPath = ([string]($candidate | Select-Object -First 1)).Trim()
    }
}

if (-not $pythonPath -and $pyCommand) {
    $candidate = & $pyCommand.Source -3.12 -c 'import sys; print(sys.executable)' 2>$null
    if ($LASTEXITCODE -eq 0 -and $candidate) {
        $pythonPath = ([string]($candidate | Select-Object -First 1)).Trim()
    }
}

if (-not $pythonPath -or -not (Test-Path -LiteralPath $pythonPath)) {
    Stop-Bootstrap "Python 3.12 was not found. Install it and rerun this script (uv users can run 'uv python install 3.12')."
}

$venvPath = Join-Path $repoRoot '.venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    if (Test-Path -LiteralPath $venvPath) {
        Stop-Bootstrap "The existing .venv directory has no Windows Python executable. Remove or rename '$venvPath' manually, then rerun this script."
    }

    if ($uvCommand) {
        Invoke-BootstrapCommand `
            -Description 'Create the Python 3.12 virtual environment' `
            -FilePath $uvCommand.Source `
            -Arguments @('venv', '--python', $pythonPath, $venvPath)
    }
    else {
        Invoke-BootstrapCommand `
            -Description 'Create the Python 3.12 virtual environment' `
            -FilePath $pythonPath `
            -Arguments @('-m', 'venv', $venvPath)
    }
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Stop-Bootstrap "Virtual environment creation did not produce '$venvPython'."
}

$venvVersion = & $venvPython -c 'import sys; print(sys.version_info.major, sys.version_info.minor, sep=chr(46))'
if ($LASTEXITCODE -ne 0) {
    Stop-Bootstrap 'Could not run the Python executable in .venv.'
}
if (([string]($venvVersion | Select-Object -First 1)).Trim() -ne '3.12') {
    Stop-Bootstrap "The existing .venv uses Python $venvVersion instead of Python 3.12. Remove or rename '$venvPath' manually, then rerun this script."
}

if ($uvCommand) {
    Invoke-BootstrapCommand `
        -Description 'Install runtime requirements' `
        -FilePath $uvCommand.Source `
        -Arguments @('pip', 'install', '--python', $venvPython, '-r', (Join-Path $repoRoot 'requirements.txt'), '--index-url', 'https://pypi.org/simple')
    Invoke-BootstrapCommand `
        -Description 'Install developer requirements' `
        -FilePath $uvCommand.Source `
        -Arguments @('pip', 'install', '--python', $venvPython, '-r', (Join-Path $repoRoot 'requirements-dev.txt'), '--index-url', 'https://pypi.org/simple')
}
else {
    Invoke-BootstrapCommand `
        -Description 'Install runtime requirements' `
        -FilePath $venvPython `
        -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $repoRoot 'requirements.txt'), '--index-url', 'https://pypi.org/simple')
    Invoke-BootstrapCommand `
        -Description 'Install developer requirements' `
        -FilePath $venvPython `
        -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $repoRoot 'requirements-dev.txt'), '--index-url', 'https://pypi.org/simple')
}

$doctor = Join-Path $venvPath 'Scripts\claude-runway-doctor.exe'
if ($uvCommand) {
    Invoke-BootstrapCommand `
        -Description 'Install/update ClaudeRunway command-line tools from this checkout' `
        -FilePath $uvCommand.Source `
        -Arguments @('pip', 'install', '--python', $venvPython, '--no-deps', '--no-build-isolation', $repoRoot)
}
else {
    Invoke-BootstrapCommand `
        -Description 'Install/update ClaudeRunway command-line tools from this checkout' `
        -FilePath $venvPython `
        -Arguments @('-m', 'pip', 'install', '--no-deps', '--no-build-isolation', $repoRoot)
}
if (-not (Test-Path -LiteralPath $doctor)) {
    Stop-Bootstrap "The doctor command was not installed at '$doctor'."
}

$currentHooksPath = & $gitCommand.Source config --get core.hooksPath 2>$null
if ($LASTEXITCODE -ne 0 -or ([string]($currentHooksPath | Select-Object -First 1)).Trim() -ne '.githooks') {
    Invoke-BootstrapCommand `
        -Description 'Enable the repository pre-push hook' `
        -FilePath $gitCommand.Source `
        -Arguments @('config', 'core.hooksPath', '.githooks')
}
else {
    Write-Host '> Repository pre-push hook already enabled'
}

Invoke-BootstrapCommand -Description 'Run the ClaudeRunway doctor check' -FilePath $doctor -Arguments @($repoRoot)

Write-Host 'bootstrap: setup complete.'
