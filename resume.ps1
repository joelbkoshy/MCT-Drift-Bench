# Resume (or start) the full MCT-DriftBench pipeline.
#
#   .\resume.ps1
#
# Safe to run any number of times, including after a crash, a reboot, or a flat
# battery. Every stage skips cells that already have a result, so a re-run picks
# up exactly where the last one stopped. Nothing is regenerated and no completed
# work is discarded.
#
# Stage order is fixed: probes -> safety -> drift -> fidelity -> judge -> analyse.

[CmdletBinding()]
param(
    [switch]$SkipReference,
    [string]$Python = "c:\Users\joel.k\Documents\PhD\Hybrid-MCT-LLM-Evaluation-Paper2\.venv\Scripts\python.exe"
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $MyInvocation.MyCommand.Definition
$results = Join-Path $repo 'results'
# The Python stages hold their own OS-level lock on results\.run.lock; this one
# only stops two copies of this wrapper script running.
$lock = Join-Path $results '.resume.lock'
New-Item -ItemType Directory -Force -Path $results | Out-Null

function Fail($message) {
    Write-Host "  $message" -ForegroundColor Red
    exit 1
}

Write-Host ''
Write-Host 'MCT-DriftBench - resume' -ForegroundColor Cyan
Write-Host ('-' * 62)

# --- refuse to run twice: two appenders would interleave rows ---------------
if (Test-Path $lock) {
    $held = Get-Content $lock -ErrorAction SilentlyContinue
    $alive = $held -and (Get-Process -Id $held -ErrorAction SilentlyContinue)
    if ($alive) {
        Fail "a run is already in progress (pid $held). Use .\status.ps1 -Watch to follow it."
    }
    Write-Host "  clearing stale lock from pid $held" -ForegroundColor DarkGray
    Remove-Item $lock -Force
}

# --- interpreter ------------------------------------------------------------
if (-not (Test-Path $Python)) {
    $local = Join-Path $repo '.venv\Scripts\python.exe'
    if (Test-Path $local) { $Python = $local }
    else { Fail "no interpreter found. Pass -Python <path to python.exe>." }
}
Write-Host "  python   $Python" -ForegroundColor DarkGray

# --- ollama must be up; after a reboot it may not be -----------------------
$null = & ollama ps 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host '  ollama not responding, starting it' -ForegroundColor Yellow
    Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2
        $null = & ollama ps 2>&1
        if ($LASTEXITCODE -eq 0) { break }
    }
    if ($LASTEXITCODE -ne 0) { Fail 'could not reach the Ollama server.' }
}

$installed = (& ollama list) -join "`n"
foreach ($model in @('qwen2.5:1.5b-instruct', 'mct-qwen', 'llama3')) {
    if ($installed -notmatch [regex]::Escape($model)) {
        Fail "model '$model' is not installed. See the README for the pull/create commands."
    }
}
Write-Host '  ollama   up, all three models present' -ForegroundColor DarkGray
Write-Host ('-' * 62)

# --- run --------------------------------------------------------------------
$env:PYTHONPATH = $repo
$extra = if ($SkipReference) { @('--no-reference') } else { @() }

$PID | Out-File -FilePath $lock -Encoding ascii
$started = Get-Date

try {
    $stages = @(
        @{ Name = 'probes';   Log = $null },
        @{ Name = 'safety';   Log = 'safety_log.txt' },
        @{ Name = 'drift';    Log = 'drift_log.txt' },
        @{ Name = 'fidelity'; Log = 'fidelity_log.txt' },
        @{ Name = 'judge';    Log = 'judge_log.txt' },
        @{ Name = 'analyse';  Log = 'analysis_log.txt' }
    )

    foreach ($stage in $stages) {
        Write-Host ("  [{0:HH:mm:ss}] {1}" -f (Get-Date), $stage.Name) -ForegroundColor Yellow
        if ($stage.Log) {
            $logPath = Join-Path $results $stage.Log
            & $Python (Join-Path $repo 'run.py') $stage.Name @extra *>> $logPath
        } else {
            & $Python (Join-Path $repo 'run.py') $stage.Name @extra
        }
        if ($LASTEXITCODE -ne 0) { Fail "stage '$($stage.Name)' failed (exit $LASTEXITCODE). Re-run this script to resume." }
    }
}
finally {
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
}

$elapsed = (Get-Date) - $started
Write-Host ('-' * 62)
Write-Host ("  COMPLETE in {0:hh\:mm\:ss}" -f $elapsed) -ForegroundColor Green
Write-Host '  results/summary.json, results/table_*.csv, figures/' -ForegroundColor DarkGray
Write-Host ''
