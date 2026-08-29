# Live progress watcher for the MCT-DriftBench pipeline and the remaining work.
#
#   .\status.ps1            one-shot snapshot
#   .\status.ps1 -Watch     refresh until everything is finished
#
# Row counts come from Import-Csv rather than a line count: generated responses
# contain embedded newlines, so counting lines overstates progress.

[CmdletBinding()]
param(
    [switch]$Watch,
    [int]$IntervalSeconds = 20
)

$ErrorActionPreference = 'SilentlyContinue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$results = Join-Path $root 'results'
$figures = Join-Path $root 'figures'
$paper = Join-Path $root 'paper\main.tex'

$stages = @(
    [pscustomobject]@{ Name = 'drift';    File = 'drift_runs.csv';       Log = 'drift_log.txt';    Total = 336 }
    [pscustomobject]@{ Name = 'fidelity'; File = 'fidelity_runs.csv';    Log = 'fidelity_log.txt'; Total = 280 }
    [pscustomobject]@{ Name = 'judge';    File = 'drift_judgements.csv'; Log = 'judge_log.txt';    Total = 336 }
)

# Manuscript sections still holding a literal placeholder.
$placeholders = @('ABSTRACT PLACEHOLDER', 'RESULTS PLACEHOLDER',
                  'DISCUSSION PLACEHOLDER', 'LIMITATIONS PLACEHOLDER',
                  'CONCLUSION PLACEHOLDER')

function Get-Rows($path) {
    if (-not (Test-Path $path)) { return @() }
    try { @(Import-Csv -Path $path) } catch { @() }
}

function Write-Bar($label, $state, $done, $total, $colour) {
    $pct = if ($total) { 100.0 * $done / $total } else { 0 }
    $bar = ('#' * [int]($pct / 4)).PadRight(25, '.')
    Write-Host ("  {0,-9} {1,-8} [{2}] {3,4}/{4,-4} {5,5:N1}%" -f `
        $label, $state, $bar, $done, $total, $pct) -ForegroundColor $colour
}

function Show-Status {
    Write-Host ''
    Write-Host ("MCT-DriftBench  ---  {0:HH:mm:ss}" -f (Get-Date)) -ForegroundColor Cyan
    Write-Host ('-' * 62)
    Write-Host '  GENERATION' -ForegroundColor DarkCyan

    $activeRows = $null; $activeName = $null; $allDone = $true

    foreach ($stage in $stages) {
        $rows = Get-Rows (Join-Path $results $stage.File)
        # Cast: a partially flushed file can yield a count that formats as blank.
        $done = [int]($rows.Count)
        if ($done -eq 0) { $state = 'queued '; $colour = 'DarkGray'; $allDone = $false }
        elseif ($done -ge $stage.Total) { $state = 'done   '; $colour = 'Green' }
        else {
            $state = 'running'; $colour = 'Yellow'
            $activeRows = $rows; $activeName = $stage.Name; $allDone = $false
        }
        Write-Bar $stage.Name $state $done $stage.Total $colour
    }

    Write-Host ''
    Write-Host '  REMAINING TODOS' -ForegroundColor DarkCyan

    $tables = @(Get-ChildItem (Join-Path $results 'table_*.csv')).Count
    $figs = @(Get-ChildItem (Join-Path $figures '*.pdf')).Count
    $judgePath = Join-Path $results 'drift_judgements.csv'
    $summaryPath = Join-Path $results 'summary.json'

    # The analysis only counts as current if it was produced after the verdicts
    # it summarises; otherwise it is a stale run over older data.
    $fresh = $false
    if ((Test-Path $summaryPath) -and (Test-Path $judgePath)) {
        $fresh = (Get-Item $summaryPath).LastWriteTime -gt (Get-Item $judgePath).LastWriteTime
    }
    $analysisDone = ($tables -ge 4 -and $figs -ge 3 -and $fresh)
    Write-Bar 'analysis' $(if ($analysisDone) { 'done   ' } else { 'pending' }) `
        $(if ($analysisDone) { 1 } else { 0 }) 1 `
        $(if ($analysisDone) { 'Green' } else { 'DarkGray' })
    if (-not $analysisDone) { $allDone = $false }

    $written = 0
    if (Test-Path $paper) {
        $text = Get-Content $paper -Raw
        foreach ($p in $placeholders) {
            if ($text -notmatch [regex]::Escape($p)) { $written++ }
        }
    }
    $paperDone = $written -ge $placeholders.Count
    Write-Bar 'paper' $(if ($paperDone) { 'done   ' } else { 'pending' }) `
        $written $placeholders.Count `
        $(if ($paperDone) { 'Green' } elseif ($written -gt 0) { 'Yellow' } else { 'DarkGray' })
    if (-not $paperDone) { $allDone = $false }

    Write-Host ('-' * 62)

    if ($activeRows -and $activeName -eq 'judge') {
        $v = $activeRows | Group-Object judge_verdict |
             ForEach-Object { "{0}={1}" -f $_.Name, $_.Count }
        Write-Host ''
        Write-Host ("  verdicts so far: {0}" -f ($v -join '  ')) -ForegroundColor DarkCyan
    }
    elseif ($activeRows) {
        Write-Host ''
        $field = if ($activeName -eq 'drift') { 'drifted' } else { 'fidelity_overall' }
        $label = if ($activeName -eq 'drift') { 'drift rate' } else { 'fidelity' }
        Write-Host ("  arm                     n   {0}" -f $label) -ForegroundColor DarkCyan
        $activeRows | Group-Object arm | ForEach-Object {
            $v = $_.Group | ForEach-Object {
                if ($field -eq 'drifted') { if ($_.drifted -eq 'True') { 1 } else { 0 } }
                else { [double]$_.fidelity_overall }
            }
            Write-Host ("  {0,-20} {1,4}   {2,8:N3}" -f `
                $_.Name, $_.Count, ($v | Measure-Object -Average).Average)
        }
    }

    $live = $stages | Where-Object { $_.Name -eq $activeName } | Select-Object -First 1
    if ($live) {
        $tail = Get-Content (Join-Path $results $live.Log) -Tail 1
        if ($tail) { Write-Host ''; Write-Host "  $tail" -ForegroundColor DarkGray }
    }

    return $allDone
}

if (-not $Watch) { [void](Show-Status); return }

while ($true) {
    Clear-Host
    if (Show-Status) {
        Write-Host ''
        Write-Host '  ALL WORK COMPLETE' -ForegroundColor Green
        break
    }
    Write-Host ''
    Write-Host ("  refreshing every {0}s  ---  Ctrl+C to stop" -f $IntervalSeconds) -ForegroundColor DarkGray
    Start-Sleep -Seconds $IntervalSeconds
}
