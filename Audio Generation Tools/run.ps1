<#
.SYNOPSIS
Runs every pipeline stage in order. Safe to stop at any point (Ctrl+C, closing the
window, killing the process, sleep) and rerun: each stage skips finished work.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File run.ps1              # run until everything is done
powershell -ExecutionPolicy Bypass -File run.ps1 -Pause       # running stages wait after the current segment
powershell -ExecutionPolicy Bypass -File run.ps1 -Resume      # ...and carry on
powershell -ExecutionPolicy Bypass -File run.ps1 -Pilot
powershell -ExecutionPolicy Bypass -File run.ps1 -MaxMinutes 60
powershell -ExecutionPolicy Bypass -File run.ps1 -Rerender
powershell -ExecutionPolicy Bypass -File run.ps1 -Root D:\ent_synth
#>
param(
    [switch]$Pause,           # tell a running pipeline to wait (model stays loaded); returns at once
    [switch]$Resume,          # let it continue
    [double]$MaxMinutes = 0,  # time-box synthesis; later stages then run on whatever is finished
    [switch]$Pilot,           # only the cases in data\pilot_ids.txt
    [switch]$Rerender,        # first clear the segments QC listed in data\rerender.txt
    [switch]$SkipQC,
    [int]$Shard = 0,          # split synthesis across machines: -Shard i -NumShards N
    [int]$NumShards = 1,
    [string]$Root = ""        # models, venvs and audio; default $env:ENT_SYNTH_ROOT, else <home>\ent_synth
)
$ErrorActionPreference = "Stop"
if (-not $Root) { $Root = if ($env:ENT_SYNTH_ROOT) { $env:ENT_SYNTH_ROOT } else { Join-Path $HOME "ent_synth" } }
$env:ENT_SYNTH_ROOT = $Root  # the Python stages read it too (synth/config.py)
$TtsPython = "$Root\index-tts\.venv\Scripts\python.exe"
$ToolsPython = "$Root\venv-tools\Scripts\python.exe"
$PauseFlag = "$Root\data\PAUSE"
if ($Pause) {
    New-Item -ItemType File -Force $PauseFlag | Out-Null
    Write-Host "Paused: synthesis/QC wait after the current segment. Continue with: run.ps1 -Resume"
    exit 0
}
if ($Resume) {
    Remove-Item $PauseFlag -ErrorAction SilentlyContinue
    Write-Host "Resumed: the running pipeline continues within a few seconds."
    exit 0
}
foreach ($py in $ToolsPython, $TtsPython) {
    if (-not (Test-Path $py)) { throw "$py not found: do the one-time setup in README.md for $Root, or pass -Root" }
}
if (Test-Path $PauseFlag) { Write-Host "Note: paused ($PauseFlag exists); it will wait until run.ps1 -Resume" -ForegroundColor Yellow }
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:USE_MODELSCOPE = "false"  # skip IndexTTS's network probe; weights are already local

function Invoke-Stage([string]$Name, [string]$Python, [string[]]$StageArgs) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Python -m @StageArgs
    if ($LASTEXITCODE -ne 0) { throw "$Name stopped (exit code $LASTEXITCODE); rerun run.ps1 to continue" }
}

$select = @()
if ($Pilot) { $select += "--pilot" }

Invoke-Stage "build cases" $ToolsPython @("synth.build_cases")

$synthArgs = @("synth.synthesize") + $select + @("--shard", "$Shard", "--num-shards", "$NumShards")
if ($MaxMinutes -gt 0) { $synthArgs += @("--max-minutes", "$MaxMinutes") }
if ($Rerender) { $synthArgs += "--rerender" }
Invoke-Stage "synthesize" $TtsPython $synthArgs

Invoke-Stage "assemble" $ToolsPython (@("synth.assemble") + $select)
if (-not $SkipQC) { Invoke-Stage "qc" $ToolsPython (@("synth.qc") + $select) }
Invoke-Stage "export" $ToolsPython (@("synth.export_ent") + $select + @("--check"))
Write-Host "`nDone. Outputs are in $Root\data" -ForegroundColor Green
