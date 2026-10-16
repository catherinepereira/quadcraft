# Launches the held game, collects studio frames, closes the game, then trains the reader at full GPU and exports embeddings
# -CollectOnly stops after closing the game, -TrainOnly skips the game and collection
# Each step waits for the one before, a failed step stops the run
param([string]$Version = "v10", [int]$Steps = 50000, [switch]$CollectOnly, [switch]$TrainOnly)
$d = Split-Path $PSScriptRoot -Parent
$py = "$d\.venv\Scripts\python.exe"

function Step([string]$name, [string[]]$arguments) {
  $p = Start-Process -FilePath $py -ArgumentList $arguments -WorkingDirectory $d `
    -RedirectStandardOutput "$d\logs\$name.log" -RedirectStandardError "$d\logs\$name.err" -WindowStyle Hidden -Wait -PassThru
  return $p.ExitCode -eq 0
}

if (-not $TrainOnly) {
  $game = & "$d\..\mod\scripts\regame.ps1"
  $game | Out-File -Encoding utf8 "$d\logs\regame-$Version.log"
  if (-not ($game -contains "gametest passed")) { exit 1 }
  $collected = Step "collect-studio-$Version" @("-m", "drone_model.collect.studio")
  # the held gametest ends once this file exists
  New-Item -ItemType File -Force "$d\..\mod\build\run\clientGameTest\e2e-done" | Out-Null
  if (-not $collected -or $CollectOnly) { exit [int](-not $collected) }
  Start-Sleep 30
}
if (-not (Step "reader-$Version" @("-m", "drone_model.train.reader", "--steps", "$Steps", "--out", "checkpoints\reader-$Version.pt"))) { exit 1 }
Step "embeddings-$Version" @("-m", "drone_model.evaluate.embeddings", "--checkpoint", "checkpoints\reader-$Version.pt", "--dest", "..\..\quadcraft-embeddings\public\data") | Out-Null
