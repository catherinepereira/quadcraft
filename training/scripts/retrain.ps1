# Overnight retrain, each step after the one before, a failed step stops the run and frees the hub booking
# 1. the held game runs the client gametest, then collect.fields takes crop field frames for the block reader
# 2. with the game closed, the reader trains from the current one on studio, render, and field frames
#    It is promoted only when it reads -ReaderClass better on the same validation frames, see train.promote_reader
# 3. the game is back for a goto DAgger round and a skill round, a skill round without harvest_crops when the reader stayed
# -SkillPlanOnly stops the skill round after its planner episodes and first retrain
param(
  [int]$GotoRound = 4, [int]$SkillRound = 11, [int]$Booking = 0, [string]$Reader = "v11", [int]$ReaderSteps = 15000,
  [string]$ReaderClass = "minecraft:wheat", [switch]$SkillPlanOnly
)
$d = Split-Path $PSScriptRoot -Parent
$hub = "C:\Users\cathe\Developer\hub\hub.py"
$py = "$d\.venv\Scripts\python.exe"
$done = "$d\..\mod\build\run\clientGameTest\e2e-done"
$stamp = Get-Date -Format "yyyyMMdd-HHmm"
Start-Transcript -Path "$d\logs\retrain-$stamp.log" | Out-Null

function Finish([string]$note) {
  New-Item -ItemType File -Force $done | Out-Null
  if ($Booking) { & $py $hub --agent quadcraft-reader done $Booking --note $note }
  "$(Get-Date -Format HH:mm:ss) $note"
  Stop-Transcript | Out-Null
  exit [int]($note -notmatch "^done")
}

function Game {
  $game = & "$d\..\mod\scripts\regame.ps1"
  $game
  if (-not ($game -contains "gametest passed")) { Finish "gametest failed" }
}

# the output goes to the host so the function returns only whether the step succeeded
function Step([string]$name, [string[]]$arguments) {
  "$(Get-Date -Format HH:mm:ss) $name" | Out-Host
  & $py -u @arguments 2>&1 | Tee-Object -FilePath "$d\logs\$name.log" | Select-String "^shard|^train|^baseline|^step|^best|promoted|kept|Traceback|Error" | Out-Host
  return $LASTEXITCODE -eq 0
}

if ($Booking) { & $py $hub --agent quadcraft-reader start $Booking }
Push-Location $d
Game
if (-not (Step "collect-fields-$Reader" @("-m", "drone_model.collect.fields"))) { Finish "field collection failed" }
New-Item -ItemType File -Force $done | Out-Null
Start-Sleep 30

$out = "checkpoints\reader-$Reader.pt"
$train = @("-m", "drone_model.train.reader", "--data", "data\studio", "data\render", "data\fields", "--init", "checkpoints\reader.pt",
  "--baseline", "checkpoints\reader.pt", "--steps", "$ReaderSteps", "--lr", "3e-4", "--out", $out)
if (-not (Step "reader-$Reader" $train)) { Finish "reader $Reader training failed" }
$promoted = Step "promote-$Reader" @("-m", "drone_model.train.promote_reader", $out, "--class", $ReaderClass)

Game
try { & "$PSScriptRoot\dagger_goto.ps1" -Round $GotoRound } catch { Finish "goto round $GotoRound failed: $_" }
$skillTasks = if ($promoted) { "" } else { "replicate_build,copy_build,mine_deposit,gather_build,hunt_mobs,patrol_area" }
try { & "$PSScriptRoot\dagger_skill.ps1" -Round $SkillRound -Tasks $skillTasks -PlanOnly:$SkillPlanOnly } catch { Finish "skill round $SkillRound failed: $_" }
Pop-Location
Finish "done: reader $Reader $(if ($promoted) { 'promoted' } else { 'not promoted' }), goto r$GotoRound, skill r$SkillRound$(if ($SkillPlanOnly) { ' plan half' })"
