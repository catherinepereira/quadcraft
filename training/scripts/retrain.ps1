# Overnight retrain: the held game runs the client gametest, then a goto DAgger round and a skill DAgger round,
# then the game closes and the hub booking is freed
# A failed gametest or round stops the run
param([int]$GotoRound = 3, [int]$SkillRound = 11, [int]$Booking = 0)
$d = Split-Path $PSScriptRoot -Parent
$hub = "C:\Users\cathe\Developer\hub\hub.py"
$py = "$d\.venv\Scripts\python.exe"
$stamp = Get-Date -Format "yyyyMMdd-HHmm"
Start-Transcript -Path "$d\logs\retrain-$stamp.log" | Out-Null

function Finish([string]$note) {
  New-Item -ItemType File -Force "$d\..\mod\build\run\clientGameTest\e2e-done" | Out-Null
  if ($Booking) { & $py $hub --agent quadcraft-reader done $Booking --note $note }
  "$(Get-Date -Format HH:mm:ss) $note"
  Stop-Transcript | Out-Null
  exit [int]($note -notmatch "^done")
}

if ($Booking) { & $py $hub --agent quadcraft-reader start $Booking }
$game = & "$d\..\mod\scripts\regame.ps1"
$game
if (-not ($game -contains "gametest passed")) { Finish "gametest failed, no retrain" }

try { & "$PSScriptRoot\dagger_goto.ps1" -Round $GotoRound } catch { Finish "goto round $GotoRound failed: $_" }
try { & "$PSScriptRoot\dagger_skill.ps1" -Round $SkillRound } catch { Finish "skill round $SkillRound failed: $_" }
Finish "done: gametest passed, goto r$GotoRound and skill r$SkillRound retrained"
