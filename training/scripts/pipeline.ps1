# The training pipeline, the -Stages given in this order, each after the one before, a failed stage stops the run
#   render    block frames drawn offline from the game's own models, on the GPU without the game (data/render)
#   studio    game frames of mobs and of the blocks the renderer can't draw (data/studio)
#   fields    game frames of harvest arena wheat fields (data/fields)
#   reader    the block reader, from ImageNet weights on studio and render frames to checkpoints/reader-base.pt,
#             then fine-tuned with the field frames into checkpoints/reader.pt
#             With -FineTune it fine-tunes the current reader.pt instead and promotes the result only when it reads
#             -ReaderClass better on the same validation frames, see train.promote_reader
#   goto      the next DAgger round of the goto policy, see dagger_goto.ps1
#   skill     the next DAgger round of the cell skill, see dagger_skill.ps1
#   evaluate  the whole drone on every training arena, see evaluate.ps1
# The game starts for the stages that need it and closes before the reader trains, which wants the whole GPU
# Logs go to logs/pipeline-<time>.log, each step's full output to logs/<step>.log
param(
  [string[]]$Stages = @("render", "studio", "fields", "reader", "goto", "skill", "evaluate"),
  [switch]$FineTune, [string]$ReaderClass = "minecraft:wheat", [int]$EvalEpisodes = 5
)
$d = Split-Path $PSScriptRoot -Parent
$py = "$d\.venv\Scripts\python.exe"
$held = "$d\..\mod\build\run\clientGameTest\e2e-done"
$script:gameUp = $false
New-Item -ItemType Directory -Force "$d\logs" | Out-Null
Start-Transcript -Path "$d\logs\pipeline-$(Get-Date -Format yyyyMMdd-HHmm).log" | Out-Null

function Finish([string]$note) {
  if ($script:gameUp) { New-Item -ItemType File -Force $held | Out-Null }
  "$(Get-Date -Format HH:mm:ss) $note"
  Stop-Transcript | Out-Null
  exit [int]($note -notmatch "^done")
}

# the held client gametest, a game whose bridge stays open for the scripts once its checks pass
function GameUp {
  if ($script:gameUp) { return }
  $game = & "$d\..\mod\scripts\regame.ps1"
  $game | Out-Host
  if (-not ($game -contains "gametest passed")) { Finish "gametest failed" }
  $script:gameUp = $true
}

function GameDown {
  if (-not $script:gameUp) { return }
  New-Item -ItemType File -Force $held | Out-Null
  $script:gameUp = $false
  # the client takes a moment to release the GPU
  Start-Sleep 30
}

# output goes to the host so the function returns only whether the step succeeded
function Step([string]$name, [string[]]$arguments) {
  "$(Get-Date -Format HH:mm:ss) $name" | Out-Host
  & $py -u @arguments 2>&1 | Tee-Object -FilePath "$d\logs\$name.log" | Select-String "^shard|^\d+ |^train|^baseline|^step|^best|promoted|kept|Traceback|Error" | Out-Host
  return $LASTEXITCODE -eq 0
}

# one past the highest round among the policy's data folders, which end in -r<round>
function NextRound([string]$policy) {
  $rounds = Get-ChildItem "$d\data\policies\$policy" -Directory -ErrorAction SilentlyContinue |
    ForEach-Object { if ($_.Name -match "-r(\d+)$") { [int]$Matches[1] } }
  return 1 + ($rounds | Measure-Object -Maximum).Maximum
}

Push-Location $d
foreach ($stage in $Stages) {
  switch ($stage) {
    "render" {
      GameDown
      if (-not (Step "collect-render" @("-m", "drone_model.collect.render"))) { Finish "render collection failed" }
    }
    "studio" {
      GameUp
      if (-not (Step "collect-studio" @("-m", "drone_model.collect.studio"))) { Finish "studio collection failed" }
    }
    "fields" {
      GameUp
      if (-not (Step "collect-fields" @("-m", "drone_model.collect.fields"))) { Finish "field collection failed" }
    }
    "reader" {
      GameDown
      $all = @("--data", "data\studio", "data\render", "data\fields")
      if ($FineTune) {
        $train = @("-m", "drone_model.train.reader") + $all + @("--init", "checkpoints\reader.pt", "--baseline", "checkpoints\reader.pt",
          "--steps", "15000", "--lr", "3e-4", "--out", "checkpoints\reader-candidate.pt")
        if (-not (Step "reader-candidate" $train)) { Finish "reader fine-tune failed" }
        Step "reader-promote" @("-m", "drone_model.train.promote_reader", "checkpoints\reader-candidate.pt", "--class", $ReaderClass) | Out-Null
      } else {
        $base = @("-m", "drone_model.train.reader", "--data", "data\studio", "data\render", "--steps", "60000", "--out", "checkpoints\reader-base.pt")
        if (-not (Step "reader-base" $base)) { Finish "reader base training failed" }
        $tune = @("-m", "drone_model.train.reader") + $all + @("--init", "checkpoints\reader-base.pt", "--baseline", "checkpoints\reader-base.pt",
          "--steps", "15000", "--lr", "3e-4", "--out", "checkpoints\reader.pt")
        if (-not (Step "reader" $tune)) { Finish "reader fine-tune failed" }
      }
    }
    "goto" {
      GameUp
      try { & "$PSScriptRoot\dagger_goto.ps1" -Round (NextRound "goto") } catch { Finish "goto round failed: $_" }
    }
    "skill" {
      GameUp
      try { & "$PSScriptRoot\dagger_skill.ps1" -Round (NextRound "skill") } catch { Finish "skill round failed: $_" }
    }
    "evaluate" {
      GameUp
      & "$PSScriptRoot\evaluate.ps1" -Episodes $EvalEpisodes
    }
    default { Finish "no stage named $stage" }
  }
}
Pop-Location
Finish "done: $($Stages -join ', ')"
