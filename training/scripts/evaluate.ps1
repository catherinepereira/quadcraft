# The whole drone, reader, planners, and both learned policies, on every training arena, -Episodes each, one run at a time
# Needs the game running with the bridge open. Reports land in reports/brain/, evaluate.summary turns them into a table
param([int]$Episodes = 5)
$model = Split-Path $PSScriptRoot -Parent
$python = Join-Path $model ".venv\Scripts\python.exe"

# task, size, terrain
$runs = @(
  @("replicate_build", 3, "flat"),
  @("copy_build", 5, "flat"),
  @("copy_build", 5, "rough"),
  @("mine_deposit", 5, "flat"),
  @("mine_deposit", 5, "rough"),
  @("gather_build", 4, "flat"),
  @("harvest_crops", 5, "flat"),
  @("hunt_mobs", 5, "flat"),
  @("hunt_mobs", 5, "rough"),
  @("hunt_mobs", 5, "cave"),
  @("patrol_area", 5, "flat"),
  @("patrol_area", 5, "rough"),
  @("goto_point", 5, "flat"),
  @("goto_point", 5, "rough"),
  @("goto_point", 5, "cave"),
  @("find_block", 5, "flat"),
  @("find_block", 5, "rough"),
  @("find_block", 5, "cave"),
  @("follow_mob", 5, "flat"),
  @("follow_mob", 5, "rough"),
  @("dock_station", 5, "flat"),
  @("dock_station", 5, "rough"),
  @("dock_station", 5, "cave")
)
$started = Get-Date
foreach ($r in $runs) {
  $line = & $python -u -m drone_model.brain --task $r[0] --size $r[1] --terrain $r[2] --episodes $Episodes 2>&1 | Select-String "flown by|Traceback|Error"
  "$(Get-Date -Format HH:mm:ss) $($r[0]) s$($r[1]) $($r[2]): $line"
}
& $python -m drone_model.evaluate.summary --since $started.ToString("yyyyMMdd-HHmmss")
