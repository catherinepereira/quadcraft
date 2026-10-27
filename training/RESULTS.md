# Results

Measured on the v1 checkpoints: `reader.pt`, `goto.pt`, and `skill.pt`. How they were trained is in [docs/TRAINING.md](../docs/TRAINING.md).

## The whole drone

`scripts/evaluate.ps1 -Episodes 5`: the brain flies each training arena with the reader, the planners, and both learned policies, on the evaluation seeds from 100000, which no training data uses. Median steps count successful episodes only.

| Task | Terrain | Success | Median steps | Notes |
| --- | --- | --- | --- | --- |
| copy_build | flat | 4 of 5 | 1914 | 1 timeout |
| copy_build | rough | 5 of 5 | 2016 |  |
| dock_station | cave | 5 of 5 | 50 |  |
| dock_station | flat | 5 of 5 | 41 |  |
| dock_station | rough | 5 of 5 | 41 |  |
| find_block | cave | 5 of 5 | 90 |  |
| find_block | flat | 5 of 5 | 69 |  |
| find_block | rough | 4 of 5 | 138 | 1 timeout |
| follow_mob | flat | 1 of 5 | 496 | 4 timeout |
| follow_mob | rough | 2 of 5 | 1492 | 3 timeout |
| gather_build | flat | 4 of 5 | 3544 | 1 timeout |
| goto_point | cave | 5 of 5 | 46 |  |
| goto_point | flat | 5 of 5 | 39 |  |
| goto_point | rough | 5 of 5 | 39 |  |
| harvest_crops | flat | 5 of 5 | 917 |  |
| hunt_mobs | cave | 5 of 5 | 1189 | 100% of prey killed |
| hunt_mobs | flat | 3 of 5 | 586 | 87% of prey killed, 2 timeout |
| hunt_mobs | rough | 2 of 5 | 1529 | 80% of prey killed, 2 timeout, 1 wrecked |
| mine_deposit | flat | 5 of 5 | 2716 |  |
| mine_deposit | rough | 5 of 5 | 4471 |  |
| patrol_area | flat | 4 of 5 | 183 | 1 timeout |
| patrol_area | rough | 2 of 5 | 523 | 3 timeout |
| replicate_build | flat | 4 of 5 | 611 | 1 timeout |

95 of 115 episodes succeed. By policy, the goto policy's tasks reach 47 of 55 and the cell skill's 48 of 60:

| Group | Success |
| --- | --- |
| goto_point, dock_station | 30 of 30 |
| find_block | 14 of 15 |
| follow_mob | 3 of 10 |
| copy_build, replicate_build, gather_build | 17 of 20 |
| mine_deposit, harvest_crops | 15 of 15 |
| hunt_mobs | 10 of 15, 89% of prey killed |
| patrol_area | 6 of 10 |

Hunts and follows keep the world running between steps so their targets move, so their episodes don't replay exactly from a seed, and a harvest arena's ripe crops vary between runs. Expect an episode or two of difference between runs of the same configuration.

## Block reader

Scored on a held-out tenth of the game shards, studio and field frames, which no training step saw.

| Metric | Base (`reader-base.pt`) | v1 (`reader.pt`) |
| --- | --- | --- |
| Pixel accuracy | 0.883 | 0.911 |
| Mean IoU over the classes present | 0.378 | 0.396 |
| Ripeness accuracy on crop pixels | 0.44 | 0.971 |
| Wheat IoU | 0.171 | 0.921 |

The base reader trained 60,000 steps on 226,754 rendered and 58,086 studio frames. The fine-tune added 3,448 field frames for 15,000 steps. Small and far subjects read worst, and blocks learned mostly from rendered frames read worse on game frames than blocks learned from game frames, see [docs/FUTURE.md](../docs/FUTURE.md#perception).

## Pilot policies

| Policy | Training data | Best validation loss |
| --- | --- | --- |
| Goto policy | 1,700 episodes in 92 runs | 0.053 |
| Cell skill | 1,209 episodes in 78 runs | 0.387 |

Validation loss is mean squared error on the move, plus the fire loss for the skill, on held-out episodes of the same data. It compares checkpoints of one policy on one dataset and says little across datasets, the table above is the measure that matters.
