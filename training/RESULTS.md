# Results

Measured on the current checkpoints: `reader.pt`, `goto.pt`, and `skill.pt`. How they were trained is in [docs/TRAINING.md](../docs/TRAINING.md).

## The whole drone

`scripts/evaluate.ps1 -Episodes 5`: the brain flies each training arena with the reader, the planners, and both learned policies, on the evaluation seeds from 100000, which no training data uses. Median steps count successful episodes only.

| Task | Terrain | Success | Median steps | Notes |
| --- | --- | --- | --- | --- |
| replicate_build | flat | 4 of 5 | 467 | 1 timeout |
| copy_build | flat | 3 of 5 | 2365 | 2 timeout |
| copy_build | rough | 0 of 5 | - | 5 timeout |

gather_build, goto_point, and dock_station haven't been evaluated with the current checkpoints yet.

The cell skill has one DAgger round of data on the square-on work pose (below), and it trails the planner it learns from. Flown by the scripted controller, the same planner finished 11 of 12 flat and 9 of 12 rough copies while collecting the skill's data. The skill it replaced, trained on several rounds with the old work geometry, scored replicate 5 of 5, copy flat 4 of 5, and copy rough 5 of 5 on 2026-10-27. More skill rounds are the next step.

## Work pose

Every break and place is worked from a fixed pose square-on to one face, see Planners in [docs/TRAINING.md](../docs/TRAINING.md#planners). Measured over every place of 2 scripted copy_build episodes on flat ground, seeds 100000 and 100001:

| | Before | Square-on work pose |
| --- | --- | --- |
| Camera distance to the face | 2.63 ± 0.05 | 2.50 ± 0.04 |
| Angle off north, south, east, or west | 24.5° mean, 40.7° at the 90th percentile | 0.0° |
| Places made square-on | 10% | 100% |
| Steps per copy | 964, 1039 | 1005, 1124 |

## Block reader

Scored on a held-out tenth of the game frames, which no training step saw: pixel accuracy 0.911 and mean IoU 0.396 over the classes present. The reader trained 60,000 steps on 226,754 rendered and 58,086 studio frames, then 15,000 steps with 3,448 frames of wheat fields added, which lifted mean IoU from 0.378. Small and far subjects read worst, and blocks learned mostly from rendered frames read worse on game frames than blocks learned from game frames, see [docs/FUTURE.md](../docs/FUTURE.md#perception).

## Pilot policies

| Policy | Training data | Best validation loss |
| --- | --- | --- |
| Goto policy | 1,220 episodes in 54 runs | 0.086 |
| Cell skill | 108 episodes in 8 runs, one round on the square-on work pose | 0.202 |

Validation loss is mean squared error on the move, plus the fire loss for the skill, on held-out episodes of the same data. It compares checkpoints of one policy on one dataset and says little across datasets, the tables above are the measure that matters.
