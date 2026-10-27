# Training

How the drone's models fit together, where their training data comes from, how each one trains, and how to reproduce the v1 checkpoints. Commands run from `training/` with its `.venv`, see [training/README.md](../training/README.md) for setup.

## Overview

The drone flies from its own camera. Each game tick it gets an RGB frame, a depth frame, and its own pose and motion, and nothing else about the world.

```
camera rgb + depth ──> block reader ──> perception core ──> planner ──> goal ──> pilot policy ──> action ──> mod
                       (learned)        voxel memory        (scripted)          goto or skill
                                        world map                               (learned)
                                        mob tracker
```

| Part | Kind | File | What it does |
| --- | --- | --- | --- |
| Block reader | learned | `checkpoints/reader.pt` | Labels every pixel with one of 1303 classes, every block a camera can show and every mob kind, and whether a crop is ripe |
| Perception core | code | `perception/core.py` | Back-projects the labels through depth into a voxel memory, a map of free space, and a set of tracked mobs |
| Planners | code | `scripted/` | Read the core and name one goal at a time, such as look at a point, break this block, or fly there |
| Goto policy | learned | `checkpoints/goto.pt` | Flies the goto planner's goals: a point, a found block, a followed mob or player, the next search point, the flight home |
| Cell skill | learned | `checkpoints/skill.pt` | Flies every other planner's goals and decides when to fire the tool |
| Battery costs | learned online | `checkpoints/energy.json` | Charge per cell for each kind of job, updated as jobs finish, see Battery in the training README |

The reader trains from labeled frames. The two pilot policies train by imitation: each planner also carries a scripted controller that can fly its goals, and that controller labels every step the policies learn from.

No part of the drone sees the true world. The bridge state carries the drone's pose, inventory, range sensors, and the job it was given, and the planners and policies read only that and the core. The true arena layout reaches only scoring and data collection, through the bridge's `arena` request. This holds for the teachers too, so the policies learn from a teacher that knew only what the drone could know.

## The drone's loop

`Planner.act` (`scripted/base.py`) runs once per step:

1. The perception core takes the frame. The reader labels it, the voxel memory and world map take the labels through depth, and the mob tracker groups mob pixels into mobs.
2. The planner's `decide` reads the core and sets its `intent`, the goal for this step: a mode (`view`, `break`, `place`, `attack`, `goto`, `follow`, or `explore`), an aim point, and for some modes a viewpoint, a standoff, or a hover height.
3. The scripted controller computes the action that flies the intent.
4. With a pilot attached, the pilot flies the intent in place of the scripted action. The scripted controller still runs, so the planner's state moves on exactly as it did while it taught the pilot.

Training and evaluation run the game in lockstep mode, frozen between steps, so a slow model never falls behind the world.

## Perception

### Block reader

`perception/reader.py`. A ConvNeXt-Tiny encoder pretrained on ImageNet, with a U-Net decoder, 32.3M parameters in all.

- Input: the 320x240 RGB frame with ImageNet normalization, and depth as a fourth channel, clipped at 64 blocks, scaled to 0 to 1, and centered on 0.5.
- The decoder upsamples the encoder's four stages back to full size. ConvNeXt starts at stride 4, so two extra side stages convolve the input at half and full resolution for the last two skip connections, which keeps thin blocks and far mobs sharp.
- Two heads over the full-size features: a 1x1 convolution to the 1303 class logits, and one to a ripeness logit, read only on crop pixels.
- Classes: `<sky>`, `<other>`, every block from the game's registry, and every mob kind. `python -m drone_model.perception.classes`, with the game running, rebuilds `perception/classes.json` after a game update. Blocks that look exactly like another, such as infested stone or waxed copper, are labeled as that block. Every shard records a hash of the class list, and training leaves out shards made for another list.

Without a reader checkpoint the core reads the mod's mask stream instead, which is ground truth. That mode is for debugging and for data the reader can't label yet (`--teacher-sees mask`).

### Voxel memory

`perception/memory.py`. Each labeled pixel is back-projected through depth into the block it shows, stepping 0.15 blocks past the hit so a face lands in the block that owns it. Each cell keeps votes per class and for ripeness, and the class with the most votes is what the drone believes is there. Break and place events update a cell at once. Planners ask the memory for a box's blocks, for every cell of a block kind, and whether a cell is solid, which comes from the game's collision shapes.

### World map

`perception/worldmap.py`. The free space to fly through, per column: a hit below the camera raises the column's known floor, a hit above lowers its known ceiling, and a ray that crosses a cell marks it empty. A climb that stalls marks a ceiling over the drone, and a bump with the front range sensor marks the column ahead. Planners route over it with A* (`scripted/route.py`).

### Mob tracker

`perception/mobs.py`. Mob pixels back-projected through depth and grouped into mobs, one kind at a time. Hits in cells the memory has as solid are left out, which drops a mob outline spilling onto the ground. Each mob keeps its identity from frame to frame and the kind it was labeled as most often, and is forgotten once the camera looks at its spot and sees past it.

### Crosshair

`DroneCore.aim` works out what the crosshair rests on from the depth at the middle of the frame and the reader's class there: the cell, the face, and the distance. Planners fire a tool only when it reports the right cell within reach.

### Scan perception

A job started with scan perception carries the server's read of its boxes, and the core loads them into the memory as certain reads. Planners then skip their surveys. This is a player option for exact copies, the training data for the policies comes from vision episodes.

## Reader training data

Three collectors write frames in one shard format, `data/<source>/shard-NNNN/` with `rgb.npy`, `depth.npy` (float16), `cls.npy`, `ripe.npy`, and `meta.json`, which names the class list hash, the frame size, the source (`game` or `render`), and each frame's subject. A shard is finished once its `done` file exists, and a stopped collector picks up at the first unfinished shard.

| Source | Collector | v1 size | Labels from | Needs |
| --- | --- | --- | --- | --- |
| `data/render` | `collect.render` | 226,754 frames in 56 shards, about 130 GB | the renderer itself | GPU, no game |
| `data/studio` | `collect.studio` | 58,086 frames in 15 shards, about 34 GB | the mod's `state` stream (block state per pixel) and the `mask` stream's entity ids | the game |
| `data/fields` | `collect.fields` | 3,448 frames in 8 shards, about 2 GB | the same streams | the game |

### Studio scenes in the game

For each subject the mod's studio arena (`mod/.../task/Studio.java`) stages a small scene: the subject with random neighbors, ground, walls, light, and time of day, on flat, rough, or cave terrain, and picks up to 8 camera poses around it at random angles and distances. The collector captures a frame from each pose and keeps those where the subject covers at least 12 pixels. The default plan is 40 scenes of every mob, 20 of every block the offline renderer can't draw, and 2 of every other block, so every block has some game frames for validation. The plan is fixed by the seed (300000).

### Offline rendering

Most block frames are drawn without the game. `drone_model.render` reads block models, textures, and colormaps from the game's client jar in Gradle's Loom cache and reproduces the game's look with moderngl: biome tints, smooth lighting and ambient occlusion over flooded sky and block light, the lightmap formula, animated texture frames, cutout and translucent passes, water and lava, and 2x supersampling box-filtered down. A label pass gives class, ripeness, and z-depth from each model element's box. Scenes follow `Studio.java`'s staging. The default is 50 scenes of each of the 1134 blocks it can draw, 4 views each. Each worker process stages and meshes scenes on the CPU with its own GPU context, 3 workers draw about 70 frames a second and the default runs 8. `--preview 12` writes a sheet of frames beside their labels. Blocks drawn by their own renderers (chests, heads, banners, shulker boxes, decorated pots) and every mob come only from studio frames.

### Field frames

`collect.fields` builds harvest arenas and captures each wheat field from 24 random poses, 1 to 7 blocks over the crops and up to 8 blocks past the field's edge, all inside the geofence. It keeps frames with at least 12 wheat pixels. This is the data that teaches the reader crops at the distances and angles the harvest planner flies. `--tasks follow_mob,hunt_mobs` captures the walled mob arenas instead, into `data/arenas`, but v1 doesn't use it.

## Reader training

`python -m drone_model.train.reader`

- Split: a tenth of the game shards (studio and fields) by shard for validation, the rest of them and every rendered shard for training. The score is always on game frames, which is what the drone sees.
- Batches of 8 frames, read on two background threads into pinned memory. Augmentation: a random horizontal flip and brightness scaled by 0.7 to 1.3.
- Class loss: cross-entropy on a random quarter of each frame's pixels, weighted by the inverse square root of each class's frequency, so floor and sky don't drown out a single block. Full-size logits over 1303 classes for every pixel don't fit in memory.
- Ripeness loss: binary cross-entropy on crop pixels.
- AdamW with weight decay 1e-4 and a one-cycle schedule, the encoder at a tenth of the decoder's learning rate. Mixed precision on CUDA.
- Every 2500 steps it scores the validation shards (pixel accuracy, mean IoU over the classes present, ripeness accuracy, per-class IoU) and keeps the checkpoint with the best mean IoU. The report goes beside the checkpoint as JSON.
- About 7.5 GB of GPU memory at batch 8, which leaves room for the desktop on a 12 GB card. `--gpu-share 0.4` keeps the GPU idle 60% of the time, for training while the game runs.

### The v1 reader

Two stages:

| Stage | Command arguments | Steps | Time (RTX 4070 SUPER) | Result |
| --- | --- | --- | --- | --- |
| Base | `--data data\studio data\render --steps 60000 --out checkpoints\reader-base.pt`, from ImageNet weights, learning rate 1e-3 | 60,000 | about 2 h 15 min | mean IoU 0.362 on studio validation frames |
| Fine-tune | `--data data\studio data\render data\fields --init checkpoints\reader-base.pt --baseline checkpoints\reader-base.pt --steps 15000 --lr 3e-4 --out checkpoints\reader.pt` | 15,000 | about 35 min | see below |

On the same validation frames, field frames included:

| | Base | v1 |
| --- | --- | --- |
| Pixel accuracy | 0.883 | 0.911 |
| Mean IoU | 0.378 | 0.396 |
| Ripeness accuracy | 0.44 | 0.971 |
| Wheat IoU | 0.171 | 0.921 |

`--init` loads a trained reader's weights, matching its class head row by row by class name, so a reader trained for an older class list still starts the new one. `--baseline` scores another checkpoint on the same validation frames before training and stores it in the report.

### Promoting a later reader

`python -m drone_model.train.promote_reader checkpoints\reader-candidate.pt --class minecraft:wheat` copies a fine-tuned candidate over `reader.pt` only when, against its baseline on the same frames, it labels the named class better and loses at most 0.005 mean IoU. It exits 1 when it keeps the baseline.

## Planners

Planners are the teachers and, in the brain, the part that decides what to do. Each is a `Planner` subclass, built per job by `scripted.jobs.make_planner` from the job's `kind`.

| Planner | File | Jobs and arenas |
| --- | --- | --- |
| `BuildPlanner` | `jobs.py` | copy, build from a schematic, either one gathering its materials from a box first (copy_region, build_schematic, replicate_build, copy_build, schematic_build, gather_build) |
| `MinePlanner` | `jobs.py` | mine the blocks of named kinds in a box (mine_region, mine_deposit) |
| `HarvestPlanner` | `jobs.py`, `harvest.py` | harvest ripe crops and replant, plant bare farmland (harvest_region, harvest_crops) |
| `PatrolPlanner` | `patrol.py` | sweep a region round after round, a guard also kills its prey (patrol_region, guard_region, patrol_area, hunt_mobs) |
| `GotoPlanner` | `goto.py` | fly to a point, find a block of a named kind, follow a player or a mob (fly_to, seek_block, follow_player, goto_point, find_block, follow_mob) |
| `DockPlanner` | `jobs.py` | fly home and land on the charging station (return_home, dock_station) |

What every planner shares, in `scripted/base.py`:

- Search: until it knows where to go, it turns on the spot, then visits a grid of points inside the geofence, nearest first, sweeping the camera at each.
- Flight: A* over the world map around tall columns and under low ceilings, at cruise height, with the scripted controller steering toward the goal and easing the camera onto the aim point.
- Work: `work_on` flies to a block, aims at it, and fires once the crosshair (`DroneCore.aim`) rests on that cell within reach. It gives a block up after 150 steps, and when the crosshair shows the remembered block isn't there.
- Getting unstuck: backing off and turning when told to move and not moving, and marking ceilings and blocked columns it bumps into.

A few behaviors worth knowing:

- A copy surveys its source from views around and over it, and the memory's read of the source is the plan. It clears the destination top down, then places bottom up, each block against a face already in place.
- Mining breaks the target blocks it can see, then digs trenches two wide every four blocks, which gives every block of the box an open face, and breaks what comes into view.
- Harvest looks the field over from above, harvests and replants one crop at a time, then sweeps the plots with no known crop and plants a plot when the majority of 3 aimed reads, or the memory, has it as bare farmland.
- A follow job names the mob's kind and either nothing, for the one nearest the drone once it has looked all around, or an `at` point, for the one nearest that point, which it flies to first.

## Pilot policies

Both policies are small CNNs over the camera at 80x60 and flat vectors, and both output the five flight commands `[forward, right, up, yaw, pitch]` in [-1, 1].

| | Goto policy (`policies/goto.py`) | Cell skill (`policies/skill.py`) |
| --- | --- | --- |
| Image | RGB and depth averaged down to 80x60, depth as 0 to 255 over 64 blocks | the same |
| Encoder | four conv blocks (32, 64, 96, 128 channels, stride 2, GroupNorm), pooled to 3x4, linear to 256 | the same |
| State | 14 values: velocity, sin and cos of yaw, pitch, position inside the geofence, mining progress, container open, selected slot, and the front and down range sensors | the first 12 of those |
| Goal | 10 values: the goal relative to the camera in the heading frame, its distance across the ground and bearing, standoff, hover, and the mode (goto, follow, explore) | 11 values: the aim point and the viewpoint relative to the camera in the heading frame, whether there is a viewpoint, and the mode (view, break, place, attack) |
| Extra inputs | none | 5 aim features worked out from the inputs: yaw and pitch error to the aim point, distance, how far past or short of it the crosshair's depth lands, and whether it is within reach |
| Heads | MLP 256, 128 to the move, plus a linear path straight from the goal to the move | the same, the direct path also taking the aim features, plus a fire logit |
| Loss | mean squared error on the move | mean squared error on the move, plus binary cross-entropy on fire for the tool modes |
| Acting | the move | the move, and the intent's tool when the fire probability is over 0.5 |

The direct linear path lets the policy learn proportional steering toward the goal before the image features are useful. On steps where the planner has no intent, such as a turn on the spot while searching, the planner's own action flies.

## How the pilots learn

The pilots learn by imitation through `drone_model.framework`.

### Labels

Each training row is one step of an episode the planner works, labeled with the scripted controller's clean action for that step. What flies is that action plus Gaussian noise of 0.2 on the five flight commands, so the data covers the drone recovering from being pushed off course (DART, Laskey et al. 2017). Steps where the planner is using a tool fly clean, since a nudged aim makes the mod reset a dig. Tool use and item transfers are never noised.

### Training

`python -m drone_model.framework.train --policy <goto|skill>`

- Every episode of the policy under `data/policies/<policy>/`, one `.npz` file per episode, split into training and validation by episode, so validation never sees a frame from a training episode.
- Every 2nd step of an episode (`stride` 2), neighboring frames are nearly the same.
- Batches of 128 rows, drawn task by task in proportion to the square root of each task's row count, so 6000-step hunts don't crowd 600-step builds out of the batches. The task is the run folder's first word.
- A smoothness penalty, 0.2 times `relu(-turn_t * turn_t+1)` summed over yaw and pitch for each row and the step after it, so turning back and forth costs and turning steadily doesn't.
- Brightness jitter, AdamW at 5e-4 with weight decay 1e-4, one-cycle schedule, gradient clipping at 1, 8000 steps. Each training run starts fresh and trains on all the policy's data. The checkpoint with the best validation loss is kept.

### DAgger rounds

A policy that only ever saw its teacher's states drifts into states the teacher never visited. Each round adds data from the policy's own flights (DAgger, Ross et al. 2011):

1. Planner episodes: the planner flies with flight noise, every step labeled.
2. Train on all the data so far.
3. DAgger episodes: the new checkpoint flies half the steps, picked at random, and the planner still labels every step.
4. Train on all the data again.

`scripts/dagger_goto.ps1 -Round N` and `scripts/dagger_skill.ps1 -Round N` run one round. Runs land in `data/policies/<policy>/<task>-<terrain>[-s<size>]-<plan|dagger>-r<N>/`, and the round number picks the seeds, so no two rounds fly the same arenas and no collection seed overlaps the evaluation seeds from 100000. `-SkipPlan` starts from step 2, `-PlanOnly` stops after it, and the skill script's `-Tasks` narrows a round to some tasks.

Goto round, about 1 h 50 min:

| Task | Terrain | Episodes per half | Step limit |
| --- | --- | --- | --- |
| goto_point | flat, rough, cave | 30 each | 600 |
| find_block | flat, rough | 15 each | 1500 |
| find_block | cave | 10 | 1500 |
| follow_mob | flat, rough | 10 each | 1500 |
| dock_station | flat, rough | 15 each | 600 |
| dock_station | cave | 10 | 600 |

Skill round, about 10 hours:

| Task | Terrain | Size | Episodes per half | Step limit |
| --- | --- | --- | --- | --- |
| replicate_build | flat | 3 | 20 | 900 |
| copy_build | flat, rough | 5 | 12 each | 4000 |
| mine_deposit | flat, rough | 5 | 10 each | 4000 |
| gather_build | flat | 4 | 10 | 6000 |
| harvest_crops | flat | 5 | 12 | 1500 |
| hunt_mobs | flat | 5 | 20 | 6000 |
| hunt_mobs | rough | 5 | 15 | 6000 |
| hunt_mobs | cave | 5 | 10 | 6000 |
| patrol_area | flat, rough | 5 | 10 each | 1500 |

The v1 policies trained on all the rounds collected so far: 1,700 goto episodes in 92 runs (best validation loss 0.053) and 1,209 skill episodes in 78 runs (best validation loss 0.387). Validation loss compares checkpoints of one policy on one dataset, it says little across datasets.

## The pipeline

`scripts/pipeline.ps1` runs the whole process, the stages given in order:

```powershell
powershell -File scripts\pipeline.ps1                                      # everything, from rendering to evaluation
powershell -File scripts\pipeline.ps1 -Stages fields,reader -FineTune       # new field frames and a gated reader fine-tune
powershell -File scripts\pipeline.ps1 -Stages goto,skill,evaluate           # the next DAgger round of each policy, then evaluation
```

| Stage | Does | Game | Time |
| --- | --- | --- | --- |
| render | `collect.render` | closed | about 1 h |
| studio | `collect.studio` | open | about 2 h |
| fields | `collect.fields` | open | a few minutes |
| reader | base and fine-tune, or with `-FineTune` a gated fine-tune of the current reader | closed | about 3 h, or 35 min |
| goto | the next goto round | open | about 2 h |
| skill | the next skill round | open | about 10 h |
| evaluate | `scripts/evaluate.ps1` | open | about 4 h |

The script starts the game for the stages that need it (`mod/scripts/regame.ps1`, the client gametest held open after its checks pass) and closes it before the reader trains, which wants the whole GPU. A failed gametest or stage stops the run. Each round takes the next number after the highest one in the policy's data folders. Logs go to `logs/pipeline-<time>.log`, with each step's full output in `logs/<step>.log`.

Disk: about 130 GB for rendered frames, 34 GB for studio frames, and 33 GB for the policies' episodes.

## Evaluation

`scripts/evaluate.ps1 -Episodes 5` flies the whole drone, reader, planners, and both policies, through `drone_model.brain` on 23 arena configurations: every training task on the terrains it trains on. Every run uses the evaluation seeds from 100000, which no collection uses. Each run writes a report to `reports/brain/`, and `drone_model.evaluate.summary` turns the newest of each into the table in [training/RESULTS.md](../training/RESULTS.md).

Narrower checks:

```powershell
.venv\Scripts\python -m drone_model.framework.evaluate --policy goto --task find_block --terrain rough --episodes 10    # one policy flying its teacher's goals
.venv\Scripts\python -m drone_model.framework.evaluate --policy goto --task find_block --terrain rough --teacher       # the scripted controller, for comparison
.venv\Scripts\python -m drone_model.brain --task copy_build --scripted --episodes 5                                    # the brain without the learned pilots
.venv\Scripts\python -m drone_model.evaluate.reader --episodes 10                                                      # the core's read of reference builds, cell by cell
.venv\Scripts\python -m drone_model.evaluate.smoothness --policy skill --task hunt_mobs --episodes 3                    # turn reversals and jerk
.venv\Scripts\python -m drone_model.evaluate.video --policy goto --task find_block --terrain rough                      # an MP4 of an episode
```

## Reproducing v1 from scratch

With the mod built, the game launchable (`mod/scripts/regame.ps1`), and about 200 GB free:

1. `pipeline.ps1 -Stages render,studio,fields,reader` for the reader, about 6 hours.
2. Goto and skill rounds until the policies stop improving on `evaluate.ps1`. v1 took the data of several rounds of each, with the run lists above, so plan on a few days of GPU time. `pipeline.ps1 -Stages goto,skill,evaluate` runs one round of each.

Arena layouts depend on the seed, but mobs keep moving between steps, so hunts and follows don't replay exactly from a seed. Results differ by an episode or two from run to run.

## Adding a policy

A policy is a `PolicySpec` subclass (`framework/spec.py`) registered in `framework/registry.py`:

| Piece | What it does |
| --- | --- |
| `model()` | the `torch.nn.Module` to train |
| `teacher(task, info, mask_ids, reader)` | the planner that labels every step, the job planner for the task by default |
| `record(step)` | one training row from a `Step` (state, observation, the teacher's action, the teacher) as named arrays, or `None` to skip the step |
| `loss(model, batch)` | the loss under `"loss"`, plus anything worth printing |
| `agent(model, mask_ids)` | an `Agent` that flies the trained model, `reset()` per episode and `act(step)` per step |

Optional settings: `streams` (what the env sends), `batch_size`, `stride`, `augment(batch)`, and `smoothness` with `look(model, batch)`, which returns the predicted yaw and pitch. A new behavior with no task to practice on needs a training arena in the mod: a `TaskKind`, a case in `TrainingArenas.placeTask`, metrics in `ArenaRecord`, and scoring in `TaskScorer`, see [PROTOCOL.md](PROTOCOL.md). `tests/test_framework.py` runs every registered policy through record, loss, and act on a made-up step.

```powershell
.venv\Scripts\python -m drone_model.framework.collect --policy <name> --task goto_point --episodes 100
.venv\Scripts\python -m drone_model.framework.train --policy <name>
.venv\Scripts\python -m drone_model.framework.collect --policy <name> --task goto_point --episodes 50 --dagger checkpoints\<name>.pt
.venv\Scripts\python -m drone_model.framework.train --policy <name>
.venv\Scripts\python -m drone_model.framework.evaluate --policy <name> --task goto_point
```
