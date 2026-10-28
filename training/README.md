# training

Two Python packages. `quadcraft` is the client for the mod's bridge: a Gymnasium env for every task, a loader for recorded episodes, and `.schem` files. `drone_model` is the drone itself: the perception core with the block reader, the planners, the learned policies and the framework that trains them, and the brain that flies the player's copy and build jobs. How the models work and train is in [docs/TRAINING.md](../docs/TRAINING.md), results in [RESULTS.md](RESULTS.md).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -e ".[dev,render]"
```

The brain, collection, and evaluation need the game running with the mod and a world open, for example `scripts\minecraft.ps1 -Arena` from the repo root. Rendering and training don't.

## Layout

```
src/quadcraft/                 bridge client, Gymnasium env, episode loader, schematic files
src/drone_model/brain.py       flies the player's jobs, and the training arenas for evaluation
src/drone_model/energy.py      learned job costs, and the battery keeper that flies home to charge
src/drone_model/perception/    the perception core: block reader, voxel memory, world map
src/drone_model/scripted/      the planners: the shared Planner, A* routes, one planner per kind of job
src/drone_model/policies/      the goto policy and the cell skill, each a framework spec
src/drone_model/framework/     every policy's collection, training, and evaluation
src/drone_model/render/        the offline block renderer
src/drone_model/collect/       block reader frames: rendered and studio
src/drone_model/train/         block reader training and promotion
src/drone_model/evaluate/      reader checks, the evaluation summary, smoothness, embeddings, MP4s
scripts/                       the training pipeline, DAgger rounds, and the evaluation run
tests/
```

Generated files stay in this folder and are gitignored:

```
checkpoints/   reader.pt, goto.pt, skill.pt, energy.json, each with a JSON training report
data/          reader frames in render/ and studio/, policy episodes in policies/<policy>/<run>/
reports/       evaluation reports in brain/, eval/, and reader/
logs/          pipeline logs
```

## The brain

```powershell
.venv\Scripts\python -m drone_model.brain                                                          # fly every job started in game or on the dashboard
.venv\Scripts\python -m drone_model.brain --task gather_build --size 4 --episodes 5                # evaluate on a training arena
.venv\Scripts\python -m drone_model.brain --task copy_build --size 8 --perception scan --episodes 3
.venv\Scripts\python -m drone_model.brain --fleet replicate_build,dock_station --episodes 5        # one drone per task in a shared arena
.venv\Scripts\python -m drone_model.brain --task replicate_build --scripted                        # the planners' scripted controller in place of the learned policies
```

The brain watches the player's drones as an observer. When a job starts on a drone nobody flies, it starts a worker with its own controller connection for that drone, which runs the drone's queued jobs one after another and lets the drone go after 20 seconds without a new one. Each drone keeps one perception core for as long as the brain flies it, so a job starts out knowing what earlier jobs saw. Training arenas get a fresh core every episode. The memory streams to the dashboard's Drone memory panel as it changes.

`--skill`, `--goto`, and `--reader` pick other checkpoints.

### Battery

With the drone's battery on and its home station inside the job's geofence, a battery keeper (`energy.py`) rides along with every planner. At a job's start it estimates the job's cost from a learned charge per cell of the job's boxes for its kind of job. If the charge, less a reserve and the flight home, won't cover it, the drone charges first. Mid-job it turns home once the charge only covers the flight there plus the reserve, waits on the station for a full charge, and hands the drone back to the planner, which picks up where its memory left off. Each finished job moves the learned rates 30% of the way toward what it measured, saved to `checkpoints/energy.json`. Training arenas don't drain the battery.

## Training

`scripts\pipeline.ps1` runs every stage in order, see [The pipeline](../docs/TRAINING.md#the-pipeline). The commands it runs:

```powershell
.venv\Scripts\python -m drone_model.collect.render                     # rendered block frames, data/render, no game
.venv\Scripts\python -m drone_model.collect.render --preview 12        # a sheet of rendered frames beside their labels
.venv\Scripts\python -m drone_model.collect.studio                     # game frames of mobs and the blocks the renderer can't draw, data/studio
.venv\Scripts\python -m drone_model.train.reader --data data\studio data\render --steps 60000 --out checkpoints\reader.pt
.venv\Scripts\python -m drone_model.train.reader --data data\studio data\render --init checkpoints\reader.pt --baseline checkpoints\reader.pt --steps 15000 --lr 3e-4 --out checkpoints\reader-candidate.pt
.venv\Scripts\python -m drone_model.train.promote_reader checkpoints\reader-candidate.pt
powershell -File scripts\dagger_goto.ps1 -Round 6                         # a goto DAgger round
powershell -File scripts\dagger_skill.ps1 -Round 12 -Tasks gather_build   # a skill DAgger round on some tasks
.venv\Scripts\python -m drone_model.framework.collect --policy skill --task gather_build --size 4 --episodes 40
.venv\Scripts\python -m drone_model.framework.train --policy skill
```

With the game running, `python -m drone_model.perception.classes` rebuilds the reader's class list after a game update.

## Evaluation

```powershell
powershell -File scripts\evaluate.ps1 -Episodes 5                                               # the brain on every training arena, then the summary table
.venv\Scripts\python -m drone_model.evaluate.summary                                            # the table from the newest reports
.venv\Scripts\python -m drone_model.framework.evaluate --policy goto --task dock_station --terrain rough --episodes 10
.venv\Scripts\python -m drone_model.framework.evaluate --policy skill --task copy_build --teacher   # the scripted controller, for comparison
.venv\Scripts\python -m drone_model.evaluate.reader --episodes 10                               # the core's read of reference builds, cell by cell
.venv\Scripts\python -m drone_model.evaluate.smoothness --policy goto --task dock_station --terrain rough
.venv\Scripts\python -m drone_model.evaluate.video --policy skill --task copy_build   # MP4, the chase camera beside the drone's own
.venv\Scripts\python -m drone_model.evaluate.embeddings --checkpoint checkpoints\reader.pt --dest <folder>
```

Videos show the drone from behind with vanilla's third-person camera (the `chase` stream) next to what its camera sees, with depth inset, or with `--view drone` the drone's camera full size with depth and the mod's semantic mask. The embeddings export averages the reader's decoder features over each block or mob region in a set of frames, with a crop of each, laid out with UMAP in 2D and 3D for an embedding explorer.

## Client (`quadcraft`)

Clients connect to `127.0.0.1:8318`, or to the host in `QUADCRAFT_HOST`.

```python
from quadcraft import DroneClient, action

with DroneClient() as drone:
    drone.set_mode("lockstep")
    obs = drone.reset(seed=7)
    obs = drone.step(action(forward=1.0, yaw=5.0))
    print(obs.state["pos"], obs.rgb.shape, obs.episode["reward"])
```

```python
from quadcraft import DroneEnv

env = DroneEnv(streams=("rgb", "depth"), record=True)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

`record=True` saves episodes to the training data, `record="test"` to the game's `quadcraft/test-recordings` for runs that shouldn't train anything. Recording stops when the env closes.

The env's action is a dict: `move` (`[forward, right, up, yaw, pitch]` in [-1, 1], with yaw and pitch scaled to 15 degrees per tick), `tool` (an index into `quadcraft.protocol.TOOLS`), `slot`, and `transfer` (`[kind, slot]`, kind 1 stores a drone stack in the open container, 2 takes one out). An optional `block`, such as `"minecraft:bricks"`, names the block to place, and the drone uses whichever slot holds it.

Observations hold the image streams, a 6-value state vector (velocity, sin and cos of yaw, pitch), `bounds`, the drone's position inside the geofence from -1 to 1 per axis, and `range`, the range sensors ahead and below as a share of their reach, 1 when nothing is in range. Tool tasks add `inventory` and `container` as `[item index, count]` per slot (indices into `DroneClient.item_ids`) and `tool_state` (mining progress, container open, selected slot). Leaving the geofence, or the drone getting wrecked by mobs, ends the episode as a failure, and `info["episode"]` tells them apart. Jobs put the drone's instruction in `info["state"]["job"]`, see [docs/PROTOCOL.md](../docs/PROTOCOL.md).

Schematics (`.schem`, the Sponge format WorldEdit uses):

```python
from quadcraft import schematic

s = schematic.load("house.schem")
s.size, s.get(0, 0, 0)  # (width, height, length), "minecraft:oak_planks"
s.save("copy.schem")
```

Recorded episodes:

```python
from quadcraft import list_episodes, iter_transitions

episodes = list_episodes("data", "goto_point", outcome="success")
for sample in iter_transitions(episodes):
    sample["rgb"], sample["depth"], sample["action"]
```

`DroneClient(log_dir=Path("logs"))` writes `py-<session>.jsonl` in the same line shape the mod uses.

## Tests

```powershell
.venv\Scripts\python -m pytest                                              # against a fake bridge, no game needed
$env:QUADCRAFT_E2E = "1"; .venv\Scripts\python -m pytest tests\test_e2e.py   # against a running game
```
