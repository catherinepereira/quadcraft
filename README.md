# Quadcraft

Pilotable drone for Minecraft Java whose camera feeds external programs, for recording flights and training models that fly it.

In your own world the drone runs jobs you set up with the tablet: copy a region, build a schematic, mine blocks, harvest a field, patrol a region or guard it from mobs, fly to a point, find a block, or follow you. A copy or build can mine its materials from a third region first. Each drone has a job queue and a battery, it flies back to its charging station to charge when the work left won't fit in the charge, using job costs it learns as it works. With BlueMap installed, the dashboard's Map tab shows the world with the regions and the drones on it. Training arenas cover the same jobs plus navigation, tool, and docking tasks, on flat, rough, or cave terrain.

Each drone works from its camera. A learned block reader labels every pixel, and the perception core builds a voxel memory of the blocks in view and tracks the mobs around the drone. A planner picks one goal at a time from that memory, and a learned policy flies it. Scan perception reads the job's blocks straight from the world in place of the camera.

| Folder | What it is |
| --- | --- |
| [mod](mod) | Fabric mod: drone, piloting, jobs, bridge server, recorder, HUD |
| [dashboard](dashboard) | Browser dev panel |
| [training](training) | `quadcraft` (bridge client, Gymnasium env, dataset loader, schematic files) and `drone_model` (perception core, planners, learned policies, the drone brain) |
| [docs](docs) | [PROTOCOL.md](docs/PROTOCOL.md), the bridge protocol all three parts follow, [SCOPE.md](docs/SCOPE.md), the plan and milestones, and [FUTURE.md](docs/FUTURE.md), known gaps |
| [scripts](scripts) | Start the containers and Minecraft |

## Generated files

| Folder | Written by |
| --- | --- |
| `training/data` | the mod's recordings in dev runs and policy training data, with the reader's frame cache in `data/cache` |
| `training/checkpoints`, `training/reports` | training and evaluation |
| `training/logs` | long training and collection runs |
| `mod/run/logs` | the mod's JSONL logs in dev runs, pruned after `logRetentionDays` |

All of them are gitignored. A normal install of the mod writes recordings and logs under `.minecraft/quadcraft/`.

## Quick start

Minecraft runs on the host, it needs the GPU and a window:

```powershell
scripts\minecraft.ps1          # dev client with the mod, create a creative superflat world
scripts\minecraft.ps1 -Arena   # or a test world with the drone and an arena ready for scripts
```

It uses a JDK 25 from `~/.jdks`, or `JAVA_HOME` when there isn't one there. In game, press N to start an episode, V to fly, B to record, and K for the job screen.

The dashboard and the training environment run in Docker (Docker Desktop with the NVIDIA GPU support for training):

```powershell
scripts\up.ps1                                              # build and start both, dashboard on http://localhost:5318
docker compose exec train python -m drone_model.brain       # run jobs started in game or on the dashboard, --scripted flies them without the learned policies
docker compose exec train bash                              # training shell in training/
```

Both containers reach the game's bridge through `host.docker.internal`. The training container mounts `training`, so data, checkpoints, and reports land on the host. `scripts/up.sh` and `scripts/minecraft.sh` do the same on Linux and macOS.

Without Docker, see each folder's README: `npm run dev` in the dashboard, and the `.venv` setup in training.
