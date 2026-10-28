# Quadcraft

A camera drone for Minecraft Java that copies builds. It reads a structure with its camera into a schematic and rebuilds it somewhere else, or builds a schematic file at a point you pick.

You set up jobs with the drone's tablet: copy a region, build a schematic, or fly home to charge. A copy or build takes its blocks from the drone's inventory and can dig the missing ones from a materials region first. Each drone has a job queue and a battery, and flies back to its charging station when the work left won't fit in its charge. You can also fly it yourself, with a camera gimbal that looks ahead or straight down. A bridge lets programs outside the game read the camera and fly the drone.

The drone works only from its own camera. A learned block reader labels every pixel of each frame, a perception core turns the labels into a voxel memory of the world, a planner picks one goal at a time from that memory, and a learned policy flies the goal. [docs/TRAINING.md](docs/TRAINING.md) covers how each part works and trains.

| Folder | What it is |
| --- | --- |
| [mod](mod) | Fabric mod: the drone, piloting, jobs, training arenas, the bridge server, recording, the HUD |
| [dashboard](dashboard) | Browser panel: live camera streams, control, the drone's memory, logs, episodes, metrics, config |
| [training](training) | `quadcraft`, the bridge client and Gymnasium env, and `drone_model`, the perception core, planners, learned policies, and the drone brain |
| [docs](docs) | [TRAINING.md](docs/TRAINING.md), how the models work and train, [PROTOCOL.md](docs/PROTOCOL.md), the bridge protocol every part follows, [FUTURE.md](docs/FUTURE.md), known limitations |
| [scripts](scripts) | Start Minecraft and the containers |

## Platform

| Item | Version |
| --- | --- |
| Minecraft | Java 26.3, singleplayer |
| Loader | Fabric Loader 0.19.5, Fabric API 0.161.0+26.3 |
| JDK | 25 |
| Training | Python 3.11, PyTorch with CUDA, a GPU with 12 GB |

| Port | Used by |
| --- | --- |
| 8318 | The mod's bridge: WebSocket for control and streams, HTTP for logs, episodes, and config |
| 5318 | The dashboard, which proxies `/api` and `/ws` to 8318 |

## Quick start

Minecraft runs on the host, it needs the GPU and a window:

```powershell
scripts\minecraft.ps1          # dev client with the mod, create a creative superflat world
scripts\minecraft.ps1 -Arena   # or a test world with a drone and an arena ready for scripts
```

It uses a JDK 25 from `~/.jdks`, or `JAVA_HOME` when there isn't one there. In game, V flies the drone, G swings its camera down or level, K opens the job screen, N starts a training episode, and B records. See [mod/README.md](mod/README.md) for every key, the jobs, and the config.

The drone's brain and the dashboard run in Docker (Docker Desktop with NVIDIA GPU support):

```powershell
scripts\up.ps1                                              # build and start both, dashboard on http://localhost:5318
docker compose exec train python -m drone_model.brain       # fly every job started in game or on the dashboard
docker compose exec train bash                              # a shell in training/
```

Both containers reach the game's bridge through `host.docker.internal`. The training container mounts `training`, so data, checkpoints, and reports land on the host. `scripts/up.sh` and `scripts/minecraft.sh` do the same on Linux and macOS. Without Docker, see [training/README.md](training/README.md) for the `.venv` setup and [dashboard/README.md](dashboard/README.md) for `npm run dev`.

## Models

| File | What it is |
| --- | --- |
| `training/checkpoints/reader.pt` | The block reader, a ConvNeXt-Tiny U-Net over RGB and depth at 320x240, 1303 classes |
| `training/checkpoints/goto.pt` | The goto policy, flies to a point and home to charge |
| `training/checkpoints/skill.pt` | The cell skill, flies the views, breaks, and places of copies and builds |
| `training/checkpoints/energy.json` | Learned battery cost per kind of job, updated as jobs finish |

Results on every training arena are in [training/RESULTS.md](training/RESULTS.md).

## Generated files

| Folder | Written by |
| --- | --- |
| `training/data` | reader training frames, policy episodes, and the mod's recordings in dev runs |
| `training/checkpoints`, `training/reports` | training and evaluation |
| `training/logs` | the training pipeline |
| `mod/run/logs` | the mod's JSONL logs in dev runs, pruned after `logRetentionDays` |

All of them are gitignored. A normal install of the mod writes recordings and logs under `.minecraft/quadcraft/`.
