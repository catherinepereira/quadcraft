"""
Collects a policy's training data. Its teacher flies the task with flight noise, none while it uses a tool, and labels
every step with its clean action (DART, Laskey et al. 2017).
With --dagger a trained policy flies a share of the steps and the teacher still labels every one, so the data covers
the states the policy gets itself into (DAgger, Ross et al. 2011)

    python -m drone_model.framework.collect --policy skill --task hunt_mobs --episodes 40
    python -m drone_model.framework.collect --policy skill --task hunt_mobs --dagger checkpoints/skill.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from quadcraft import DroneEnv

from ..paths import CHECKPOINTS
from ..perception.reader import FRAME, Reader
from ..scripted.base import TOOLS
from ..torch_utils import pick_device
from .data import run_dir, save_episode
from .evaluate import outcome
from .registry import get
from .spec import PolicySpec, Step


def with_flight_noise(action: dict, rng: np.random.Generator, scale: float) -> dict:
    """The action with Gaussian noise on its flight, tools and transfers stay the teacher's own"""
    return dict(action, move=np.clip(action["move"] + rng.normal(0.0, scale, size=5).astype(np.float32), -1.0, 1.0))


def collect(spec: PolicySpec, args, out: Path) -> None:
    reader = Reader(args.reader) if args.teacher_sees == "reader" else None
    rng = np.random.default_rng(args.seed)
    options = {"obstacles": args.obstacles, "terrain": args.terrain, "size": args.size, "perception": args.perception}
    if args.max_steps:
        options["maxSteps"] = args.max_steps
    if args.prey:
        options["prey"] = args.prey
    env = DroneEnv(task=args.task, tools=True, streams=spec.streams, action_pause_ms=0, width=FRAME[1], height=FRAME[0], task_options=options)
    agent = None
    try:
        for i in range(args.episodes):
            seed = args.seed + i
            obs, info = env.reset(seed=seed)
            teacher = spec.teacher(args.task, info, env.client.mask_ids, reader)
            if args.dagger is not None:
                agent = agent or spec.agent(spec.load(args.dagger, pick_device()), env.client.mask_ids)
                agent.reset()
            rows = []
            # an arena can be done before its first step
            terminated, truncated = bool(info["episode"].get("done")), False
            while not (terminated or truncated):
                state = info["state"]
                label = teacher.act(state, obs)
                step = Step(state, obs, label, env.client.mask_ids, teacher)
                row = spec.record(step)
                if row is not None:
                    rows.append(row)
                if agent is not None and rng.random() < args.beta:
                    action = agent.act(step)
                elif (teacher.intent or {}).get("mode") in TOOLS[1:]:
                    # noise knocks the aim off a block being dug, which resets the dig, so a tool's hold flies clean
                    action = label
                else:
                    action = with_flight_noise(label, rng, args.noise)
                obs, _, terminated, truncated, info = env.step(action)
            save_episode(out / f"{seed}.npz", rows)
            print(f"episode {i + 1}/{args.episodes}: {len(rows)} rows, {outcome(info['episode'])}", flush=True)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--task", default="goto_point")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=300_000, help="first env seed, evaluation uses 100000 and up")
    parser.add_argument("--terrain", choices=["flat", "rough", "cave"], default="flat")
    parser.add_argument("--obstacles", type=int, default=4)
    parser.add_argument("--size", type=int, default=5, help="structure or deposit side for the copy, build, and mine arenas")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--perception", choices=["vision", "scan"], default="vision", help="scan hands the drone the job's blocks or mobs")
    parser.add_argument("--teacher-sees", choices=["reader", "mask"], default="reader", help="mask is the mod's ground truth, for data the reader can't label yet")
    parser.add_argument("--reader", type=Path, default=CHECKPOINTS / "reader.pt")
    parser.add_argument("--noise", type=float, default=0.2, help="flight noise on the teacher's executed action")
    parser.add_argument("--dagger", type=Path, default=None, help="a trained checkpoint that flies a share of the steps")
    parser.add_argument("--beta", type=float, default=0.5, help="share of steps the --dagger policy flies")
    parser.add_argument("--run", default=None, help="data folder under data/policies/<policy>, task-terrain by default")
    parser.add_argument("--prey", default=None, help="a hunt's prey: hostile, all, or kinds such as cow,zombie, the arena picks when left out")
    args = parser.parse_args()

    spec = get(args.policy)
    default = f"{args.task}-{args.terrain}" + ("-dagger" if args.dagger else "")
    collect(spec, args, run_dir(spec.name, args.run or default))


if __name__ == "__main__":
    main()
