"""
What the block reader tells apart: every block a camera can show and every kind of mob, from the game's registry.
The list is built once into classes.json so recordings, the reader, and the planners agree on it, and rebuilt when the
game adds blocks or mobs. The studio scenes (mod task/Studio.java) stage the same blocks and mobs

    python -m drone_model.perception.classes     # with the game running, reads the registry over the bridge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASS_FILE = Path(__file__).with_name("classes.json")

# the hidden blocks and look-alikes match the mod's task/Studio.java
# blocks no camera shows
HIDDEN = {
    "minecraft:air", "minecraft:cave_air", "minecraft:void_air", "minecraft:barrier", "minecraft:light", "minecraft:structure_void",
    "minecraft:moving_piston",
}
# look exactly like the block after the prefix
LOOK_ALIKE_PREFIXES = ("minecraft:infested_", "minecraft:waxed_")
LOOK_ALIKES = {"minecraft:petrified_oak_slab": "minecraft:oak_slab"}
# mobs bigger than the arena, or never spawned outside commands
SKIPPED_MOBS = {"minecraft:ender_dragon", "minecraft:giant"}
# entities of the misc category that are still mobs to the reader
MISC_MOBS = {"minecraft:villager", "minecraft:iron_golem", "minecraft:snow_golem", "minecraft:copper_golem"}


def alias(name: str) -> str:
    """The block a look-alike is read as, the name itself for the rest"""
    for prefix in LOOK_ALIKE_PREFIXES:
        if name.startswith(prefix):
            return "minecraft:" + name[len(prefix):]
    return LOOK_ALIKES.get(name, name)


def build(mask_ids: dict) -> dict:
    """The class list from a bridge welcome's mask ids, which carry the registry's blocks, their solidity, and entities"""
    blocks = [
        {"name": name, "solid": solid}
        for name, solid in zip(mask_ids["blocks"], mask_ids["solid"], strict=True)
        if name not in HIDDEN and alias(name) == name
    ]
    mobs = [
        name for name, category in zip(mask_ids["entities"], mask_ids["categories"], strict=True)
        if (category != "misc" or name in MISC_MOBS) and name not in SKIPPED_MOBS
    ]
    return {"blocks": blocks, "mobs": mobs}


def load() -> dict:
    return json.loads(CLASS_FILE.read_text(encoding="utf-8"))


def main() -> None:
    from quadcraft import DroneClient

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-ids", type=Path, default=None, help="a recorded mask_ids.json from a game with solidity, in place of the bridge")
    args = parser.parse_args()
    if args.mask_ids is not None:
        mask_ids = json.loads(args.mask_ids.read_text(encoding="utf-8"))
    else:
        with DroneClient(role="observer") as client:
            mask_ids = client.mask_ids
    spec = build(mask_ids)
    CLASS_FILE.write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    print(f"{len(spec['blocks'])} blocks, {len(spec['mobs'])} mobs to {CLASS_FILE}")


if __name__ == "__main__":
    main()
