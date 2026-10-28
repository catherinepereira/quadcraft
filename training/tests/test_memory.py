import numpy as np

from drone_model.perception.memory import VoxelMemory
from drone_model.perception.reader import CLASSES, SKY

BRICKS = CLASSES.index("minecraft:bricks")
STATE = {"pos": [0.5, 3.0, 0.5], "yaw": 0.0, "pitch": 90.0, "camera": {"eyeHeight": 0.0, "fov": 40.0, "windowAspect": 1.0}}


def frame(cls_index: int):
    """Looking straight down from 3 blocks over a floor whose top is at y 1, so every pixel lands in block y 0"""
    depth = np.full((16, 16), 2.0, np.float32)
    return depth, np.full((16, 16), cls_index, np.uint8)


def test_seen_blocks_become_beliefs_with_changes():
    memory = VoxelMemory()
    depth, cls = frame(BRICKS)
    changes = memory.observe(STATE, depth, cls)
    assert changes and all(c.after == "minecraft:bricks" and c.cause == "seen" for c in changes)
    assert memory.label(changes[0].cell) == "minecraft:bricks"
    # seeing the same thing again changes nothing
    assert memory.observe(STATE, depth, cls) == []


def test_breaks_and_placements_update_without_looking():
    memory = VoxelMemory()
    depth, cls = frame(BRICKS)
    cell = memory.observe(STATE, depth, cls)[0].cell
    broke = memory.apply_events([{"type": "break", "pos": list(cell)}])
    assert broke[0].before == "minecraft:bricks" and broke[0].after == "minecraft:air" and broke[0].cause == "broke"
    placed = memory.apply_events([{"type": "place", "pos": list(cell), "block": "minecraft:oak_planks"}])
    assert placed[0].after == "minecraft:oak_planks" and placed[0].cause == "placed"


def test_sky_adds_nothing():
    memory = VoxelMemory()
    depth, cls = frame(SKY)
    assert memory.observe(STATE, depth, cls) == [] and not memory.cells
