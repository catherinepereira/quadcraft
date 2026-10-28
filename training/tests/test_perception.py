import math

import numpy as np

from drone_model.perception.worldmap import DRONE_FRAMES, WorldMap

MASK_IDS = {"blocks": ["minecraft:stone"], "entities": ["quadcraft:drone", "minecraft:zombie", "minecraft:cow"], "categories": ["misc", "monster", "creature"], "entityBase": 32768}


def look(map_: WorldMap, pitch: float, depth: float, eye_y: float = 3.0) -> None:
    """One frame from (0.5, eye_y, 0.5) facing +z, every pixel hitting stone at the same depth"""
    state = {"pos": [0.5, eye_y, 0.5], "yaw": 0.0, "pitch": pitch, "camera": {"eyeHeight": 0.0, "fov": 70.0, "windowAspect": 1.0}}
    map_.update(state, np.full((16, 16), depth, np.float32), np.ones((16, 16), np.uint8), 64.0)


def test_roof_seen_from_below_is_a_ceiling():
    map_ = WorldMap(MASK_IDS)
    look(map_, pitch=-60.0, depth=4.0)
    assert map_.ceilings and not map_.heights
    map_.fly_y = 2.0
    assert not any(map_.blocked(c) for c in map_.ceilings)
    map_.fly_y = min(map_.ceilings.values())
    assert all(map_.blocked(c) for c in map_.ceilings if map_.ceilings[c] == map_.fly_y)


def test_ceiling_under_a_known_floor_is_ignored():
    map_ = WorldMap(MASK_IDS)
    map_.ceilings[(0, 3)] = 4.0
    map_.heights[(0, 3)] = 9.0
    assert map_.ceiling((0, 3)) == math.inf
    map_.fly_y = 10.0
    assert not map_.blocked((0, 3))
    map_.fly_y = 5.0
    assert map_.blocked((0, 3))


def test_headroom_reads_the_column_overhead():
    map_ = WorldMap(MASK_IDS)
    map_.ceilings[(0, 0)] = 7.0
    map_.ceilings[(3, 3)] = 4.0
    assert map_.headroom(0.5, 0.5) == 7.0
    assert map_.headroom(1.5, 1.5) == math.inf


def test_a_seen_drone_blocks_its_column_for_a_while():
    map_ = WorldMap(MASK_IDS)
    state = {"pos": [0.5, 3.0, 0.5], "yaw": 0.0, "pitch": 0.0, "camera": {"eyeHeight": 0.0, "fov": 70.0, "windowAspect": 1.0}}
    drone = np.full((16, 16), 32768, np.int64)
    map_.update(state, np.full((16, 16), 3.0, np.float32), drone, 64.0)
    assert map_.drones and not map_.heights and not map_.ceilings
    map_.fly_y = 3.0
    ahead = (0, 3)
    assert ahead in map_.drones and map_.blocked(ahead) and not map_.free(0.5, 3.5)
    # out of the way above or below the flying height
    map_.fly_y = 9.0
    assert not map_.blocked(ahead)
    map_.fly_y = 3.0
    for _ in range(DRONE_FRAMES):
        map_.update(state, np.full((16, 16), 64.0, np.float32), np.zeros((16, 16), np.int64), 64.0)
    assert not map_.blocked(ahead)


def test_memory_calls_a_voted_block_solid_and_a_crop_not():
    from drone_model.perception.memory import MIN_VOTES, Cell, VoxelMemory
    from drone_model.perception.reader import INDEX

    memory = VoxelMemory()
    for key, block in (((0, 0, 0), "minecraft:stone_bricks"), ((1, 0, 0), "minecraft:wheat")):
        cell = Cell()
        cell.votes[INDEX[block]] = MIN_VOTES
        memory.cells[key] = cell
    assert memory.solid((0, 0, 0))
    assert not memory.solid((1, 0, 0))
    assert not memory.solid((2, 0, 0))


def test_project_lands_on_the_pixel_backproject_came_from():
    from drone_model.perception.worldmap import backproject, project

    state = {"pos": [2.0, 5.0, -1.0], "yaw": 30.0, "pitch": 20.0, "camera": {"eyeHeight": 0.2, "fov": 70.0, "windowAspect": 1.6}}
    depth = np.full((24, 32), 6.0, np.float32)
    view = backproject(state, depth, 64.0, stride=1)
    row, col, z = project(state, view.points[5 * 32 + 7], depth.shape)
    assert (row, col) == (5, 7) and abs(z - 6.0) < 1e-6


def test_reader_labels_each_mob_kind_and_nothing_else():
    from drone_model.perception.reader import INDEX, SKY, entity_table

    names = ["quadcraft:drone", "minecraft:zombie", "minecraft:cow", "minecraft:item", "minecraft:spider", "minecraft:iron_golem"]
    assert list(entity_table(names)) == [SKY, INDEX["minecraft:zombie"], INDEX["minecraft:cow"], SKY, INDEX["minecraft:spider"], INDEX["minecraft:iron_golem"]]


def test_a_misread_block_gives_way_once_later_reads_outvote_it():
    from drone_model.perception.memory import MIN_VOTES, Cell, VoxelMemory
    from drone_model.perception.reader import INDEX

    memory = VoxelMemory()
    cell = Cell()
    cell.votes[INDEX["minecraft:cobblestone"]] = MIN_VOTES
    memory.cells[(3, 1, 4)] = cell
    memory.touch((3, 1, 4))
    assert memory.cells_of("minecraft:cobblestone") == [(3, 1, 4)]
    cell.votes[INDEX["minecraft:stone"]] = 2 * MIN_VOTES
    memory.touch((3, 1, 4))
    assert memory.cells_of("minecraft:cobblestone") == []
    assert memory.cells_of("minecraft:stone") == [(3, 1, 4)]


def test_planners_built_on_one_core_share_its_memory_and_map():
    from drone_model.perception.core import DroneCore
    from drone_model.scripted.goto import GotoPlanner
    from drone_model.scripted.jobs import BuildPlanner

    core = DroneCore(MASK_IDS)
    goto, build = GotoPlanner(MASK_IDS, core=core), BuildPlanner(MASK_IDS, core=core)
    assert goto.memory is build.memory is core.memory
    assert goto.map is build.map is core.map


def test_a_ground_truth_frame_fills_the_memory_and_leaves_mobs_out():
    from drone_model.perception.core import DroneCore

    core = DroneCore(MASK_IDS)
    state = {"pos": [0.5, 3.0, 0.5], "yaw": 0.0, "pitch": 0.0, "camera": {"eyeHeight": 0.0, "fov": 70.0, "windowAspect": 1.0}}
    depth = np.full((16, 16), 6.0, np.float32)
    mask = np.ones((16, 16), np.int64)
    # a zombie in front of a stone wall
    depth[6:10, 6:10] = 4.0
    mask[6:10, 6:10] = 32769
    for _ in range(8):
        core.see(state, {"depth": depth, "mask": mask})
    assert core.memory.label((4, 7, 6)) == "minecraft:stone"
    # mobs move, the zombie's pixels never become a cell
    assert core.memory.label((0, 3, 4)) == "minecraft:air"


def test_look_alikes_read_as_the_block_they_copy_and_solidity_comes_from_the_game():
    from drone_model.perception.memory import MIN_VOTES, Cell, VoxelMemory
    from drone_model.perception.reader import INDEX, OTHER, state_tables

    classes, _ = state_tables(["minecraft:infested_stone", "minecraft:waxed_copper_block[]", "minecraft:petrified_oak_slab[type=top]", "minecraft:no_such_block"])
    assert list(classes[1:]) == [INDEX["minecraft:stone"], INDEX["minecraft:copper_block"], INDEX["minecraft:oak_slab"], OTHER]
    memory = VoxelMemory()
    for key, block in (((0, 0, 0), "minecraft:stone"), ((1, 0, 0), "minecraft:poppy"), ((2, 0, 0), "minecraft:oak_slab")):
        memory.cells[key] = Cell()
        memory.cells[key].reset(INDEX[block], MIN_VOTES)
    assert [memory.solid(k) for k in ((0, 0, 0), (1, 0, 0), (2, 0, 0))] == [True, False, True]
