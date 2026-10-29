import pytest

from drone_model.scripted.jobs import BuildPlanner, base_name, box_views

MASK_IDS = {"blocks": ["minecraft:stone", "minecraft:bricks"], "entities": ["quadcraft:drone", "minecraft:zombie", "minecraft:cow"], "categories": ["misc", "monster", "creature"], "entityBase": 32768}
JOB = {"kind": "copy", "source": [0, 1, 0, 2, 3, 2], "dest": [10, 1, 0, 12, 3, 2], "size": [3, 3, 3]}


def planner() -> BuildPlanner:
    # a stand-in reader, these tests feed memory directly
    return BuildPlanner(MASK_IDS, reader=object())


def remember(p: BuildPlanner, cell, block: str) -> None:
    p.memory.apply_events([{"type": "place", "pos": list(cell), "block": block}])


def test_base_name_drops_properties():
    assert base_name("minecraft:oak_stairs[facing=north]") == "minecraft:oak_stairs"


def test_box_views_cover_every_side_and_the_top():
    views = box_views(JOB["source"])
    vias = [v for v, _ in views]
    assert any(v[2] > 3 for v in vias) and any(v[2] < 0 for v in vias)
    assert any(v[0] > 3 for v in vias) and any(v[0] < 0 for v in vias)
    assert any(v[1] > 4 for v in vias)


def test_read_plan_moves_the_source_onto_the_destination():
    p = planner()
    remember(p, (1, 1, 1), "minecraft:bricks")
    remember(p, (1, 2, 1), "minecraft:oak_planks")
    assert p.read_plan(JOB, {}) == {(11, 1, 1): "minecraft:bricks", (11, 2, 1): "minecraft:oak_planks"}
    assert p.read_schematic.get(1, 1, 1) == "minecraft:oak_planks"


def test_placement_works_square_on_from_the_block_below_then_the_sides():
    p = planner()
    state = {"pos": [11.5, 3.0, 4.5]}
    remember(p, (11, 0, 1), "minecraft:stone")
    support, face, normal, via, point = p.placement(state, (11, 1, 1))
    # straight over the support's top face, 2.5 up
    assert (support, face, normal) == ((11, 0, 1), "up", (0, 1, 0))
    assert point == pytest.approx((11.5, 0.99, 1.5)) and via == pytest.approx((11.5, 3.49, 1.5))
    remember(p, (12, 2, 1), "minecraft:bricks")
    support, face, normal, via, point = p.placement(state, (11, 2, 1))
    # level with the support's west face and straight out from it, the drone faces east
    assert (support, face, normal) == ((12, 2, 1), "west", (-1, 0, 0))
    assert via == pytest.approx((point[0] - 2.5, point[1], point[2])) and point[1] == pytest.approx(2.5)
    assert p.placement(state, (5, 5, 5)) is None


def test_a_dig_keeps_its_face_and_never_works_from_a_diagonal():
    p = planner()
    remember(p, (4, 0, 4), "minecraft:stone")
    state = {"pos": [9.0, 2.0, 4.6]}
    first = p.work_pose(state, ("break", (4, 0, 4)), [((4, 0, 4), n) for n in p.toward(state, (4.5, 0.5, 4.5), ((1, 0, 0), (0, 0, 1)))])
    assert first[1] == (1, 0, 0) and first[2] == pytest.approx((4.99 + 2.5, 0.5, 4.5))
    # the drone moving around doesn't switch the face once picked
    state["pos"] = [4.6, 2.0, 9.0]
    again = p.work_pose(state, ("break", (4, 0, 4)), [((4, 0, 4), (0, 0, 1)), ((4, 0, 4), (1, 0, 0))])
    assert again == first


def test_trenches_open_a_face_of_every_block():
    p = planner()
    box = [0, 0, 0, 6, 3, 9]
    state = {"pos": [0.5, 10.0, 0.5]}
    p.next_dig(state, box)
    trench = set(p.trenches[tuple(box)])
    for x in range(7):
        for y in range(4):
            for z in range(10):
                if (x, y, z) in trench:
                    continue
                neighbors = [(x + 1, y, z), (x - 1, y, z), (x, y, z + 1), (x, y, z - 1), (x, y + 1, z)]
                outside = x in (0, 6) or z in (0, 9) or y == 3
                assert outside or any(n in trench for n in neighbors), (x, y, z)


def test_scans_fill_memory_and_dig_shafts_to_buried_materials(tmp_path):
    from quadcraft.schematic import Schematic

    s = Schematic.empty(3, 3, 1)
    s.blocks[:, :, :] = "minecraft:stone"
    s.blocks[0, 0, 1] = "minecraft:diamond_ore"
    (tmp_path / "scans").mkdir()
    s.save(tmp_path / "scans" / "r.schem")
    p = BuildPlanner(MASK_IDS, reader=object(), schematics=tmp_path)
    job = {"kind": "copy", "gather": [0, 0, 0, 2, 2, 0], "scan": {"gather": "scans/r.schem"}}
    p.core.take_scans(job)
    p.note_job(job)
    assert p.scanned_in(job["gather"], {"minecraft:diamond_ore"}) == [(1, 0, 0)]
    # buried under two layers: the shaft starts at the top of the ore's column
    assert p.shaft_blocker((1, 0, 0), job["gather"]) == (1, 2, 0)


def test_a_pocket_under_stone_gets_a_shaft_not_a_dig_through_it(tmp_path):
    from quadcraft.schematic import Schematic

    # a column of ore, an air pocket, then stone on top, with stone beside it
    s = Schematic.empty(2, 3, 1)
    s.blocks[:, :, :] = "minecraft:stone"
    s.blocks[0, 0, 0] = "minecraft:coal_ore"
    s.blocks[1, 0, 0] = "minecraft:air"
    (tmp_path / "scans").mkdir()
    s.save(tmp_path / "scans" / "p.schem")
    p = BuildPlanner(MASK_IDS, reader=object(), schematics=tmp_path)
    job = {"kind": "copy", "gather": [0, 0, 0, 1, 2, 0], "scan": {"gather": "scans/p.schem"}}
    p.core.take_scans(job)
    p.note_job(job)
    p.dig({"pos": [0.5, 5.0, 0.5], "yaw": 0.0, "pitch": 0.0, "camera": {"eyeHeight": 0.2}}, (0, 0, 0), "minecraft:coal_ore", job["gather"])
    # the drone works the stone capping the pocket first
    assert p.working_on == (0, 2, 0)


def test_aim_angle_is_small_looking_straight_down_at_any_yaw():
    from drone_model.scripted.base import aim_angle

    for yaw in (0.0, 90.0, -137.0):
        state = {"pos": [0.5, 3.0, 0.5], "yaw": yaw, "pitch": 89.5, "camera": {"eyeHeight": 0.2}}
        # a point a hair off the vertical, where the yaw error alone swings wildly
        assert aim_angle(state, (0.51, 0.0, 0.5)) < 1.0
    assert aim_angle({"pos": [0, 0, 0], "yaw": 0.0, "pitch": 0.0, "camera": {"eyeHeight": 0.0}}, (5.0, 0.0, 0.0)) > 80.0


def test_a_work_pose_needs_room_for_the_whole_drone_and_a_clear_line():
    p = planner()
    remember(p, (1, 2, 1), "minecraft:stone")
    assert p.can_work_from((2, 0, 2), (2.5, 2.5, 2.5), (2.5, 0.99, 2.5))
    assert not p.can_work_from((1, 0, 1), (1.5, 2.5, 1.5), (1.5, 0.99, 1.5))
    # straddling a cell edge touches the stone too
    assert not p.can_work_from((2, 0, 1), (2.1, 2.5, 1.5), (2.5, 0.99, 1.5))
    # the stone stands between the camera and the face
    assert not p.can_work_from((1, 0, 1), (1.5, 4.5, 1.5), (1.5, 0.99, 1.5))


def test_a_shaft_through_a_block_the_tier_cant_harvest_is_given_up(tmp_path):
    from quadcraft.schematic import Schematic

    # coal buried under diamond ore, which a copper drone would break for nothing
    s = Schematic.empty(2, 3, 1)
    s.blocks[:, :, :] = "minecraft:stone"
    s.blocks[0, 0, 0] = "minecraft:coal_ore"
    s.blocks[2, 0, 0] = "minecraft:diamond_ore"
    (tmp_path / "scans").mkdir()
    s.save(tmp_path / "scans" / "d.schem")
    p = BuildPlanner(MASK_IDS, reader=object(), schematics=tmp_path)
    job = {
        "kind": "copy", "gather": [0, 0, 0, 1, 2, 0],
        "scan": {"gather": "scans/d.schem"}, "unharvestable": ["minecraft:diamond_ore"],
    }
    p.core.take_scans(job)
    p.note_job(job)
    assert p.wasted((0, 2, 0)) and not p.wasted((1, 2, 0))
    p.dig({"pos": [0.5, 5.0, 0.5], "yaw": 0.0, "pitch": 0.0, "camera": {"eyeHeight": 0.2}}, (0, 0, 0), "minecraft:coal_ore", job["gather"])
    assert (0, 0, 0) in p.unreachable


def test_trenches_go_around_blocks_the_tier_cant_harvest(tmp_path):
    from quadcraft.schematic import Schematic

    s = Schematic.empty(3, 1, 1)
    s.blocks[:, :, :] = "minecraft:stone"
    s.blocks[0, 0, 1] = "minecraft:obsidian"
    (tmp_path / "scans").mkdir()
    s.save(tmp_path / "scans" / "t.schem")
    p = BuildPlanner(MASK_IDS, reader=object(), schematics=tmp_path)
    job = {"kind": "copy", "gather": [0, 0, 0, 2, 0, 0], "scan": {"gather": "scans/t.schem"}, "unharvestable": ["minecraft:obsidian"]}
    p.core.take_scans(job)
    p.note_job(job)
    state = {"pos": [1.5, 3.0, 0.5]}
    dug = []
    while (cell := p.next_dig(state, [0, 0, 0, 2, 0, 0])) is not None:
        dug.append(cell)
        p.dug.add(cell)
    assert sorted(dug) == [(0, 0, 0), (2, 0, 0)]


def test_gathering_skips_materials_the_tier_cant_harvest():
    p = planner()
    p.unharvestable = frozenset({"minecraft:obsidian"})
    p.plan = {(10, 1, 0): "minecraft:obsidian"}
    state = {"pos": [0.5, 5.0, 0.5], "inventory": [], "job": {**JOB, "gather": [20, 0, 0, 22, 2, 2]}}
    assert p.gather(state, list(p.plan)) is None
