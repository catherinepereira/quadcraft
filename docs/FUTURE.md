# Future work

Known gaps and things tried so far, to pick up later. Per-job limitations of the brain are in [training/README.md](../training/README.md#known-limitations).

## Drones avoiding each other

Postponed. Learning to steer around other drones is the goal, scripted traffic rules aren't.

- Drones are solid to each other, and every episode counts `droneCollisions`.
- Each charging station takes one drone.
- In mask perception the experts' world map blocks a column for 15 frames after seeing another drone in it near flying height. The brain sees through the block reader, which has no drone class, so it doesn't see other drones.
- A drone class for the reader was tried and backed out. Picking it up again needs reader frames with drones in view and a retrain.

## Fleets

- A fleet (several drones, one job each, one shared arena that resets once every drone is done) works for training arenas only. A fleet reset that names a player job is refused.
- Player drones already run different jobs at once from their queues, and a job can't overlap another drone's running job. This hasn't been tested with three or more drones on different jobs in one world.
- Fleet evaluation tops out at 8 drones (`FleetResetPayload.MAX_MEMBERS`).

## Capture

- Every drone shares the client's one camera, so each drone's step rate drops with the number flying: one drone runs at about 147 steps a second, two at about 79 each. A second camera or render target per drone wasn't tried.
- Depth and mask raycasts run on the client thread, about 3.7 ms a frame at 160x120.

## Combat and patrols

- Hunts haven't been rerun since reader v10 added a class for every mob kind, a guard may now hunt spiders, endermen, and the rest in vision.
- A misread on a block the voxel memory hasn't voted on yet still becomes a target until the map catches up, about 1 in 10 views of tracked prey in an all-mob hunt.
- The learned skill gets wrecked by creepers on all-mob hunts, it flies closer than the planner's standoff.
- A skill round on hunts alone costs builds (7 of 10 down to 1 of 10). Builds last had DAgger data in round 4, the next skill round has to cover every task.
- Hunts and follows keep the world running in lockstep so their targets move, which makes those episodes differ from run to run on the same seed.
- A drone has 20 health and no armor or upgrades. Tiers change the beam's damage, not the drone's toughness.
- Creepers only come after a drone that hit them, and a provoked one explodes like it would at a player, blocks included.
- Patrol cells are a fixed 8 blocks across, a region can't set its own spacing or route.

## Block reader

- The find, hunt, and copy checks haven't been run with reader v10 yet, the numbers for vision jobs in RESULTS.md come from older readers.
- v10 reads subjects under 300 pixels at 320x240 right less than a fifth of the time for blocks.
- Blocks learned mostly from rendered frames read about 13 points worse on game frames than blocks learned from game frames. The renderer has no sunrise or sunset colors or clouds, scatters plants on any ground, and its trees and hills look unlike the game's.
- At 320x240 the voxel memory, world map, and mob tracker take 4 times the pixels per frame. Their cost in the brain hasn't been measured.
- The offline renderer (`drone_model.render`) labels each block by its model elements' boxes, the mod's raycasts hit outline shapes, which can differ for plants and thin blocks. Exporting the outline shapes from the game would make them match.
- The renderer has no clouds, sun, moon, waterlogged blocks, or block entities. Chests, heads, banners, shulker boxes, decorated pots, and mobs come only from the game, their geometry could be exported from the game's model layers the same way.
- Light levels for the renderer are a table copied from the game's Blocks.java for 26.3, a game update needs it checked.

## Getting somewhere

- A follow job keeps the drone within 48 blocks across of where it started, a player walking farther ends it as out of bounds.
- follow_mob trains on a villager. Players move faster and more sharply.
- A followed mob is found by its kind through the camera and kept by the mob tracker. With several of its kind in view the drone may end up on another one, and the goto policy's follow data came from a planner that was told where the target was.
- A followed player is still a position the drone is handed, the tablet's beacon. Following one by sight needs a player class in the reader.
- Planners judge what the crosshair rests on from the frame's depth and the reader's class there, which replaced the mod telling them. Builds, mining, and harvests haven't been rerun since.
- The goto policy turns back and forth more than the skill, 9 to 14 reversals per 100 steps on rough goto_point, find_block, and dock_station.
- Docking is checked in the dock_station arena, the flight home in a player's world hasn't been flown end to end since the arenas started moving off stations.
- The scripted goto planner reaches cave points 4 times in 6. It climbs into pockets under low dips in the uneven roof and keeps trying a gap too low for it. Flat and rough points it reaches every time.

## Map

- BlueMap re-renders the changed regions every 5 seconds after saving the chunks, so arena changes show up on the map within about 5 seconds.
- Drone markers move every second from `quadcraft/drones.json` in BlueMap's web root. BlueMap's own marker push still comes every 10 seconds and carries the names and details.
