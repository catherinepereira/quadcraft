# Known limitations and future work

What v1 doesn't do well, and where to pick up. Measured results are in [training/RESULTS.md](../training/RESULTS.md).

## Perception

- The reader labels parts of the training arenas' gray concrete walls as mobs. Those pixels never become solid cells in the voxel memory, so the mob tracker keeps phantom mobs along the walls, and a follow can lock onto one. Frames of the walled arenas (`collect.fields --tasks follow_mob,hunt_mobs`) in a fine-tune didn't fix it on their own and lowered villager IoU. A mob-free check against the world map's floor and walls, or more wall frames weighted up, are the next things to try.
- One mob can show up as several tracked mobs when the reader's kind for it changes from frame to frame, and the follow planner can drop the one it was on.
- Small or far subjects read worst. Blocks learned mostly from rendered frames read worse on game frames than blocks learned from game frames. The renderer has no sunrise or sunset colors, clouds, sun, or moon, scatters plants on any ground, and its trees and hills look unlike the game's.
- The renderer labels blocks by their model elements' boxes, while the mod's raycasts hit outline shapes, which differ for some plants and thin blocks. Exporting outline shapes from the game would make them match.
- Blocks with their own renderers (chests, heads, banners, shulker boxes, decorated pots) and every mob come only from game frames. Their geometry could be exported from the game's model layers the way block models are read now.
- Light levels for the renderer are a table copied from the game's `Blocks.java` for 26.3, a game update needs it checked.
- The voxel memory files a young crop, which is only a few pixels tall, in the farmland cell under it. Harvest only acts on ripe crops, which are full height, so it doesn't matter there.
- The reader has no drone class, so a drone doesn't see other drones.
- A followed player is a position the drone is handed, the tablet's beacon. Following a player by sight needs a player class.

## Jobs

- follow_mob is the weakest arena, 3 of 10, every failure a timeout. Phantom mobs and dropped mobs, see Perception above, are the causes found in logged episodes.
- Patrols and hunts on rough ground time out more than on flat ground (patrol_area 2 of 5, hunt_mobs 2 of 5). The cause isn't traced yet.
- Vision copies read only what the survey views show. A cell hidden inside a solid build reads as air, and copies of 60 or more blocks miss a few from misreads. Build jobs and scanned jobs place any block exactly.
- Mining and gathering dig open trenches or shafts from the top, the drone doesn't fly enclosed tunnels.
- The planners know the drone's own pose exactly, the way a drone with good onboard positioning would.
- Hunts and follows keep the world running between steps so their targets move, which makes those episodes differ from run to run on the same seed. A harvest arena's ripe crops also vary between runs of a seed.
- The cell skill can fly closer to a creeper than the planner's standoff and get wrecked.
- A drone has 20 health and no armor. Tiers change the beam's damage, not the drone's toughness. Creepers only come after a drone that hit them, and a provoked one explodes the way it would at a player, blocks included.
- Patrol cells are a fixed 8 blocks across, a region can't set its own spacing or route.
- A follow job keeps the drone within 48 blocks of where it started, a player walking farther ends it as out of bounds. follow_mob trains on villagers, players move faster and turn more sharply.
- The flight home in a player's world is checked in the dock_station arena, not end to end over a long job.

## Several drones

- Drones are solid to each other and every episode counts `droneCollisions`, but they don't steer around each other. Learning to avoid other drones is the goal, scripted traffic rules aren't, and it needs the reader to see drones.
- A fleet (several drones, one job each, one shared arena that resets once every drone is done) works for training arenas only, up to 8 drones. Player drones run their own queues side by side, and a job can't overlap another drone's running job.
- Every drone shares the client's one camera, so each drone's step rate drops with the number flying. A render target per drone would remove that.

## Map

- BlueMap re-renders changed regions about 5 seconds after their chunks save, so arena changes reach the map with that delay. Drone markers move every second from `quadcraft/drones.json` in BlueMap's web root, while BlueMap's own marker push, with names and details, comes every 10 seconds.
