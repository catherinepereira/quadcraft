# Known limitations and future work

What v1 doesn't do well, and where to pick up. Measured results are in [training/RESULTS.md](../training/RESULTS.md).

## Perception

- Small or far subjects read worst. Blocks learned mostly from rendered frames read worse on game frames than blocks learned from game frames. The renderer has no sunrise or sunset colors, clouds, sun, or moon, scatters plants on any ground, and its trees and hills look unlike the game's.
- The renderer labels blocks by their model elements' boxes, while the mod's raycasts hit outline shapes, which differ for some plants and thin blocks. Exporting outline shapes from the game would make them match.
- Blocks with their own renderers (chests, heads, banners, shulker boxes, decorated pots) and every mob come only from game frames. Their geometry could be exported from the game's model layers the way block models are read now.
- Light levels for the renderer are a table copied from the game's `Blocks.java` for 26.3, a game update needs it checked.
- The reader has no drone class, so a drone doesn't see other drones.

## Jobs

- Vision copies read only what the survey views show. A cell hidden inside a solid build reads as air, and copies of 60 or more blocks miss a few from misreads. Build jobs and scanned jobs place any block exactly.
- Gathering digs open trenches from the top, the drone doesn't fly enclosed tunnels.
- The planners know the drone's own pose exactly, the way a drone with good onboard positioning would.
- A drone has 20 health and no armor, and doesn't steer clear of mobs. The training arenas have none.
- Copy and build scores ignore block properties such as stair facing, and a vision copy places each block in its default state, since the reader labels blocks without their properties.
- The flight home in a player's world is checked in the dock_station arena, not end to end over a long job.

## Several drones

- Drones are solid to each other and every episode counts `droneCollisions`, but they don't steer around each other. Learning to avoid other drones is the goal, scripted traffic rules aren't, and it needs the reader to see drones.
- A fleet (several drones, one job each, one shared arena that resets once every drone is done) works for training arenas only, up to 8 drones. Player drones run their own queues side by side, and a job can't overlap another drone's running job, so several drones can't yet split one copy or build.
- Every drone shares the client's one camera, so each drone's step rate drops with the number flying. A render target per drone would remove that.

## Map

- BlueMap re-renders changed regions about 5 seconds after their chunks save, so arena changes reach the map with that delay. Drone markers move every second from `quadcraft/drones.json` in BlueMap's web root, while BlueMap's own marker push, with names and details, comes every 10 seconds.
