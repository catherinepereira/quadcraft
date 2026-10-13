package com.catherinepereira.quadcraft.task;

import com.catherinepereira.quadcraft.Quadcraft;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.tags.FluidTags;
import net.minecraft.world.clock.ServerClockManager;
import net.minecraft.world.entity.AgeableMob;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntitySpawnReason;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.Mob;
import net.minecraft.world.entity.MobCategory;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.AttachFace;
import net.minecraft.world.level.block.state.properties.BedPart;
import net.minecraft.world.level.block.state.properties.BlockStateProperties;
import net.minecraft.world.level.block.state.properties.DoubleBlockHalf;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.phys.shapes.CollisionContext;
import org.jspecify.annotations.Nullable;

/**
 * A studio scene for block reader frames: one subject, a block or a mob, with random neighbors around it, a random
 * backdrop, light, and time of day, and camera views of it from random angles and distances.
 * The views are picked once the arena's obstacles, trees, and stalactites stand, see addViews.
 * The client puts the drone's camera at each view and captures a frame, see the view message in docs/PROTOCOL.md.
 * Lockstep keeps the world frozen, so nothing falls, flows, or wanders off
 */
final class Studio {
	static final int RADIUS = 9;
	// no shape updates, so placing a neighbor never knocks out the subject or half of it
	private static final int FLAGS = TrainingArenas.FLAGS | Block.UPDATE_KNOWN_SHAPE;
	private static final int VIEWS = 8;
	private static final int VIEW_TRIES = 60;
	// camera distance from the subject, sampled evenly in log space so close and far views both come up
	private static final double NEAR = 1.4;
	private static final double FAR = 10.0;
	// camera elevation over the subject in degrees, mostly from above, where a flying drone's camera is
	private static final double LOW = -15.0;
	private static final double HIGH = 70.0;
	// aim this far off the subject in degrees, so it lands anywhere in the frame and sometimes half out of it
	private static final float AIM_JITTER_YAW = 14.0F;
	private static final float AIM_JITTER_PITCH = 10.0F;
	private static final int MAX_NEIGHBORS = 8;
	private static final int BASIN = 5;
	// half the drone's 0.6 wide box, the camera cell has to fit it
	private static final double CAMERA_CLEARANCE = 0.32;
	// the hidden blocks and look-alikes match training's perception/classes.py, which builds the reader's classes
	// blocks no camera shows
	private static final Set<String> HIDDEN = Set.of(
		"minecraft:air", "minecraft:cave_air", "minecraft:void_air", "minecraft:barrier", "minecraft:light", "minecraft:structure_void", "minecraft:moving_piston"
	);
	// look exactly like another block, the reader learns them under that block's name
	private static final List<String> LOOK_ALIKE_PREFIXES = List.of("minecraft:infested_", "minecraft:waxed_", "minecraft:petrified_");
	private static final Set<String> FLYERS = Set.of(
		"minecraft:bat", "minecraft:bee", "minecraft:allay", "minecraft:vex", "minecraft:parrot", "minecraft:phantom", "minecraft:ghast", "minecraft:happy_ghast", "minecraft:blaze"
	);
	private static final Set<MobCategory> SWIMMERS = Set.of(MobCategory.WATER_CREATURE, MobCategory.WATER_AMBIENT, MobCategory.UNDERGROUND_WATER_CREATURE, MobCategory.AXOLOTLS);
	// what a block can need under it, tried after whatever is already there
	private static final List<Block> BASES = List.of(
		Blocks.STONE, Blocks.GRASS_BLOCK, Blocks.SAND, Blocks.FARMLAND, Blocks.SOUL_SAND, Blocks.MYCELIUM, Blocks.END_STONE, Blocks.NETHERRACK,
		Blocks.MUD, Blocks.CRIMSON_NYLIUM, Blocks.WARPED_NYLIUM, Blocks.MOSS_BLOCK, Blocks.JUNGLE_LOG, Blocks.WATER
	);
	private static final List<Direction> HORIZONTAL = List.of(Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST);
	private static final List<Block> LIGHTS = List.of(Blocks.TORCH, Blocks.LANTERN, Blocks.GLOWSTONE, Blocks.SEA_LANTERN, Blocks.SHROOMLIGHT, Blocks.SOUL_LANTERN);

	private Studio() {
	}

	static void build(ServerLevel level, Random rng, ArenaRecord record, List<Vec3> keyPoints, Terrain terrain, String subject) {
		Identifier id = Identifier.tryParse(subject);
		boolean isBlock = id != null && BuiltInRegistries.BLOCK.containsKey(id);
		boolean isMob = id != null && !isBlock && BuiltInRegistries.ENTITY_TYPE.containsKey(id);
		if (!isBlock && !isMob) {
			throw new IllegalArgumentException("studio takes a subject, a block or mob name, not '" + subject + "'");
		}
		setTime(level, rng);
		List<Block> fullCubes = visibleBlocks().stream().filter(b -> b.defaultBlockState().isCollisionShapeFullBlock(level, BlockPos.ZERO)).toList();
		BlockPos base = TrainingArenas.freeCell(rng, record, keyPoints, terrain, 1, 0.0);
		backdrop(level, rng, terrain, base, fullCubes);

		// the subject's box, aimed at by every view
		AABB box;
		if (isBlock) {
			box = placeSubject(level, rng, terrain, base, BuiltInRegistries.BLOCK.getValue(id), fullCubes);
		} else {
			box = spawnSubject(level, rng, terrain, base, BuiltInRegistries.ENTITY_TYPE.getValue(id), fullCubes);
		}
		if (box == null) {
			throw new IllegalArgumentException("studio couldn't stage " + subject);
		}
		neighbors(level, rng, box, visibleBlocks());
		record.subject = subject;
		record.subjectBox = box;
		record.subjectIsBlock = isBlock;
		keyPoints.add(box.getCenter());
	}

	/** Up to VIEWS camera poses of the staged subject into the record */
	static void addViews(ServerLevel level, Random rng, ArenaRecord record, Terrain terrain) {
		BlockPos corner = BlockPos.containing(record.subjectBox.minX, record.subjectBox.minY, record.subjectBox.minZ);
		if (record.subjectIsBlock && !BuiltInRegistries.BLOCK.getKey(level.getBlockState(corner).getBlock()).toString().equals(record.subject)) {
			Quadcraft.LOGGER.warn("studio lost its subject {}, {} stands in its place", record.subject, level.getBlockState(corner));
		}
		for (int i = 0; i < VIEWS; i++) {
			float[] view = view(level, rng, record, terrain, record.subjectBox, record.subjectIsBlock);
			if (view != null) {
				record.views.add(view);
			}
		}
	}

	/** Every block a camera can show, less the look-alikes */
	static List<Block> visibleBlocks() {
		List<Block> out = new ArrayList<>();
		for (Block block : BuiltInRegistries.BLOCK) {
			String name = BuiltInRegistries.BLOCK.getKey(block).toString();
			if (!HIDDEN.contains(name) && LOOK_ALIKE_PREFIXES.stream().noneMatch(name::startsWith)) {
				out.add(block);
			}
		}
		return out;
	}

	// mostly day, some dawn and dusk, some night
	private static void setTime(ServerLevel level, Random rng) {
		int roll = rng.nextInt(10);
		long ticks = roll < 6 ? 1000 + rng.nextInt(10000) : roll < 8 ? (rng.nextBoolean() ? 22500 + rng.nextInt(1500) : 11500 + rng.nextInt(1500)) : 13500 + rng.nextInt(9000);
		ServerClockManager clocks = level.clockManager();
		level.dimensionType().defaultClock().ifPresent(clock -> {
			clocks.setTotalTicks(clock, ticks);
			level.getServer().getPlayerList().broadcastAll(clocks.createFullSyncPacket());
		});
	}

	/** Some scenes get a ground of another block around the subject, a wall nearby, or a light */
	private static void backdrop(ServerLevel level, Random rng, Terrain terrain, BlockPos base, List<Block> fullCubes) {
		if (rng.nextFloat() < 0.5F) {
			BlockState ground = fullCubes.get(rng.nextInt(fullCubes.size())).defaultBlockState();
			int reach = 2 + rng.nextInt(4);
			for (int dx = -reach; dx <= reach; dx++) {
				for (int dz = -reach; dz <= reach; dz++) {
					int x = base.getX() + dx;
					int z = base.getZ() + dz;
					level.setBlock(new BlockPos(x, terrain.ground(x, z), z), ground, FLAGS);
				}
			}
		}
		if (rng.nextFloat() < 0.4F) {
			BlockState wall = fullCubes.get(rng.nextInt(fullCubes.size())).defaultBlockState();
			Direction side = HORIZONTAL.get(rng.nextInt(HORIZONTAL.size()));
			Direction along = side.getClockWise();
			BlockPos middle = base.relative(side, 2 + rng.nextInt(3));
			int half = 1 + rng.nextInt(3);
			int height = 2 + rng.nextInt(3);
			for (int i = -half; i <= half; i++) {
				BlockPos column = middle.relative(along, i);
				int ground = terrain.ground(column.getX(), column.getZ());
				for (int y = ground + 1; y <= ground + height && y < terrain.ceiling(column.getX(), column.getZ()); y++) {
					level.setBlock(new BlockPos(column.getX(), y, column.getZ()), wall, FLAGS);
				}
			}
		}
		if (rng.nextFloat() < 0.3F) {
			int dx = rng.nextInt(5) - 2;
			int dz = rng.nextInt(5) - 2;
			if (dx != 0 || dz != 0) {
				int x = base.getX() + dx;
				int z = base.getZ() + dz;
				place(level, rng, new BlockPos(x, terrain.ground(x, z) + 1, z), LIGHTS.get(rng.nextInt(LIGHTS.size())), false);
			}
		}
	}

	/** The block on a pedestal 0 to 2 high, as one block or a 2x2x2 cube, the water ones in a basin. Returns its box */
	private static AABB placeSubject(ServerLevel level, Random rng, Terrain terrain, BlockPos base, Block block, List<Block> fullCubes) {
		if (block.defaultBlockState().getFluidState().is(FluidTags.WATER)) {
			BlockPos floor = basin(level, rng, base, fullCubes);
			place(level, rng, floor, block, false);
			return new AABB(floor);
		}
		BlockState full = fullState(level, rng, block);
		if (full != null && rng.nextFloat() < 0.3F) {
			for (int dx = 0; dx < 2; dx++) {
				for (int dz = 0; dz < 2; dz++) {
					int x = base.getX() + dx;
					int z = base.getZ() + dz;
					// down to each column's own ground, so the cube stands on rough terrain too
					for (int y = terrain.ground(x, z) + 1; y < base.getY() + 2; y++) {
						level.setBlock(new BlockPos(x, y, z), full, FLAGS);
					}
				}
			}
			return new AABB(base.getX(), base.getY(), base.getZ(), base.getX() + 2, base.getY() + 2, base.getZ() + 2);
		}
		int room = terrain.ceiling(base.getX(), base.getZ()) - base.getY() - 3;
		int height = Math.clamp(rng.nextInt(3), 0, Math.max(0, room));
		BlockState pedestal = fullCubes.get(rng.nextInt(fullCubes.size())).defaultBlockState();
		for (int h = 0; h < height; h++) {
			level.setBlock(base.above(h), pedestal, FLAGS);
		}
		BlockPos top = base.above(height);
		place(level, rng, top, block, true);
		// a two-block block shows both halves
		BlockState placed = level.getBlockState(top);
		return placed.hasProperty(BlockStateProperties.DOUBLE_BLOCK_HALF) ? new AABB(top).expandTowards(0, 1, 0) : new AABB(top);
	}

	private static @Nullable BlockState fullState(ServerLevel level, Random rng, Block block) {
		List<BlockState> states = new ArrayList<>(block.getStateDefinition().getPossibleStates());
		states.removeIf(state -> !state.isCollisionShapeFullBlock(level, BlockPos.ZERO));
		return states.isEmpty() ? null : states.get(rng.nextInt(states.size()));
	}

	/**
	 * A basin BASIN across of a random rim around water one deep, returns its middle cell.
	 * The label rays stop at a water surface, so a block under water reads as water, one in the top layer as itself
	 */
	private static BlockPos basin(ServerLevel level, Random rng, BlockPos base, List<Block> fullCubes) {
		BlockState rim = rng.nextBoolean() ? Blocks.GLASS.defaultBlockState() : fullCubes.get(rng.nextInt(fullCubes.size())).defaultBlockState();
		// stone, grass, or sand, water plants need a sturdy floor
		BlockState floor = BASES.get(rng.nextInt(3)).defaultBlockState();
		BlockPos min = base.offset(-BASIN / 2, 0, -BASIN / 2);
		for (int x = 0; x < BASIN; x++) {
			for (int z = 0; z < BASIN; z++) {
				boolean edge = x == 0 || z == 0 || x == BASIN - 1 || z == BASIN - 1;
				level.setBlock(min.offset(x, -1, z), edge ? rim : floor, FLAGS);
				level.setBlock(min.offset(x, 0, z), edge ? rim : Blocks.WATER.defaultBlockState(), FLAGS);
			}
		}
		return base;
	}

	/** The mob with its variant and maybe young, without AI so it holds still, swimmers and flyers in the air. Returns its box */
	private static @Nullable AABB spawnSubject(ServerLevel level, Random rng, Terrain terrain, BlockPos base, EntityType<?> type, List<Block> fullCubes) {
		if (!(type.create(level, EntitySpawnReason.EVENT) instanceof Mob mob)) {
			return null;
		}
		String name = BuiltInRegistries.ENTITY_TYPE.getKey(type).toString();
		boolean swimmer = SWIMMERS.contains(type.getCategory());
		boolean flyer = FLYERS.contains(name);
		Vec3 at;
		if (swimmer) {
			// over the water, a swimmer under the surface would read as water
			BlockPos spot = rng.nextBoolean() ? basin(level, rng, base, fullCubes) : base;
			at = Vec3.atBottomCenterOf(spot).add(0, 1.0 + rng.nextDouble(), 0);
		} else {
			at = Vec3.atBottomCenterOf(base).add(0, flyer ? 1 + rng.nextInt(3) : 0, 0);
		}
		mob.snapTo(at.x, at.y, at.z, rng.nextFloat() * 360.0F, 0.0F);
		mob.finalizeSpawn(level, level.getCurrentDifficultyAt(BlockPos.containing(at)), EntitySpawnReason.EVENT, null);
		if (mob instanceof AgeableMob young && rng.nextInt(4) == 0) {
			young.setBaby(true);
		}
		calm(mob, flyer || swimmer);
		level.addFreshEntity(mob);
		// a jockey's mount or rider came with it
		for (Entity other : mob.getIndirectPassengers()) {
			if (other instanceof Mob rider) {
				calm(rider, false);
			}
		}
		if (mob.getVehicle() instanceof Mob mount) {
			calm(mount, false);
		}
		return mob.getBoundingBox();
	}

	private static void calm(Mob mob, boolean floats) {
		mob.setNoAi(true);
		mob.setSilent(true);
		mob.setNoGravity(floats);
		mob.setPersistenceRequired();
		mob.addTag(TrainingArenas.ARENA_TAG);
	}

	/** Up to MAX_NEIGHBORS random blocks in the cells around the subject, touching it or a step away */
	private static void neighbors(ServerLevel level, Random rng, AABB box, List<Block> blocks) {
		int count = rng.nextInt(MAX_NEIGHBORS + 1);
		for (int i = 0; i < count; i++) {
			int reach = rng.nextFloat() < 0.6F ? 1 : 2;
			BlockPos cell = BlockPos.containing(
				box.minX + rng.nextDouble() * box.getXsize() + rng.nextInt(2 * reach + 1) - reach,
				box.minY + rng.nextInt(Math.max(1, (int) Math.ceil(box.getYsize())) + 1) - 1,
				box.minZ + rng.nextDouble() * box.getZsize() + rng.nextInt(2 * reach + 1) - reach
			);
			if (box.intersects(new AABB(cell)) || !level.getBlockState(cell).isAir()) {
				continue;
			}
			place(level, rng, cell, blocks.get(rng.nextInt(blocks.size())), false);
		}
	}

	/**
	 * A camera pose, x, y, z, yaw, and pitch, looking near the subject from a random angle and distance, inside the arena
	 * and in the open. A ray from the camera has to reach some part of the subject, the rest may be hidden.
	 * Null when no try worked
	 */
	private static float @Nullable [] view(ServerLevel level, Random rng, ArenaRecord record, Terrain terrain, AABB box, boolean isBlock) {
		Vec3 center = box.getCenter();
		for (int t = 0; t < VIEW_TRIES; t++) {
			double dist = Math.exp(Math.log(NEAR) + rng.nextDouble() * (Math.log(FAR) - Math.log(NEAR))) + box.getXsize() / 2;
			double heading = rng.nextDouble() * 2 * Math.PI;
			double elevation = Math.toRadians(LOW + rng.nextDouble() * (HIGH - LOW));
			Vec3 cam = center.add(Math.cos(elevation) * Math.sin(heading) * dist, Math.sin(elevation) * dist, Math.cos(elevation) * Math.cos(heading) * dist);
			int cx = (int) Math.floor(cam.x);
			int cz = (int) Math.floor(cam.z);
			if (Math.abs(cx - record.origin.getX()) > record.radius - 1 || Math.abs(cz - record.origin.getZ()) > record.radius - 1) {
				continue;
			}
			if (cam.y < terrain.ground(cx, cz) + 1.2 || cam.y > terrain.ceiling(cx, cz) - 0.6) {
				continue;
			}
			if (!level.noCollision(new AABB(cam, cam).inflate(CAMERA_CLEARANCE)) || !level.getFluidState(BlockPos.containing(cam)).isEmpty()) {
				continue;
			}
			Vec3 spot = new Vec3(box.minX + rng.nextDouble() * box.getXsize(), box.minY + rng.nextDouble() * box.getYsize(), box.minZ + rng.nextDouble() * box.getZsize());
			BlockHitResult hit = level.clip(new ClipContext(cam, spot, ClipContext.Block.VISUAL, ClipContext.Fluid.NONE, CollisionContext.empty()));
			boolean seen = hit.getType() == HitResult.Type.MISS || isBlock && box.intersects(new AABB(hit.getBlockPos()));
			if (!seen) {
				continue;
			}
			Vec3 d = center.subtract(cam);
			float yaw = (float) Math.toDegrees(Math.atan2(-d.x, d.z)) + (rng.nextFloat() * 2 - 1) * AIM_JITTER_YAW;
			float pitch = (float) -Math.toDegrees(Math.atan2(d.y, Math.sqrt(d.x * d.x + d.z * d.z))) + (rng.nextFloat() * 2 - 1) * AIM_JITTER_PITCH;
			return new float[] {(float) cam.x, (float) cam.y, (float) cam.z, yaw, Math.clamp(pitch, -89.0F, 89.0F)};
		}
		return null;
	}

	/**
	 * Sets a random state of block at pos with whatever it needs under, behind, or over it, trying the bases when swapBase.
	 * A block nothing lets stay is set anyway
	 */
	private static void place(ServerLevel level, Random rng, BlockPos pos, Block block, boolean swapBase) {
		List<BlockState> states = new ArrayList<>(block.getStateDefinition().getPossibleStates());
		Collections.shuffle(states, rng);
		List<BlockState> tries = new ArrayList<>(states.subList(0, Math.min(3, states.size())));
		tries.add(block.defaultBlockState());
		BlockState under = level.getBlockState(pos.below());
		List<BlockState> bases = new ArrayList<>(List.of(under));
		if (swapBase) {
			BASES.forEach(b -> bases.add(b.defaultBlockState()));
		}
		for (BlockState pick : tries) {
			BlockState state = firstPart(pick);
			BlockPos back = backing(state, pos);
			for (BlockState base : bases) {
				level.setBlock(pos.below(), base, FLAGS);
				if (state.canSurvive(level, pos)) {
					set(level, pos, state);
					return;
				}
				// only an empty or water backing cell is filled, a block there may be the subject or a neighbor
				if (back != null && (level.getBlockState(back).isAir() || level.getBlockState(back).is(Blocks.WATER))) {
					BlockState was = level.getBlockState(back);
					level.setBlock(back, Blocks.STONE.defaultBlockState(), FLAGS);
					if (state.canSurvive(level, pos)) {
						set(level, pos, state);
						return;
					}
					level.setBlock(back, was, FLAGS);
				}
			}
		}
		level.setBlock(pos.below(), under, FLAGS);
		BlockState state = firstPart(tries.getFirst());
		// a hanging block gets a block over it
		if (level.getBlockState(pos.above()).isAir()) {
			level.setBlock(pos.above(), Blocks.STONE.defaultBlockState(), FLAGS);
			if (state.canSurvive(level, pos)) {
				set(level, pos, state);
				return;
			}
			level.setBlock(pos.above(), Blocks.AIR.defaultBlockState(), FLAGS);
		}
		// what still can't stay is set anyway, nothing ticks in the frozen world to break it
		set(level, pos, state);
	}

	// the half or part a two-block block is placed from
	private static BlockState firstPart(BlockState state) {
		if (state.hasProperty(BlockStateProperties.DOUBLE_BLOCK_HALF)) {
			state = state.setValue(BlockStateProperties.DOUBLE_BLOCK_HALF, DoubleBlockHalf.LOWER);
		}
		if (state.hasProperty(BlockStateProperties.BED_PART)) {
			state = state.setValue(BlockStateProperties.BED_PART, BedPart.FOOT);
		}
		return state;
	}

	// the cell a wall or ceiling block hangs from, null for one that stands on the block under it
	private static @Nullable BlockPos backing(BlockState state, BlockPos pos) {
		if (state.hasProperty(BlockStateProperties.ATTACH_FACE)) {
			AttachFace face = state.getValue(BlockStateProperties.ATTACH_FACE);
			return face == AttachFace.CEILING ? pos.above() : face == AttachFace.WALL ? pos.relative(state.getValue(BlockStateProperties.HORIZONTAL_FACING).getOpposite()) : null;
		}
		if (state.hasProperty(BlockStateProperties.FACING)) {
			Direction facing = state.getValue(BlockStateProperties.FACING);
			return facing == Direction.UP ? null : pos.relative(facing.getOpposite());
		}
		if (state.hasProperty(BlockStateProperties.HORIZONTAL_FACING)) {
			return pos.relative(state.getValue(BlockStateProperties.HORIZONTAL_FACING).getOpposite());
		}
		return null;
	}

	// a two-block block gets its other half
	private static void set(ServerLevel level, BlockPos pos, BlockState state) {
		level.setBlock(pos, state, FLAGS);
		if (state.hasProperty(BlockStateProperties.DOUBLE_BLOCK_HALF)) {
			level.setBlock(pos.above(), state.setValue(BlockStateProperties.DOUBLE_BLOCK_HALF, DoubleBlockHalf.UPPER), FLAGS);
		}
		if (state.hasProperty(BlockStateProperties.BED_PART)) {
			BlockPos head = pos.relative(state.getValue(BlockStateProperties.HORIZONTAL_FACING));
			level.setBlock(head, state.setValue(BlockStateProperties.BED_PART, BedPart.HEAD), FLAGS);
		}
	}
}
