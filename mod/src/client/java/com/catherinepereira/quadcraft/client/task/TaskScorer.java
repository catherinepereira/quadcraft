package com.catherinepereira.quadcraft.client.task;

import com.catherinepereira.quadcraft.Json;
import com.catherinepereira.quadcraft.task.TaskKind;
import com.catherinepereira.quadcraft.task.TrainingArenas;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.util.function.Predicate;
import net.minecraft.core.BlockPos;
import net.minecraft.world.phys.Vec3;
import org.jspecify.annotations.Nullable;

/**
 * Client-side scoring for every task.
 * Movement shaping runs per simulated tick from the drone pose, which only the client knows exactly.
 * Build progress (cells that match, blocks that don't belong) comes from the server's metrics on each tool sync
 */
public final class TaskScorer {
	public static final double SUCCESS_BONUS = 10.0;
	// degrees per tick, slower turns don't count toward reversals
	private static final float REVERSAL_RATE = 1.0F;
	// tool tasks reward approach only down to roughly reach distance, the rest is the tool's job
	private static final double TOOL_STANDOFF = 3.0;

	private boolean active;
	private String episodeId;
	private TaskKind kind = TaskKind.GOTO_POINT;
	private long seed;
	// the task's first key point, the station for return_home
	private BlockPos primary = BlockPos.ZERO;
	private JsonObject arena = new JsonObject();
	private int maxSteps;
	private double collisionPenalty;
	private int step;
	private int collisions;
	// the collisions that were with another drone
	private int droneCollisions;
	private @Nullable Vec3 goal;
	private double prevDist;
	private double totalReward;
	private double stepReward;
	private boolean success;
	private boolean truncated;
	private int[] metrics = new int[0];
	private double[] bounds = new double[6];
	private double outOfBoundsPenalty;
	private boolean outOfBounds;
	// health lost, at damagePenalty per point, and whether the drone was wrecked, which ends the episode
	private double damagePenalty;
	private float lastHealth = -1;
	private float damage;
	private boolean wrecked;
	// back-and-forth turning: each reversal of the yaw or pitch turn costs jitterPenalty
	private double jitterPenalty;
	private float lastYawRate;
	private float lastPitchRate;
	private int turnReversals;
	// a goto's point and how close counts as there
	private @Nullable Vec3 point;
	private double arrive;

	public void begin(
		String episodeId, TaskKind kind, long seed, BlockPos primary, JsonObject arena, int maxSteps, double collisionPenalty,
		int boundsPadding, double outOfBoundsPenalty, double damagePenalty, double jitterPenalty
	) {
		JsonArray origin = arena.getAsJsonArray("origin");
		int radius = arena.get("radius").getAsInt() + boundsPadding;
		int ox = origin.get(0).getAsInt();
		int floor = origin.get(1).getAsInt();
		int oz = origin.get(2).getAsInt();
		this.bounds = new double[] {ox - radius, floor, oz - radius, ox + radius + 1, floor + TrainingArenas.HEIGHT + 1, oz + radius + 1};
		if (arena.has("fence")) {
			// jobs bring their own geofence around the source and destination boxes
			JsonArray fence = arena.getAsJsonArray("fence");
			for (int i = 0; i < 6; i++) {
				this.bounds[i] = fence.get(i).getAsDouble() + (i < 3 ? -boundsPadding : boundsPadding);
			}
		}
		this.outOfBoundsPenalty = outOfBoundsPenalty;
		this.outOfBounds = false;
		this.damagePenalty = damagePenalty;
		this.jitterPenalty = jitterPenalty;
		this.lastYawRate = 0;
		this.lastPitchRate = 0;
		this.turnReversals = 0;
		this.lastHealth = -1;
		this.damage = 0;
		this.wrecked = false;
		this.active = true;
		this.episodeId = episodeId;
		this.kind = kind;
		this.seed = seed;
		this.primary = primary;
		this.arena = arena;
		this.maxSteps = maxSteps;
		this.collisionPenalty = collisionPenalty;
		this.step = 0;
		this.collisions = 0;
		this.droneCollisions = 0;
		this.goal = null;
		this.totalReward = 0;
		this.stepReward = 0;
		this.success = false;
		this.truncated = false;
		this.metrics = new int[0];
		JsonObject job = arena.has("job") ? arena.getAsJsonObject("job") : null;
		this.point = job != null && job.has("point") ? Vec3.atCenterOf(Json.readPos(job.getAsJsonArray("point"))) : null;
		this.arrive = job != null && job.has("arrive") ? job.get("arrive").getAsDouble() : 0;
	}

	/** The arena's charging station for a dock_station episode, null otherwise */
	public @Nullable BlockPos dockStation() {
		if (this.kind != TaskKind.DOCK_STATION || !this.arena.has("targets") || this.arena.getAsJsonArray("targets").isEmpty()) {
			return null;
		}
		return Json.readPos(this.arena.getAsJsonArray("targets").get(0).getAsJsonArray());
	}

	public void end() {
		this.active = false;
	}

	public boolean active() {
		return this.active;
	}

	public boolean done() {
		return this.success || this.truncated || this.outOfBounds || this.wrecked;
	}

	/** The drone's health once a tick: lost health costs damagePenalty a point, and a wreck ends the episode like leaving the geofence */
	public void health(float health, boolean wrecked) {
		if (this.active && !this.done() && this.lastHealth > health) {
			double reward = -this.damagePenalty * (this.lastHealth - health);
			this.damage += this.lastHealth - health;
			if (wrecked) {
				this.wrecked = true;
				reward -= this.outOfBoundsPenalty;
			}
			this.stepReward += reward;
			this.totalReward += reward;
		}
		this.lastHealth = health;
	}

	/** The drone's turn rates once a tick, in degrees per tick */
	public void turn(float yawRate, float pitchRate) {
		if (!this.active || this.done()) {
			return;
		}
		int flips = reversed(this.lastYawRate, yawRate) + reversed(this.lastPitchRate, pitchRate);
		// slower turns keep the last counted rate, so easing through zero still counts the reversal once
		if (Math.abs(yawRate) >= REVERSAL_RATE) {
			this.lastYawRate = yawRate;
		}
		if (Math.abs(pitchRate) >= REVERSAL_RATE) {
			this.lastPitchRate = pitchRate;
		}
		if (flips > 0) {
			this.turnReversals += flips;
			this.stepReward -= this.jitterPenalty * flips;
			this.totalReward -= this.jitterPenalty * flips;
		}
	}

	private static int reversed(float last, float now) {
		return Math.abs(now) >= REVERSAL_RATE && Math.abs(last) >= REVERSAL_RATE && Math.signum(now) != Math.signum(last) ? 1 : 0;
	}

	public double[] bounds() {
		return this.bounds;
	}

	public String episodeId() {
		return this.episodeId;
	}

	public TaskKind kind() {
		return this.kind;
	}

	public JsonObject arena() {
		return this.arena;
	}

	/** Called once per applied action, reward then accumulates across the ticks that action covers */
	public void beginAction() {
		this.stepReward = 0;
	}

	/**
	 * One simulated tick. isTarget tells whether a target block still stands in the client's world copy,
	 * so shaping heads for the nearest unmined one
	 */
	public void update(Vec3 droneCenter, boolean collided, boolean hitDrone, Predicate<BlockPos> isTarget) {
		if (!this.active || this.done()) {
			return;
		}
		this.step++;
		double reward = 0;
		if (collided) {
			this.collisions++;
			reward -= this.collisionPenalty;
			if (hitDrone) {
				this.droneCollisions++;
			}
		}
		if (!this.inBounds(droneCenter)) {
			this.outOfBounds = true;
			reward -= this.outOfBoundsPenalty;
			this.stepReward += reward;
			this.totalReward += reward;
			return;
		}
		Vec3 target = this.currentGoal(droneCenter, isTarget);
		boolean reaching = this.point != null;
		if (target != null) {
			double standoff = reaching ? 0 : TOOL_STANDOFF;
			double dist = Math.max(0, droneCenter.distanceTo(target) - standoff);
			if (this.goal != null && this.goal.equals(target)) {
				reward += this.prevDist - dist;
			}
			this.goal = target;
			this.prevDist = dist;
		}
		if (reaching && target != null && droneCenter.distanceTo(target) <= this.arrive) {
			this.succeed();
			reward += SUCCESS_BONUS;
		}
		this.stepReward += reward;
		this.totalReward += reward;
		if (!this.success && this.step >= this.maxSteps) {
			this.truncated = true;
		}
	}

	/** Reward from world progress the server reports, see ArenaRecord.metrics for the layout */
	public void applyMetrics(int[] next) {
		if (!this.active || this.done()) {
			this.metrics = next;
			return;
		}
		int[] prev = this.metrics.length == next.length ? this.metrics : next;
		double reward = 0;
		boolean won = false;
		if (isBuild(this.kind) && next.length >= 3) {
			double total = Math.max(1, next[1]);
			reward += 10.0 * (next[0] - prev[0]) / total;
			reward -= 1.0 * (next[2] - prev[2]);
			won = next[0] >= next[1] && next[2] == 0;
		}
		this.metrics = next;
		if (won) {
			this.succeed();
			reward += SUCCESS_BONUS;
		}
		this.stepReward += reward;
		this.totalReward += reward;
	}

	private boolean inBounds(Vec3 p) {
		double[] b = this.bounds;
		return p.x >= b[0] && p.y >= b[1] && p.z >= b[2] && p.x <= b[3] && p.y <= b[4] && p.z <= b[5];
	}

	/** Ends the episode as a success for a goal the client checks itself, such as docking */
	public void complete() {
		if (this.active && !this.done()) {
			this.succeed();
			this.stepReward += SUCCESS_BONUS;
			this.totalReward += SUCCESS_BONUS;
		}
	}

	private void succeed() {
		this.success = true;
		this.truncated = false;
	}

	private @Nullable Vec3 currentGoal(Vec3 drone, Predicate<BlockPos> isTarget) {
		return switch (this.kind) {
			// the drone docks on top of the station
			case RETURN_HOME -> Vec3.atCenterOf(this.primary.above());
			// no distance shaping, the drone has to study the source before heading to the build site
			case REPLICATE_BUILD, COPY_REGION, BUILD_SCHEMATIC, COPY_BUILD, SCHEMATIC_BUILD, GATHER_BUILD -> null;
			case GOTO_POINT -> this.point;
			case DOCK_STATION -> {
				Vec3 station = this.nearestTarget(drone, isTarget);
				yield station == null ? null : station.add(0, 1, 0);
			}
			// nothing to reach, a studio scene is only looked at
			case STUDIO -> null;
		};
	}

	private static boolean isBuild(TaskKind kind) {
		return switch (kind) {
			case REPLICATE_BUILD, COPY_REGION, BUILD_SCHEMATIC, COPY_BUILD, SCHEMATIC_BUILD, GATHER_BUILD -> true;
			default -> false;
		};
	}

	private @Nullable Vec3 nearestTarget(Vec3 drone, Predicate<BlockPos> isTarget) {
		Vec3 best = null;
		if (this.arena.has("targets")) {
			for (JsonElement el : this.arena.getAsJsonArray("targets")) {
				BlockPos pos = Json.readPos(el.getAsJsonArray());
				Vec3 center = Vec3.atCenterOf(pos);
				if (isTarget.test(pos) && (best == null || drone.distanceTo(center) < drone.distanceTo(best))) {
					best = center;
				}
			}
		}
		return best;
	}

	public JsonObject snapshot() {
		JsonObject json = new JsonObject();
		json.addProperty("id", this.episodeId);
		json.addProperty("task", this.kind.id);
		json.addProperty("seed", this.seed);
		json.addProperty("step", this.step);
		json.addProperty("reward", this.stepReward);
		json.addProperty("totalReward", this.totalReward);
		json.addProperty("distance", this.goal == null ? -1 : this.prevDist);
		json.addProperty("done", this.done());
		json.addProperty("success", this.success);
		json.addProperty("truncated", this.truncated);
		json.addProperty("collisions", this.collisions);
		json.addProperty("turnReversals", this.turnReversals);
		json.addProperty("droneCollisions", this.droneCollisions);
		json.addProperty("outOfBounds", this.outOfBounds);
		json.addProperty("damage", this.damage);
		json.addProperty("wrecked", this.wrecked);
		JsonArray m = new JsonArray();
		for (int v : this.metrics) {
			m.add(v);
		}
		json.add("metrics", m);
		return json;
	}
}
