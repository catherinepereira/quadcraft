package com.catherinepereira.quadcraft.task;

import com.catherinepereira.quadcraft.Json;
import com.catherinepereira.quadcraft.entity.DroneEntity;
import com.catherinepereira.quadcraft.entity.DroneTier;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.List;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.phys.AABB;
import org.jspecify.annotations.Nullable;

/** One player's current arena, the server's source of truth for task progress */
public final class ArenaRecord {
	public final TaskKind kind;
	public final BlockPos origin;
	public final int radius;
	public List<int[]> obstacles = List.of();
	public List<int[]> trees = List.of();
	public List<int[]> stalactites = List.of();
	public String terrain = "flat";
	public final List<BlockPos> targets = new ArrayList<>();
	public final List<BlockPos> distractors = new ArrayList<>();
	public @Nullable Blueprint blueprint;
	public @Nullable BlockPos referenceBase;
	public @Nullable BlockPos buildBase;
	public @Nullable DroneJob job;
	// the drone's tier, its job lists the blocks it would break for nothing
	public DroneTier tier = DroneTier.COPPER;
	// the job's boxes as schematic files by box name, set in scan perception
	public @Nullable JsonObject scan;
	// overrides the geofence the client derives from origin and radius, as x0, y0, z0, x1, y1, z1
	public int @Nullable [] fence;
	// a studio scene's block or mob, its box, and the camera poses of it as x, y, z, yaw, pitch
	public @Nullable String subject;
	public @Nullable AABB subjectBox;
	public boolean subjectIsBlock;
	public final List<float[]> views = new ArrayList<>();

	ArenaRecord(TaskKind kind, BlockPos origin, int radius) {
		this.kind = kind;
		this.origin = origin;
		this.radius = radius;
	}

	/**
	 * Task progress counters sent with every tool sync.
	 * replicate_build, copy_region, build_schematic: destination cells that match, target blocks, destination blocks that don't belong.
	 * mine_region: blocks of the kind left in the region, how many there were at the start.
	 * harvest_crops, harvest_region: crops ripe at the start still standing, ripe ones harvested, ripe at the start,
	 * empty farmland cells, farmland cells.
	 * patrol_area, hunt_mobs, patrol_region, guard_region: prey left, mobs the drone killed, prey at the start.
	 * goto_point, find_block, follow_mob, fly_to, seek_block, follow_player: none, the client scores them from the drone's pose
	 */
	public int[] metrics(ServerLevel level, DroneEntity drone) {
		return switch (this.kind) {
			case STUDIO -> new int[0];
			case REPLICATE_BUILD, COPY_REGION, BUILD_SCHEMATIC, MINE_REGION, HARVEST_CROPS, HARVEST_REGION, COPY_BUILD, SCHEMATIC_BUILD, MINE_DEPOSIT, GATHER_BUILD, RETURN_HOME, PATROL_AREA,
				HUNT_MOBS, PATROL_REGION, GUARD_REGION, GOTO_POINT, FIND_BLOCK, FOLLOW_MOB, FLY_TO, SEEK_BLOCK, FOLLOW_PLAYER, DOCK_STATION -> this.job.score(level);
		};
	}

	public JsonObject toJson() {
		JsonObject json = new JsonObject();
		json.addProperty("task", this.kind.id);
		json.add("origin", Json.pos(this.origin));
		json.addProperty("radius", this.radius);
		json.add("obstacles", rows(this.obstacles));
		json.addProperty("terrain", this.terrain);
		json.add("trees", rows(this.trees));
		json.add("stalactites", rows(this.stalactites));
		if (!this.targets.isEmpty()) {
			json.add("targets", positions(this.targets));
			json.add("distractors", positions(this.distractors));
		}
		if (this.job != null) {
			JsonObject job = this.job.toJson();
			if (this.scan != null) {
				job.add("scan", this.scan);
			}
			job.addProperty("tier", this.tier.id);
			JsonArray unharvestable = new JsonArray();
			this.tier.unharvestable().forEach(unharvestable::add);
			job.add("unharvestable", unharvestable);
			json.add("job", job);
		}
		if (this.fence != null) {
			JsonArray f = new JsonArray();
			for (int v : this.fence) {
				f.add(v);
			}
			json.add("fence", f);
		}
		if (this.blueprint != null) {
			json.add("referenceBase", Json.pos(this.referenceBase));
			json.add("buildBase", Json.pos(this.buildBase));
			json.add("blueprint", this.blueprint.toJson());
		}
		if (this.subject != null) {
			json.addProperty("subject", this.subject);
			if (this.subjectBox != null) {
				AABB b = this.subjectBox;
				JsonArray box = new JsonArray();
				for (double v : new double[] {b.minX, b.minY, b.minZ, b.maxX, b.maxY, b.maxZ}) {
					box.add(v);
				}
				json.add("subjectBox", box);
			}
			JsonArray views = new JsonArray();
			for (float[] view : this.views) {
				JsonArray v = new JsonArray();
				for (float f : view) {
					v.add(f);
				}
				views.add(v);
			}
			json.add("views", views);
		}
		return json;
	}

	private static JsonArray rows(List<int[]> list) {
		JsonArray rows = new JsonArray();
		for (int[] row : list) {
			JsonArray a = new JsonArray();
			for (int v : row) {
				a.add(v);
			}
			rows.add(a);
		}
		return rows;
	}

	private static JsonArray positions(List<BlockPos> list) {
		JsonArray a = new JsonArray();
		list.forEach(p -> a.add(Json.pos(p)));
		return a;
	}
}
