package com.catherinepereira.quadcraft.client;

import com.catherinepereira.quadcraft.Json;
import com.catherinepereira.quadcraft.client.task.TaskScorer;
import com.catherinepereira.quadcraft.entity.DroneEntity;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.mojang.blaze3d.pipeline.RenderTarget;
import net.minecraft.client.Minecraft;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.BlockPos;
import net.minecraft.util.Mth;
import net.minecraft.world.entity.Mob;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

/** The drone's state as every observation carries it, see the state section of docs/PROTOCOL.md */
public final class StateJson {
	private StateJson() {
	}

	/** Empty without a drone. Drains the tool events, so each one goes out with a single observation */
	public static JsonObject of(Minecraft mc, DroneController controller, TaskScorer task, ToolState tools, int selectedSlot, float sensorRange) {
		JsonObject json = new JsonObject();
		DroneEntity drone = controller.drone();
		if (drone == null || mc.level == null) {
			return json;
		}
		json.addProperty("droneId", drone.getId());
		json.add("pos", vec(drone.position()));
		json.add("vel", vec(controller.velocity()));
		json.addProperty("tier", drone.tier().id);
		json.add("battery", drone.batteryJson());
		json.addProperty("health", drone.getHealth());
		json.addProperty("maxHealth", drone.getMaxHealth());
		json.addProperty("yaw", drone.getYRot());
		json.addProperty("pitch", drone.getXRot());
		json.addProperty("collided", controller.lastCollided());
		json.add("range", range(mc, drone, sensorRange));
		// lets scripts project world points into the frame, same frustum math as Raycaster
		RenderTarget frame = mc.gameRenderer.mainRenderTarget();
		JsonObject camera = new JsonObject();
		camera.addProperty("fov", mc.gameRenderer.mainCamera().getFov());
		camera.addProperty("windowAspect", (float) frame.width / frame.height);
		camera.addProperty("eyeHeight", drone.getEyeHeight());
		json.add("camera", camera);
		boolean episode = task.episodeId() != null;
		// the job (boxes and schematic name) is the drone's instruction, so policies may read it too
		// The rest of the arena layout is never in the state, tools that need it ask the bridge for it
		JsonObject arena = episode ? task.arena() : null;
		json.add("job", arena != null && arena.has("job") ? arena.get("job") : JsonNull.INSTANCE);
		JsonObject job = arena != null && arena.has("job") ? arena.getAsJsonObject("job") : null;
		// a player carries the drone's tablet, so a followed player is a beacon the drone knows, a followed mob it finds with its camera
		if (job != null && job.has("target") && mc.level.getEntity(job.get("target").getAsInt()) instanceof Player target) {
			json.add("followTarget", vec(target.getBoundingBox().getCenter()));
		}
		if (job != null && job.has("scan") && job.has("hunt") && job.get("hunt").getAsBoolean()) {
			json.add("mobs", mobs(mc, Json.readBox(job.getAsJsonArray("region"))));
		}
		// the geofence is part of the task, so policies may read it, unlike the arena layout
		if (episode) {
			JsonArray box = new JsonArray();
			for (double v : task.bounds()) {
				box.add(v);
			}
			json.add("bounds", box);
		} else {
			json.add("bounds", JsonNull.INSTANCE);
		}
		tools.addTo(json, mc.level, selectedSlot);
		json.add("events", tools.drainEvents());
		return json;
	}

	/**
	 * Range sensors like a real drone's: the gap from the drone's body to the nearest block straight ahead along its
	 * heading and straight down, null when nothing is within range
	 */
	private static JsonObject range(Minecraft mc, DroneEntity drone, float range) {
		Vec3 center = drone.getBoundingBox().getCenter();
		double halfWidth = drone.getBbWidth() / 2.0;
		double halfHeight = drone.getBbHeight() / 2.0;
		float yaw = drone.getYRot() * Mth.DEG_TO_RAD;
		Vec3 ahead = new Vec3(-Mth.sin(yaw), 0.0, Mth.cos(yaw));
		JsonObject json = new JsonObject();
		json.add("front", gap(mc, drone, center, ahead, halfWidth, range));
		json.add("down", gap(mc, drone, center, new Vec3(0.0, -1.0, 0.0), halfHeight, range));
		json.addProperty("max", range);
		return json;
	}

	private static JsonElement gap(Minecraft mc, DroneEntity drone, Vec3 from, Vec3 dir, double inset, float range) {
		Vec3 end = from.add(dir.scale(inset + range));
		BlockHitResult hit = mc.level.clip(new ClipContext(from, end, ClipContext.Block.COLLIDER, ClipContext.Fluid.NONE, drone));
		if (hit.getType() != HitResult.Type.BLOCK) {
			return JsonNull.INSTANCE;
		}
		return new JsonPrimitive(Math.max(0.0, hit.getLocation().distanceTo(from) - inset));
	}

	// scan perception for hunts: every mob in the region with its kind, where the client sees it, the planner picks out its prey
	private static JsonArray mobs(Minecraft mc, BlockPos[] region) {
		AABB box = new AABB(Vec3.atLowerCornerOf(region[0]), Vec3.atLowerCornerOf(region[1]).add(1.0, 1.0, 1.0));
		JsonArray out = new JsonArray();
		for (Mob mob : mc.level.getEntitiesOfClass(Mob.class, box, Mob::isAlive)) {
			JsonObject entry = new JsonObject();
			entry.addProperty("entity", BuiltInRegistries.ENTITY_TYPE.getKey(mob.getType()).toString());
			entry.add("pos", vec(mob.getBoundingBox().getCenter()));
			out.add(entry);
		}
		return out;
	}

	private static JsonArray vec(Vec3 v) {
		JsonArray a = new JsonArray();
		a.add(v.x);
		a.add(v.y);
		a.add(v.z);
		return a;
	}
}
