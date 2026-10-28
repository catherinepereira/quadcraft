package com.catherinepereira.quadcraft.task;

import com.catherinepereira.quadcraft.Json;
import com.google.gson.JsonObject;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;

/** Get to a point given as coordinates, the goto policy's training task. The client scores it from the drone's pose */
public final class GotoJob implements DroneJob {
	// a point counts as reached this close to its center
	public static final double ARRIVE = 1.5;

	private final BlockPos point;

	private GotoJob(BlockPos point) {
		this.point = point;
	}

	public static GotoJob point(BlockPos point) {
		return new GotoJob(point);
	}

	@Override
	public int[] score(ServerLevel level) {
		return new int[0];
	}

	@Override
	public boolean allows(BlockPos pos) {
		return false;
	}

	@Override
	public boolean changesBlocks() {
		return false;
	}

	@Override
	public BlockPos[] corners() {
		return new BlockPos[] {this.point, this.point};
	}

	@Override
	public JsonObject toJson() {
		JsonObject json = new JsonObject();
		json.addProperty("kind", "goto");
		json.add("point", Json.pos(this.point));
		json.addProperty("arrive", ARRIVE);
		return json;
	}
}
