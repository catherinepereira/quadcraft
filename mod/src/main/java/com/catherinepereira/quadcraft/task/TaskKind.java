package com.catherinepereira.quadcraft.task;

public enum TaskKind {
	REPLICATE_BUILD("replicate_build"),
	COPY_REGION("copy_region"),
	BUILD_SCHEMATIC("build_schematic"),
	COPY_BUILD("copy_build"),
	SCHEMATIC_BUILD("schematic_build"),
	GATHER_BUILD("gather_build"),
	RETURN_HOME("return_home"),
	GOTO_POINT("goto_point"),
	DOCK_STATION("dock_station"),
	STUDIO("studio");

	/** Jobs run in the player's own world, the rest build a training arena */
	public boolean isJob() {
		return this == COPY_REGION || this == BUILD_SCHEMATIC || this == RETURN_HOME;
	}

	public final String id;

	TaskKind(String id) {
		this.id = id;
	}

	public static TaskKind parse(String id) {
		for (TaskKind kind : values()) {
			if (kind.id.equals(id)) {
				return kind;
			}
		}
		throw new IllegalArgumentException("unknown task '" + id + "'");
	}
}
