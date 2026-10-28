package com.catherinepereira.quadcraft.client.render;

import com.catherinepereira.quadcraft.entity.DroneTier;
import net.minecraft.client.renderer.entity.state.EntityRenderState;

public final class DroneRenderState extends EntityRenderState {
	public DroneTier tier = DroneTier.COPPER;
	public float yRot;
	public float cameraPitch;
	// degrees the body leans toward its motion, forward and to the right
	public float leanForward;
	public float leanRight;
	// flashes red while hurt, like a mob
	public boolean hurt;
}
