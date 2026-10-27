package com.catherinepereira.quadcraft.client;

import com.catherinepereira.quadcraft.Quadcraft;
import com.catherinepereira.quadcraft.ModContent;
import com.catherinepereira.quadcraft.TabletInput;
import com.catherinepereira.quadcraft.client.hud.DroneHud;
import com.catherinepereira.quadcraft.client.render.DroneModel;
import com.catherinepereira.quadcraft.client.render.DroneRenderer;
import com.catherinepereira.quadcraft.net.DroneSyncPayload;
import com.catherinepereira.quadcraft.net.RegionsPayload;
import com.catherinepereira.quadcraft.net.TaskReadyPayload;
import com.mojang.blaze3d.platform.InputConstants;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayNetworking;
import net.fabricmc.fabric.api.client.rendering.v1.ModelLayerRegistry;
import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;
import net.fabricmc.fabric.api.client.rendering.v1.level.LevelRenderEvents;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.entity.EntityRenderers;

public class QuadcraftClient implements ClientModInitializer {
	private static final KeyMapping.Category CATEGORY = KeyMapping.Category.register(Quadcraft.id("main"));

	@Override
	public void onInitializeClient() {
		ModelLayerRegistry.registerModelLayer(DroneRenderer.LAYER, DroneModel::createLayer);
		EntityRenderers.register(ModContent.DRONE, DroneRenderer::new);

		KeyMapping pilot = KeyMappingHelper.registerKeyMapping(new KeyMapping("key.quadcraft.pilot", InputConstants.Type.KEYBOARD, InputConstants.KEY_V, CATEGORY));
		KeyMapping newEpisode = KeyMappingHelper.registerKeyMapping(new KeyMapping("key.quadcraft.new_episode", InputConstants.Type.KEYBOARD, InputConstants.KEY_N, CATEGORY));
		KeyMapping record = KeyMappingHelper.registerKeyMapping(new KeyMapping("key.quadcraft.record", InputConstants.Type.KEYBOARD, InputConstants.KEY_B, CATEGORY));
		KeyMapping inventory = KeyMappingHelper.registerKeyMapping(new KeyMapping("key.quadcraft.inventory", InputConstants.Type.KEYBOARD, InputConstants.KEY_R, CATEGORY));
		KeyMapping hud = KeyMappingHelper.registerKeyMapping(new KeyMapping("key.quadcraft.hud", InputConstants.Type.KEYBOARD, InputConstants.KEY_F8, CATEGORY));
		KeyMapping copy = KeyMappingHelper.registerKeyMapping(new KeyMapping("key.quadcraft.copy", InputConstants.Type.KEYBOARD, InputConstants.KEY_J, CATEGORY));
		KeyMapping jobs = KeyMappingHelper.registerKeyMapping(new KeyMapping("key.quadcraft.jobs", InputConstants.Type.KEYBOARD, InputConstants.KEY_K, CATEGORY));
		KeyMapping gimbal = KeyMappingHelper.registerKeyMapping(new KeyMapping("key.quadcraft.gimbal", InputConstants.Type.KEYBOARD, InputConstants.KEY_G, CATEGORY));

		HudElementRegistry.addLast(Quadcraft.id("hud"), (graphics, delta) -> {
			if (Holder.hud != null) {
				Holder.hud.extractRenderState(graphics, delta);
			}
		});

		ClientLifecycleEvents.CLIENT_STARTED.register(mc -> {
			ClientRuntime runtime = new ClientRuntime(mc);
			Holder.runtime = runtime;
			Holder.hud = new DroneHud(runtime);
			TabletInput.handler = new TabletInput.Handler() {
				@Override
				public void select(TabletInput.Target target, net.minecraft.core.BlockPos pos) {
					runtime.select(target, pos);
				}

				@Override
				public void openTablet() {
					runtime.openTablet();
				}
			};
			runtime.start();
		});
		ClientLifecycleEvents.CLIENT_STOPPING.register(mc -> {
			if (Holder.runtime != null) {
				Holder.runtime.stop();
			}
		});

		ClientTickEvents.END_CLIENT_TICK.register(mc -> {
			ClientRuntime runtime = Holder.runtime;
			if (runtime == null) {
				return;
			}
			while (pilot.consumeClick()) {
				runtime.togglePiloting();
			}
			while (newEpisode.consumeClick()) {
				runtime.newEpisodeFromKeyboard();
			}
			while (record.consumeClick()) {
				runtime.toggleRecording();
			}
			while (inventory.consumeClick()) {
				runtime.openInventory();
			}
			while (hud.consumeClick()) {
				runtime.toggleHud();
			}
			while (copy.consumeClick()) {
				runtime.startCopyFromKeyboard();
			}
			while (jobs.consumeClick()) {
				runtime.openJobs();
			}
			while (gimbal.consumeClick()) {
				runtime.toggleGimbal();
			}
			runtime.tick();
		});

		LevelRenderEvents.END_MAIN.register(context -> {
			if (Holder.runtime != null) {
				Holder.runtime.onFrameRendered(Minecraft.getInstance().gameRenderer.mainCamera());
			}
		});

		ClientPlayNetworking.registerGlobalReceiver(TaskReadyPayload.TYPE, (payload, context) -> {
			if (Holder.runtime != null) {
				Holder.runtime.onTaskReady(payload);
			}
		});

		ClientPlayNetworking.registerGlobalReceiver(RegionsPayload.TYPE, (payload, context) -> {
			if (Holder.runtime != null) {
				Holder.runtime.onRegions(payload.json());
			}
		});

		ClientPlayNetworking.registerGlobalReceiver(DroneSyncPayload.TYPE, (payload, context) -> {
			if (Holder.runtime != null) {
				Holder.runtime.onSync(payload);
			}
		});

		ClientPlayConnectionEvents.DISCONNECT.register((handler, mc) -> {
			ClientRuntime runtime = Holder.runtime;
			if (runtime != null) {
				mc.execute(() -> {
					runtime.setMode(ClientRuntime.Mode.REALTIME);
					runtime.controller().setPiloting(false);
					runtime.broadcastStatus();
				});
			}
		});
	}

	public static ClientRuntime runtime() {
		return Holder.runtime;
	}

	private static final class Holder {
		static ClientRuntime runtime;
		static DroneHud hud;
	}
}
