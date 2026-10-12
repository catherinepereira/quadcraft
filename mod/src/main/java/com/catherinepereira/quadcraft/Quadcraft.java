package com.catherinepereira.quadcraft;

import com.catherinepereira.quadcraft.map.BlueMapMarkers;
import com.catherinepereira.quadcraft.net.ModNetworking;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.resources.Identifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Quadcraft implements ModInitializer {
	public static final String MOD_ID = "quadcraft";
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	@Override
	public void onInitialize() {
		ModContent.register();
		ModNetworking.register();
		// BlueMap is optional, its API classes only load when it's installed
		if (FabricLoader.getInstance().isModLoaded("bluemap")) {
			BlueMapMarkers.register();
		}
	}

	public static Identifier id(String path) {
		return Identifier.fromNamespaceAndPath(MOD_ID, path);
	}
}
