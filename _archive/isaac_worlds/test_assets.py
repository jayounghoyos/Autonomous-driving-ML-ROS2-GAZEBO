#!/usr/bin/env python3
"""
Test Isaac Sim Asset Loading
Verifies that Nucleus/cloud assets are accessible
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.storage.native import get_assets_root_path
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api import World
from pxr import UsdLux, Gf

print("="*60)
print("Isaac Sim Asset Loading Test")
print("="*60)

# Get assets root
assets_root = get_assets_root_path()
print(f"\nAssets root: {assets_root}")

if assets_root is None:
    print("\n❌ ERROR: Assets root is None!")
    print("\nNucleus is not configured. Follow these steps:")
    print("\n1. Edit: ~/.local/share/ov/data/Kit/Isaac-Sim/5.0/user.config.json")
    print('2. Add: "persistent": {"isaac": {"asset_root": {"default": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0"}}}')
    print("3. Restart this script")
    simulation_app.close()
    exit(1)

print("✅ Assets root configured!")

# Create world
world = World()
world.scene.add_default_ground_plane()
print("\n✓ Ground plane added")

# Add lighting
stage = world.stage
dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(600.0)
sun_light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
sun_light.CreateIntensityAttr(1200.0)
sun_light.AddOrientOp().Set(Gf.Quatf(0.7071, 0.7071, 0, 0))
print("✓ Lighting added")

# Try to load Leatherback
print("\nAttempting to load Leatherback robot...")
try:
    leatherback_path = assets_root + "/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd"
    print(f"Path: {leatherback_path}")
    
    stage_utils.add_reference_to_stage(leatherback_path, "/World/Leatherback")
    print("✅ Leatherback loaded successfully!")
    leatherback_loaded = True
    
except Exception as e:
    print(f"❌ Failed to load Leatherback: {e}")
    leatherback_loaded = False

# Try to load environment
print("\nAttempting to load Grid environment...")
try:
    env_path = assets_root + "/Isaac/Environments/Grid/default_environment.usd"
    print(f"Path: {env_path}")
    
    stage_utils.add_reference_to_stage(env_path, "/World/Environment")
    print("✅ Environment loaded successfully!")
    env_loaded = True
    
except Exception as e:
    print(f"❌ Failed to load environment: {e}")
    env_loaded = False

# Reset world
world.reset()

print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"Assets Root: {'✅ Configured' if assets_root else '❌ Not configured'}")
print(f"Leatherback: {'✅ Loaded' if leatherback_loaded else '❌ Failed'}")
print(f"Environment: {'✅ Loaded' if env_loaded else '❌ Failed'}")
print("="*60)

if leatherback_loaded and env_loaded:
    print("\n🎉 SUCCESS! All assets loaded correctly!")
    print("\nYou should see:")
    print("  - Leatherback robot")
    print("  - Grid environment")
    print("\nReady to proceed with Ackermann controller setup!")
else:
    print("\n⚠️  Some assets failed to load.")
    print("Check the error messages above.")

print("\nPress Ctrl+C to exit")

# Keep simulation running
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
