#!/usr/bin/env python3
"""
Isaac Sim - Training Environment with Camera
Simple environment with camera sensor for RL training
"""

from isaacsim import SimulationApp

# Launch Isaac Sim with GUI
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualSphere, VisualCuboid
from isaacsim.sensor.camera import Camera
from pxr import UsdLux, Gf

print("Creating training environment with camera...")

# Create world
world = World()

# Add ground plane
world.scene.add_default_ground_plane()
print("✓ Ground plane added")

# Add lighting
stage = world.stage

# Dome light
dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(600.0)
print("✓ Dome light added")

# Sun light
sun_light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
sun_light.CreateIntensityAttr(1200.0)
sun_light.AddOrientOp().Set(Gf.Quatf(0.7071, 0.7071, 0, 0))
print("✓ Sun light added")

# Add goal marker (green sphere)
goal = world.scene.add(
    VisualSphere(
        prim_path="/World/goal_marker",
        name="goal",
        position=np.array([5.0, 0.0, 0.3]),
        radius=0.3,
        color=np.array([0, 1, 0])
    )
)
print("✓ Goal marker added")

# Add robot (blue box)
robot = world.scene.add(
    VisualCuboid(
        prim_path="/World/robot",
        name="robot",
        position=np.array([0, 0, 0.2]),
        size=0.4,
        color=np.array([0, 0, 1])
    )
)
print("✓ Robot added")

# Add camera to robot
camera = Camera(
    prim_path="/World/robot/camera",
    name="robot_camera",
    frequency=20,  # 20 Hz
    resolution=(84, 84),  # Small resolution for RL
    position=np.array([0.3, 0, 0.1]),  # In front of robot
    orientation=np.array([1, 0, 0, 0])  # Looking forward
)
world.scene.add(camera)
print("✓ Camera added to robot")

# Initialize camera
camera.initialize()
print("✓ Camera initialized")

# Reset world
world.reset()
print("✓ World ready!")

print("\n🎉 Training environment with camera created!")
print("\nYou should see:")
print("  - Gray ground plane")
print("  - Green goal sphere")
print("  - Blue cube robot with camera")
print("\n✅ Camera sensor ready!")
print("✅ Ready for RL training!")
print("\nPress Ctrl+C to exit")

# Keep simulation running and test camera
frame_count = 0
while simulation_app.is_running():
    world.step(render=True)
    
    # Every 30 frames, capture and print camera info
    if frame_count % 30 == 0:
        camera_data = camera.get_current_frame()
        if camera_data is not None:
            rgba = camera_data["rgba"]
            print(f"Camera frame {frame_count}: shape={rgba.shape}, min={rgba.min()}, max={rgba.max()}")
    
    frame_count += 1

simulation_app.close()
