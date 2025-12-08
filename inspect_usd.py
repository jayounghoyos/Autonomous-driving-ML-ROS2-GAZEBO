#!/usr/bin/env python3
"""
Diagnostic script to inspect the Leatherback USD asset
and find the exact joint paths needed for control.
"""

from isaacsim import SimulationApp

# Initialize minimal app
print("Starting Isaac Sim...")
sim_app = SimulationApp({"headless": True})

from isaacsim.core.api import World
import isaacsim.core.utils.stage as stage_utils
from pxr import Usd, UsdGeom, UsdPhysics

def inspect_stage():
    world = World()
    world.scene.add_default_ground_plane()
    
    # Use cloud assets since local LFS file is broken
    from isaacsim.storage.native import get_assets_root_path
    assets_root = get_assets_root_path()
    if assets_root is None:
        print("Error: Cloud assets not configured!")
        return
        
    leatherback_path = assets_root + "/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd"
    
    print(f"\nLoading asset: {leatherback_path}")
    stage_utils.add_reference_to_stage(leatherback_path, "/World/Leatherback")
    
    print("\n" + "="*50)
    print("INSPECTING USD HIERARCHY")
    print("="*50)
    
    stage = world.stage
    robot_prim = stage.GetPrimAtPath("/World/Leatherback")
    
    if not robot_prim.IsValid():
        print("ERROR: Robot prim not found!")
        return

    # Helper to traverse
    def print_hierarchy(prim, depth=0):
        indent = "  " * depth
        name = prim.GetName()
        type_name = prim.GetTypeName()
        
        # Highlight interesting things
        marker = ""
        if "Physics" in type_name or "Joint" in type_name or "Drive" in type_name:
            marker = "  <-- PHYSICS"
        
        # Checking for API schemas (like LimitAPI, DriveAPI)
        schemas = prim.GetAppliedSchemas()
        if schemas:
            marker += f" [Schemas: {schemas}]"

        print(f"{indent}{name} ({type_name}){marker}")
        
        # Recurse
        for child in prim.GetChildren():
            print_hierarchy(child, depth + 1)
            
    print_hierarchy(robot_prim)
    print("="*50 + "\n")
    
    sim_app.close()

if __name__ == "__main__":
    inspect_stage()
