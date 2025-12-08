#!/usr/bin/env python3
"""
Test Isaac Sim installation
Verifies that Isaac Sim can be imported and basic functionality works
"""

from isaacsim import SimulationApp

# Launch Isaac Sim (headless for testing)
simulation_app = SimulationApp({"headless": True})

print("✓ Isaac Sim imported successfully!")
print(f"✓ SimulationApp created")

# Import Isaac Sim core modules
try:
    from omni.isaac.core import World
    print("✓ Isaac Core modules available")
    
    # Create a simple world
    world = World()
    print("✓ World created")
    
    # Add ground plane
    world.scene.add_default_ground_plane()
    print("✓ Ground plane added")
    
    # Reset world
    world.reset()
    print("✓ World reset successful")
    
    print("\n🎉 Isaac Sim is working perfectly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    simulation_app.close()
    print("✓ Simulation closed")
