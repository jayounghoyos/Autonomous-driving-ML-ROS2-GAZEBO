from isaacsim import SimulationApp

# Start simulation
sim = SimulationApp({"headless": True})

from isaacsim.core.api import World
import isaacsim.core.utils.stage as stage_utils
from pxr import Usd

def main():
    world = World()
    
    # Load Leatherback (Same logic as env)
    assets_root = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1" # Fallback or use local
    # Actually let's use the local file or the one in env if possible, but simplest is to just use the one we know works
    # The env uses:
    leatherback_path = "/home/jayoungh/PersonalPorjects/Leatherback-main/source/Leatherback/Leatherback/tasks/direct/leatherback/custom_assets/leatherback_simple_better.usd"
    
    try:
        stage_utils.add_reference_to_stage(leatherback_path, "/World/Leatherback")
    except:
        # Fallback to logic in env if that file is missing (env has fallback)
        from isaacsim.storage.native import get_assets_root_path
        assets_root = get_assets_root_path()
        leatherback_path = assets_root + "/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd"
        stage_utils.add_reference_to_stage(leatherback_path, "/World/Leatherback")

    print("\n" + "="*50)
    print("PRINTING HIERARCHY")
    print("="*50)
    
    stage = world.stage
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Leatherback")):
        print(prim.GetPath())
        
    sim.close()

if __name__ == "__main__":
    main()
