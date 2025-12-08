from omni.isaac.core.articulations import Articulation
import inspect
from isaacsim.core.prims.impl.single_articulation import SingleArticulation

print("Articulation signature:")
try:
    print(inspect.signature(Articulation.set_joint_velocities))
except Exception as e:
    print(e)
    
print("SingleArticulation signature:")
try:
    print(inspect.signature(SingleArticulation.set_joint_velocities))
except Exception as e:
    print(e)
