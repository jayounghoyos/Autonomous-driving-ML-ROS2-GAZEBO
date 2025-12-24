"""
Isaac Lab integration for Leatherback autonomous vehicle RL.

This package provides Isaac Sim 5.1.0 + Isaac Lab compatible environments
for training autonomous navigation with the NVIDIA Leatherback vehicle.
"""

from isaac_lab.envs import (
    LeatherbackEnv,
    LeatherbackEnvCfg,
    LeatherbackEnvCfgDebug,
    LeatherbackEnvCfgHeadless,
    LeatherbackEnvCfgWithSensors,
)

__all__ = [
    "LeatherbackEnv",
    "LeatherbackEnvCfg",
    "LeatherbackEnvCfgHeadless",
    "LeatherbackEnvCfgWithSensors",
    "LeatherbackEnvCfgDebug",
]
