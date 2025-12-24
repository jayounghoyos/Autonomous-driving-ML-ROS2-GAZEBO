"""
Leatherback environment module.

Provides Gymnasium-compatible environments for RL training
with the NVIDIA Leatherback vehicle in Isaac Sim.
"""

from .leatherback_env import LeatherbackEnv
from .leatherback_env_cfg import (
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
