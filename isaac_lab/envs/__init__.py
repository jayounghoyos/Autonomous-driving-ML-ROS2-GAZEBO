"""
Isaac Lab Navigation Environments.

Provides Gymnasium-compatible environments for RL training
in Isaac Sim 5.1.0.

Available robots:
- LeatherbackEnv: NVIDIA Leatherback (Ackermann steering, 4-wheel)
- DifferentialDriveEnv: 4-wheel skid-steer robot (matches real hardware)
"""

from .leatherback_env import LeatherbackEnv
from .leatherback_env_cfg import (
    LeatherbackEnvCfg,
    LeatherbackEnvCfgDebug,
    LeatherbackEnvCfgHeadless,
    LeatherbackEnvCfgWithSensors,
)

from .differential_drive_env import DifferentialDriveEnv
from .differential_drive_env_cfg import (
    DifferentialDriveEnvCfg,
    DifferentialDriveEnvCfgFullSensors,
    DifferentialDriveEnvCfgTest,
    DifferentialDriveEnvCfgBARN,
)

__all__ = [
    # Leatherback (Ackermann)
    "LeatherbackEnv",
    "LeatherbackEnvCfg",
    "LeatherbackEnvCfgHeadless",
    "LeatherbackEnvCfgWithSensors",
    "LeatherbackEnvCfgDebug",
    # Differential Drive (Skid-Steer)
    "DifferentialDriveEnv",
    "DifferentialDriveEnvCfg",
    "DifferentialDriveEnvCfgFullSensors",
    "DifferentialDriveEnvCfgTest",
    "DifferentialDriveEnvCfgBARN",
]
