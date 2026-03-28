"""
Isaac Lab integration for Jackal autonomous robot RL.

Provides Isaac Sim 5.1.0 compatible environments for training
autonomous navigation with the Clearpath Jackal (skid-steer).
"""

from isaac_lab.envs import (
    DifferentialDriveEnv,
    DifferentialDriveEnvCfg,
    DifferentialDriveEnvCfgFullSensors,
    DifferentialDriveEnvCfgTest,
    DifferentialDriveEnvCfgBARN,
)

__all__ = [
    "DifferentialDriveEnv",
    "DifferentialDriveEnvCfg",
    "DifferentialDriveEnvCfgFullSensors",
    "DifferentialDriveEnvCfgTest",
    "DifferentialDriveEnvCfgBARN",
]
