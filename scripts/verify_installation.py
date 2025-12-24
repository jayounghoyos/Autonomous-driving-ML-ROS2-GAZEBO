#!/usr/bin/env python3
"""
Installation Verification Script.

Verifies that all components are properly installed:
- Python version
- PyTorch + CUDA
- Isaac Sim
- Isaac Lab
- ROS2 Jazzy

Usage:
    python scripts/verify_installation.py
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def ok(msg: str) -> None:
    """Print success message."""
    print(f"  {Colors.GREEN}[OK]{Colors.RESET} {msg}")


def fail(msg: str) -> None:
    """Print failure message."""
    print(f"  {Colors.RED}[FAIL]{Colors.RESET} {msg}")


def warn(msg: str) -> None:
    """Print warning message."""
    print(f"  {Colors.YELLOW}[WARN]{Colors.RESET} {msg}")


def header(msg: str) -> None:
    """Print section header."""
    print(f"\n{Colors.BOLD}{msg}{Colors.RESET}")
    print("-" * 50)


def check_python() -> bool:
    """Verify Python version is 3.11."""
    header("Python Version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor == 11:
        ok(f"Python {version_str}")
        return True
    else:
        fail(f"Python {version_str} (requires 3.11)")
        return False


def check_torch() -> bool:
    """Verify PyTorch and CUDA."""
    header("PyTorch + CUDA")
    try:
        import torch

        ok(f"PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            ok(f"CUDA Available: Yes")
            ok(f"CUDA Version: {torch.version.cuda}")
            ok(f"GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok(f"VRAM: {vram:.1f} GB")
            return True
        else:
            warn("CUDA not available (CPU only)")
            return True

    except ImportError as e:
        fail(f"PyTorch not installed: {e}")
        return False


def check_gymnasium() -> bool:
    """Verify Gymnasium is installed."""
    header("Gymnasium")
    try:
        import gymnasium

        ok(f"Gymnasium {gymnasium.__version__}")
        return True
    except ImportError as e:
        fail(f"Gymnasium not installed: {e}")
        return False


def check_stable_baselines() -> bool:
    """Verify Stable-Baselines3 is installed."""
    header("Stable-Baselines3")
    try:
        import stable_baselines3

        ok(f"Stable-Baselines3 {stable_baselines3.__version__}")
        return True
    except ImportError as e:
        fail(f"Stable-Baselines3 not installed: {e}")
        return False


def check_isaac_sim() -> bool:
    """Verify Isaac Sim can be imported."""
    header("Isaac Sim")
    try:
        from isaacsim import SimulationApp

        ok("Isaac Sim importable")
        return True
    except ImportError as e:
        fail(f"Isaac Sim not installed: {e}")
        print("    Install with: pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com")
        return False


def check_ros2() -> bool:
    """Verify ROS2 Jazzy is available."""
    header("ROS2 Jazzy")
    try:
        import rclpy

        ok("rclpy importable")

        # Check ROS_DISTRO
        import os
        distro = os.environ.get("ROS_DISTRO", "not set")
        if distro == "jazzy":
            ok(f"ROS_DISTRO: {distro}")
        elif distro == "not set":
            warn("ROS_DISTRO not set (source /opt/ros/jazzy/setup.bash)")
        else:
            warn(f"ROS_DISTRO: {distro} (expected jazzy)")

        return True
    except ImportError as e:
        fail(f"ROS2 not available: {e}")
        print("    Install with: sudo apt install ros-jazzy-desktop")
        return False


def check_project_structure() -> bool:
    """Verify project structure is correct."""
    header("Project Structure")
    project_root = Path(__file__).parent.parent

    required_dirs = [
        "isaac_lab/envs",
        "training/configs",
        "ros2_ws/src",
        "scripts",
        "models",
        "logs",
    ]

    required_files = [
        "isaac_lab/envs/leatherback_env.py",
        "isaac_lab/envs/leatherback_env_cfg.py",
        "training/train_ppo.py",
        "training/evaluate.py",
        "training/configs/ppo_config.yaml",
        "requirements.txt",
        "pyproject.toml",
        "README.md",
    ]

    all_ok = True

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.is_dir():
            ok(f"Directory: {dir_path}/")
        else:
            fail(f"Missing directory: {dir_path}/")
            all_ok = False

    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.is_file():
            ok(f"File: {file_path}")
        else:
            fail(f"Missing file: {file_path}")
            all_ok = False

    return all_ok


def main() -> int:
    """Run all verification checks."""
    print("=" * 60)
    print(f"{Colors.BOLD}Leatherback RL - Installation Verification{Colors.RESET}")
    print("=" * 60)

    checks = [
        ("Python", check_python),
        ("PyTorch", check_torch),
        ("Gymnasium", check_gymnasium),
        ("Stable-Baselines3", check_stable_baselines),
        ("Project Structure", check_project_structure),
    ]

    # Optional checks (don't fail if not present)
    optional_checks = [
        ("Isaac Sim", check_isaac_sim),
        ("ROS2", check_ros2),
    ]

    results = {}

    # Required checks
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            fail(f"{name} check failed with error: {e}")
            results[name] = False

    # Optional checks
    for name, check_fn in optional_checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            warn(f"{name} check skipped: {e}")
            results[name] = None

    # Summary
    header("Summary")
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    print(f"\n  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")

    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All required checks passed!{Colors.RESET}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}Some checks failed. Please fix the issues above.{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
