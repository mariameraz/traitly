# traitly/utils/metadata.py
# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import Dict, Optional
import importlib.metadata
import sys
from packaging.requirements import Requirement
from datetime import datetime
import platform
import psutil
import torch

#############################################
# GPU
#############################################
try:
    _GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    _GPU_AVAILABLE = False
except Exception:
    _GPU_AVAILABLE = False

#########################################
## Get info about package dependencies ##
#########################################

def get_package_versions(
    package_name: str = "traitly"
) -> Dict[str, str]:
    # Import all the dependencies required by traitly
    requires = importlib.metadata.requires(package_name) or []
    # Clean the returnt and only keep the package's name
    packages = [Requirement(req).name for req in requires]

    # Check if they are installed and return their version
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"

    return versions

def get_session_metadata(
    img_path: Optional[str] = None,
) -> list[str]:
    traitly_version = importlib.metadata.version("traitly")
    lines = [
        f"traitly: {traitly_version}",
        f"python: {platform.python_version()}",
        f"run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if img_path is not None:
        lines.insert(2, f"image: {img_path}")
    return lines


def get_system_metadata() -> list[str]:
    return [
        f"os: {platform.system()} {platform.release()}",
        f"architecture: {platform.machine()}",
        f"cpu cores: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count(logical=True)} threads)",
        f"ram: {psutil.virtual_memory().total / (1024**3):.1f} GB",
        f"gpu: {'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'none (cpu only)'}",
    ]
