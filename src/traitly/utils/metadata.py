from typing import Dict
import importlib.metadata
import sys
from packaging.requirements import Requirement

def get_package_versions(
    package_name: str = "traitly"
) -> Dict[str, str]:
    # Import all the dependencies required by traitly
    requires = importlib.metadata.requires(package_name) or []
    # Clean the returnt and only keep the package's name
    packages = [Requirement(req).name for req in requires]
    # Include Python info
    versions = {"python": sys.version.split()[0]}

    # Check if they are installed and return their version
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"

    return versions
