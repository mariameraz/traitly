# setup.py

from setuptools import setup, find_packages
import os
import sys

# Read README if it exists
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Traitly: Morphological analysis of fruits in images"

# Version
VERSION = "0.1.0"

# Base dependencies
INSTALL_REQUIRES = [
    # Image processing
    "opencv-python>=4.5.0",
    "numpy>=1.20.0,<2.0.0",  # Compatibility with opencv
    "Pillow>=8.0.0",
    
    # Scientific analysis
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    
    # Visualization
    "matplotlib>=3.4.0",
    
    # OCR and detection
    "pyzbar>=0.1.9",          # QR code detection
    "easyocr>=1.6.0",         # OCR text detection
    
    # Machine Learning
    "ultralytics>=8.0.0",     # YOLO models
    "torch>=1.10.0",          # Required by ultralytics
    "torchvision>=0.11.0",
    
    # PDF processing
    "PyMuPDF>=1.19.0",        # fitz
    
    # Utilities
    "tqdm>=4.62.0",           # Progress bars
    "psutil>=5.8.0",          # System monitoring
]

# Optional dependencies for development
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
        "ipython>=8.0.0",
        "jupyter>=1.0.0",
    ],
    "docs": [
        "sphinx>=4.5.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
    # For systems without GPU (CPU-only torch)
    "cpu": [
        "torch==2.0.1+cpu",
        "torchvision==0.15.2+cpu",
    ],
}

# All optional dependencies combined
EXTRAS_REQUIRE["all"] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

# Platform-specific configuration
PLATFORM_SPECIFIC = {}

if sys.platform == "darwin":  # macOS
    # On macOS, opencv-python sometimes needs special configuration
    PLATFORM_SPECIFIC["opencv-python-headless"] = ">=4.5.0"
elif sys.platform == "win32":  # Windows
    # Windows-specific packages if needed
    pass
elif sys.platform.startswith("linux"):  # Linux
    # Linux-specific packages if needed
    pass

setup(
    name="traitly",
    version=0.1.0-alpha,
    
    # Metadata
    author="Maria Alejandra Torres Meraz",
    author_email="ma.torresmeraz@gmail.com",
    description="Morphological analysis of fruits in images using computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/traitly",
    license="MIT",
    
    # Packages
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "traitly": [
            "package_data/models/size_reference.pt",  # YOLO models if you include them
            "config/*.yaml",
        ],
    },
    
    # Dependencies
    python_requires=">=3.8,<3.12",  # Python 3.8-3.11
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Classification
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    # Entry points (optional - for CLI)
    entry_points={
        "console_scripts": [
            "traitly=traitly.cli:main",  # If you create a CLI
        ],
    },
    
    # Keywords for PyPI search
    keywords=[
        "computer-vision",
        "image-analysis",
        "fruit-analysis",
        "locule-analysis"
        "phenotyping",
        "morphology",
        "opencv",
        "machine-learning",
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/mariameraz/traitly/issues",
        "Documentation": "",
        "Source Code": "https://github.com/mariameraz/traitly",
    },
)
