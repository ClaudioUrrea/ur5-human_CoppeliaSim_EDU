"""
LBCF-MPC: Learning-Based Control Barrier Functions for Safe HRC
Setup script for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]

setup(
    name="lbcf-mpc",
    version="1.0.0",
    author="Claudio Urrea",
    author_email="claudio.urrea@usach.cl",
    description="Learning-Based Control Barrier Functions for Safe Human-Robot Collaboration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ClaudioUrrea/ur5-human_CoppeliaSim_EDU",
    project_urls={
        "Bug Tracker": "https://github.com/ClaudioUrrea/ur5-human_CoppeliaSim_EDU/issues",
        "Documentation": "https://github.com/ClaudioUrrea/ur5-human_CoppeliaSim_EDU/docs",
        "Source Code": "https://github.com/ClaudioUrrea/ur5-human_CoppeliaSim_EDU",
        "Paper": "https://doi.org/10.3390/mathXXXXXXX",
        "Dataset": "https://doi.org/10.6084/m9.figshare.30282127",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.6b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.13.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lbcf-mpc=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yaml", "*.yml"],
    },
    keywords=[
        "control barrier functions",
        "human-robot collaboration",
        "safety-critical systems",
        "model predictive control",
        "deep learning",
        "robotics",
        "manufacturing automation",
        "cobots",
        "collision avoidance",
    ],
    zip_safe=False,
)