# Reproducing Paper Results

This document provides step-by-step instructions to reproduce all results from the paper "Learning-Based Control Barrier Functions for Safety-Critical Multi-Objective Optimization in Human-Robot Collaborative Manufacturing Systems."

## Prerequisites

- Python 3.8+
- CoppeliaSim Edu 4.10.0 (optional, for new experiments)
- 16GB RAM
- CUDA-capable GPU (recommended, not required)

## Installation
```bash
# Clone repository
git clone https://github.com/claudiourrea/LBCF-MPC-HRC.git
cd LBCF-MPC-HRC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package
pip install -e .