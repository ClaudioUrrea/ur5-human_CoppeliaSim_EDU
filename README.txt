# LBCF-MPC: Learning-Based Control Barrier Functions for Safe HRC

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Mathematics-green.svg)](https://doi.org/10.3390/mathXXXXXXX)
[![DOI](https://doi.org/10.6084/m9.figshare.30282127)

Official implementation of **"Learning-Based Control Barrier Functions for Safety-Critical Multi-Objective Optimization in Human-Robot Collaborative Manufacturing Systems"** published in *Mathematics* (MDPI), 2026.

![CoppeliaSim Simulation](coppeliasim/Figure 1.png)
*Figure 1: UR5 robot collaborating with human operator in CoppeliaSim simulation (Scenario 3: Simultaneous Operation)*

---

## Overview

This repository provides a complete implementation of LBCF-MPC (Learning-Based Control Barrier Functions with Model Predictive Control) for safe human-robot collaboration in manufacturing environments.

### Key Features

- **Formal Safety Guarantees**: 100% safety compliance (zero violations) across 120 experimental runs
- **Conservatism Reduction**: 58% reduction in unnecessary safety margins compared to manual CBF design
- **Multi-Objective Optimization**: Simultaneous optimization of 6 conflicting objectives (throughput, cycle time, energy, ergonomics, equipment wear, task fairness)
- **Real-Time Performance**: 18.1ms average solution time (50Hz control rate)
- **Learning-Based Barriers**: Lipschitz-constrained neural networks (98.3% validation accuracy)

### Main Contributions

1. **Learned CBFs with Formal Guarantees**: Neural barrier functions maintaining forward invariance under probabilistic guarantees
2. **Efficient Multi-Objective MPC**: O(H³m³) complexity algorithm suitable for real-time control
3. **Conservatism Quantification**: Theoretical characterization of safety margin reduction (52-63% across scenarios)
4. **Experimental Validation**: High-fidelity CoppeliaSim simulation with realistic human motion from CMU MoCap

---

## Data Availability

All experimental data and pre-trained models are publicly available on **FigShare**:

**DOI:** [10.6084/m9.figshare.30282127](https://doi.org/10.6084/m9.figshare.30282127)

### Quick Download

```bash
# Download complete dataset (~505 MB)
wget https://figshare.com/ndownloader/articles/30282127/versions/1 -O lbcfmpc-data.zip

# Extract to data directory
unzip lbcfmpc-data.zip -d data/

# Verify contents
ls -lh data/
```

### Dataset Contents

| File | Size | Description |
|------|------|-------------|
| `training_dataset_50k.npz` | 287 MB | 50,000 labeled state samples for CBF training |
| `cbf_lipschitz_epoch500.pth` | 2.3 MB | Pre-trained CBF neural network (98.3% validation accuracy) |
| `gp_human_predictor.pkl` | 18 MB | Gaussian Process human motion predictor |
| `experimental_results_120runs.pkl` | 145 MB | Complete experimental results (4 scenarios × 6 methods × 30 runs) |
| `mocap_human_motions.pkl` | 50 MB | CMU MoCap human motion sequences |
| `statistical_analysis.csv` | 3.2 MB | Statistical analysis (Friedman & Wilcoxon tests) |

**Total:** ~505 MB

See the [FigShare dataset](https://doi.org/10.6084/m9.figshare.30282127) for complete documentation.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ClaudioUrrea/ur5-human_CoppeliaSim_EDU.git
cd ur5-human_CoppeliaSim_EDU

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Download Pre-trained Models

```bash
# Download from FigShare
wget https://figshare.com/ndownloader/articles/30282127/versions/1 -O data.zip
unzip data.zip -d data/
```

### Run Demo

```bash
# Test with pre-trained models
python examples/demo_lbcf_mpc.py --scenario 3

# Expected output:
# ✓ CBF loaded (98.3% validation accuracy)
# ✓ GP predictor loaded (0.045m RMSE)
# ✓ Running Scenario 3: Simultaneous Operation
# ✓ Safety: 0 violations, min distance: 0.21m
# ✓ Throughput: 142 pieces/hour
```

---

## Reproduce Paper Results

### Complete Reproduction (Using Pre-trained Models)

```bash
# Reproduce all tables and figures from paper
python scripts/reproduce_paper_results.py \
    --skip-training \
    --skip-data-gen \
    --data-dir data/

# Generates:
# - results/tables/table2_safety.csv
# - results/tables/table3_efficiency.csv
# - results/tables/table4_conservatism.csv
# - results/tables/table5_computational.csv
# - results/figures/*.png
```

**Expected runtime:** ~5 minutes

### Validate Against Paper Statistics

```bash
python scripts/validate_reproduction.py

# Output:
# ✓ Table 2: LBCF-MPC violation rate 0.00% (paper: 0.00%)
# ✓ Table 2: Min distance 0.21m (paper: 0.21±0.06m)
# ✓ Table 3: Throughput 142 pc/h (paper: 142±8 pc/h)
# ✓ Table 4: Conservatism reduction 58% (paper: 52-63%)
# All checks passed! ✓
```

---

## Training from Scratch

If you want to retrain models instead of using pre-trained ones:

### Step 1: Generate Training Data

```bash
python scripts/generate_training_data.py \
    --n-samples 50000 \
    --output data/raw/training_dataset.npz

# Runtime: ~10 minutes
# Output: 287 MB dataset
```

### Step 2: Train CBF Neural Network

```bash
python experiments/train_cbf.py \
    --data data/raw/training_dataset.npz \
    --output models/checkpoints \
    --epochs 500 \
    --batch-size 256 \
    --lr 3e-4

# Runtime: ~2-3 hours (GPU), ~8-10 hours (CPU)
# Final metrics:
#   Validation accuracy: >98%
#   Lipschitz constant: ~1.0
#   Validation loss: <0.05
```

### Step 3: Run Experiments

```bash
# Run all experiments (4 scenarios × 6 methods × 30 runs)
python experiments/run_experiments.py \
    --scenarios 1 2 3 4 \
    --methods all \
    --runs 30

# Runtime: ~10-15 hours
# Output: 720 experimental runs
```

### Step 4: Analyze Results

```bash
python experiments/analyze_results.py \
    --results-dir results/ \
    --output results/tables/

# Generates all paper tables
```

---

## Repository Structure

```
ur5-human_CoppeliaSim_EDU/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── cbf/                          # Control Barrier Functions
│   │   ├── lipschitz_network.py     # Lipschitz-constrained neural network
│   │   ├── cbf_trainer.py           # CBF training loop
│   │   └── cbf_loss.py              # Loss functions
│   ├── control/                      # Control algorithms
│   │   ├── mpc_solver.py            # LBCF-MPC implementation
│   │   └── qp_formulation.py        # QP problem formulation
│   ├── prediction/                   # Human motion prediction
│   │   └── gp_predictor.py          # Gaussian Process predictor
│   ├── simulation/                   # Simulation interface
│   │   └── coppeliasim_interface.py # CoppeliaSim API wrapper
│   └── utils/                        # Utilities
│       ├── metrics.py               # Performance metrics
│       └── visualization.py         # Plotting functions
│
├── experiments/                      # Experimental scripts
│   ├── train_cbf.py                 # Train CBF neural network
│   ├── run_experiments.py           # Run validation experiments
│   ├── baseline_methods.py          # Baseline implementations
│   └── analyze_results.py           # Statistical analysis
│
├── scripts/                          # Utility scripts
│   ├── generate_training_data.py    # Generate CBF training data
│   ├── reproduce_paper_results.py   # One-command reproduction
│   ├── validate_reproduction.py     # Verify against paper
│   └── export_figures.py            # Generate paper figures
│
├── examples/                         # Usage examples
│   ├── demo_lbcf_mpc.py            # Basic demo
│   ├── demo_scenario_1.py          # Scenario 1: Coexistence
│   ├── demo_scenario_2.py          # Scenario 2: Sequential
│   ├── demo_scenario_3.py          # Scenario 3: Simultaneous
│   └── demo_scenario_4.py          # Scenario 4: Adaptive
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb    # Dataset analysis
│   ├── 02_cbf_training.ipynb        # Training visualization
│   └── 03_results_visualization.ipynb # Results analysis
│
├── coppeliasim/                      # CoppeliaSim files
│   ├── Scene_in_CoppeliaSim_for_Mathematics.ttt # Simulation scene
│   ├── Escena_3.png                 # Scene screenshot
│   └── python_remote_api.py         # Remote API
│
├── docs/                             # Documentation
│   ├── API.md                       # API documentation
│   ├── EXPERIMENTS.md               # Experimental setup
│   ├── REPRODUCING.md               # Reproduction guide
│   └── ARCHITECTURE.md              # System architecture
│
├── tests/                            # Unit tests
│   ├── test_cbf.py
│   ├── test_mpc.py
│   └── test_gp.py
│
└── results/                          # Results (generated)
    ├── tables/                       # Paper tables (CSV/LaTeX)
    └── figures/                      # Paper figures (PNG/PDF)
```

---

## Cite This Work

If you use this code or dataset in your research, please cite:

```bibtex
@article{urrea2026lbcfmpc,
  title={Learning-Based Control Barrier Functions for Safety-Critical 
         Multi-Objective Optimization in Human-Robot Collaborative 
         Manufacturing Systems},
  author={Urrea, Claudio},
  journal={Mathematics},
  year={2026},
  volume={XX},
  number={YY},
  pages={ZZZ},
  publisher={MDPI},
  doi={10.3390/mathXXXXXXX}
}
```

**Dataset citation:**

```bibtex
@dataset{urrea2025lbcfmpc_data,
  author = {Urrea, Claudio},
  title = {{LBCF-MPC: Training Data and Pre-trained Models 
            for Safe Human-Robot Collaboration}},
  year = {2025},
  publisher = {figshare},
  version = {1.0},
  doi = {10.6084/m9.figshare.30282127},
  url = {https://doi.org/10.6084/m9.figshare.30282127}
}
```

---

## System Requirements

### Minimum Requirements

- **OS:** Ubuntu 20.04+ / Windows 10+ / macOS 11+
- **Python:** 3.8+
- **RAM:** 8 GB
- **Storage:** 2 GB free space
- **CPU:** 4 cores

### Recommended Requirements

- **Python:** 3.10+
- **RAM:** 16 GB
- **Storage:** 5 GB free space
- **GPU:** NVIDIA with 4GB+ VRAM (for training)
- **CPU:** 8 cores @ 3.5GHz+

### Dependencies

```
numpy >= 1.21.0
scipy >= 1.7.0
torch >= 1.10.0
cvxpy >= 1.1.0
osqp >= 0.6.2
casadi >= 3.5.5
matplotlib >= 3.4.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
GPy >= 1.10.0
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The dataset on FigShare is licensed under CC BY 4.0.

---

## Acknowledgments

- **CoppeliaSim** for providing an educational license for high-fidelity simulation
- **CMU Motion Capture Database** for human motion data
- **Faculty of Engineering**, Universidad de Santiago de Chile
- Anonymous reviewers for their constructive feedback

---

## 📧 Contact

**Claudio Urrea, Ph.D.**  
Electrical Engineering Department  
Faculty of Engineering  
Universidad de Santiago de Chile  
Email: claudio.urrea@usach.cl  
GitHub: [@ClaudioUrrea](https://github.com/ClaudioUrrea)  
Institution: [USACH](https://www.usach.cl/)

---

## Links

- **Paper:** [DOI:10.3390/mathXXXXXXX](https://doi.org/10.3390/mathXXXXXXX)
- **Dataset:** [DOI:10.6084/m9.figshare.30282127](https://doi.org/10.6084/m9.figshare.30282127)
- **Documentation:** [docs/](docs/)
- **Issues:** [GitHub Issues](https://github.com/ClaudioUrrea/ur5-human_CoppeliaSim_EDU/issues)

---

** If you find this work useful, please consider starring the repository!**

Last updated: October 19, 2025
