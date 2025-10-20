# Changelog

All notable changes to the LBCF-MPC project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2025-10-19

### Added

#### Core Implementation
- **Lipschitz-Constrained Neural Networks**: Spectral normalization for CBF
- **LBCF-MPC Algorithm**: Multi-objective MPC with learned barriers
- **GP Human Predictor**: Gaussian Process motion forecasting (0.4s horizon)
- **Four HRC Scenarios**: Coexistence, Sequential, Simultaneous, Adaptive
- **Six Baseline Methods**: Manual CBF, MPC-Soft, Reactive CBF, Safe-RL, PID

#### Data and Models
- Training dataset: 50,000 labeled samples (287 MB)
- Pre-trained CBF model: 98.3% validation accuracy, L ≤ 1.04
- GP predictor trained on CMU MoCap (subjects 07, 13, 20)
- Complete experimental results: 720 runs (4 scenarios × 6 methods × 30 runs)
- Statistical analysis: Friedman and Wilcoxon tests with effect sizes

#### Simulation
- CoppeliaSim integration with UR5 robot model
- Realistic human motion from CMU Motion Capture Database
- ISO/TS 15066 compliant safety distance calculations
- High-fidelity collision detection

#### Documentation
- Comprehensive README with quick start guide
- API documentation for all modules
- Jupyter notebooks for data exploration and visualization
- Reproduction scripts for all paper results
- Contributing guidelines

#### Testing
- Unit tests for CBF, MPC, and GP modules
- Integration tests for full pipeline
- Validation against paper statistics

### Results

- **Safety**: 100% compliance (0 violations) across all LBCF-MPC runs
- **Conservatism Reduction**: 58% average (52-63% across scenarios)
- **Multi-Objective**: 28% hypervolume improvement over baselines
- **Computational**: 18.1ms average solution time (50Hz capable)
- **Statistical**: Large effect sizes (Cohen's d = 1.18-3.42, p < 0.001)

### Infrastructure

- Python 3.8+ support
- PyTorch deep learning framework
- CVXPY/OSQP for convex optimization
- CasADi for nonlinear optimization
- GitHub Actions CI/CD (coming soon)

### Published

- **Paper**: Mathematics (MDPI), 2025
- **Dataset**: FigShare DOI:10.6084/m9.figshare.30282127
- **Code**: GitHub - MIT License
- **Data**: FigShare - CC BY 4.0 License

---

## [Unreleased]

### Planned Features

- [ ] ROS/ROS2 integration for real robot deployment
- [ ] Online adaptation with meta-learning
- [ ] Multi-agent HRC scenarios
- [ ] Transfer learning to different robot platforms
- [ ] Real-time visualization dashboard
- [ ] Docker containers for easy deployment
- [ ] Benchmark suite for comparing CBF methods

### Under Development

- [ ] Extended documentation with tutorials
- [ ] Additional baseline comparisons
- [ ] Sensitivity analysis tools
- [ ] Performance profiling utilities

---

## Version History

### Version Numbering

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Comparison Links

- [Unreleased](https://github.com/ClaudioUrrea/ur5-human_CoppeliaSim_EDU/compare/v1.0.0...HEAD)
- [1.0.0](https://github.com/ClaudioUrrea/ur5-human_CoppeliaSim_EDU/releases/tag/v1.0.0)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to this project.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{urrea2025lbcfmpc,
  title={Learning-Based Control Barrier Functions for Safety-Critical 
         Multi-Objective Optimization in Human-Robot Collaborative 
         Manufacturing Systems},
  author={Urrea, Claudio},
  journal={Mathematics},
  year={2025},
  publisher={MDPI}
}
```