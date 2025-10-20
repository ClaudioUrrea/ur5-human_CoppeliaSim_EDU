# Contributing to LBCF-MPC

Thank you for your interest in contributing to the LBCF-MPC project! This document provides guidelines for contributing to the codebase.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Setup](#development-setup)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

---

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ur5-human_CoppeliaSim_EDU.git
   cd ur5-human_CoppeliaSim_EDU
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ClaudioUrrea/ur5-human_CoppeliaSim_EDU.git
   ```

---

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)
- Relevant logs or error messages

### Suggesting Enhancements

For feature requests:

- Explain the motivation and use case
- Describe the proposed solution
- Consider implementation complexity
- Discuss alternatives you've considered

### Code Contributions

We welcome:

- Bug fixes
- New features
- Documentation improvements
- Performance optimizations
- Test coverage improvements

---

## Development Setup

### 1. Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt
```

### 2. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### 3. Download Test Data

```bash
# Download sample data for testing
python scripts/download_test_data.py
```

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- Maximum line length: 100 characters
- Use type hints for function signatures
- Docstrings: Google style

### Code Formatting

We use **black** for code formatting:

```bash
# Format all Python files
black src/ tests/ experiments/

# Check formatting
black --check src/
```

### Linting

We use **flake8** for linting:

```bash
# Run linter
flake8 src/ tests/

# Configuration in .flake8
```

### Type Checking

We use **mypy** for static type checking:

```bash
# Run type checker
mypy src/
```

### Example Code Style

```python
from typing import Optional, Tuple
import numpy as np
import torch


def compute_cbf_value(
    state: np.ndarray,
    model: torch.nn.Module,
    threshold: float = 0.15
) -> Tuple[float, bool]:
    """
    Compute Control Barrier Function value for given state.
    
    Args:
        state: Robot-human state vector [38D]
        model: Trained CBF neural network
        threshold: Safety distance threshold (meters)
    
    Returns:
        Tuple of (barrier_value, is_safe)
        
    Raises:
        ValueError: If state dimension is incorrect
        
    Example:
        >>> state = np.random.randn(38)
        >>> h_value, is_safe = compute_cbf_value(state, cbf_model)
        >>> print(f"Barrier: {h_value:.3f}, Safe: {is_safe}")
    """
    if len(state) != 38:
        raise ValueError(f"Expected state dim 38, got {len(state)}")
    
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        h_value = model(state_tensor).item()
    
    is_safe = h_value >= 0
    
    return h_value, is_safe
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_cbf.py

# Run specific test
pytest tests/test_cbf.py::test_lipschitz_constraint
```

### Writing Tests

Place tests in `tests/` directory:

```python
# tests/test_cbf.py
import pytest
import numpy as np
from src.cbf.lipschitz_network import LipschitzCBFNetwork


def test_cbf_forward_pass():
    """Test CBF network forward pass"""
    model = LipschitzCBFNetwork(input_dim=38)
    state = torch.randn(1, 38)
    output = model(state)
    
    assert output.shape == (1, 1)
    assert not torch.isnan(output).any()


def test_lipschitz_constant():
    """Test Lipschitz constant constraint"""
    model = LipschitzCBFNetwork(input_dim=38)
    L = model.lipschitz_constant()
    
    assert L <= 1.1  # Allow small tolerance
    assert L >= 0.9


@pytest.mark.parametrize("input_dim", [10, 38, 50])
def test_different_dimensions(input_dim):
    """Test CBF with different input dimensions"""
    model = LipschitzCBFNetwork(input_dim=input_dim)
    state = torch.randn(5, input_dim)
    output = model(state)
    
    assert output.shape == (5, 1)
```

---

## Pull Request Process

### 1. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, documented code
- Add tests for new functionality
- Update documentation as needed
- Follow coding standards

### 3. Commit Changes

Use clear, descriptive commit messages:

```bash
git add src/cbf/new_feature.py tests/test_new_feature.py
git commit -m "Add Lipschitz network with spectral normalization

- Implement spectral normalization layers
- Add tests for Lipschitz constant verification
- Update documentation with usage examples

Closes #123"
```

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- **Title**: Clear, concise description
- **Description**: 
  - What changes were made
  - Why they were made
  - How to test them
  - Related issues
- **Checklist**:
  - [ ] Tests pass locally
  - [ ] Code follows style guidelines
  - [ ] Documentation updated
  - [ ] No breaking changes (or documented)

### 5. Code Review

- Respond to reviewer comments
- Make requested changes
- Push updates to the same branch
- Request re-review when ready

### 6. Merge

Once approved:
- Squash commits if needed
- Maintainer will merge the PR
- Delete your feature branch

---

## Development Workflow

### Typical Development Cycle

1. **Issue Creation**: Discuss what you want to work on
2. **Branch**: Create feature branch from `main`
3. **Develop**: Write code following standards
4. **Test**: Add and run tests
5. **Document**: Update docs and docstrings
6. **Commit**: Make clear, atomic commits
7. **Push**: Push to your fork
8. **PR**: Create pull request
9. **Review**: Address feedback
10. **Merge**: Maintainer merges when ready

### Branch Naming Convention

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation only
- `refactor/` - Code refactoring
- `test/` - Test additions/changes

Examples:
- `feature/gp-online-adaptation`
- `fix/mpc-constraint-violation`
- `docs/api-documentation`

---

## Questions?

If you have questions about contributing:

- Open a GitHub issue with the `question` label
- Email: claudio.urrea@usach.cl
- Check existing issues and discussions

---

## Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in future publications (if substantial contribution)

Thank you for contributing to LBCF-MPC!