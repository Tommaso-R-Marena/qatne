# Contributing to QATNE

Thank you for your interest in contributing to QATNE! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Testing Guidelines](#testing-guidelines)
5. [Documentation](#documentation)
6. [Pull Request Process](#pull-request-process)
7. [Coding Standards](#coding-standards)
8. [Areas for Contribution](#areas-for-contribution)

---

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

### Summary

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect differing viewpoints and experiences
- Accept responsibility and apologize for mistakes

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Familiarity with quantum computing concepts (helpful but not required)
- Basic understanding of tensor networks (for core development)

### Setting Up Development Environment

1. **Fork the repository**

   Click the "Fork" button on GitHub to create your own copy.

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/qatne.git
   cd qatne
   ```

3. **Add upstream remote**

   ```bash
   git remote add upstream https://github.com/Tommaso-R-Marena/qatne.git
   ```

4. **Create development environment**

   Using conda (recommended):
   ```bash
   conda env create -f environment.yml
   conda activate qatne
   ```

   Or using pip:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .[dev]
   ```

5. **Verify installation**

   ```bash
   pytest tests/ -v
   ```

---

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or improvements

### 2. Make Your Changes

- Write clear, concise commit messages
- Keep commits focused and atomic
- Follow the coding standards (see below)

### 3. Write Tests

All new code should include tests:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_qatne_solver.py -v

# Run with coverage
pytest tests/ --cov=qatne --cov-report=html
```

### 4. Update Documentation

- Add docstrings to all public functions/classes
- Update relevant markdown files in `docs/`
- Add examples to notebooks if appropriate

### 5. Commit and Push

```bash
git add .
git commit -m "Add feature: brief description"
git push origin feature/your-feature-name
```

---

## Testing Guidelines

### Test Categories

**Unit Tests** - Test individual functions/methods
```python
def test_tensor_network_initialization():
    tn = TensorNetwork(num_sites=4, bond_dim=2)
    assert tn.num_sites == 4
    assert tn.bond_dim == 2
```

**Integration Tests** - Test component interactions
```python
@pytest.mark.integration
def test_full_optimization_pipeline():
    # Test complete workflow
    pass
```

**Slow Tests** - Long-running tests
```python
@pytest.mark.slow
def test_convergence_on_large_system():
    # Tests that take > 10 seconds
    pass
```

### Running Tests

```bash
# Fast tests only (default)
pytest tests/ -m "not slow"

# All tests including slow ones
pytest tests/

# Specific marker
pytest tests/ -m integration

# Parallel execution
pytest tests/ -n auto
```

### Test Coverage

We aim for >80% code coverage:

```bash
pytest tests/ --cov=qatne --cov-report=html
open htmlcov/index.html
```

### Writing Good Tests

✅ **DO:**
- Test edge cases and boundary conditions
- Use descriptive test names
- Include docstrings explaining what's being tested
- Use fixtures for common setup
- Test both success and failure cases

❌ **DON'T:**
- Write tests that depend on external services
- Use hardcoded paths or credentials
- Write tests that depend on test execution order
- Test implementation details instead of behavior

---

## Documentation

### Docstring Format

We use Google-style docstrings:

```python
def compute_energy(
    self, 
    params: np.ndarray
) -> float:
    """
    Compute expectation value of Hamiltonian.
    
    Uses parameter shift rule for gradient estimation.
    Measurements are performed in the computational basis.
    
    Args:
        params: Circuit parameters, shape (num_params,)
    
    Returns:
        Energy expectation value in Hartree
    
    Raises:
        ValueError: If params has incorrect shape
    
    Example:
        >>> solver = QATNESolver(hamiltonian=H, num_qubits=4)
        >>> params = np.random.randn(20)
        >>> energy = solver.compute_energy(params)
        >>> print(f"Energy: {energy:.6f} Ha")
    """
    # Implementation
```

### Documentation Structure

- **Theory** (`docs/theory/`) - Mathematical foundations, proofs
- **Tutorials** (`docs/tutorials/`) - Step-by-step guides
- **API Reference** (`docs/api/`) - Function/class documentation
- **Examples** (`notebooks/`) - Jupyter notebooks

### Building Documentation

```bash
cd docs/
make html
open _build/html/index.html
```

---

## Pull Request Process

### Before Submitting

1. ✅ All tests pass
2. ✅ Code follows style guidelines
3. ✅ Documentation is updated
4. ✅ Commit messages are clear
5. ✅ Branch is up to date with main

### Submitting a Pull Request

1. **Update your branch**

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create PR on GitHub**

   - Click "New Pull Request"
   - Select your branch
   - Fill out the template (see below)

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No breaking changes (or documented)

## Related Issues
Closes #123
```

### Review Process

1. **Automated Checks** - CI runs tests automatically
2. **Code Review** - Maintainer reviews your code
3. **Revisions** - Address feedback if needed
4. **Approval** - Maintainer approves and merges

---

## Coding Standards

### Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Imports**: Grouped and sorted with `isort`
- **Formatting**: Use `black` for consistent formatting
- **Type hints**: Use type hints for function signatures

### Code Formatting

```bash
# Format code
black qatne/ tests/

# Sort imports
isort qatne/ tests/

# Check style
flake8 qatne/ tests/ --max-line-length=100
```

### Type Hints

```python
from typing import Tuple, List, Optional
import numpy as np

def solve(
    self,
    initial_params: Optional[np.ndarray] = None,
    max_iterations: int = 1000
) -> Tuple[float, np.ndarray]:
    """Type hints for clarity."""
    pass
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `QATNESolver`)
- **Functions/Methods**: `snake_case` (e.g., `compute_energy`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`)
- **Private**: Prefix with `_` (e.g., `_internal_method`)

### Best Practices

✅ **DO:**
```python
# Clear variable names
ground_energy = solver.solve(max_iterations=100)

# Explicit comparisons
if result is not None:
    process(result)

# List comprehensions for simple cases
squares = [x**2 for x in range(10)]
```

❌ **DON'T:**
```python
# Unclear abbreviations
e = s.slv(mi=100)

# Implicit comparisons
if result:
    process(result)

# Complex nested comprehensions
result = [[f(x, y) for x in range(10) if g(x)] for y in range(20) if h(y)]
```

---

## Areas for Contribution

### High Priority

1. **Algorithm Improvements**
   - Alternative tensor network structures (tree, PEPS)
   - Advanced optimization methods (ADAM, L-BFGS)
   - Error mitigation techniques

2. **Hardware Integration**
   - Real quantum hardware testing (IBM, Rigetti, IonQ)
   - Noise modeling and characterization
   - Hardware-efficient ansätze

3. **Performance Optimization**
   - Parallel gradient computation
   - GPU acceleration for classical parts
   - Memory-efficient tensor operations

4. **Testing and Validation**
   - More molecular benchmarks (H₂O, N₂, etc.)
   - Comparison with other VQE implementations
   - Statistical analysis tools

### Medium Priority

5. **Documentation**
   - Tutorial notebooks for beginners
   - Advanced usage examples
   - Video tutorials
   - Blog posts

6. **Visualization**
   - Interactive circuit visualizations
   - Real-time optimization dashboards
   - 3D molecular structure overlays

7. **Extensions**
   - Time-evolution (VQE → QITE)
   - Excited states (VQE → SSVQE)
   - Open quantum systems

### Good First Issues

- 📝 Improve docstrings
- 🐛 Fix typos in documentation
- ✨ Add type hints to existing code
- 🧪 Write additional unit tests
- 📊 Create visualization examples
- 📚 Translate documentation

---

## Questions?

- **GitHub Discussions**: Ask questions, share ideas
- **Issues**: Report bugs, request features
- **Email**: [Contact maintainer directly]

---

## Recognition

Contributors will be:

- Listed in `CONTRIBUTORS.md`
- Acknowledged in release notes
- Invited to co-author papers using QATNE (if significant contribution)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to QATNE! 🚀
