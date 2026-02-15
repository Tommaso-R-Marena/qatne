# Contributing to QATNE

Thank you for your interest in contributing to QATNE!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/qatne.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit: `git commit -m "Add your feature"`
7. Push: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate qatne

# Install in development mode
pip install -e .

# Install dev dependencies
pip install pytest pytest-cov black flake8 mypy
```

## Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Write unit tests for new features
- Maintain >80% test coverage
- Use type hints where appropriate

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=qatne tests/

# Run specific test
pytest tests/test_tensor_network.py::TestTensorNetwork::test_initialization
```

## Documentation

- Update docstrings for any API changes
- Add examples to tutorials if adding new features
- Update README.md if changing core functionality

## Pull Request Guidelines

- Describe what your PR does and why
- Reference any related issues
- Ensure all tests pass
- Update documentation as needed
- Keep PRs focused on a single feature/fix

## Questions?

Feel free to open an issue for questions or discussions!
