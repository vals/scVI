# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

scVI-tools is a package for probabilistic modeling and analysis of single-cell omics data, built on PyTorch and AnnData. It provides both high-level APIs for common analyses and building blocks for developing novel probabilistic models.

## Development Environment Setup

### Installation
```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"
# or with uv
uv pip install -e ".[dev]"

# Optional: Set up pre-commit hooks
pre-commit install
```

### Python Version Support
- Minimum: Python 3.10
- Tested on: Python 3.10, 3.11, 3.12

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
coverage run -m pytest -v --color=yes
coverage report

# Run specific test categories (use pytest markers)
pytest -m "not optional"  # Skip optional tests
pytest -m internet --internet-tests  # Run internet-dependent tests
pytest -m multigpu --multigpu-tests  # Run multi-GPU tests
pytest -m autotune --autotune-tests  # Run autotune tests
pytest -m "custom dataloaders" --custom-dataloader-tests  # Run custom dataloader tests

# Run specific test files or functions
pytest tests/test_my_change.py
pytest tests/test_my_change.py::test_my_change
```

### Test Categories (Pytest Markers)
- `internet`: Tests requiring internet access
- `optional`: Optional tests (usually slower)
- `private`: Tests using private keys (e.g., HuggingFace)
- `multigpu`: Multi-GPU performance tests
- `autotune`: Ray autotune capability tests
- `custom dataloaders`: Custom data loader tests

### Test Configuration
- Tests are located in `tests/` directory
- Configuration in `pyproject.toml` under `[tool.pytest.ini_options]`
- Test fixtures and configuration in `tests/conftest.py`

## Code Quality and Linting

### Tools Used
- **Ruff**: Code formatting and linting (replaces Black + flake8)
- **Pre-commit**: Automated code quality checks
- **Prettier**: YAML file formatting
- **Mdformat**: Markdown formatting

### Running Code Quality Checks
```bash
# Run pre-commit on modified files
pre-commit

# Run pre-commit on all files
pre-commit run --all

# Manual ruff usage
ruff check .  # Linting
ruff format .  # Formatting
```

### Code Style Requirements
- Line length: 99 characters
- Docstring style: numpydoc format
- Type hints recommended for new code (PEP 484/526)
- Conventional Commits for main branch commits

## Package Architecture

The scVI codebase is organized into several key modules:

### Core Structure (`src/scvi/`)

1. **`data/`**: Data handling and preprocessing
   - `_manager.py`: AnnDataManager for data registration
   - `fields/`: Data field definitions for different data types
   - `_built_in_data/`: Built-in datasets
   - `_preprocessing.py`: Data preprocessing utilities

2. **`model/`**: High-level model implementations
   - `_scvi.py`: Main scVI model
   - `_scanvi.py`: Semi-supervised scVI
   - `_totalvi.py`: Multi-modal scVI
   - `base/`: Base model classes and mixins
   - Core models: scVI, scANVI, totalVI, MULTIVI, PeakVI, etc.

3. **`module/`**: PyTorch Lightning modules (model implementations)
   - `_vae.py`: Variational autoencoder implementations
   - `base/`: Base module classes
   - Neural network architectures for various models

4. **`external/`**: External model implementations
   - Community-contributed models
   - Each subdirectory contains model-specific implementations
   - Examples: CellAssign, Stereoscope, GIMVI, etc.

5. **`train/`**: Training infrastructure
   - `_trainer.py`: Training orchestration
   - `_trainingplans.py`: Training plan definitions
   - `_callbacks.py`: Training callbacks

6. **`dataloaders/`**: Data loading utilities
   - `_ann_dataloader.py`: AnnData-specific data loaders
   - `_data_splitting.py`: Data splitting utilities

7. **`nn/`**: Neural network components
   - `_base_components.py`: Reusable neural network layers
   - `_embedding.py`: Embedding layers

8. **`distributions/`**: Custom probability distributions
   - PyTorch-compatible distributions for single-cell data

9. **`hub/`**: Model sharing and versioning
   - Integration with HuggingFace Hub

10. **`autotune/`**: Hyperparameter optimization
    - Ray Tune integration

11. **`criticism/`**: Model criticism tools
    - Posterior predictive checks

### Key Design Patterns

1. **Model-Module Separation**:
   - Models (`model/`) provide high-level APIs
   - Modules (`module/`) implement PyTorch Lightning modules

2. **AnnDataManager System**:
   - Centralized data registration and validation
   - Field-based data handling for different data types

3. **Base Classes**:
   - `BaseModelClass`: Foundation for all models
   - Mixins for common functionality (VAE, training, etc.)

4. **PyTorch Lightning Integration**:
   - All training uses Lightning for GPU/multi-GPU support
   - Training plans define optimization strategies

## Build System

- **Build backend**: Hatchling
- **Source directory**: `src/scvi/`
- **Package name**: `scvi-tools`

## Dependencies

### Core Dependencies
- PyTorch ecosystem: `torch`, `lightning>=2.0`, `torchmetrics>=0.11.0`
- Data science: `numpy`, `scipy`, `pandas`, `scikit-learn>=0.21.2`
- Single-cell: `anndata>=0.11`, `mudata>=0.1.2`, `scanpy>=1.10` (optional)
- Probabilistic ML: `pyro-ppl>=1.6.0`, `jax`, `jaxlib`, `numpyro>=0.12.1`

### Optional Dependencies
- `[tests]`: `pytest`, `coverage`
- `[docs]`: Sphinx documentation tools
- `[autotune]`: `hyperopt`, `ray[tune]`
- `[hub]`: `huggingface_hub`, model sharing tools
- `[optional]`: All optional features combined

## Common Development Commands

```bash
# Development installation
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
coverage run -m pytest -v && coverage report

# Code formatting and linting
pre-commit run --all

# Install pre-commit hooks
pre-commit install

# Build documentation locally
pip install -e ".[docs]"
cd docs && make html

# Clean up generated files
git clean -fdx
```

## Working with Models

### Adding a New Model
1. Create model class in `model/` (inherits from base classes)
2. Create corresponding module in `module/` (PyTorch Lightning module)
3. Add tests in appropriate `tests/` subdirectory
4. Update API documentation if public
5. Add entry to changelog

### Model Development Pattern
```python
# High-level model API (model/)
class MyModel(BaseModelClass, VAEMixin):
    def __init__(self, adata, **kwargs):
        # Model initialization
        
# Neural network implementation (module/)  
class MyVAE(BaseModuleClass):
    def __init__(self, **kwargs):
        # Neural network architecture
```

## Important Files for Development

- `pyproject.toml`: Build configuration, dependencies, tool settings
- `.pre-commit-config.yaml`: Code quality automation
- `tests/conftest.py`: Test configuration and fixtures
- `src/scvi/__init__.py`: Package initialization
- `src/scvi/model/base/`: Base classes for model development
- `src/scvi/data/_manager.py`: Data registration system

## CI/CD

The project uses GitHub Actions with multiple test workflows:
- `test_linux.yml`: Main test suite on Ubuntu
- Platform-specific tests: Windows, macOS
- Feature-specific tests: CUDA, multi-GPU, autotune, etc.
- Documentation building and deployment

## Tips for Effective Development

1. **Always run tests**: Both existing and new tests for your changes
2. **Follow the architecture**: Use existing base classes and patterns
3. **Document thoroughly**: Use numpydoc format for all public APIs
4. **Check compatibility**: Test with different Python versions if possible
5. **Use type hints**: Especially for new code
6. **Leverage AnnDataManager**: For data handling in new models
7. **Follow existing patterns**: Look at similar models for implementation guidance

## Troubleshooting

- If tests fail, check test markers and run appropriate subset
- For import errors, ensure proper installation with development dependencies
- For pre-commit issues, run `pre-commit clean` and reinstall hooks
- Check Python version compatibility (3.10-3.12)
- GPU tests require appropriate hardware setup