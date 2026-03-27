# dgd Python bindings

Python bindings for the [Differentiable Growth Distance](https://github.com/HybridRobotics/differentiable-growth-distance) library.

The bindings are implemented with `pybind11` and packaged with `scikit-build-core`.
The recommended workflow is an editable install from the repository's `python/`
directory so `import dgd` works without setting `PYTHONPATH`.

## Requirements

- Python 3.8+
- CMake 3.15+
- C++17 compiler
- NumPy

## Development install

This is the primary workflow for local development.
From the repository root:

```bash
pip install --no-build-isolation -e python/[test]
```

After installation, `import dgd` works from any directory as long as the active
virtual environment or conda environment is the same one used for installation.

## Manual CMake build

This workflow is mainly useful for debugging the extension build directly.
It does not install the package into the Python environment.

```bash
cmake -B build -DDGD_BUILD_PYTHON=ON -DPython3_EXECUTABLE=$(which python)
cmake --build build --target _dgd_core -j
export PYTHONPATH=/path/to/repo/python:$PYTHONPATH
```

Use `PYTHONPATH` only with this manual workflow. It is not needed after an editable
install or wheel install.

## Building a wheel

From the repository root:

```bash
pip install build
python -m build python/ --wheel
```

The generated wheel can then be installed with `pip install dist/dgd-*.whl`.

## Running the tests

```bash
cd python
pytest
```

If the active environment auto-loads unrelated global pytest plugins, use:

```bash
cd python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

## Style checks

The project uses Ruff to enforce PEP 8 and Google-style Python docstrings.

```bash
cd python
ruff check .
```

## Pre-commit hooks

The repository-level `.pre-commit-config.yaml` includes Python hooks for
`ruff` and `black` (scoped to `python/*.py`), along with existing C++ hooks.

`ruff` covers linting, import sorting, and many `flake8` checks.

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Package layout

- `dgd/`: Python package and compiled extension module
- `src/`: pybind11 binding translation units
- `tests/`: Python-side smoke tests for the bindings surface
- `CMakeLists.txt`: extension build definition used by both direct CMake and scikit-build
- `pyproject.toml`: packaging metadata and build backend configuration
