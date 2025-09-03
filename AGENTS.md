# Repository Guidelines

## Project Structure & Module Organization
- `simulator.py`: Core engine (`Tensor`, `BinOpx`, `Cache`).
- `bandwidth.py`: Bandwidth link between caches with input/output counters and transfer time.
- `simulate.py`: Example kernels and small driver; run to see behavior.
- `simtensor.py`: Lightweight module stub reserved for future API surfacing.
- `tests/`: Unit tests (`unittest`) such as `tests/test_simulator.py`.
- `.github/workflows/ci.yml`: CI for Python 3.9–3.12 (build + tests).

## Build, Test, and Development Commands
- Install (editable): `python -m pip install -e .`
- Build sdist/wheel: `python -m pip install build && python -m build`
- Run tests (verbose): `python -m unittest discover -v`
- Run examples locally: `python simulate.py`

Each command runs from the repository root. CI runs the same build and test steps.

## Coding Style & Naming Conventions
- Language: Python 3.9+; follow PEP 8 with 4‑space indentation.
- Names: modules/functions `snake_case`; classes `CamelCase`; tests `test_*.py`.
- Structure: keep functions small and focused; prefer explicit data flow over hidden mutation.
- Formatting: no enforced tool yet; if used locally, format with `black` (88 cols) and lint with `ruff` before submitting.
- Types: do not use type annotations or the `typing` module; prefer clear names and brief docstrings/comments.
- Error handling: avoid silent fallbacks or swallowing exceptions; let unexpected states crash loudly to expose bugs.

## Testing Guidelines
- Framework: `unittest` (`unittest.TestCase`).
- Layout: place new tests in `tests/test_<area>.py`; name classes `Test*` and methods `test_*`.
- Scope: include success and error‑path coverage (e.g., cache capacity, residency checks).
- Run locally with `python -m unittest discover -v`; ensure compatibility across 3.9–3.12.

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood. Use optional scope tags (e.g., `simulator:`, `tests:`, `CI:`). Example: `simulator: fix load/store level check`.
- PRs: include a clear description, linked issues, test updates, and sample output or reproduction (e.g., snippet from `python simulate.py`).
- Requirements: all tests pass, CI green, and docs (`README.md`/this file) updated when behavior changes.

## Architecture Overview (Quick)
- Computation modeled via `BinOpx`; memory via `Cache` levels bridged by `Bandwidth`.
- Algorithms operate on `Tensor` views; residency in the correct cache level is enforced during `run`.
