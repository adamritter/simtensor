# Roadmap and Ideas

This file collects concrete tasks and longer-term ideas to evolve simtensor.
Items are grouped by area; feel free to propose additions or re-prioritize.

## Core Model and API
- [ ] Clarify memory layout: document current stride convention and decide on row- vs column-major as default; expose explicit strides on Tensor.
- [ ] Tighten slicing semantics: handle negative indices/slices, empty ranges, and step values; add comprehensive docstrings and examples.
- [ ] Add dtype support (e.g., float32/64, int types) and basic validation.
- [ ] Separate logical value vs storage: consider immutable TensorView vs mutable backing store to make aliasing clearer.
- [ ] Normalize naming (parent/parentcache, Bandwidth.cache): unify attribute names and add type hints.

## Algorithms and Scheduling
- [ ] Generalize matmul tiling: parameterize block sizes; implement register/L1/L2 blocking with double-buffering.
- [ ] Add additional kernels: vector add, GEMV, batched GEMM, convolution (1D/2D), and simple fused ops (e.g., GEMM+Bias+ReLU).
- [ ] Overlap compute and data movement: pipeline load/compute/store at different levels when bandwidth allows.
- [ ] Introduce a simple scheduler IR (ops + tensor views + dependencies) to model execution order explicitly.

## Hardware and Cost Model
- [ ] Hardware profiles: JSON/YAML descriptions for cache levels (size, bandwidth), compute throughput, and latency knobs.
- [ ] Contention/latency modeling: optional penalties for concurrent loads/stores; configurable read/write asymmetry.
- [ ] Vectorization model: lanes/width, issue rate, and utilization; allow architecture presets (e.g., AVX2/AVX-512/NEON).

## Auto-Tuning
- [ ] Search tile sizes given a hardware profile; heuristics first, then grid/random search.
- [ ] Emit a small report (best tile, predicted time, utilization breakdown).

## CLI and UX
- [ ] Add a console script simtensor with subcommands:
  - simtensor run config.yaml to simulate a scenario
  - simtensor bench to try common shapes/kernels on a profile
  - simtensor viz to output simple timelines/ASCII charts
- [ ] Human-readable summaries: pretty tables for bandwidth vs compute time and utilization.

## Validation, Tests, and Benchmarks
- [ ] Property tests (Hypothesis) for slicing/indexing equivalence across shapes.
- [ ] Golden-value tests for kernels vs NumPy implementations for correctness (small sizes).
- [ ] Tests for memory accounting: Cache.load/store/free invariants, aliasing, and capacity errors.
- [ ] Microbenchmarks for the simulator itself; ensure scaling is acceptable for larger shapes.

## Visualization
- [ ] ASCII timeline of events (load/compute/store) with per-level occupancy.
- [ ] Memory usage over time per cache level; export CSV/JSON for external plotting.
- [ ] Roofline-style summary: predicted FLOP/s vs bandwidth ceilings.

## Docs and Examples
- [ ] Expand README with install, quickstart, and examples.
- [ ] Add examples/ with small scripts (e.g., 4x4 GEMM, L1/L2 only, mixed placements).
- [ ] Host docs (MkDocs or Sphinx) with API reference and design notes.

## Packaging and Releases
- [ ] Expose __version__ and basic metadata in a simtensor module (or package the code under a simtensor/ package).
- [ ] Add console entry point in pyproject.toml when CLI exists.
- [ ] Release workflow: build wheels/sdist on tags, publish to PyPI (with provenance/signing if desired).

## CI and Tooling
- [ ] Expand CI matrix to include macOS/Windows.
- [ ] Add linting/formatting (ruff/black), typing (mypy), and coverage reporting (coverage.py + codecov).
- [ ] Pre-commit hooks to keep style consistent.

## Stretch / Future Directions
- [ ] GPU model: SMs, shared memory, warps, occupancy; simple kernel execution model.
- [ ] Multi-threaded CPU: cores, NUMA, shared caches; thread scheduling impact.
- [ ] Auto-derivation of optimal algorithms from constraints (search/explore space with pruning).

---
Contributions and ideas are welcome. Open an issue or PR to discuss any item.
