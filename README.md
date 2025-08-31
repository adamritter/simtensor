This is a program that simulates / outputs optimal kernels.

It can define vector operations, cache hierary with bandwidth and parallelization.

Simulator:
===========

Simulates the time to run the operations:
- Loading / saving / discarding data in caches are explicit in the simulation
- The simulator makes sure that the cache constraints are kept and calculates bandwidth per cache level and FLOPS timing
- From these data it calculates utilization (just FLOPS time divided by max time from these constraints)


Optimal algorithms
==================
These are a set of algorithms that achieve tasks with 100% utilization with the minimal amount of bandwidth.
Some example tasks are:

- Multiplying 2 matrices that are in L2 cache where the output fits in L1 cache and outputting it there.
- Matrix multiplication (From L2 to L2)
- Matrix multiplication when one input is in L1 cache, other in L2 and output is in L2
- Triple matrix multiplication that is faster than just 2 matmuls in some cases.

The interesting part of these algorithms is not that they have 100% utilization, but that they find the optimal execution with shapes that are not the easiest ones, and keep track of the constraints that are necessary
for optimal execution.
Design Overview
===============

Components
- Tensor: n-D views over flat storage with simple strides and slicing.
- Cache: capacity and residency for a memory level; verifies data locality when running ops.
- Bandwidth: link between a parent cache (higher level) and a child cache (lower level), tracking input/output words and transfer time.
- BinOpx: a simple compute endpoint. Each call to `run` increments a configurable `time` cost.

Hierarchy
- Typical setup chains multiple caches via Bandwidth to a compute node:

  L1 Cache  --[ Bandwidth ]-->  L0 Cache  --( BinOpx compute )

Data Flow
- load(view): copies a Tensor view down one level (parent -> child), increasing `Bandwidth.input` and child `used` capacity.
- run(op, ...): executes compute at the cache level where tensors are resident, increasing `BinOpx.time`.
- store(view): moves a view up one level (child -> parent), increasing `Bandwidth.output` and allocating in the parent.
- store_to(src, dst): like `store` but writes into an existing parent view `dst` (no new parent allocation).

Timing & Utilization
- Compute: accumulated in `BinOpx.time`.
- Bandwidth: accumulated per link in `Bandwidth.time` using either a shared-line or separate-line model.
- Utilization: `cpu_time / max(cpu_time, max_bandwidth_time)` along the chain.

Utilization & Counters
======================

You can compute utilization of a run and reset counters between experiments.

Example:

```
import simulator

# Build a two-level hierarchy
op = simulator.BinOpx([], 0, [], 0, [], 0, lambda a,b,c: None, t=1)
L0 = simulator.Cache(32, op)
bw = simulator.Bandwidth(L0, input_clocks_per_word=2, output_clocks_per_word=3)
L1 = simulator.Cache(1000, bw)

# Allocate and move a small view
mat = L1.calloc(4, 4)  # resides in L1
v0 = L1.load(mat[:, 0:2])  # move 8 words to L0
L1.store(v0)               # move back up

# Do some compute at L0
a = simulator.Tensor([1], 0, [])
b = simulator.Tensor([2], 0, [])
c = simulator.Tensor([3], 0, [])
L0.alloc(a); L0.alloc(b); L0.alloc(c)
for _ in range(10):
    op.run(a, b, c)

u = simulator.utilization(L1)
print("utilization:", u)

# Reset for another experiment
simulator.reset_counters(L1)
```

Bandwidth timing model:

- Shared line (default): `time = (input + output) * input_clocks_per_word`
- Separate lines: `time = max(input * input_clocks_per_word, output * output_clocks_per_word)`

API Reference
=============

Tensor
------
- data: underlying Python list buffer (flat storage)
- level: integer cache level of the tensor
- sz: shape list (e.g., `[rows, cols]`, `[]` for scalar)
- offset/skips: view offset and per-dimension strides in elements
- size(): total element count of the view
- sum(): recursive sum across the view
- `t[i][j]`/slices: returns a Tensor view with adjusted offset/strides
- assignment: `t[i][j] = value` writes to underlying buffer

BinOpx
------
- ctor: `BinOpx(as0, alevel, bs, blevel, cs, clevel, f, t=0)`
- run(a, b, c): asserts levels, calls `f(a,b,c)`, increments `time` by `t`
- time: accumulated compute time units

Cache
-----
- ctor: `Cache(size, parent)` where `parent` is `Bandwidth` or `BinOpx`
- level/used/size: tracking for this cache
- alloc(t): mark tensor's storage as resident in this cache (capacity check)
- free(t): release tensor storage from this cache
- calloc(n, m): allocate zero matrix `[n, m]` in this cache
- alloc_diag(n): allocate identity matrix `[n, n]` in this cache
- load(m): move a view one level down; increments `Bandwidth.input`
- store(m): move a view one level up; increments `Bandwidth.output` and allocs in parent cache
- store_to(src, dst): copy from child view `src` into existing parent view `dst`; updates bandwidth and frees child view, no new parent alloc
- run(op, *args): run an op against tensors resident in this cache

Bandwidth
---------
- ctor: `Bandwidth(cache, input_clocks_per_word=1, output_clocks_per_word=None)`
- input/output: total words moved down/up through this link
- time: aggregate transfer time
  - shared line: `(input + output) * input_clocks_per_word`
  - separate lines: `max(input * input_clocks_per_word, output * output_clocks_per_word)`
- load(t)/store(t): update counters/time and forward to `cache`

Top-level Functions
-------------------
- utilization(cache): `cpu_time / max(cpu_time, max_bandwidth_time)` across the chain
- reset_counters(cache): zero all bandwidth counters and the compute node time



Dynamic-programming helpers
===========================

This project includes a small dynamic-programming (DP) facility to enumerate
matrix-chain shapes and select efficient execution variants across cache/bandwidth
levels. It consists of two parts:

- Enumeration: BinOpx.dynamic_times, Cache.dynamic_times, Bandwidth.dynamic_times
  - BinOpx.dynamic_times(n_mats, limit) enumerates all chains of n_mats matrices
    with power-of-two dimensions whose total product is bounded by limit. Each
    entry maps a key of alternating (shape, level_marker) pairs to a CPU-time
    estimate for a left-associated execution. Example key for a 2-mat chain:
      ((a,b), 0, (b,c), 0, (a,c), 0)
    Values are ["BinOpx", cpu_time].
  - Cache.dynamic_times filters entries by capacity at a given cache level by
    summing the element counts for shapes whose marker equals that level.
  - Bandwidth.dynamic_times augments the mapping with bandwidth-aware variants
    by promoting operands across the link ("LDST") and then applies a DP
    expansion ("DBL") that doubles shared dimensions when both sides reside at
    the bandwidth level. Values are either ["BinOpx", cpu] or
    [("LDST", ...), cpu, bw] or [("DBL", tag), cpu, bw].

- Execution: dynamic.run_dynamic
  Given a DP results mapping and operands, run_dynamic chooses the matching key
  (ignoring level markers when necessary) and dispatches to the correct runner:
  - BinOpx: compute-only, left-associated execution at the compute node.
  - LDST: load selected operands down one level, compute at the lower cache,
    then store the result according to the output level marker.
  - DBL: execute as LDST and account for the extra transfer implied by the
    expansion so validation matches the DP table.

Usage example
-------------

```
from simulator import Cache, Bandwidth
from simulate import muladd
from dynamic import run_dynamic

# Build hierarchy: L1 --bw--> L0 -- muladd (compute)
L0 = Cache(24, muladd)
bw = Bandwidth(L0)
L1 = Cache(1000, bw)

# Enumerate and pick a case
results = bw.dynamic_times(2, 4096)
A = L1.calloc(2, 2)
B = L1.calloc(2, 2)
out = run_dynamic(results, L1, A, B)
```

Helper notes
------------
- Keys always end with the output dims pair. Level markers are integers that
  refer to cache levels; 0 is the compute level, higher numbers are higher
  caches across bandwidth links.
- LDST tuple lists indices of operands that are expected to be resident at the
  higher cache level before loading down for compute.
- DBL tag indicates which shared dimension was doubled during expansion; this
  is used for debugging and does not affect execution.
