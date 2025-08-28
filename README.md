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
