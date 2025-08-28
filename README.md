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
