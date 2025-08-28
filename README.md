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
