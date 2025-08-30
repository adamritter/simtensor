# Dynamic programming for matmul computation with minimal time
# The trick is to use the simulator and use only powers of 2 for now.
# limit time to 100 before running the simulator for now (a*b*c < 100, a*b*c+b*c*d < 100)

# OK, now let's start with just the op (simple 2 powers):
from simulator import Cache, Bandwidth, utilization, Tensor
from simulate import muladd, matmulsimple


def run_dynamic(results, node, *tensors):
    """
    Execute a matrix-chain multiplication using the algorithm indicated by a
    precomputed dynamic result entry.

    For now only supports entries produced by BinOpx.enumerate_power_chain
    (type tag "BinOpx"). We run a simple left-associated matmul chain on the
    provided CPU node `node` using `simulate.matmulsimple` and verify that the
    CPU operation count matches the expected ops from `results`.

    Args:
        results: dict as returned by BinOpx.dynamic_times()/enumerate_power_chain
        node: a simulator.BinOpx instance (the CPU muladd node)
        tensors: sequence of simulator.Tensor operands [A, B, C, ...]

    Returns:
        simulator.Tensor: the final output tensor
    """
    if len(tensors) < 2:
        raise ValueError("run_dynamic requires at least two tensors")

    # Build dims list [d0, d1, d2, ...] from input tensors of shapes (d0,d1),(d1,d2),...
    dims = [tensors[0].sz[0], tensors[0].sz[1]]
    for t in tensors[1:]:
        if len(t.sz) != 2:
            raise ValueError("Only 2D tensors are supported")
        if dims[-1] != t.sz[0]:
            raise ValueError("Incompatible chain shapes: {} vs {}".format(dims[-1], t.sz[0]))
        dims.append(t.sz[1])

    # Construct the canonical key used by BinOpx.enumerate_power_chain
    def make_key_from_dims(dims_):
        flat = []
        for i in range(len(dims_) - 1):
            flat.extend([(dims_[i], dims_[i + 1]), 0])
        flat.extend([(dims_[0], dims_[-1]), 0])
        return tuple(flat)

    key = make_key_from_dims(dims)
    entry = results.get(key)
    if entry is None:
        # Attempt to find a matching entry ignoring the level markers
        for k, v in results.items():
            try:
                shapes = [k[i] for i in range(0, len(k), 2)]
            except Exception:
                continue
            want_shapes = [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)] + [
                (dims[0], dims[-1])
            ]
            if shapes == want_shapes:
                key = k
                entry = v
                break

    if entry is None:
        raise KeyError("No dynamic result found for dims {}".format(dims))

    # Only support BinOpx entries for now
    algo = None
    cpu_expected = None
    if isinstance(entry, list) and entry:
        if isinstance(entry[0], str):
            algo = entry[0]
            cpu_expected = entry[1] if len(entry) > 1 else None
        else:
            # Back-compat: just a [cpu_time] list
            cpu_expected = entry[0]
            algo = "BinOpx"

    if algo != "BinOpx":
        raise NotImplementedError("Only BinOpx dynamic entries are supported for now")

    # Compute left-associated chain using CPU node and matmulsimple
    time_before = getattr(node, "time", 0)

    # Allocate initial output at node.clevel (or 0 if absent)
    clevel = getattr(node, "clevel", 0)
    out = Tensor.zeros(tensors[0].sz[0], tensors[1].sz[1], level=clevel)
    matmulsimple(node, tensors[0], tensors[1], out)
    for t in tensors[2:]:
        nxt = Tensor.zeros(out.sz[0], t.sz[1], level=clevel)
        matmulsimple(node, out, t, nxt)
        out = nxt

    # Verify CPU operations/time matches expected if provided
    time_after = getattr(node, "time", 0)
    cpu_used = time_after - time_before

    if cpu_expected is not None:
        if cpu_used != cpu_expected:
            raise AssertionError(
                "CPU time mismatch: computed {} != expected {}".format(cpu_used, cpu_expected)
            )

    return out

def pp(results):
    for k, v in results.items():
        cpu = 0
        bw_time = 0
        if isinstance(v, list) and v:
            # Formats:
            # - BinOpx: ["BinOpx", cpu]
            # - Bandwidth: [(op,...), cpu, bw]
            if isinstance(v[0], str):
                cpu = v[1] if len(v) > 1 else 0
            elif isinstance(v[0], tuple):
                cpu = v[1]
                bw_time = v[2] if len(v) > 2 else 0
        util = 0.0 if (cpu == 0 and bw_time == 0) else (cpu / max(cpu, bw_time))
        print(f"{k}: {v} | util={util:.3f}")



if __name__ == "__main__":
    cache = Cache(12, muladd)
    bw = Bandwidth(cache)
    results = bw.dynamic_times(2, 1000)
    pp(results)







