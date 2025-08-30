# Dynamic programming for matmul computation with minimal time
# The trick is to use the simulator and use only powers of 2 for now.
# limit time to 100 before running the simulator for now (a*b*c < 100, a*b*c+b*c*d < 100)

# OK, now let's start with just the op (simple 2 powers):
from simulator import Cache, Bandwidth, utilization, Tensor, reset_counters
from simulate import muladd, matmulsimple


def run_dynamic(results, node, *tensors, reset_counter=True, accumulate_output=None):
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

    Args:
        accumulate_output: optional Tensor to accumulate results into. If
            provided, it will be loaded (in LDST/DBL mode) and used as the
            destination buffer without zero-initialization, and then stored
            back. If None, a new output buffer is allocated as before.

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

    # Construct keys used in dynamic maps
    def make_key_from_dims(dims_):
        flat = []
        for i in range(len(dims_) - 1):
            flat.extend([(dims_[i], dims_[i + 1]), 0])
        flat.extend([(dims_[0], dims_[-1]), 0])
        return tuple(flat)

    def make_key_from_tensors(dims_, tensors_, out_level):
        flat = []
        for i in range(len(dims_) - 1):
            flat.extend([(dims_[i], dims_[i + 1]), getattr(tensors_[i], 'level', 0)])
        flat.extend([(dims_[0], dims_[-1]), out_level])
        return tuple(flat)

    # Prefer a key that encodes current tensor levels (to capture LDST variants)
    out_level = getattr(node, 'level', 0)
    key_levels = make_key_from_tensors(dims, tensors, out_level)
    entry = results.get(key_levels)
    key = key_levels
    # Fallback to canonical all-zero-levels key
    if entry is None:
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

    # Determine entry type and expected CPU
    algo = None
    cpu_expected = None
    bw_op = None
    if isinstance(entry, list) and entry:
        if isinstance(entry[0], str):
            # e.g., ["BinOpx", cpu]
            algo = entry[0]
            cpu_expected = entry[1] if len(entry) > 1 else None
        elif isinstance(entry[0], tuple):
            # e.g., [("LDST", ...), cpu, bw]
            bw_op = entry[0]
            algo = bw_op[0]
            cpu_expected = entry[1] if len(entry) > 1 else None
        else:
            # Back-compat: just a [cpu_time]
            algo = "BinOpx"
            cpu_expected = entry[0]

    # Reset counters if requested
    if reset_counter:
        if isinstance(node, Cache):
            reset_counters(node)
        else:
            # BinOpx
            if hasattr(node, 'time'):
                node.time = 0

    if algo in ("LDST", "DBL"):
        if not isinstance(node, Cache):
            raise TypeError("LDST execution requires a Cache node for load/store")
        # Identify indices of inputs to load
        out_pos = len(tensors)
        if algo == "LDST":
            idxs = list(bw_op[1:]) if bw_op and len(bw_op) > 1 else []
        else:
            # For DBL (or other bandwidth-level ops), infer loads based on current levels
            cur_level = getattr(node, 'level', 0)
            idxs = [i for i in range(len(tensors)) if getattr(tensors[i], 'level', 0) == cur_level]
        # Load specified inputs to the next lower level
        loaded = {}
        load_idxs = [i for i in idxs if i < out_pos]
        for i in load_idxs:
            loaded[i] = node.load(tensors[i])
        # Prepare tensors for recursive call (use loaded ones where applicable)
        rec_args = [loaded.get(i, tensors[i]) for i in range(len(tensors))]
        # Compute directly at the compute level using the child cache for outputs
        comp_cache = getattr(node, 'parentcache', None)
        if comp_cache is None or not hasattr(comp_cache, 'parent'):
            raise RuntimeError("Cannot locate compute node under cache for LDST execution")
        comp_node = comp_cache.parent  # BinOpx

        # Expected ops for left-associated chain
        def _expected_ops(ds):
            a0 = ds[0]
            ops = 0
            for i in range(1, len(ds) - 1):
                ops += a0 * ds[i] * ds[i + 1]
            return ops
        exp_ops = _expected_ops(dims)

        # Optionally load the accumulation output to the compute level
        loaded_out = None
        if accumulate_output is not None:
            loaded_out = node.load(accumulate_output)

        # Compute chain: allocate outputs in comp_cache so they are resident
        if accumulate_output is not None and loaded_out is not None:
            out_low = loaded_out
        else:
            out_low = comp_cache.calloc(dims[0], dims[2])
        matmulsimple(comp_node, rec_args[0], rec_args[1], out_low)
        for t in rec_args[2:]:
            nxt = comp_cache.calloc(out_low.sz[0], t.sz[1])
            matmulsimple(comp_node, out_low, t, nxt)
            comp_cache.free(out_low)
            out_low = nxt
        # Store result back to this cache
        out_rows, out_cols = out_low.sz
        if accumulate_output is None:
            out_high = node.calloc(out_rows, out_cols)
            node.store_to(out_low, out_high)
        else:
            out_high = accumulate_output
            node.store_to(out_low, out_high)
        # Free loaded inputs in child cache
        for i in load_idxs:
            node.parentcache.free(loaded[i])
        # Validate CPU and bandwidth if available
        comp = node.parentcache.parent if hasattr(node, 'parentcache') else None
        if reset_counter:
            if cpu_expected is not None and comp is not None:
                if comp.time != cpu_expected:
                    raise AssertionError("CPU time mismatch: {} != {}".format(comp.time, cpu_expected))
            # Bandwidth time expected for this link if provided in entry
            link = getattr(node, 'parent', None)
            if isinstance(entry, list) and len(entry) > 2 and link is not None and hasattr(link, 'time'):
                bw_expected = entry[2]
                # If we additionally loaded the output for accumulation, add its size
                if accumulate_output is not None and hasattr(out_low, 'sz'):
                    bw_expected += out_low.sz[0] * out_low.sz[1]
                if link.time != bw_expected:
                    raise AssertionError(
                        "Bandwidth time mismatch: {} != {}".format(link.time, bw_expected)
                    )
        return out_high
    elif algo == "BinOpx":
        # Compute left-associated chain using CPU node and matmulsimple
        if not hasattr(node, 'run') or not isinstance(node, Tensor) and not isinstance(node, Cache):
            pass  # just proceed; node is expected to be BinOpx
        time_before = getattr(node, "time", 0)

        # Allocate initial output at node.clevel (or 0 if absent)
        clevel = getattr(node, "clevel", 0)
        n_mats = len(tensors)
        if accumulate_output is not None:
            # Validate shape and level
            if accumulate_output.sz != [tensors[0].sz[0], tensors[-1].sz[1]]:
                raise ValueError("accumulate_output shape mismatch")
            if getattr(accumulate_output, 'level', clevel) != clevel:
                raise ValueError("accumulate_output must reside at CPU level {}".format(clevel))

        if n_mats == 2:
            out = accumulate_output if accumulate_output is not None else Tensor.zeros(
                tensors[0].sz[0], tensors[1].sz[1], level=clevel
            )
            matmulsimple(node, tensors[0], tensors[1], out)
        else:
            # Build intermediates until the last multiply; then write into final dest
            temp = Tensor.zeros(tensors[0].sz[0], tensors[1].sz[1], level=clevel)
            matmulsimple(node, tensors[0], tensors[1], temp)
            for t in tensors[2:-1]:
                nxt = Tensor.zeros(temp.sz[0], t.sz[1], level=clevel)
                matmulsimple(node, temp, t, nxt)
                temp = nxt
            last = tensors[-1]
            if accumulate_output is not None:
                out = accumulate_output
                matmulsimple(node, temp, last, out)
            else:
                out = Tensor.zeros(temp.sz[0], last.sz[1], level=clevel)
                matmulsimple(node, temp, last, out)

        # Verify CPU operations/time matches expected if provided
        time_after = getattr(node, "time", 0)
        cpu_used = time_after - time_before if not reset_counter else time_after
        if cpu_expected is not None:
            if cpu_used != cpu_expected:
                raise AssertionError(
                    "CPU time mismatch: computed {} != expected {}".format(cpu_used, cpu_expected)
                )
        return out
    else:
        raise NotImplementedError("Unsupported dynamic entry type: {}".format(algo))

def pp(results):
    for k, v in results.items():
        cpu = 0
        bw_time = 0
        if isinstance(v, list) and v:
            # Formats:
            # - BinOpx: ["BinOpx", cpu]
            # - Bandwidth: [(op,...), cpu, bw]
            # - Back-compat: [cpu, bw]
            if isinstance(v[0], str):
                cpu = v[1] if len(v) > 1 else 0
            elif isinstance(v[0], tuple):
                cpu = v[1]
                bw_time = v[2] if len(v) > 2 else 0
            else:
                cpu = v[0]
                bw_time = v[1] if len(v) > 1 else 0
        util = 0.0 if (cpu == 0 and bw_time == 0) else (cpu / max(cpu, bw_time))
        print(f"{k}: {v} | util={util:.3f}")



if __name__ == "__main__":
    cache = Cache(12, muladd)
    bw = Bandwidth(cache)
    results = bw.dynamic_times(2, 1000)
    pp(results)







