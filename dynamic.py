# Dynamic programming for matmul computation with minimal time
# The trick is to use the simulator and use only powers of 2 for now.
# limit time to 100 before running the simulator for now (a*b*c < 100, a*b*c+b*c*d < 100)

# OK, now let's start with just the op (simple 2 powers):
from simulator import Cache, Bandwidth, utilization, Tensor, reset_counters, get_counters
from simulate import muladd, matmulsimple
import os
DEBUG = int(os.environ.get("DEBUG", 0))


def run_dynamic(results, node, *tensors, out_level=None, reset_counter=True, accumulate_output=None):
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
    if node is None:
        raise Exception("Node is None")
    if len(tensors) < 2:
        raise ValueError("run_dynamic requires at least two tensors")

    if DEBUG > 0:
        print("run_dynamic: ", node, tensors, out_level, reset_counter, accumulate_output)

    # Build dims list [d0, d1, d2, ...] from input tensors of shapes (d0,d1),(d1,d2),...
    dims = [tensors[0].sz[0], tensors[0].sz[1]]
    for t in tensors[1:]:
        if len(t.sz) != 2:
            raise ValueError("Only 2D tensors are supported")
        if dims[-1] != t.sz[0]:
            raise ValueError("Incompatible chain shapes: {} vs {}".format(dims[-1], t.sz[0]))
        dims.append(t.sz[1])

    def make_key_from_tensors(dims_, tensors_, out_level):
        flat = []
        for i in range(len(dims_) - 1):
            flat.extend([(dims_[i], dims_[i + 1]), getattr(tensors_[i], 'level', 0)])
        flat.extend([(dims_[0], dims_[-1]), out_level])
        return tuple(flat)

    key = make_key_from_tensors(dims, tensors, out_level)
    entry = results.get(key)
    if entry is None:
        raise KeyError("No dynamic result found for key {}".format(key))

    if reset_counter:
        reset_counters(node)
    
    orig_counter = get_counters(node)
    if entry[0] == "BinOpx":
        # Just run it
        out = _run_dynamic_binopx(node, tensors, accumulate_output)
    elif entry[0][0] == "LDST":
        out = _run_dynamic_ldst(node, tensors, accumulate_output, entry[0][1:], out_level=out_level, key=key, results=results)
    elif entry[0][0] == "DBL":
        out = _run_dynamic_dbl(node, tensors, accumulate_output, entry[0][1], out_level=out_level, key=key, results=results)
    else:
        raise NotImplementedError("Unsupported dynamic entry type: {}".format(entry[0]))

    counters = get_counters(node)
    # decrement counters by orig_counter
    while len(counters) < len(orig_counter):
        counters.append(0)
    while len(orig_counter) < len(counters):
        orig_counter.append(0)
    counters = [counters[i] - orig_counter[i] for i in range(len(counters))]
    while len(counters) > len(entry[1:]) and counters[-1] == 0:
        counters.pop()
    if counters != entry[1:]:
        raise AssertionError("Counters mismatch: {} != {}, key: {}, entry: {}".format(counters, entry[1:], key, entry))
    return out


def _run_dynamic_binopx(node, tensors, accumulate_output):
    root_node = node.root_node()

    n_mats = len(tensors)
    if accumulate_output is not None:
        if accumulate_output.sz != [tensors[0].sz[0], tensors[-1].sz[1]]:
            raise ValueError("accumulate_output shape mismatch: {} != {}".format(accumulate_output.sz, [tensors[0].sz[0], tensors[-1].sz[1]]))
            raise ValueError("accumulate_output shape mismatch")

    # Locate the highest Cache ancestor (if any) of the provided node.
    highest_cache = None
    cur = node
    if isinstance(cur, Bandwidth):
        cur = cur.cache
    while cur is not None:
        if isinstance(cur, Cache):
            highest_cache = cur
        cur = getattr(cur, "parentcache", None)

    left = tensors[0]
    out = None
    for i in range(1, n_mats):
        right = tensors[i]
        is_last = (i == n_mats - 1)
        if is_last:
            if accumulate_output is not None:
                # Use the caller‑supplied output tensor.
                dest = accumulate_output
            else:
                # Allocate the output in the highest cache if one exists;
                # otherwise fall back to a plain Tensor allocation.
                if highest_cache is not None:
                    dest = highest_cache.calloc(left.sz[0], right.sz[1])
                    if dest is None:
                        raise ValueError("calloc error: dest is None, accumulate_output is ", accumulate_output, ", highest_cache is ", highest_cache, ", is_last is ", is_last)
                else:
                    # No cache in the hierarchy (pure BinOp tree).
                    dest = Tensor.zeros(left.sz[0], right.sz[1], level=0)
        else:
            # For intermediate results we still allocate at level‑0.
            dest = Tensor.zeros(left.sz[0], right.sz[1], level=0)
        if dest is None:
            raise ValueError("dest is None, accumulate_output is ", accumulate_output, ", highest_cache is ", highest_cache, ", is_last is ", is_last)
        if is_last:
            out = dest
        matmulsimple(root_node, left, right, dest)
        left = dest

    return out


def _run_dynamic_ldst(node, tensors, accumulate_output, bw_op, out_level=None, key=None, results=None):
    """Execute one LDST step by loading listed operands, then running the
    predecessor dynamic entry using run_dynamic. This avoids re-implementing
    the matmul here and ensures consistent accounting.
    """
    if not isinstance(node, Bandwidth) and not isinstance(node, Cache):
        raise TypeError("LDST execution requires a Bandwidth or Cache node for load/store, type is ", type(node))

    # Load selected operands down one level
    loaded = list(tensors).copy()
    for i in bw_op:
        if i < len(loaded):
            loaded[i] = node.load(tensors[i])
    
    loaded_out = accumulate_output
    if loaded_out and len(tensors) in bw_op:
        loaded_out = node.load(accumulate_output)

    run_node = node.cache.parent if isinstance(node, Bandwidth) else node.parent

    out_low = run_dynamic(
        results,
        run_node,
        *loaded,
        reset_counter=False,
        accumulate_output=loaded_out,
        out_level=out_level-1 if (len(tensors) in bw_op) else out_level
    )

    if accumulate_output is None:
        if out_level != out_low.level:
            out_high = node.calloc(out_low.sz[0], out_low.sz[1])
            node.store_to(out_low, out_high)
        else:
            out_high = out_low
    else:
        out_high = accumulate_output
        if len(tensors) in bw_op:
            node.store_to(out_low, out_high)

    for i in bw_op:
        if i < len(loaded):
            node.free(loaded[i], allow_lower_level=True)

    return out_high


def _run_dynamic_dbl(node, tensors, accumulate_output, out_level_marker=None, key=None, dbl_op=None, results=None):
    """Execute DBL by splitting into two predecessor runs.

    Strategy:
    - Use previous_key(current_key, ("DBL", j)) to obtain the predecessor key.
    - Slice the operand tensors into two halves corresponding to the DBL
      expansion position j.
    - Call run_dynamic twice on the predecessor problem:
        * boundary j in {0, n}: write into disjoint output slices (no accumulate)
        * interior 0 < j < n: second call uses accumulation into the same dst
    - Both internal calls run with reset_counter=False so that outer verification
      checks the aggregate CPU/BW against the DBL entry.
    """
    if not isinstance(node, Cache):
        raise TypeError("DBL execution requires a Cache node for load/store")
    if not isinstance(dbl_op, tuple) or len(dbl_op) < 2 or dbl_op[0] != "DBL":
        raise ValueError("_run_dynamic_dbl requires dbl_op ('DBL', j)")
    # Build the current key if not provided; rely on operand/output levels.
    if key is None:
        dims = [tensors[0].sz[0], tensors[0].sz[1]]
        for t in tensors[1:]:
            dims.append(t.sz[1])
        out_level = getattr(node, 'level', 0)
        flat = []
        for i in range(len(dims) - 1):
            flat.extend([(dims[i], dims[i + 1]), getattr(tensors[i], 'level', 0)])
        flat.extend([(dims[0], dims[-1]), out_level])
        key = tuple(flat)

    # Compute predecessor key using the provided tag
    prev = previous_key(key, dbl_op)
    # Determine split position and sizes from prev key shapes
    pairs_prev = [(prev[i], prev[i + 1]) for i in range(0, len(prev), 2)]
    ops_prev = pairs_prev[:-1]
    n = len(ops_prev)
    j = int(dbl_op[1])
    # Convenience: output shape
    out_rows = tensors[0].sz[0]
    out_cols = tensors[-1].sz[1]

    # Prepare the destination buffer at the target high level
    if accumulate_output is not None:
        base_out = accumulate_output
    else:
        base_out = node.calloc(out_rows, out_cols)

    # Helper to run one half with a chosen set of operands and an output view
    def run_half(ops_list, out_view, use_accumulate):
        # Ensure the out_view resides in the high-level cache
        if use_accumulate:
            return run_dynamic(
                results,
                node,
                *ops_list,
                reset_counter=False,
                accumulate_output=out_view,
            )
        tmp = run_dynamic(
            results,
            node,
            *ops_list,
            reset_counter=False,
            accumulate_output=None,
        )
        if tmp.sz != out_view.sz:
            raise ValueError("Mismatched temporary output shape during DBL run")
        for i in range(tmp.sz[0]):
            for j2 in range(tmp.sz[1]):
                out_view[i, j2] = tmp[i][j2].value if tmp[i][j2].sz == [] else tmp[i][j2]
        return out_view

    # Build operand lists for both halves
    ops1 = list(tensors)
    ops2 = list(tensors)

    if j == 0:
        # Split first matrix rows: top and bottom halves write into disjoint row blocks
        half = ops_prev[0][0][0]  # rows of first op in predecessor
        a0 = tensors[0]
        ops1[0] = a0[0:half, :]
        ops2[0] = a0[half:half * 2, :]
        # Outputs: first half rows and second half rows
        out1 = base_out[0:half, :]
        out2 = base_out[half:half * 2, :]
        run_half(ops1, out1, use_accumulate=False)
        run_half(ops2, out2, use_accumulate=False)
    elif j == n:
        # Split last matrix columns: left and right column blocks
        half = ops_prev[-1][0][1]  # cols of last op in predecessor
        bl = tensors[-1]
        ops1[-1] = bl[:, 0:half]
        ops2[-1] = bl[:, half:half * 2]
        out1 = base_out[:, 0:half]
        out2 = base_out[:, half:half * 2]
        run_half(ops1, out1, use_accumulate=False)
        run_half(ops2, out2, use_accumulate=False)
    else:
        # Interior split on the shared dimension between ops[j-1] and ops[j]
        half = ops_prev[j - 1][0][1]  # shared K in predecessor
        left = tensors[j - 1]
        right = tensors[j]
        ops1[j - 1] = left[:, 0:half]
        ops1[j] = right[0:half, :]
        ops2[j - 1] = left[:, half:half * 2]
        ops2[j] = right[half:half * 2, :]
        # Both halves produce the full output; accumulate on the second
        out_full = base_out
        run_half(ops1, out_full, use_accumulate=False)
        run_half(ops2, out_full, use_accumulate=True)

    return base_out

def previous_key(key, op=None):
    """Return the logical previous key for a dynamic-programming entry.

    If `op` is provided, use it to construct the specific predecessor:
    - LDST: `op` is ("LDST", i, j, ...). Decrease by 1 the level markers for
      each operand index in the tuple, leaving other markers and all shapes
      unchanged.
    - DBL: `op` is ("DBL", tag). Reverse the DBL expansion by halving the
      affected shape dimensions indicated by `tag` (0..n) and, for boundary
      tags 0 or n, also halving the corresponding output dimension. Levels are
      left unchanged.

    If `op` is None or not recognized, fall back to decreasing all integer
    level markers by 1 (not below 0), preserving shapes.
    """
    # Helper to split/join the flat key
    pairs = [(key[i], key[i + 1]) for i in range(0, len(key), 2)]
    ops = pairs[:-1]
    outp = pairs[-1]

    if isinstance(op, tuple) and op:
        kind = op[0]
        if kind == "LDST":
            # Decrease listed operand levels by 1
            idxs = list(op[1:])
            kl = list(key)
            for oi in idxs:
                pos = 2 * oi + 1
                if pos < len(kl) - 1:
                    lvl = kl[pos]
                    if isinstance(lvl, int) and lvl > 0:
                        kl[pos] = lvl - 1
            return tuple(kl)
        if kind == "DBL":
            # Reverse the doubling using the tag position
            try:
                tag = op[1]
            except Exception:
                tag = None
            if isinstance(tag, int):
                new_ops = list(ops)
                (odims, olvl) = outp
                new_out = (odims, olvl)
                n = len(new_ops)
                if n >= 1:
                    if tag == 0:
                        # ops[0].rows was doubled; halve it, and halve out.rows
                        (a0, a1), al = new_ops[0]
                        new_ops[0] = ((a0 // 2, a1), al)
                        new_out = ((odims[0] // 2, odims[1]), olvl)
                    elif tag == n:
                        # ops[-1].cols was doubled; halve it, and halve out.cols
                        (b0, b1), bl = new_ops[-1]
                        new_ops[-1] = ((b0, b1 // 2), bl)
                        new_out = ((odims[0], odims[1] // 2), olvl)
                    elif 0 < tag < n:
                        # Shared dim between ops[tag-1] and ops[tag] was doubled; halve both
                        (a0, a1), al = new_ops[tag - 1]
                        (b0, b1), bl = new_ops[tag]
                        new_ops[tag - 1] = ((a0, a1 // 2), al)
                        new_ops[tag] = ((b0 // 2, b1), bl)
                    # Reassemble
                    flat = []
                    for shp, lvl in new_ops:
                        flat.extend([shp, lvl])
                    flat.extend([new_out[0], new_out[1]])
                    return tuple(flat)
    return None


def extras(key, results):
    """Compute extra cpu/bandwidth for a row.

    - For BinOpx entries: return None.
    - For LDST entries: subtract previous row numbers (cpu and bandwidth) from
      the current row.
    - For DBL entries: first double the previous row numbers, then subtract
      that doubled number from the current row.
    """
    v = results.get(key)
    if not isinstance(v, list) or not v:
        return None
    # Determine the op kind, if any
    kind = None
    if isinstance(v[0], str):
        kind = v[0]
    elif isinstance(v[0], tuple):
        kind = v[0][0] if v[0] else None
    # BinOpx rows do not report extras
    if kind == "BinOpx" or kind is None:
        return None

    prev = results.get(previous_key(key, v[0]))
    if prev is None:
        return None
    # Extras are computed over the numeric tail: bwcpu is just value[1:]
    cur_tail = list(v[1:])
    prev_tail = list(prev[1:])
    # Pad the shorter list (e.g., previous BinOpx with no bw slots) with zeros
    n = max(len(cur_tail), len(prev_tail))
    if len(cur_tail) < n:
        cur_tail += [0] * (n - len(cur_tail))
    if len(prev_tail) < n:
        prev_tail += [0] * (n - len(prev_tail))
    factor = 2 if kind == "DBL" else 1
    return [cur_tail[i] - factor * prev_tail[i] for i in range(n)]


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
        extra = extras(k, results)
        if extra is None:
            print(f"{k}: {v} | util={util:.3f}")
        else:
            print(f"{k}: {v} | util={util:.3f} | extras={extra}")                   
        print(f"    previous_key: {previous_key(k, v[0])} = {results.get(previous_key(k, v[0]))}")


if __name__ == "__main__":
    cache = Cache(12, muladd)
    bw = Bandwidth(cache)
    results = bw.dynamic_times(3, 8)
    pp(results)







