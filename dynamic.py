# Dynamic programming for matmul computation with minimal time
# The trick is to use the simulator and use only powers of 2 for now.
# limit time to 100 before running the simulator for now (a*b*c < 100, a*b*c+b*c*d < 100)

# OK, now let's start with just the op (simple 2 powers):
from simulator import Cache, Bandwidth, utilization, Tensor, reset_counters, get_counters
from simulate import muladd, matmulsimple, matmul
import os
import math
from itertools import zip_longest
DEBUG = int(os.environ.get("DEBUG", 0))
def run_dynamic_best(node, *tensors, reset_counter=True, accumulate_output=None, only_store=False):
    """
    Choose the output cache level that yields the shortest predicted runtime
    and execute the matrix‑chain multiplication at that level.

    The routine builds the dynamic‑programming table via
    ``node.dynamic_times`` for the given operand chain, evaluates every
    reachable cache level, and finally calls :func:`run_dynamic` once with
    the fastest level.

    Parameters
    ----------
    node : simulator.BinOpx | simulator.Bandwidth | simulator.Cache
        Compute node from which the cache hierarchy is reachable.
    *tensors : simulator.Tensor
        Operands of the matrix chain (must form a valid chain: A@B@C…).
    reset_counter : bool, default ``True``
        Whether to reset performance counters before the final execution.
    accumulate_output : simulator.Tensor or ``None``
        Forwarded to :func:`run_dynamic`.
    only_store : bool, default ``False``
        Forwarded to :func:`run_dynamic`.

    Returns
    -------
    simulator.Tensor
        The resulting output tensor located at the chosen cache level.
    """
    if len(tensors) < 2:
        raise ValueError("run_dynamic_best requires at least two tensors")

    # Allow the final positional argument to be an output buffer (accumulator).
    ts = list(tensors)

    # Construct the dimension list (d0, d1, d2, …) from the chain shapes.
    dims = [ts[0].sz[0], ts[0].sz[1]]
    for t in ts[1:]:
        if len(t.sz) != 2 or dims[-1] != t.sz[0]:
            raise ValueError("Incompatible tensor chain")
        dims.append(t.sz[1])

    # Use a power-of-two limit covering the product of chain dims so that the
    # enumeration includes the exact shapes we want to run.
    prod = 1
    for d in dims:
        prod *= d
    limit = 1
    while limit <= prod:
        limit *= 2

    # Build the DP table (same convention as in the __main__ demo).
    results = node.dynamic_times(len(ts), limit)
    #print("limit: ", limit)
    #pp(results)

    # Helper to reproduce the exact key encoding expected by run_dynamic.
    def _make_key(level: int):
        flat = []
        for i in range(len(dims) - 1):
            flat.extend([(dims[i], dims[i + 1]), getattr(ts[i], 'level', 0)])
        flat.extend([(dims[0], dims[-1]), level])
        return tuple(flat)

    # Collect all reachable cache levels starting at `node`.
    levels = set()
    cur = node
    if isinstance(cur, Bandwidth):
        cur = cur.cache
    while cur is not None:
        if isinstance(cur, Cache):
            levels.add(cur.level)
        cur = getattr(cur, "parentcache", None)
    if not levels:
        levels.add(0)  # Fall back to level‑0 (DRAM) if no cache hierarchy.

    best_level = None
    if accumulate_output is not None:
        best_level = accumulate_output.level
    else:
        best_time = None
        for lvl in sorted(levels):
            entry = results.get(_make_key(lvl))
            if entry is None:
                continue
            numeric_tail = [e for e in entry[1:] if isinstance(e, (int, float))]
            if not numeric_tail:
                continue
            runtime = max(numeric_tail)
            if best_time is None or runtime < best_time:
                best_time = runtime
                best_level = lvl
    # pp(results)

    # Execute once at the best level.
    return run_dynamic(
        results,
        node,
        *ts,
        out_level=best_level,
        reset_counter=reset_counter,
        accumulate_output=accumulate_output,
        only_store=only_store,
    )


def run_dynamic(results, node, *tensors, out_level=None, reset_counter=True, accumulate_output=None, only_store=False):
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
        out = _run_dynamic_binopx(node, tensors, accumulate_output, only_store=only_store)
    elif entry[0][0] == "LDST":
        out = _run_dynamic_ldst(node, tensors, accumulate_output, entry[0][1:], out_level=out_level, key=key, results=results, only_store=only_store)
    elif entry[0][0] == "JOIN":
        interm_lvl = entry[0][2] if len(entry[0]) > 2 else getattr(tensors[entry[0][1] - 1], "level", 0)
        out = _run_dynamic_join(
            node,
            tensors,
            accumulate_output,
            entry[0][1],
            interm_lvl,
            out_level=out_level,
            key=key,
            results=results,
            only_store=only_store,
        )
    elif entry[0][0] == "DBL":
        out = _run_dynamic_dbl(node, tensors, accumulate_output, entry[0][1], out_level=out_level, key=key, results=results, only_store=only_store)
    else:
        raise NotImplementedError("Unsupported dynamic entry type: {}".format(entry[0]))

    counters = get_counters(node)
    # decrement counters by orig_counter
    counters = [c - o for c, o in zip_longest(counters, orig_counter, fillvalue=0)]
    while len(counters) > len(entry[1:]) and counters[-1] == 0:
        counters.pop()
    if accumulate_output is not None and not only_store:
        _account_accumulate_load(node, counters, accumulate_output)
    if counters != entry[1:]:
            ppkey(results, key)
            raise AssertionError("Old: Counters mismatch: {} != {}, key: {}, entry: {}, has_accumulate_output: {}, only_store: {}".format(counters, entry[1:], key, entry, accumulate_output is not None, only_store))
                
    # When accumulating into a caller-provided output buffer, loads/stores can
    # be orchestrated at a surrounding LDST level, so bandwidth deltas observed
    # within this call may not exactly match the compact dynamic entry. In that
    # case, validate only the CPU count. For standard runs, enforce full match.
    if accumulate_output is not None:
        if len(entry) > 1 and (len(counters) == 0 or counters[0] != entry[1]):
            raise AssertionError("CPU counter mismatch under accumulation: {} != {}, key: {}, entry: {}".format(counters, entry[1:], key, entry))
    else:
        if counters != entry[1:]:
            raise AssertionError("Counters mismatch: {} != {}, key: {}, entry: {}".format(counters, entry[1:], key, entry))
    if isinstance(node, Bandwidth) or isinstance(node, Cache):
        assert node.cachecontains(out, allow_lower_level=True), "Output tensor not in cache: " + str(out)  + " for key: " + str(key) + ", value: " + str(entry)+ ", accumulate_output: " + str(accumulate_output)+ ", in " + str(node)
    return out


def _account_accumulate_load(node, counters, accumulate_output):
    """Decrement bandwidth counters for loads of ``accumulate_output``."""
    cur = node
    while cur is not None:
        if isinstance(cur, Cache):
            if cur.level < accumulate_output.level:
                idx = cur.level + 1  # counters[0] is CPU; bandwidth links start at 1
                while idx >= len(counters):
                    counters.append(0)
                counters[idx] -= accumulate_output.size()
            cur = cur.parent
        elif isinstance(cur, Bandwidth):
            cur = cur.cache
        else:
            cur = None


def _run_dynamic_binopx(node, tensors, accumulate_output, only_store=False):
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
    if root_node.clevel != 0:
        raise ValueError("Root node level is not 0: ", root_node.clevel)
    for i in range(1, n_mats):
        right = tensors[i]
        is_last = (i == n_mats - 1)
        if is_last:
            if accumulate_output is not None:
                if accumulate_output.level != 0:
                    raise ValueError("Accumulate output level is not 0: ", accumulate_output.level)
                # If the provided output buffer resides at a different level
                # than the compute node expects, allocate a local destination
                # at the compute level and let the caller (e.g., LDST layer)
                # handle any stores to the higher level.
                if accumulate_output.level != 0:
                    if highest_cache is not None:
                        dest = highest_cache.calloc(left.sz[0], right.sz[1])
                    else:
                        dest = Tensor.zeros(left.sz[0], right.sz[1], level=0)
                else:
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


def _run_dynamic_ldst(node, tensors, accumulate_output, bw_op, out_level=None, key=None, results=None, only_store=False):
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
            loaded[i] = node.load(tensors[i], allow_lower_level=True)
    
    loaded_out = accumulate_output

    if accumulate_output and len(tensors) in bw_op:
        if only_store:
            loaded_out = None
        else:
            loaded_out = node.load(accumulate_output, allow_lower_level=True)

    out_low = run_dynamic(
        results,
        node,
        *loaded,
        reset_counter=False,
        accumulate_output=loaded_out,
        out_level=out_level-1 if (len(tensors) in bw_op) else out_level,
        only_store=only_store
    )

    if accumulate_output is None:
        if out_level != out_low.level:
            out_high = node.calloc(out_low.sz[0], out_low.sz[1], level=out_level)
            node.store_to(out_low, out_high, allow_lower_level=True)
        else:
            out_high = out_low
    else:
        out_high = accumulate_output
        if len(tensors) in bw_op:
            node.store_to(out_low, out_high, allow_lower_level=True)

    for i in bw_op:
        if i < len(loaded):
            node.free(loaded[i], allow_lower_level=True)

    return out_high


def _run_dynamic_dbl(node, tensors, accumulate_output, j, out_level=None, key=None, results=None, only_store=False):
    """Execute a DBL split by running two half-problems and writing directly
    into the appropriate output slices. No temps, minimal branching."""
    m = len(tensors)
    rows, cols = tensors[0].sz[0], tensors[-1].sz[1]
    out = accumulate_output
    ops1, ops2 = list(tensors), list(tensors)

    if j in (0, m):
        if out is None:
            out = node.calloc(rows, cols, level=out_level)
        h = rows // 2 if j == 0 else cols // 2
        if j == 0:
            ops1[0], ops2[0] = tensors[0][0:h, :], tensors[0][h:2 * h, :]
            out_slices = (out[0:h, :], out[h:2 * h, :])
        else:
            ops1[-1], ops2[-1] = tensors[-1][:, 0:h], tensors[-1][:, h:2 * h]
            out_slices = (out[:, 0:h], out[:, h:2 * h])
        store = only_store or accumulate_output is None
        for ops, out_slice in zip((ops1, ops2), out_slices):
            run_dynamic(
                results,
                node,
                *ops,
                reset_counter=False,
                accumulate_output=out_slice,
                out_level=out_level,
                only_store=store,
            )
    else:
        h = tensors[j].sz[0] // 2
        ops1[j - 1], ops1[j] = tensors[j - 1][:, 0:h], tensors[j][0:h, :]
        ops2[j - 1], ops2[j] = tensors[j - 1][:, h:2 * h], tensors[j][h:2 * h, :]
        if out is None:
            out = run_dynamic(results, node, *ops1, reset_counter=False, out_level=out_level, only_store=only_store)
        else:
            run_dynamic(
                results,
                node,
                *ops1,
                reset_counter=False,
                accumulate_output=out,
                out_level=out_level,
                only_store=only_store,
            )
        run_dynamic(
            results,
            node,
            *ops2,
            reset_counter=False,
            accumulate_output=out,
            out_level=out_level,
            only_store=False,
        )

    return out


def _run_dynamic_join(
    node,
    tensors,
    accumulate_output,
    n_inputs,
    interm_level,
    out_level=None,
    key=None,
    results=None,
    only_store=False,
):
    """Execute a JOIN step by concatenating two matmul chains.

    ``n_inputs`` specifies how many operands belong to the first chain and
    ``interm_level`` gives the cache level for the intermediate result. The
    first subproblem runs without an explicit output buffer and its result is
    then used as the leading operand of the second chain, which runs with
    ``accumulate_output`` if provided.
    """

    if n_inputs < 1 or n_inputs >= len(tensors):
        raise ValueError("JOIN split out of range")

    # First subchain: compute the left part to obtain the intermediate result.
    ops1 = list(tensors[:n_inputs])
    interm = run_dynamic(
        results,
        node,
        *ops1,
        reset_counter=False,
        out_level=interm_level,
        only_store=only_store or accumulate_output is None,
    )

    # Second subchain: use the intermediate result as the first operand.
    ops2 = [interm] + list(tensors[n_inputs:])
    out = run_dynamic(
        results,
        node,
        *ops2,
        reset_counter=False,
        accumulate_output=accumulate_output,
        out_level=out_level,
        only_store=only_store,
    )

    if isinstance(node, Cache) or isinstance(node, Bandwidth):
        node.free(interm, allow_lower_level=True)

    return out


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
            tag = op[1]
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
        if kind == "JOIN":
            # Reconstruct the first subproblem that produced this JOIN entry.
            n_inputs = op[1]
            interm_lvl = op[2]
            if isinstance(n_inputs, int) and 0 < n_inputs <= len(ops):
                first_ops = ops[:n_inputs]
                # Output dims: rows from first operand, cols from last operand
                out_rows = first_ops[0][0][0]
                out_cols = first_ops[-1][0][1]
                out_lvl = interm_lvl if isinstance(interm_lvl, int) else first_ops[-1][1]
                flat = []
                for shp, lvl in first_ops:
                    flat.extend([shp, lvl])
                flat.extend([(out_rows, out_cols), out_lvl])
                return tuple(flat)
    return None


def previous_key2(key, op=None):
    """Variant of :func:`previous_key` for JOIN entries.

    For JOIN rows this returns the key of the *second* subproblem, i.e. the
    right-hand matrix chain. For all other operations it delegates to
    :func:`previous_key`.
    """

    pairs = [(key[i], key[i + 1]) for i in range(0, len(key), 2)]
    ops = pairs[:-1]
    outp = pairs[-1]

    if isinstance(op, tuple) and op and op[0] == "JOIN":
        n_inputs = op[1]
        interm_lvl = op[2]
        if isinstance(n_inputs, int) and 0 < n_inputs < len(ops):
            left_ops = ops[:n_inputs]
            right_ops = ops[n_inputs:]
            # Build intermediate result from the left subchain
            interm_rows = left_ops[0][0][0]
            interm_cols = left_ops[-1][0][1]
            interm_lvl = interm_lvl if isinstance(interm_lvl, int) else left_ops[-1][1]
            chain = [((interm_rows, interm_cols), interm_lvl)] + list(right_ops)
            flat = []
            for shp, lvl in chain:
                flat.extend([shp, lvl])
            flat.extend([outp[0], outp[1]])
            return tuple(flat)
        return None

    return previous_key(key, op)


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

    prev1 = results.get(previous_key(key, v[0]))
    if prev1 is None:
        return None
    cur_tail = list(v[1:])
    prev1_tail = list(prev1[1:])
    if kind == "JOIN":
        prev2 = results.get(previous_key2(key, v[0]))
        prev2_tail = list(prev2[1:]) if prev2 is not None else []
        prev_tail = [p1 + p2 for p1, p2 in zip_longest(prev1_tail, prev2_tail, fillvalue=0)]
    else:
        prev_tail = prev1_tail
    factor = 2 if kind == "DBL" else 1
    return [c - factor * p for c, p in zip_longest(cur_tail, prev_tail, fillvalue=0)]


def ppkey(results, k):
    """Pretty-print a single dynamic-programming result entry."""

    v = results[k]
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
    pk = previous_key(k, v[0])
    print(f"    previous_key: {pk} = {results.get(pk)}")
    if isinstance(v[0], tuple) and v[0] and v[0][0] == "JOIN":
        pk2 = previous_key2(k, v[0])
        print(f"    previous_key2: {pk2} = {results.get(pk2)}")


def pp(results):
    for k in results:
        ppkey(results, k)


if __name__ == "__main__":
    cache = Cache(12, muladd)
    bw = Bandwidth(cache)
    results = bw.dynamic_times(3, 8)
    pp(results)
