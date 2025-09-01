# Dynamic programming for matmul computation with minimal time
# The trick is to use the simulator and use only powers of 2 for now.
# limit time to 100 before running the simulator for now (a*b*c < 100, a*b*c+b*c*d < 100)

# OK, now let's start with just the op (simple 2 powers):
from simulator import Cache, Bandwidth, utilization, Tensor, reset_counters
from simulate import muladd, matmulsimple


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

    # Prepare an execution node depending on the algorithm. For bandwidth-level
    # entries (LDST/DBL) we need a Cache above a Bandwidth link. If the provided
    # `node` is just the compute BinOpx, synthesize a minimal two-level cache
    # hierarchy so we can perform loads/stores and accumulate bandwidth time.
    exec_node = node
    operands_count = len(dims) - 1
    # Parse the key we actually matched to recover operand/output levels
    key_pairs = [(key[i], key[i + 1]) for i in range(0, len(key), 2)]
    operand_levels = [lvl for (_shp, lvl) in key_pairs[:-1]]
    out_pair = key_pairs[-1]
    out_level_marker = out_pair[1]

    if algo in ("LDST", "DBL") and not isinstance(node, Cache):
        # Compute conservative capacities for the synthetic caches
        def shape_elems(shp):
            n = 1
            for d in shp:
                n *= d
            return n
        # High-level cache needs to hold any tensors marked at level 1 in the key
        high_words = 0
        for (shp, lvl) in key_pairs:
            if isinstance(lvl, int) and lvl == 1:
                high_words += shape_elems(shp)
        if high_words == 0:
            # Fallback: be generous
            high_words = sum(shape_elems(shp) for (shp, _lvl) in key_pairs)
        low_words = sum(shape_elems(shp) for (shp, _lvl) in key_pairs)
        # Account for a transient extra buffer during chained multiplies: when
        # we allocate the next intermediate before freeing the previous one,
        # peak usage can exceed (inputs + final output) by up to the size of a
        # single matrix. Reserve that headroom using the largest shape present
        # in the key to avoid "Not enough memory" during LDST/DBL runs.
        peak_extra = 0
        for (shp, _lvl) in key_pairs:
            words = shape_elems(shp)
            if words > peak_extra:
                peak_extra = words
        # Build: BinOpx (node) <- Cache(low) <- Bandwidth <- Cache(high)
        # Intermediate allocation can temporarily require holding two
        # intermediates at once (previous out_low and newly allocated nxt)
        # in addition to any loaded inputs. Reserve headroom for up to two
        # such matrices using the largest shape observed to avoid transient
        # overcommit during chained multiplies.
        low_cache = Cache(max(1, low_words + 2 * peak_extra), node)
        bw_link = Bandwidth(low_cache)
        high_cache = Cache(max(1, max(high_words, low_words)), bw_link)
        exec_node = high_cache

        # Align tensor levels with the matched key so DBL/LDST selection works.
        # Clamp any higher logical levels down to the actual cache level we
        # synthesized. Dynamic tables from multi-link hierarchies can contain
        # levels > exec_node.level; we execute with a single link here, so
        # keep tensors at most at exec_node.level to ensure loads produce
        # compute-level views with the expected levels.
        exec_level = getattr(exec_node, 'level', 0)
        for i in range(operands_count):
            try:
                lvl = operand_levels[i]
                tensors[i].level = lvl if lvl <= exec_level else exec_level
            except Exception:
                pass
        # If an accumulation tensor is provided, align its level as well with
        # the same clamping behaviour.
        if accumulate_output is not None and isinstance(out_level_marker, int):
            try:
                lvl = out_level_marker
                accumulate_output.level = lvl if lvl <= exec_level else exec_level
            except Exception:
                pass

    # Reset counters if requested (use the actual execution node)
    if reset_counter:
        if isinstance(exec_node, Cache):
            reset_counters(exec_node)
        else:
            if hasattr(exec_node, 'time'):
                exec_node.time = 0
    else:
        # If we are not resetting and we're executing an LDST/DBL run without accumulation,
        # adjust for a prior accumulation pass that loaded the output once.
        if accumulate_output is None and isinstance(exec_node, Cache):
            link = getattr(exec_node, 'parent', None)
            if link is not None and hasattr(link, '_last_output_load_words') and getattr(link, '_last_output_load_words'):
                try:
                    link.input -= getattr(link, '_last_output_load_words')
                except Exception:
                    pass

    # Dispatch to specific runner implementations using exec_node
    if algo == "BinOpx":
        out = _run_dynamic_binopx(exec_node, tensors, accumulate_output)
    elif algo == "LDST":
        out = _run_dynamic_ldst(
            exec_node,
            tensors,
            accumulate_output,
            bw_op,
            out_level_marker,
            key=key,
            results=results,
        )
    elif algo == "DBL":
        # Pass through the matched key and op tag so the DBL runner can use
        # previous_key to recover the predecessor entry and slice tensors
        # correctly. Also pass the full results map so recursive calls can
        # resolve the predecessor algorithm.
        out = _run_dynamic_dbl(
            exec_node,
            tensors,
            accumulate_output,
            out_level_marker,
            key,
            bw_op,
            results,
        )
    else:
        raise NotImplementedError("Unsupported dynamic entry type: {}".format(algo))

    # Verification (common)
    if reset_counter:
        # CPU node depends on dispatch
        if isinstance(exec_node, Cache):
            comp_node = exec_node.parentcache.parent if hasattr(exec_node, 'parentcache') else None
            link = getattr(exec_node, 'parent', None)
        else:
            comp_node = exec_node
            link = None
        if cpu_expected is not None and comp_node is not None:
            if comp_node.time != cpu_expected:
                raise AssertionError("CPU time mismatch: {} != {}".format(comp_node.time, cpu_expected))
        # Bandwidth time
        if isinstance(entry, list) and len(entry) == 3 and link is not None and hasattr(link, 'time'):
            # Single-link case: entry is [(op,...), cpu, bw]. Validate bw.
            bw_expected = entry[2]
            # If we additionally loaded the output for accumulation, add its size
            if accumulate_output is not None:
                bw_expected += dims[0] * dims[-1]
            if link.time != bw_expected:
                raise AssertionError("Bandwidth time mismatch: {} != {}".format(link.time, bw_expected))

    return out


def _run_dynamic_binopx(node, tensors, accumulate_output):
    # Compute left-associated chain using a single loop of matmulsimple calls
    # Supports being invoked with either a compute BinOpx or a Cache sitting
    # above a BinOpx. In the latter case, allocate intermediates in the child
    # cache and call matmulsimple on the compute node.
    if isinstance(node, Cache):
        # Two cases:
        # 1) node.parent is Bandwidth -> compute BinOpx is node.parentcache.parent
        # 2) node.parent is BinOpx     -> compute BinOpx is node.parent
        if hasattr(node, 'parentcache') and node.parentcache is not None:
            comp_cache = node.parentcache
            if not hasattr(comp_cache, 'parent'):
                raise RuntimeError("Cannot locate compute node under cache for BinOpx execution")
            comp_node = comp_cache.parent
            cache_for_alloc = comp_cache
        else:
            # Directly above the compute node
            if not hasattr(node, 'parent'):
                raise RuntimeError("Cannot locate compute node under cache for BinOpx execution")
            comp_node = node.parent
            cache_for_alloc = node
        clevel = getattr(comp_node, "clevel", 0)
        alloc_fn = lambda r, c: cache_for_alloc.calloc(r, c)
    else:
        comp_node = node
        clevel = getattr(node, "clevel", 0)
        alloc_fn = lambda r, c: Tensor.zeros(r, c, level=clevel)

    n_mats = len(tensors)
    if accumulate_output is not None:
        if accumulate_output.sz != [tensors[0].sz[0], tensors[-1].sz[1]]:
            raise ValueError("accumulate_output shape mismatch")
        if getattr(accumulate_output, 'level', clevel) != clevel:
            raise ValueError("accumulate_output must reside at CPU level {}".format(clevel))

    left = tensors[0]
    out = None
    for i in range(1, n_mats):
        right = tensors[i]
        is_last = (i == n_mats - 1)
        if is_last and accumulate_output is not None:
            dest = accumulate_output
        else:
            dest = alloc_fn(left.sz[0], right.sz[1])
        if is_last:
            out = dest
        matmulsimple(comp_node, left, right, dest)
        left = dest

    return out


def _run_dynamic_ldst(node, tensors, accumulate_output, bw_op, out_level_marker=None, key=None, results=None):
    """Execute one LDST step by loading listed operands, then running the
    predecessor dynamic entry using run_dynamic. This avoids re-implementing
    the matmul here and ensures consistent accounting.
    """
    if not isinstance(node, Cache):
        raise TypeError("LDST execution requires a Cache node for load/store")

    # Determine which operands to load at this step
    out_pos = len(tensors)
    idxs = list(bw_op[1:]) if bw_op and len(bw_op) > 1 else []
    load_idxs = [i for i in idxs if i < out_pos]

    # Load selected operands down one level
    loaded = {}
    for i in load_idxs:
        try:
            node.alloc(tensors[i])
        except Exception:
            pass
        loaded[i] = node.load(tensors[i])

    # If accumulation is requested, load the destination as well
    loaded_out = None
    if accumulate_output is not None:
        try:
            node.alloc(accumulate_output)
        except Exception:
            pass
        loaded_out = node.load(accumulate_output)

    # Build argument list for the predecessor run: use loaded views where
    # applicable so the next dynamic step sees reduced levels.
    rec_args = [loaded.get(i, tensors[i]) for i in range(len(tensors))]

    # Compute predecessor key and delegate to run_dynamic without resetting
    # counters so bandwidth inputs accumulate across LDST steps.
    if key is None:
        # Reconstruct best-effort key from current tensors and node
        dims = [tensors[0].sz[0], tensors[0].sz[1]]
        for t in tensors[1:]:
            dims.append(t.sz[1])
        out_level = getattr(node, 'level', 0)
        flat = []
        for i in range(len(dims) - 1):
            flat.extend([(dims[i], dims[i + 1]), getattr(tensors[i], 'level', 0)])
        flat.extend([(dims[0], dims[-1]), out_level])
        key = tuple(flat)

    prev = previous_key(key, bw_op)

    # The results table is required to resolve the predecessor algorithm
    if results is None:
        raise ValueError("_run_dynamic_ldst requires results mapping for recursion")

    out_low = run_dynamic(
        results,
        node.parentcache,
        *rec_args,
        reset_counter=False,
        accumulate_output=loaded_out,
    )

    # If the dynamic key expects the output to reside at this (high) cache
    # level, commit the low-level result back up. This accounts for the
    # bandwidth time associated with writing the output across the link.
    target_high = False
    if out_level_marker is not None:
        target_high = isinstance(out_level_marker, int) and out_level_marker == getattr(node, 'level', 0)
    out_view = out_low
    if target_high:
        # Choose destination view: either reuse the provided high-level output
        # or allocate a fresh one.
        if accumulate_output is None:
            out_high = node.calloc(out_low.sz[0], out_low.sz[1])
        else:
            out_high = accumulate_output
        node.store_to(out_low, out_high)
        out_view = out_high

    # Free loaded inputs in the child cache to keep capacity bounded
    for i in load_idxs:
        try:
            node.parentcache.free(loaded[i])
        except Exception:
            pass

    # Track whether we loaded the output for accumulation to let outer wrapper
    # adjust counters on subsequent non-accumulate calls
    link = getattr(node, 'parent', None)
    if accumulate_output is not None and link is not None:
        try:
            setattr(link, '_last_output_load_words', (accumulate_output.sz[0] * accumulate_output.sz[1]))
        except Exception:
            pass
    elif link is not None:
        try:
            setattr(link, '_last_output_load_words', 0)
        except Exception:
            pass

    return out_view


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







