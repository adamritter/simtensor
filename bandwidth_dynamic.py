"""Helper routines for Bandwidth.dynamic_times."""

import heapq
from collections import defaultdict
from itertools import combinations, zip_longest


def _shape_elems(shape):
    """Return the product of dimensions in ``shape``."""

    n = 1
    for d in shape:
        n *= d
    return n


def _dp_split_key(key):
    """Split a DP key into operand pairs and an output pair."""

    pairs = [(key[i], key[i + 1]) for i in range(0, len(key), 2)]
    return pairs[:-1], pairs[-1]


def _dp_short_key(key):
    """Return a shortened key of input dimensions and level.

    ``key`` alternates ``(shape, level)`` pairs for each input matrix
    followed by the output pair. At least two input matrices must be
    present.

    Returns the first input's dimensions followed by its cache level,
    e.g. ``(m, k, level)``.
    """

    ops, _ = _dp_split_key(key)
    if len(ops) < 2:
        raise ValueError("need at least two input matrices")
    shp, lvl = ops[0]
    return (*shp, lvl)


def _dp_join_key(ops, outp):
    """Assemble a DP key from operand and output pairs."""

    flat = []
    for shp, lvl in ops:
        flat.extend([shp, lvl])
    flat.extend([outp[0], outp[1]])
    return tuple(flat)


def _dp_ops_count(ops):
    """Return left-associated matmul op count for a chain."""

    if len(ops) < 2:
        return 0
    a0 = ops[0][0][0]
    total = 0
    for i in range(len(ops) - 1):
        total += a0 * ops[i][0][1] * ops[i + 1][0][1]
    return total


def _dp_bw_time_for_level(bw, key, level_here):
    """Return bandwidth time for all shapes at ``level_here``."""

    ops, outp = _dp_split_key(key)
    words = 0
    for shp, lvl in ops + [outp]:
        if isinstance(lvl, int) and lvl == level_here:
            words += _shape_elems(shp)
    return words * bw.input_clocks_per_word


def _dp_update_mapping(mapping, new_key, new_cpu, new_bws, tag, keyinfo, heap=None):
    """Insert or improve a DP entry with computed times and a tag.

    ``tag`` is stored verbatim as the first element in the mapping entry.
    The remaining elements are the CPU time followed by any bandwidth times.
    Entries are compared by the maximum bandwidth time; ties fall back to the
    summed bandwidth times.
    """

    new = [tag, new_cpu] + new_bws

    if new_key not in mapping:
        mapping[new_key] = new
        keyinfo[_dp_short_key(new_key)].add(new_key)
        if heap is not None:
            heapq.heappush(heap, (new_cpu, new_key))
        return
    cur = mapping[new_key]

    cur_max = max(cur[1:])
    new_max = max(new[1:])

    if new_max < cur_max or (new_max == cur_max and sum(new[1:]) < sum(cur[1:])):
        mapping[new_key] = new
        if heap is not None:
            heapq.heappush(heap, (new_cpu, new_key))


def _dp_expand_key(key, times, mapping, level_here, max_cpu_time, heap=None, keyinfo=None):
    """Attempt DBL expansion for one key.

    For each legal position ``j`` where adjacent operands are at
    ``level_here``, construct a new key with the shared dimension doubled and
    compute timing for the expanded problem. CPU time doubles; bandwidth times
    double with an extra traversal of the output for interior splits.
    """

    ops, outp = _dp_split_key(key)
    if len(ops) < 2:
        return
    old_ops = _dp_ops_count(ops)
    if old_ops == 0:
        return
    cur_cpu = times[1]
    n = len(ops)
    for j in range(n + 1):
        new_ops_list = list(ops)
        new_outp = outp
        if j == 0:
            (shp, lvl) = ops[0]
            if lvl != level_here or outp[1] != level_here:
                continue
            a0, a1 = shp
            new_ops_list[0] = ((a0 * 2, a1), lvl)
            (od, ol) = outp
            new_outp = ((od[0] * 2, od[1]), ol)
        elif j == n:
            (shp, lvl) = ops[-1]
            if lvl != level_here or outp[1] != level_here:
                continue
            b0, b1 = shp
            new_ops_list[-1] = ((b0, b1 * 2), lvl)
            (od, ol) = outp
            new_outp = ((od[0], od[1] * 2), ol)
        else:
            (ashp, alvl) = ops[j - 1]
            (bshp, blvl) = ops[j]
            if alvl != level_here or blvl != level_here:
                continue
            a0, a1 = ashp
            b0, b1 = bshp
            if a1 != b0:
                continue
            new_ops_list[j - 1] = ((a0, a1 * 2), alvl)
            new_ops_list[j] = ((b0 * 2, b1), blvl)

        new_cpu = cur_cpu * 2
        if new_cpu > max_cpu_time:
            continue

        new_key = _dp_join_key(new_ops_list, new_outp)
        new_bws = list(map(lambda x: x * 2, times[2:]))

        if 0 < j < n:
            (odims, olvl) = new_outp
            for level in range(olvl):
                if level <= level_here:
                    new_bws[level] += _shape_elems(odims)
        _dp_update_mapping(mapping, new_key, new_cpu, new_bws, ("DBL", j), keyinfo, heap)


def _dp_expand(mapping, level_here, max_cpu_time, keyinfo=None):
    """Run the bandwidth-level DP expansion over all current entries.

    Uses a min-heap ordered by CPU time. While the smallest entry can be
    doubled without exceeding ``max_cpu_time``, attempt DBL expansions and
    enqueue any new or improved mappings.
    """

    heap = []
    for key, times in mapping.items():
        cpu = times[1]
        heapq.heappush(heap, (cpu, key))

    while heap:
        cpu, key = heapq.heappop(heap)
        times = mapping.get(key)
        if not times:
            continue
        cur_cpu = times[1]
        if cur_cpu > max_cpu_time / 2:
            break
        _dp_expand_key(key, times, mapping, level_here, max_cpu_time, heap, keyinfo)
    return mapping


def _dp_join_short_keys(key, keyinfo, mapping, heap, maxn, max_cpu_time):
    """Join ``key`` with other keys sharing its output short key.

    ``keyinfo`` maps short keys to full keys. Using the output dimensions and
    level of ``key`` we look up candidate keys whose first operand matches.
    If the joined chain does not exceed ``maxn`` inputs and the resulting CPU
    time stays within ``max_cpu_time``, the combined entry is inserted into the
    result mapping via :func:`_dp_update_mapping`.
    """

    if keyinfo is None:
        return
    value1 = mapping.get(key)
    if value1 is None:
        return
    _, outp = _dp_split_key(key)
    short = (*outp[0], outp[1])
    candidates = keyinfo.get(short, set())
    for other in list(candidates):
        if other == key:
            continue
        value2 = mapping.get(other)
        if value2 is None:
            continue
        joined_key, joined_value = join_matmuls(key, value1, other, value2)
        n_inputs = len(joined_key) // 2 - 1
        if n_inputs > maxn:
            continue
        new_cpu = joined_value[1]
        if new_cpu > max_cpu_time:
            continue
        tag = joined_value[0]
        new_bws = joined_value[2:]
        _dp_update_mapping(mapping, joined_key, new_cpu, new_bws, tag, keyinfo, heap)


def dynamic_times_impl(bw, nmatmuls, max_cpu):
    """Augment compute dynamic_times with bandwidth variants and timing.

    First delegates to the downstream cache to obtain the base CPU-time
    mapping. For each entry, produces a base variant and additional variants
    where operands at the previous level are promoted across ``bw`` with
    bandwidth time accounted. Keys follow the convention from
    ``BinOpx.dynamic_times``: a tuple alternating ``(shape, level)`` pairs.
    Returns a mapping of key variant to a list beginning with the last operation
    tag followed by CPU and bandwidth times.
    """

    base = bw.cache.dynamic_times(nmatmuls, max_cpu)
    prev_level = bw.cache.level
    out = {}
    keyinfo = defaultdict(set)
    for key, v in base.items():
        base_key = tuple(key)
        out[base_key] = v
        keyinfo[_dp_short_key(base_key)].add(base_key)
        candidate_idxs = []
        for i in range(0, len(key), 2):
            lvl = key[i + 1]
            if lvl == prev_level:
                candidate_idxs.append(i)
        for r in range(1, len(candidate_idxs) + 1):
            for subset in combinations(candidate_idxs, r):
                kl = list(key)
                bw_words = 0
                pair_idxs = []
                for i in subset:
                    kl[i + 1] = prev_level + 1
                    bw_words += _shape_elems(kl[i])
                    pair_idxs.append(i // 2)
                bw_time = bw_words * bw.input_clocks_per_word
                last_op = ("LDST",) + tuple(pair_idxs)
                new_key = tuple(kl)
                out[new_key] = [last_op] + v[1:] + [bw_time]
                keyinfo[_dp_short_key(new_key)].add(new_key)
    _dp_expand(out, prev_level + 1, max_cpu, keyinfo)
    out["_key_index"] = keyinfo
    return out


def join_matmuls(key1, value1, key2, value2):
    """Join two matmul entries and sum their counters.

    ``key1`` and ``key2`` are tuples alternating ``(shape, level)`` pairs with
    the last pair representing the output. ``value1`` and ``value2`` begin with a
    tag followed by CPU and bandwidth counters. The output of ``key1`` must match
    the first input of ``key2``.
    """

    if key1[-2:] != key2[:2]:
        raise AssertionError("mismatched intermediate output/input")

    joined_key = key1[:-2] + key2[2:]
    n_inputs = len(key1) // 2 - 1

    counters = []
    for a, b in zip_longest(value1[1:], value2[1:], fillvalue=0):
        counters.append(a + b)

    joined_value = [("JOIN", n_inputs)] + counters
    return joined_key, joined_value

