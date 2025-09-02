"""Bandwidth link between caches with simple timing model."""

import copy
import heapq


class Bandwidth:
    """Bandwidth link between caches with simple timing model.

    Args
    - cache: downstream cache level
    - input_clocks_per_word: clocks to transfer one input word
    - output_clocks_per_word: clocks to transfer one output word; if None, a
      shared line is assumed and time scales with total words moved.

    Time model
    - Shared line: time = (input + output) * input_clocks_per_word
    - Separate lines: time = max(input * input_cpw, output * output_cpw)
    """
    def __init__(self, cache, input_clocks_per_word=1, output_clocks_per_word=None):
        self.input = 0
        self.output = 0
        self.cache = cache
        self.input_clocks_per_word = input_clocks_per_word
        self.output_clocks_per_word = output_clocks_per_word
        self.time = 0  # aggregate transfer time in 'clocks'
        # No resident-tracking by default
        self._resident = None
    
    def free(self, tensor, allow_lower_level=False):
        if tensor.level == self.cache.level + 1:
            return tensor # Noop
        if not allow_lower_level:
            raise Exception("Bandwidth.free does not support allow_lower_level=False")
        return self.cache.free(tensor, allow_lower_level)

    def alloc(self, tensor, allow_lower_level=False):
        if tensor.level == self.cache.level + 1:
            return tensor # Noop
        if not allow_lower_level:
            raise Exception("Bandwidth.alloc does not support allow_lower_level=False")
        return self.cache.alloc(tensor, allow_lower_level)
    
    def cachecontains(self, tensor, allow_lower_level=False):
        if tensor.level == self.cache.level + 1:
            return True # Noop
        if not allow_lower_level:
            raise Exception("Bandwidth.cachecontains does not support allow_lower_level=False")
        return self.cache.cachecontains(tensor, allow_lower_level)

    def store_to(self, src, dst, allow_lower_level=False):
        """Route or execute a store_to across this link.

        Cases:
        - If `src.level` is below `self.cache.level`, delegate downstream to
          allow promotion towards this link.
        - If `dst.level` equals the level above this link and `src.level` is
          exactly `self.cache.level`, move the data across (accounting output)
          and then copy element-wise into `dst`.
        - Otherwise, if `allow_lower_level` is True, delegate to downstream
          cache for further handling.
        """
        if not allow_lower_level:
            raise Exception("Bandwidth.store_to does not support allow_lower_level=False")

        # Delegate deeper if source is below our child cache level
        if src.level < self.cache.level:
            return self.cache.store_to(src, dst, allow_lower_level)

        # If destination is the level above this link, we can perform the
        # transfer and local copy.
        upper_level = self.cache.level + 1
        if dst.level == upper_level:
            if src.level != self.cache.level:
                # Nothing to do at this link; delegate down if permitted
                return self.cache.store_to(src, dst, allow_lower_level)
            # Move src across the link and free in child
            self.store(src)
            # Sizes must match
            if src.size() != dst.size():
                raise Exception("Source and destination sizes must match for store_to")
            # Copy element-wise
            sz = src.sz
            if len(sz) != len(dst.sz):
                raise Exception("Source and destination ranks must match for store_to")
            def rec(dim, off_s, off_d):
                if dim == len(sz):
                    dst.data[off_d] = src.data[off_s]
                    return
                step_s = src.skips[dim]
                step_d = dst.skips[dim]
                for i in range(sz[dim]):
                    rec(dim + 1, off_s + i * step_s, off_d + i * step_d)
            rec(0, src.offset, dst.offset)
            return dst

        # Fallback: delegate to downstream cache for further routing
        return self.cache.store_to(src, dst, allow_lower_level)

    def calloc(self, n, m, level=None):
        """Allocate a zero tensor at a requested level relative to this link.

        - If level equals the level above this link, return a standalone Tensor
          (no residency tracking at bandwidth-only level).
        - If level is below, delegate to the child cache.
        - If level is None, default to the level above this link.
        """
        if level is None:
            level = self.cache.level + 1
        from simulator import Tensor  # local import to avoid circular dependency
        if level == self.cache.level + 1:
            data = [0] * (n * m)
            return Tensor(data, level, [n, m])
        if level <= self.cache.level:
            return self.cache.calloc(n, m, level)
        # No knowledge of higher caches; return a tensor at the requested level
        data = [0] * (n * m)
        return Tensor(data, level, [n, m])

    def _update_time(self):
        if self.output_clocks_per_word is None:
            # Shared line: time scales with total words moved
            self.time = (self.input + self.output) * self.input_clocks_per_word
        else:
            # Separate lines: overlap, take the slower side
            tin = self.input * self.input_clocks_per_word
            tout = self.output * self.output_clocks_per_word
            self.time = max(tin, tout)
        return self.time


    def _shape_elems(self, shape):
        n = 1
        for d in shape:
            n *= d
        return n

    def _dp_split_key(self, key):

        """Split a dynamic-programming key into operand pairs and output.

Keys are tuples of alternating (shape, level) pairs as produced by
dynamic_times. This helper returns:
- ops: list[((d_i, d_{i+1}), level_i)] for all adjacent matrices
- outp: ((d_0, d_n), level_out)"""
        pairs = [(key[i], key[i + 1]) for i in range(0, len(key), 2)]
        return pairs[:-1], pairs[-1]

    def _dp_join_key(self, ops, outp):

        """Assemble a dynamic-programming key from ops and output pair.

The inverse of _dp_split_key. Accepts:
- ops: list of ((rows, cols), level)
- outp: ((rows, cols), level)
Returns a flat tuple alternating (shape, level)."""
        flat = []
        for shp, lvl in ops:
            flat.extend([shp, lvl])
        flat.extend([outp[0], outp[1]])
        return tuple(flat)

    def _dp_ops_count(self, ops):

        """Compute total scalar multiply-add count for a left-associated chain.

ops is a list of ((d_i, d_{i+1}), level) entries describing adjacent
matrices A0@A1@... in a chain. The cost model used throughout is the
standard left-associated matmul count:
    sum_{i=1..n-1} d0 * d_i * d_{i+1}"""
        if len(ops) < 2:
            return 0
        a0 = ops[0][0][0]
        total = 0
        for i in range(len(ops) - 1):
            total += a0 * ops[i][0][1] * ops[i + 1][0][1]
        return total

    def _dp_bw_time_for_level(self, key, level_here):

        """Return bandwidth time for all shapes at a given cache level.

Sums the element counts for each (shape, level) pair in `key` whose
level equals `level_here` (including the output pair), then multiplies
by this link's input_clocks_per_word to obtain a time in 'clocks'."""
        ops, outp = self._dp_split_key(key)
        words = 0
        for shp, lvl in ops + [outp]:
            if isinstance(lvl, int) and lvl == level_here:
                words += self._shape_elems(shp)
        return words * self.input_clocks_per_word

    def _dp_update_mapping(self, mapping, new_key, new_cpu, new_bws, tag, heap=None):
        """Insert or improve a DP entry with computed times and a tag.

        Supports multiple bandwidth times (for multi-link hierarchies). The
        mapping values have the form:
            [ ("DBL", tag), cpu_time, bw_time_link0, bw_time_link1, ... ]

        Args:
            mapping: dict to mutate
            new_key: DP key to insert/update
            new_cpu: proposed compute time
            new_bws: list of bandwidth times for all
                     links seen so far; order must match existing entries
            tag: descriptor for where the DBL expansion occurred

        Update policy:
        - CPU time: keep the minimum observed.
        - Bandwidth times: compare by the maximum across links; keep the list
          whose maximum is smaller. This mirrors utilization being dominated by
          the slowest link.
        """
        new = [("DBL", tag), new_cpu] + new_bws

        if new_key not in mapping:
            mapping[new_key] = new
            # Requeue brand-new entries for further expansion if a heap is provided
            if heap is not None:
                heapq.heappush(heap, (new_cpu, new_key))
            return
        cur = mapping[new_key]
        
        cur_max = max(cur[1:])
        new_max = max(new[1:])


        # Update if either CPU lowered or bandwidth improved
        if new_max < cur_max or (new_max == cur_max and sum(new[1:]) < sum(cur[1:])):
            mapping[new_key] = new
            # Requeue improved entries so they can be expanded further
            if heap is not None:
                heapq.heappush(heap, (new_cpu, new_key))

    def _dp_expand_key(self, key, times, mapping, level_here, max_cpu_time, heap=None):
        """Attempt DBL expansion for one key.

        For each legal position j in the chain where adjacent operands are at
        this bandwidth level, construct a new key with the relevant dimension
        doubled and compute the timing for the expanded problem.

        CPU model for DBL (chain splitting):
        - Treat DBL as executing two predecessor subproblems (halves) whose
          total compute equals 2x the predecessor CPU time, independent of the
          proportional change in naive matmul FLOPs. This matches the execution
          scheme where the chain is split and each half is computed separately
          (accumulating if interior), which yields cur_cpu = 2*prev_cpu.

        Bandwidth model:
        - Double all predecessor bandwidth times for this link.
        - Additionally, for interior splits (0 < j < n) where the output is at
          this link level, account for one extra traversal of the full output
          words.

        Only insert entries that do not exceed `max_cpu_time`."""
        ops, outp = self._dp_split_key(key)
        if len(ops) < 2:
            return
        old_ops = self._dp_ops_count(ops)
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

            new_key = self._dp_join_key(new_ops_list, new_outp)
            # Base bandwidth time for this link computed from the new key.
            new_bws = list(map(lambda x: x*2, times[2:]))

            if 0 < j < n:
                (odims, olvl) = new_outp
                for level in range(olvl):
                    if level <= level_here:
                        new_bws[level] += self._shape_elems(odims)
            self._dp_update_mapping(mapping, new_key, new_cpu, new_bws, j, heap)

    def _dp_expand(self, mapping, level_here, max_cpu_time):
        """Run the bandwidth-level DP expansion over all current entries.

        Uses a min-heap (priority queue) ordered by CPU time. While the smallest
        CPU-time entry is within half the allowed budget (so that doubling stays
        within `max_cpu_time`), attempt DBL expansions. New or improved entries
        are pushed back to the heap via `_dp_update_mapping`.
        """
        heap = []
        for key, times in mapping.items():
            try:
                cpu = times[1]
            except Exception:
                continue
            heapq.heappush(heap, (cpu, key))

        # Process entries in ascending CPU order; stop when doubling would exceed budget
        while heap:
            cpu, key = heapq.heappop(heap)
            # Always consult the latest mapping value
            times = mapping.get(key)
            if not times:
                continue
            cur_cpu = times[1]
            # If the smallest CPU exceeds half the budget, doubling would overflow
            if cur_cpu > max_cpu_time / 2:
                break
            self._dp_expand_key(key, times, mapping, level_here, max_cpu_time, heap)
        return mapping
    
    def root_node(self):
        return self.cache.root_node()

    def load(self, tensor, allow_lower_level=False):
        if allow_lower_level and tensor.level <= self.cache.level:
            return self.cache.load(tensor, allow_lower_level)
        # Deep-copy into the child cache and account input
        t2 = copy.deepcopy(tensor)
        import simulator  # local import for shared uid counter
        t2.uid = simulator.lastid
        simulator.lastid += 1
        self.input += t2.size()
        self.cache.alloc(t2)
        t2.level -= 1
        self._update_time()
        return t2

    def store(self, tensor):
        self.output += tensor.size()
        self.cache.free(tensor)
        tensor.level += 1
        self._update_time()
        return tensor

    def run(self, op, *args):
        return self.cache.run(op, *args)

    def __repr__(self):
        return (
            f"<Bandwidth input: {self.input}, output: {self.output}, time: {self.time}, parent: {self.cache}>"
        )

    def dynamic_times(self, nmatmuls, max_cpu):
        """Augment compute dynamic_times with bandwidth-level variants and timing.

        First delegates to the downstream cache (or compute node) to obtain the
        base CPU time mapping. Then, for each entry, produces:
        - A base variant with zero bandwidth time for this link.
        - For each operand in the key that is at the previous level, a duplicated
        entry with that operand promoted one level and bandwidth time added for
        moving its words across this link.

        Keys follow the convention from BinOpx.dynamic_times: tuple alternating
        (shape, level_marker), e.g., ((a,b), 0, (b,d), 0, ...).
        Returns a dict: key_variant -> [cpu_time, bw_time_for_this_link].
        """
        # Delegate to the child cache to get compute-time map
        base = self.cache.dynamic_times(nmatmuls, max_cpu)

        def shape_elems(shape):
            n = 1
            for d in shape:
                n *= d
            return n

        prev_level = self.cache.level
        out = {}
        for key, v in base.items():
            base_key = tuple(key)
            out[base_key] = v
            # Collect even indices (shape positions) at prev_level
            candidate_idxs = []
            for i in range(0, len(key), 2):
                lvl = key[i + 1]
                if lvl == prev_level:
                    candidate_idxs.append(i)
            from itertools import combinations

            for r in range(1, len(candidate_idxs) + 1):
                for subset in combinations(candidate_idxs, r):
                    kl = list(key)
                    bw_words = 0
                    pair_idxs = []
                    for i in subset:
                        kl[i + 1] = prev_level + 1
                        bw_words += shape_elems(kl[i])
                        pair_idxs.append(i // 2)
                    bw_time = bw_words * self.input_clocks_per_word
                    last_op = ("LDST",) + tuple(pair_idxs)
                    out[tuple(kl)] = [last_op] + v[1:] + [bw_time]

        # Dynamic programming expansion at the bandwidth level.
        # Use a priority queue ordered by CPU time; while the smallest CPU time
        # key is within half the allowed budget, attempt to double the shared
        # dimension between adjacent matrices that both reside at this bandwidth
        # cache level. Update times if improved and push new keys if added.
        out = self._dp_expand(out, prev_level + 1, max_cpu)
        return out

