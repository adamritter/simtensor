"""Core simulation primitives for simtensor.

Classes:
- Tensor: n-D view over a flat Python list with simple slicing/indexing.
- BinOpx: binary operation wrapper that tracks compute time.
- Cache: capacity accounting and residency tracking for tensor storage.
- Bandwidth: link between caches with input/output counters and transfer time.

Top-level helpers:
- utilization(cache): CPU utilization relative to bandwidth limits.
- reset_counters(cache): Reset bandwidth and compute timers.
"""

import copy

# Simulates the time to run the operations:
# - Loading / saving / discarding data in caches are explicit in the simulation
# - The simulator makes sure that the cache constraints are kept and calculates bandwidth per cache level and FLOPS timing
# - From these data it calculates utilization (just FLOPS time divided by max time from these constraints)
# - It also does the calculation itself to be able to check the result.
# - Needs to have vectorization support
# - Separate hardware from algorithm

class Tensor:
    """Lightweight n-D tensor view over a flat buffer.

    Attributes
    - data: underlying Python list (flat storage)
    - level: integer cache level where the data currently resides
    - sz: list[int] shape; empty list denotes a scalar
    - offset: starting index into `data`
    - skips: per-dimension strides (elements) for the view
    """
    def __init__(self, data, level, sz, offset=0, skips=None):
        self.data = data
        self.level = level
        self.sz = sz
        self.offset = offset
        if skips is None:
            skips = []
            m = 1
            for c in sz:
                skips.append(m)
                m*=c
        self.skips = skips
        if self.sz == []:
            self.value = self.data[self.offset]

    def size(self):
        """Return the total element count of the view."""
        r = 1
        for i in self.sz:
            r*=i
        return r

    def sum(self):
        """Return the sum of elements in the view (recursive)."""
        if self.sz == []:
            return self.data[self.offset]
        r=0
        for i in range(self.sz[0]):
            r+=self[i].sum()
        return r

    def __getitem__(self, key):
        """Return a Tensor view using basic int/slice indexing.

        Supports 1- or 2-D slicing via `t[i]`, `t[i:j]`, `t[:, k]`, etc.
        """
        key2=None
        if isinstance(key, tuple):
            key, key2 = key
        r = self
        if isinstance(key, slice):
            start=key.start or 0
            stop=key.stop
            if stop is None:
                stop = self.sz[0]
            if stop < 0:
                stop = stop + self.sz[0]
            offset = self.offset + self.skips[0]*start
            r = Tensor(self.data, self.level, [stop-start] + self.sz[1:], offset, self.skips)
        elif isinstance(key, int):
            offset = self.offset + self.skips[0]*key
            r = Tensor(self.data, self.level, self.sz[1:], offset, self.skips[1:])

        if isinstance(key2, slice):
            start=key2.start or 0
            stop=key2.stop
            if stop is None:
                stop = r.sz[1]
            if stop < 0:
                stop = stop + r.sz[1]
            offset = r.offset + r.skips[1]*start
            r = Tensor(r.data, r.level, [r.sz[0]]+[stop-start] + r.sz[2:], offset, r.skips)
        elif isinstance(key2, int):
            offset = r.offset + r.skips[1]*key2
            r = Tensor(r.data, r.level, [r.sz[0]]+r.sz[2:], offset, [r.skips[0]]+ r.skips[2:])
        return r

    def __setitem__(self, key, value):
        """Assign a scalar value via integer indexing into the view."""
        # Scalar assignment: accept any key (e.g., (), 0, slice(None))
        if self.sz == []:
            self.value = value
            self.data[self.offset] = value
            return
        # Normalize key to tuple for multidimensional indexing
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) == 0:
            raise IndexError("Empty index not supported")
        # Only support integer indexing assignment to a single element
        off = self.offset
        for i, idx in enumerate(key):
            if not isinstance(idx, int):
                raise NotImplementedError("Slice or advanced assignment not supported")
            if idx < 0:
                idx += self.sz[i]
            off += self.skips[i] * idx
        # Ensure fully indexed to a scalar element
        if len(key) != len(self.sz):
            raise NotImplementedError("Partial indexing assignment not supported")
        self.data[off] = value

    # No setv anymore; use item assignment
        
    def __repr__(self):
        if self.sz == []:
            return str(self.data[self.offset])
        r=["["]
        for i in range(self.sz[0]):
            if  i > 0:
                r.append(', ')
            r.append(self[i].__repr__())
        r.append("]")
        return "".join(r)

    @classmethod
    def zeros(cls, *shape, level=0):
        """Create a new zero-initialized Tensor with the given shape.

        Usage:
        - Tensor.zeros(2, 3) -> 2x3 zeros at level 0
        - Tensor.zeros([2, 3], level=1) -> 2x3 zeros at level 1
        - Tensor.zeros() -> scalar zero
        """
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        sz = list(shape)
        if len(sz) == 0:
            data = [0]
            return cls(data, level, [])
        count = 1
        for d in sz:
            count *= d
        data = [0] * count
        return cls(data, level, sz)

    def zero_(self):
        """In-place: set all elements in this Tensor view to zero."""
        if self.sz == []:
            self.data[self.offset] = 0
            self.value = 0
            return
        def rec(dim, off):
            if dim == len(self.sz):
                self.data[off] = 0
                return
            step = self.skips[dim]
            for i in range(self.sz[dim]):
                rec(dim + 1, off + i * step)
        rec(0, self.offset)
class BinOpx:

    """Binary operation wrapper with timing.

    Run enforces operand cache levels, calls the provided function `f(a,b,c)`,
    and accumulates `time` by `t` per invocation.
    """
    def __init__(self, as0, alevel, bs, blevel, cs, clevel, f, t=0):
        self.as0 = as0
        self.alevel = alevel
        self.bs = bs
        self.blevel = blevel
        self.cs = cs
        self.clevel = clevel
        self.f = f
        self.t = t
        self.time = 0

    def run(self, a, b, c):
      """Execute the op and accumulate time; returns `c.data`."""
      assert a.level == self.alevel
      assert b.level == self.blevel
      assert c.level == self.clevel
      self.f(a, b, c)
      self.time += self.t
      return c.data

    def __repr__(self):
        return f"<BinOpx time: {self.time}>"




    def dynamic_times(self, params):
        """Return compute time for matmul chains on this BinOpx.

        Each item p in `params` is a sequence of matrix shapes forming a valid chain:
        p = ((d0,d1), (d1,d2), ..., (d{n-1}, d{n})), with n >= 2.

        We assume left-associative evaluation for time estimation:
        total_ops = sum_{i=0..n-2} d0 * d{i+1} * d{i+2}
        time = total_ops * self.t

        The key mirrors prior convention by interleaving a trailing 0 marker after each shape:
        key = ((d0,d1), 0, (d1,d2), 0, ..., (d{n-1}, d{n}), 0)
        """
        r = {}
        for p in params:
            if len(p) < 2:
                continue
            d0 = p[0][0]
            # Sum left-associated matmul costs along the chain
            ops = 0
            for i in range(len(p) - 1):
                k = p[i][1]      # inner/shared dim for this step
                m = p[i + 1][1]  # columns of the right matrix at this step
                ops += d0 * k * m
            time = ops * self.t
            # Build the key with trailing zeros for compatibility, and append
            # (output_dims, cache_level) where output_dims=(d0, d_last)
            key_core = tuple([item for shp in p for item in (shp, 0)])
            out_dims = (p[0][0], p[-1][1])
            key = key_core + (out_dims, self.clevel)
            r[key] = [time]
        return r


class Cache:
    """A cache level with capacity and residency accounting.

    Parent can be a `Bandwidth` (for multi-level hierarchies) or a `BinOpx`
    (the compute endpoint). The `datas` set tracks resident storage ids.
    """
    def __init__(self, size, parent):
        self.size = size
        self.used = 0
        self.parent = parent
        self.datas = set()
        self.parentcache=None
        self.level = 0
        if isinstance(self.parent, Bandwidth):
            self.parentcache = self.parent.cache
            self.level = self.parentcache.level + 1

    def alloc(self, tensor):
        """Mark tensor storage as resident; raises if capacity exceeded."""
        if id(tensor.data) in self.datas:
            return
        self.used += tensor.size()
        if self.used > self.size:
            raise Exception("Not enough memory")
        self.datas.add(id(tensor.data))
        return tensor

    def free(self, tensor):
        """Release tensor storage from this cache."""
        self.datas.remove(id(tensor.data))
        self.used -= tensor.size()

    def run(self, op, *args):
        """Run an op with tensors that must be resident at this level."""
        for arg in args:
            if type(arg) == Tensor and id(arg.data) not in self.datas:
                raise Exception("Data not found in cache: ", arg.data, " for tensor ", arg)
        return op(self.parent, *args)

    def load(self, m):
        """Move a view one level down; updates bandwidth input and residency."""
        if id(m.data) not in self.datas:
            raise Exception("Data not found in cache during load: ", m.data, " for tensor ", m)
        return self.parent.load(m)

    def store(self, m):
        """Move a view one level up; updates bandwidth output and parent alloc."""
        m2 = self.parent.store(m)
        if m2.level != self.level:
            raise Exception("Tensor levels don't match")
        self.alloc(m2)
    
    def store_to(self, src, dst):
        """
        Store data from a one-level-lower tensor `src` into an existing
        destination view `dst` that already resides in this cache.

        Semantics mirror `store` for accounting:
        - Increments parent bandwidth output by the number of elements copied.
        - Frees the source tensor from the child cache (capacity decreases there).
        - Does NOT allocate new memory in this cache (writes into existing `dst`).

        Requirements:
        - `src.level == self.level - 1`
        - `dst.level == self.level`
        - `id(src.data)` is resident in the child cache
        - `id(dst.data)` is resident in this cache
        - `src.size() == dst.size()` (same number of elements)
        """
        # Validate cache relationship
        if not isinstance(self.parent, Bandwidth) or self.parentcache is None:
            raise Exception("store_to requires a parent Bandwidth and child Cache")

        # Validate levels
        if src.level != self.level - 1:
            raise Exception("Source tensor must be exactly one level below destination cache")
        if dst.level != self.level:
            raise Exception("Destination tensor level must match this cache level")

        # Validate residency
        if id(src.data) not in self.parentcache.datas:
            raise Exception("Source data not found in child cache during store_to")
        if id(dst.data) not in self.datas:
            raise Exception("Destination data not found in this cache during store_to")

        # Accounting like `store`: this bumps bandwidth.output and frees from child.
        self.parent.store(src)  # src.level becomes self.level; child capacity updated

        # Size compatibility
        if src.size() != dst.size():
            raise Exception("Source and destination sizes must match for store_to")

        # Copy element-wise from src view into dst view using offsets and strides.
        # Works for arbitrary dimensionality described by sz and skips.
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

    def alloc_diag(self, n):
        data = []
        for i in range(n):
            for j in range(n):
                data.append(i==j)
        return self.alloc(Tensor(data, self.level, [n, n]))

    def calloc(self, n, m):
        data = []
        for i in range(n):
            for j in range(m):
                data.append(0)
        return self.alloc(Tensor(data, self.level, [n, m]))

    def __repr__(self):
        return f"<Cache level {self.level}, used: {self.used}, size: {self.size}, parent: {self.parent}>"



    def dynamic_times(self, params):
        """Filter params by capacity and delegate to compute node dynamic_times.

        Filters out shapes where a*b*d >= self.size (using ((a,b),(b,d)) format),
        then calls the compute node (BinOpx) dynamic_times.
        """
        # capacity filter
        params2 = [((a, b), (c, d)) for ((a, b), (c, d)) in params if a * b * d < self.size]
        # find compute node: either parent is BinOpx or Bandwidth->Cache->BinOpx
        return self.parent.dynamic_times(params2)
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

    def load(self, tensor):
        tensor = copy.deepcopy(tensor)
        self.input += tensor.size()
        self.cache.alloc(tensor)
        tensor.level -= 1
        self._update_time()
        return tensor

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


    def dynamic_times(self, params):
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
        base = {}
        if hasattr(self.cache, 'dynamic_times'):
            base = self.cache.dynamic_times(params)

        def shape_elems(shape):
            n = 1
            for d in shape:
                n *= d
            return n

        prev_level = getattr(self.cache, 'level', 0)
        out = {}
        for key, v in base.items():
            base_key = tuple(key)
            out[base_key] = v + [0]
            if len(key) % 2 != 0:
                # Unexpected; skip duplication
                continue
            # Collect even indices (shape positions) at prev_level
            candidate_idxs = []
            for i in range(0, len(key), 2):
                lvl = key[i + 1]
                if isinstance(lvl, int) and lvl == prev_level:
                    candidate_idxs.append(i)
            from itertools import combinations

            for r in range(1, len(candidate_idxs) + 1):
                for subset in combinations(candidate_idxs, r):
                    kl = list(key)
                    bw_words = 0
                    for i in subset:
                        kl[i + 1] = prev_level + 1
                        bw_words += shape_elems(kl[i])
                    bw_time = bw_words * self.input_clocks_per_word
                    out[tuple(kl)] = v + [bw_time]
        return out


def utilization(cache):
    """
    Compute utilization for a cache hierarchy rooted at `cache`.

    Utilization = cpu_time / max(cpu_time, max_bandwidth_time)

    Walks down through parent Bandwidth links to the compute node (BinOpx),
    collecting each Bandwidth's aggregate `time` and the BinOpx `time`.
    """
    # Accumulate the maximum bandwidth time along the chain
    max_bw_time = 0
    c = cache
    # Be defensive in case `cache` is not a Cache or has no parent
    while hasattr(c, 'parent') and isinstance(getattr(c, 'parent', None), Bandwidth):
        bw = c.parent
        if bw.time > max_bw_time:
            max_bw_time = bw.time
        c = bw.cache

    # The compute node is expected to be the parent of the lowest cache
    cpu_time = getattr(c.parent, 'time', 0)
    denom = max(cpu_time, max_bw_time)
    if denom == 0:
        return 0.0
    return cpu_time / denom


def reset_counters(cache):
    """
    Reset timing/throughput counters along the hierarchy rooted at `cache`.

    - Sets each Bandwidth's input/output/time to 0
    - Sets the compute node's (BinOpx) time to 0
    """
    c = cache
    # Walk down bandwidth links if present; tolerate inputs without `.parent`
    while hasattr(c, 'parent') and isinstance(getattr(c, 'parent', None), Bandwidth):
        bw = c.parent
        bw.input = 0
        bw.output = 0
        bw.time = 0
        c = bw.cache
    # Reset compute node time if present
    if hasattr(c, 'parent') and hasattr(c.parent, 'time'):
        c.parent.time = 0
