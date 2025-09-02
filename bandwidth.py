"""Bandwidth link between caches with simple timing model."""

import copy
from bandwidth_dynamic import dynamic_times_impl


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
        """Augment compute dynamic_times with bandwidth-level variants and timing."""
        return dynamic_times_impl(self, nmatmuls, max_cpu)

