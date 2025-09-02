import unittest

from simulator import BinOpx, Tensor, Cache, Bandwidth, reset_counters, get_counters
from simulate import muladdsimple
from dynamic import run_dynamic_best


class TestRunDynamicBest(unittest.TestCase):
    def _new_node(self):
        return BinOpx([], 0, [], 0, [], 0, muladdsimple, 1)

    def _best_level_from_results(self, results, tensors, levels):
        # Reproduce the key encoding used by run_dynamic_best
        dims = [tensors[0].sz[0], tensors[0].sz[1]]
        for t in tensors[1:]:
            dims.append(t.sz[1])

        def make_key(out_level):
            flat = []
            for i in range(len(dims) - 1):
                flat.extend([(dims[i], dims[i + 1]), getattr(tensors[i], 'level', 0)])
            flat.extend([(dims[0], dims[-1]), out_level])
            return tuple(flat)

        best_level = None
        best_time = None
        for lvl in sorted(levels):
            entry = results.get(make_key(lvl))
            if entry is None:
                continue
            numeric_tail = [e for e in entry[1:] if isinstance(e, (int, float))]
            if not numeric_tail:
                continue
            runtime = max(numeric_tail)
            if best_time is None or runtime < best_time:
                best_time = runtime
                best_level = lvl
        return best_level

    def test_best_picks_fastest_out_level_two_mats(self):
        # Hierarchy: BinOpx <- L0 (size 24) <- bw <- L1
        node = self._new_node()
        L0 = Cache(24, node)
        bw = Bandwidth(L0)
        L1 = Cache(1000, bw)

        # Operands at L1
        A = L1.calloc(2, 2)
        B = L1.calloc(2, 2)

        # Build the same dynamic table that run_dynamic_best will consult
        # Limit must cover the product of chain dims (2*2*2=8)
        results = L1.dynamic_times(2, 8)

        # Candidate output levels reachable from L1: {0, 1}
        levels = {0, 1}
        expected_level = self._best_level_from_results(results, [A, B], levels)
        self.assertIn(expected_level, levels)

        reset_counters(L1)
        out = run_dynamic_best(L1, A, B, reset_counter=True)
        self.assertEqual(out.sz, [2, 2])

        # Counters should match the selected dynamic entry for the expected level
        # Reconstruct the chosen key and compare numeric tails to measured counters
        key = ((2, 2), A.level, (2, 2), B.level, (2, 2), expected_level)
        self.assertIn(key, results)
        numeric_tail = [e for e in results[key][1:] if isinstance(e, (int, float))]
        self.assertEqual(get_counters(L1), numeric_tail)


if __name__ == '__main__':
    unittest.main()
