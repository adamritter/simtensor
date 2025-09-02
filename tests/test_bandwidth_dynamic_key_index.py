import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator import Cache, Bandwidth
from simulate import muladd
from bandwidth_dynamic import _dp_first_input_output_count


class TestBandwidthDynamicKeyIndex(unittest.TestCase):
    def test_index_contains_keys(self):
        bw = Bandwidth(Cache(12, muladd))
        res = bw.dynamic_times(2, 1000)
        idx = res.get("_key_index")
        self.assertIsNotNone(idx)
        # Each key should appear in the index under its shortened form
        for k in res:
            if k == "_key_index":
                continue
            short = _dp_first_input_output_count(k)
            self.assertIn(k, idx.get(short, set()))
        # Two variants share the same shortened key
        ab_L1 = ((2, 2), 1, (2, 1), 0, (2, 1), 0)
        ab_bc_L1 = ((2, 2), 1, (2, 1), 1, (2, 1), 0)
        short = _dp_first_input_output_count(ab_L1)
        self.assertIn(ab_L1, idx[short])
        self.assertIn(ab_bc_L1, idx[short])
        # Expanded key from DBL should also be indexed
        expanded = ((2, 4), 1, (4, 1), 1, (2, 1), 0)
        short_exp = _dp_first_input_output_count(expanded)
        self.assertIn(expanded, idx[short_exp])


if __name__ == "__main__":
    unittest.main()
