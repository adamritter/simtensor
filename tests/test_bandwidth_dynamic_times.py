import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator import Cache, Bandwidth
from simulate import muladd


class TestBandwidthDynamicTimes(unittest.TestCase):
    def test_bw_dynamic_times_enumeration_and_variants(self):
        bw = Bandwidth(Cache(12, muladd))
        res = bw.dynamic_times(2, 1000)

        # Base key (all level 0) for a=2, b=2, c=1
        base_key = ((2, 2), 0, (2, 1), 0, (2, 1), 0)
        self.assertIn(base_key, res)
        self.assertEqual(res[base_key], ["Bandwidth", 4, 0])

        # Duplicate with only ab at level 1 -> bandwidth time = a*b = 4
        ab_L1 = ((2, 2), 1, (2, 1), 0, (2, 1), 0)
        self.assertIn(ab_L1, res)
        self.assertEqual(res[ab_L1], ["Bandwidth", 4, 4])

        # Duplicate with ab and bc at level 1 -> bw time = a*b + b*c = 4 + 2 = 6
        ab_bc_L1 = ((2, 2), 1, (2, 1), 1, (2, 1), 0)
        self.assertIn(ab_bc_L1, res)
        self.assertEqual(res[ab_bc_L1], ["Bandwidth", 4, 6])

        # DP expansion should create doubled shared-dimension when both are at level 1
        expanded = ((2, 4), 1, (4, 1), 1, (2, 1), 0)
        self.assertIn(expanded, res)
        self.assertEqual(res[expanded], ["Bandwidth", 8, 12])  # cpu doubles; bw sums elements at level 1


if __name__ == "__main__":
    unittest.main()
