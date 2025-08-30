import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator import Cache, Bandwidth
from simulate import muladd


class TestCacheDynamicTimesCapacity(unittest.TestCase):
    def test_capacity_filters_sum_at_level(self):
        # L0 large so its own filtering doesn't remove entries
        L0 = Cache(1000, muladd)
        bw = Bandwidth(L0)
        # L1 small to exercise sum-of-level filtering
        L1 = Cache(5, bw)

        res = L1.dynamic_times(2, 8)

        # Base shapes to look for: a=2, b=2, c=1
        base = ((2, 2), 0, (2, 1), 0, (2, 1), 0)
        self.assertIn(base, res)  # Bandwidth adds [cpu, bw], Cache keeps it

        # Accept single promotion that fits: ab (4 elements) at level 1
        ab_L1 = ((2, 2), 1, (2, 1), 0, (2, 1), 0)
        self.assertIn(ab_L1, res)

        # Reject combined promotions that exceed L1.size: ab (4) + bc (2) = 6
        ab_bc_L1 = ((2, 2), 1, (2, 1), 1, (2, 1), 0)
        self.assertNotIn(ab_bc_L1, res)

        # Combinations that sum to <= 5 should be kept, e.g., bc (2) + out (2) = 4
        bc_out_L1 = ((2, 2), 0, (2, 1), 1, (2, 1), 1)
        self.assertIn(bc_out_L1, res)


if __name__ == "__main__":
    unittest.main()
