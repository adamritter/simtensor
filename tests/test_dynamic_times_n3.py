import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import simulate


class TestDynamicTimesThreeMatrices(unittest.TestCase):
    def test_muladd_dynamic_times_three_mats_limit4(self):
        # Now the API interprets first arg as number of matrices (base fixed to 2)
        res = simulate.muladd.dynamic_times(3, 4)

        # limit=4 => k=2 (since 2^2 <= 4 < 2^3). We enumerate exponents e0..e3 with sum<=2
        # Count of solutions to e0+e1+e2+e3<=2 with nonnegative ints is C(2+4,4)=15
        self.assertIsInstance(res, dict)
        self.assertEqual(len(res), 15)

        # Pick a few sequences and verify keys and CPU time (left-associated):
        # dims (1,2,1,1): ops = 1*2*1 + 1*1*1 = 2 + 1 = 3
        key1 = ((1, 2), 0, (2, 1), 0, (1, 1), 0, (1, 1), 0)
        self.assertIn(key1, res)
        self.assertEqual(res[key1], [3])

        # dims (2,2,1,1): ops = 2*2*1 + 2*1*1 = 4 + 2 = 6
        key2 = ((2, 2), 0, (2, 1), 0, (1, 1), 0, (2, 1), 0)
        self.assertIn(key2, res)
        self.assertEqual(res[key2], [6])

        # dims (1,1,2,2): ops = 1*1*2 + 1*2*2 = 2 + 4 = 6
        key3 = ((1, 1), 0, (1, 2), 0, (2, 2), 0, (1, 2), 0)
        self.assertIn(key3, res)
        self.assertEqual(res[key3], [6])


if __name__ == "__main__":
    unittest.main()
