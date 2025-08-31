import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator import Cache, Bandwidth
from simulate import muladd


class TestBandwidthInternalDoubleExtraBW(unittest.TestCase):
    def test_internal_double_adds_output_and_last_matrix(self):
        bw = Bandwidth(Cache(12, muladd))
        res = bw.dynamic_times(2, 1000)

        key_out0 = ((1, 2), 1, (2, 4), 1, (1, 4), 0)
        key_out1 = ((1, 2), 1, (2, 4), 1, (1, 4), 1)

        self.assertIn(key_out0, res)
        self.assertIn(key_out1, res)

        v0 = res[key_out0]
        v1 = res[key_out1]

        self.assertIsInstance(v0, list)
        self.assertEqual(v0[0], ("DBL", 1))
        self.assertEqual(v0[2], 10)

        self.assertIsInstance(v1, list)
        self.assertEqual(v1[0], ("DBL", 2))
        self.assertEqual(v1[2], 16)


if __name__ == "__main__":
    unittest.main()
