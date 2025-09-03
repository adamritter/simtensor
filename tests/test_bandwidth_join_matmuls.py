import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bandwidth_dynamic import join_matmuls


class TestBandwidthJoinMatmuls(unittest.TestCase):
    def test_join_adds_counters_and_concatenates_keys(self):
        key1 = ((2, 3), 0, (3, 4), 0, (2, 4), 0)
        value1 = [("LDST", 0), 10, 5]
        key2 = ((2, 4), 0, (4, 5), 0, (2, 5), 0)
        value2 = [("DBL", 1), 7]

        joined_key, joined_value = join_matmuls(key1, value1, key2, value2)

        expect_key = ((2, 3), 0, (3, 4), 0, (4, 5), 0, (2, 5), 0)
        expect_value = [("JOIN", 2, 0), 17, 5]
        self.assertEqual(joined_key, expect_key)
        self.assertEqual(joined_value, expect_value)

    def test_join_mismatched_keys_raises(self):
        key1 = ((2, 3), 0, (3, 4), 0, (2, 4), 0)
        value1 = ["BinOpx", 1]
        key2 = ((2, 5), 0, (5, 6), 0, (2, 6), 0)
        value2 = ["BinOpx", 2]

        with self.assertRaises(AssertionError):
            join_matmuls(key1, value1, key2, value2)


if __name__ == "__main__":
    unittest.main()
