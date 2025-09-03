import os
import sys
import unittest
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bandwidth_dynamic import _dp_join_matmuls, _dp_short_key


class TestBandwidthJoinMatmuls(unittest.TestCase):
    def setUp(self):
        self.key1 = ((2, 3), 0, (3, 4), 0, (2, 4), 0)
        self.value1 = [("LDST", 0), 10, 5]
        self.key2 = ((2, 4), 0, (4, 5), 0, (2, 5), 0)
        self.value2 = [("LDST", 1), 7]
        self.joined_key = (
            (2, 3),
            0,
            (3, 4),
            0,
            (4, 5),
            0,
            (2, 5),
            0,
        )
        self.expect_value = [("JOIN", 2), 17, 5]

    def test_join_matmuls_adds_entry_within_limits(self):
        mapping = {self.key1: self.value1, self.key2: self.value2}
        keyinfo = defaultdict(set)
        keyinfo[_dp_short_key(self.key1)].add(self.key1)
        keyinfo[_dp_short_key(self.key2)].add(self.key2)
        heap = []

        _dp_join_matmuls(self.key1, keyinfo, mapping, heap, 4, 20)

        self.assertEqual(mapping[self.joined_key], self.expect_value)
        self.assertIn((17, self.joined_key), heap)

    def test_join_matmuls_respects_cpu_limit(self):
        mapping = {self.key1: self.value1, self.key2: self.value2}
        keyinfo = defaultdict(set)
        keyinfo[_dp_short_key(self.key1)].add(self.key1)
        keyinfo[_dp_short_key(self.key2)].add(self.key2)
        heap = []

        _dp_join_matmuls(self.key1, keyinfo, mapping, heap, 4, 16)

        self.assertNotIn(self.joined_key, mapping)
        self.assertEqual(heap, [])


if __name__ == "__main__":
    unittest.main()
