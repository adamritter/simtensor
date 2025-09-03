import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator import Cache, Bandwidth
from simulate import muladd
import bandwidth_dynamic


class TestBandwidthDynamicJoinShortKeys(unittest.TestCase):
    def test_join_entries_present_when_enabled(self):
        flag = bandwidth_dynamic.ENABLE_DP_JOIN_SHORT_KEYS
        bandwidth_dynamic.ENABLE_DP_JOIN_SHORT_KEYS = True
        try:
            bw = Bandwidth(Cache(12, muladd))
            res = bw.dynamic_times(3, 20)
        finally:
            bandwidth_dynamic.ENABLE_DP_JOIN_SHORT_KEYS = flag
        found = False
        for v in res.values():
            if isinstance(v, list) and v and isinstance(v[0], tuple) and v[0][0] == "JOIN":
                found = True
                break
        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
