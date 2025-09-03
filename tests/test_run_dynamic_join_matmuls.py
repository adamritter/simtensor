import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator import Cache, Bandwidth, Tensor
from simulate import muladd
from dynamic import run_dynamic
import bandwidth_dynamic


class TestRunDynamicJoinMatmuls(unittest.TestCase):
    def test_run_dynamic_handles_join_entries(self):
        flag = bandwidth_dynamic.ENABLE_DP_JOIN_MATMULS
        bandwidth_dynamic.ENABLE_DP_JOIN_MATMULS = True
        try:
            bw = Bandwidth(Cache(12, muladd))
            results = bw.dynamic_times(3, 20)
            join_key = None
            for k, v in results.items():
                if isinstance(v, list) and v and isinstance(v[0], tuple) and v[0][0] == "JOIN":
                    join_key = k
                    break
            self.assertIsNotNone(join_key)

            pairs = [(join_key[i], join_key[i + 1]) for i in range(0, len(join_key), 2)]
            ops = pairs[:-1]
            out_dims, out_level = pairs[-1]

            tensors = [Tensor.zeros(shp[0], shp[1], level=lvl) for shp, lvl in ops]
            tensors = [bw.alloc(t, allow_lower_level=True) for t in tensors]

            out = run_dynamic(results, bw, *tensors, out_level=out_level, reset_counter=True)
            self.assertEqual(out.sz, [out_dims[0], out_dims[1]])
            bw.free(out, allow_lower_level=True)
            for t in tensors:
                bw.free(t, allow_lower_level=True)
        finally:
            bandwidth_dynamic.ENABLE_DP_JOIN_MATMULS = flag


if __name__ == "__main__":
    unittest.main()
