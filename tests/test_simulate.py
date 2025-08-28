import os
import sys
import unittest

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import simulator
import simulate


def matrix_to_lists(t):
    return [[t[i][j].value for j in range(t.sz[1])] for i in range(t.sz[0])]


class TestSimulateMatmul(unittest.TestCase):
    def make_hierarchy(self, l0_size=256, l1_size=4096):
        op = simulator.BinOpx([], 0, [], 0, [], 0, simulate.muladdsimple, t=1)
        L0 = simulator.Cache(l0_size, op)
        bw = simulator.Bandwidth(L0)
        L1 = simulator.Cache(l1_size, bw)
        return L1, L0, bw, op

    def test_matmul_identity_small(self):
        L1, L0, bw, op = self.make_hierarchy()
        # A is identity, so A @ B = B
        A = L1.alloc_diag(4)
        B = L1.calloc(4, 4)
        # Fill B with a simple pattern
        for i in range(4):
            for j in range(4):
                B[i][j] = i * 10 + j
        C = L1.calloc(4, 4)

        simulate.matmul(L1, A, B, C)
        self.assertEqual(matrix_to_lists(C), matrix_to_lists(B))

    def test_matmul_edge_tiles_non_multiple_of_block(self):
        # This catches edge handling when dimensions are not multiples of 4
        L1, L0, bw, op = self.make_hierarchy(l0_size=2048, l1_size=20000)
        n = 6  # not a multiple of 4
        A = L1.alloc_diag(n)
        B = L1.calloc(n, n)
        for i in range(n):
            for j in range(n):
                B[i][j] = (i + 1) * 100 + j
        C = L1.calloc(n, n)

        # Should not raise and should equal B when A is identity
        simulate.matmul(L1, A, B, C)
        self.assertEqual(matrix_to_lists(C), matrix_to_lists(B))


if __name__ == "__main__":
    unittest.main()

