import os
import sys
import unittest

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import simulator
import simulate
import matmul3


def matrix_to_lists(t):
    return [[t[i][j].value for j in range(t.sz[1])] for i in range(t.sz[0])]


class TestMatmul3(unittest.TestCase):
    def make_hierarchy(self, l0_size=256, l1_size=4096):
        op = simulator.BinOpx([], 0, [], 0, [], 0, simulate.muladdsimple, t=1)
        L0 = simulator.Cache(l0_size, op)
        bw = simulator.Bandwidth(L0)
        L1 = simulator.Cache(l1_size, bw)
        return L1

    def test_matmul_store_matches_matmul(self):
        L1_a = self.make_hierarchy()
        A1 = L1_a.alloc_diag(4)
        B1 = L1_a.calloc(4, 4)
        for i in range(4):
            for j in range(4):
                B1[i][j] = i * 10 + j
        C1 = L1_a.calloc(4, 4)
        simulate.matmul(L1_a, A1, B1, C1)

        L1_b = self.make_hierarchy()
        A2 = L1_b.alloc_diag(4)
        B2 = L1_b.calloc(4, 4)
        for i in range(4):
            for j in range(4):
                B2[i][j] = i * 10 + j
        C2 = L1_b.calloc(4, 4)
        matmul3.matmul_store(L1_b, A2, B2, C2, tile=4)

        self.assertEqual(matrix_to_lists(C1), matrix_to_lists(C2))

    def test_fused_matches_two(self):
        N, M, P, R = 4, 1, 4, 1

        L1_two = self.make_hierarchy()
        A = L1_two.calloc(N, M)
        B = L1_two.calloc(M, P)
        C = L1_two.calloc(P, R)
        for i in range(N):
            A[i][0] = i + 1
        for j in range(P):
            B[0][j] = j + 2
        for k in range(P):
            C[k][0] = k + 3
        OUT_two = L1_two.calloc(N, R)
        matmul3.matmul3_two(L1_two, A, B, C, OUT_two)

        L1_fused = self.make_hierarchy()
        A2 = L1_fused.calloc(N, M)
        B2 = L1_fused.calloc(M, P)
        C2 = L1_fused.calloc(P, R)
        for i in range(N):
            A2[i][0] = i + 1
        for j in range(P):
            B2[0][j] = j + 2
        for k in range(P):
            C2[k][0] = k + 3
        OUT_fused = L1_fused.calloc(N, R)
        matmul3.matmul3_fused(L1_fused, A2, B2, C2, OUT_fused, tile=4)

        self.assertEqual(matrix_to_lists(OUT_two), matrix_to_lists(OUT_fused))


if __name__ == "__main__":
    unittest.main()
