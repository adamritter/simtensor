import os
import sys
import unittest

# Ensure project root is on sys.path for direct test invocation
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import simulator


def muladdsimple(a, b, c):
    c.setv(a.value * b.value + c.value)


class TestTensorBasics(unittest.TestCase):
    def test_scalar_set_and_sum(self):
        t = simulator.Tensor([2], 0, [])
        self.assertEqual(t.size(), 1)
        self.assertEqual(t.sum(), 2)
        t.setv(5)
        self.assertEqual(t.sum(), 5)

    def test_matrix_size_sum_and_indexing(self):
        # Column-major mapping with current strides implementation
        t = simulator.Tensor([1, 2, 3, 4], 0, [2, 2])
        self.assertEqual(t.size(), 4)
        self.assertEqual(t.sum(), 10)
        # Int indexing (nested) is relied upon by the algorithms
        self.assertEqual(t[0][0].value, 1)  # (row 0, col 0)
        self.assertEqual(t[0][1].value, 3)  # (row 0, col 1)
        self.assertEqual(t[1][0].value, 2)  # (row 1, col 0)
        self.assertEqual(t[1][1].value, 4)  # (row 1, col 1)


class TestBinOpx(unittest.TestCase):
    def test_muladd_runs_and_accumulates_time(self):
        op = simulator.BinOpx([], 0, [], 0, [], 0, muladdsimple, t=1)
        a = simulator.Tensor([2], 0, [])
        b = simulator.Tensor([3], 0, [])
        c = simulator.Tensor([1], 0, [])
        out_data = op.run(a, b, c)
        # BinOpx.run returns the underlying data buffer of c
        self.assertIsInstance(out_data, list)
        self.assertEqual(c.value, 2 * 3 + 1)
        self.assertEqual(op.time, 1)
        # Run again to confirm time accumulates
        op.run(a, b, c)
        self.assertEqual(op.time, 2)


class TestCacheAndBandwidth(unittest.TestCase):
    def setUp(self):
        # Two-level hierarchy: L0 cache attached to a BinOpx via Bandwidth, and L1 above it
        self.op = simulator.BinOpx([], 0, [], 0, [], 0, muladdsimple, t=1)
        self.L0 = simulator.Cache(24, self.op)
        self.bw = simulator.Bandwidth(self.L0)
        self.L1 = simulator.Cache(1000, self.bw)

    def test_alloc_and_capacity(self):
        a = self.L1.alloc_diag(4)  # 4x4 identity => 16 elements
        self.assertEqual(self.L1.used, 16)
        b = self.L1.calloc(2, 3)
        self.assertEqual(self.L1.used, 16 + 6)

        # Capacity overflow on too-small cache
        small = simulator.Cache(2, self.bw)
        with self.assertRaises(Exception):
            small.calloc(2, 2)  # needs 4 elements, only 2 available

    def test_load_and_store_accounting(self):
        a = self.L1.calloc(4, 4)
        # Load a column view into L0
        view = a[:, 0:1]
        self.assertEqual(view.size(), 4)
        self.assertEqual(self.bw.input, 0)
        self.assertEqual(self.L0.used, 0)

        vL0 = self.L1.load(view)
        self.assertEqual(vL0.level, self.L0.level)  # moved one level down
        self.assertEqual(self.bw.input, 4)
        self.assertEqual(self.L0.used, 4)

        # Store it back up: frees L0 and allocs in L1 (by current semantics)
        self.assertEqual(self.bw.output, 0)
        self.L1.store(vL0)
        self.assertEqual(self.bw.output, 4)
        self.assertEqual(self.L0.used, 0)
        # L1 usage increases by 4 for the stored copy
        self.assertEqual(self.L1.used, a.size() + 4)

    def test_run_requires_resident_data(self):
        # op should be called via L0 cache, but only if tensors are resident in L0
        a = self.L1.alloc_diag(2)
        b = self.L1.alloc_diag(2)
        c = self.L1.calloc(2, 2)

        # Load a, b, c to L0
        a0 = self.L1.load(a)
        b0 = self.L1.load(b)
        c0 = self.L1.load(c)

        # Running at L0 with resident tensors should succeed
        def noop_muladd(op, aa, bb, cc):
            pass

        def matmulsimple_local(muladdop, aa, bb, cc):
            # Just touch a few elements via nested indexing
            _ = aa[0][0].value
            _ = bb[0][0].value
            cc[0][0].setv(0)

        # ok path
        self.L0.run(matmulsimple_local, a0, b0, c0)

        # Free one buffer from L0 and then expect an error if passed to L0.run
        self.L0.free(a0)
        with self.assertRaises(Exception):
            self.L0.run(matmulsimple_local, a0, b0, c0)

    def test_store_to_copies_into_existing_parent_view(self):
        # Prepare parent (L1) destination buffer and a child (L0) source view
        # Destination: a zero matrix at L1; take a column view as target
        dest_full = self.L1.calloc(4, 4)
        dest_col = dest_full[:, 1:2]  # shape (4,1)

        # Source: identity at L1, load a column view to L0
        src_full = self.L1.alloc_diag(4)
        src_col_L1 = src_full[:, 2:3]
        # Load down to L0
        src_col_L0 = self.L1.load(src_col_L1)

        # Check initial accounting
        used_L0_before = self.L0.used
        used_L1_before = self.L1.used
        out_before = self.bw.output

        # Perform store_to from L0 -> L1 into existing dest view
        self.L1.store_to(src_col_L0, dest_col)

        # Bandwidth output increased by number of elements moved
        self.assertEqual(self.bw.output, out_before + src_col_L0.size())

        # Child cache freed the source
        self.assertEqual(self.L0.used, used_L0_before - src_col_L0.size())

        # Parent cache did not allocate new memory; usage unchanged
        self.assertEqual(self.L1.used, used_L1_before)

        # Data copied: dest column should match an identity column at index 2
        # Identity at column 2 has a single 1 at row 2
        # Verify via direct data reads respecting strides
        ones = [dest_full[i][1].value for i in range(4)]
        self.assertEqual(ones, [0, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
