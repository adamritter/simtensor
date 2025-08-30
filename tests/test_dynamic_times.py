import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import simulate


class TestDynamicTimesEnumeration(unittest.TestCase):
    def test_muladd_dynamic_times_power_enumeration(self):
        res = simulate.muladd.dynamic_times(2, 8)

        self.assertIsInstance(res, dict)
        self.assertEqual(len(res), 20)

        for k, v in res.items():
            self.assertIsInstance(k, tuple)
            self.assertEqual(len(k) % 2, 0)
            (ab, _lvl1, bc, _lvl2, ac, _lvl3) = k
            a, b1 = ab
            b2, c = bc
            a2, c2 = ac
            self.assertEqual(b1, b2)
            self.assertEqual(a, a2)
            self.assertEqual(c, c2)
            self.assertEqual(v, [a * b1 * c])

        cases = [
            ((1, 1), 0, (1, 1), 0, (1, 1), 0, 1),
            ((2, 1), 0, (1, 1), 0, (2, 1), 0, 2),
            ((1, 2), 0, (2, 1), 0, (1, 1), 0, 2),
            ((2, 2), 0, (2, 1), 0, (2, 1), 0, 4),
            ((4, 2), 0, (2, 1), 0, (4, 1), 0, 8),
            ((2, 2), 0, (2, 2), 0, (2, 2), 0, 8),
        ]
        for ab, l1, bc, l2, ac, l3, prod in cases:
            key = (ab, l1, bc, l2, ac, l3)
            self.assertIn(key, res)
            self.assertEqual(res[key], [prod])


if __name__ == "__main__":
    unittest.main()
