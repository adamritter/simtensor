import unittest
from bandwidth_dynamic import _dp_short_key


class TestShortKey(unittest.TestCase):
    def test_basic_key(self):
        key = ((2, 3), 0, (3, 4), 0, (2, 4), 0)
        first_in, out, n = _dp_short_key(key)
        self.assertEqual(first_in, ((2, 3), 0))
        self.assertEqual(out, ((2, 4), 0))
        self.assertEqual(n, 2)

    def test_requires_two_inputs(self):
        key = ((4, 4), 0, (4, 4), 0)
        with self.assertRaises(ValueError):
            _dp_short_key(key)
