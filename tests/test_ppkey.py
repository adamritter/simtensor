import simulate
from dynamic import ppkey


def test_ppkey_prints_single_entry():
    results = simulate.muladd.dynamic_times(2, 4)
    key = next(iter(results))
    ppkey(results, key)

