import os
import sys

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import simulate
from simulator import Tensor, Cache, Bandwidth
from dynamic import run_dynamic, pp, previous_key, extras


def verify_result(key, results):
    # Key is alternating (shape, level); last pair is output dims
    pairs = [(key[i], key[i + 1]) for i in range(0, len(key), 2)]
    operand_pairs = pairs[:-1]
    out_dims, out_level = pairs[-1]

    # Build input tensors at level 0 for each operand shape
    tensors = [Tensor.zeros(shp[0], shp[1], level=lvl) for shp, lvl in operand_pairs]

    print(f"Running dynamic for {key}, value {results.get(key)}, extras {extras(key, results)}")
    print(f"    previous_key: {previous_key(key, results.get(key)[0])} = {results.get(previous_key(key, results.get(key)[0]))}")
    out = run_dynamic(results, simulate.muladd, *tensors, out_level=out_level)
    # Output shape should match trailing dims in the key
    assert out.sz == [out_dims[0], out_dims[1]]

    # Additionally verify that for all non-BinOpx rows, the previous_key exists.
    v = results.get(key)
    if isinstance(v, list) and v:
        head = v[0]
        # Skip pure compute rows
        if not (isinstance(head, str) and head == 'BinOpx'):
            op = head if isinstance(head, tuple) else None
            pk = previous_key(key, op)
            assert pk in results, f"previous_key missing for {key} with op {head} -> {pk}"


def verify_reults(results):
    for key in results.keys():
        verify_result(key, results)


def test_run_dynamic_for_all_muladd_dynamic_times_three():
    # Enumerate all 2- and 3-matrix chains up to limit 8 and run them
    results = simulate.muladd.dynamic_times(3, 100)

    verify_reults(results)


def test_run_dynamic_for_all_muladd_dynamic_times_three_cache():
    # Enumerate all 2- and 3-matrix chains up to limit 8 and run them
    cache = Cache(12, simulate.muladd)
    results = cache.dynamic_times(3, 100)

    verify_reults(results)



def test_run_dynamic_for_all_muladd_dynamic_times_three_bandwidth():
    # Enumerate all 2- and 3-matrix chains up to limit 8 and run them
    bw = Bandwidth(Cache(12, simulate.muladd))
    results = bw.dynamic_times(3, 1000)
    pp(results)

    verify_reults(results)


# def test_run_dynamic_for_all_muladd_dynamic_times_three_bandwidth():
#     # Enumerate all 2- and 3-matrix chains up to limit 8 and run them
#     bw = Bandwidth(Cache(12, simulate.muladd))
#     results = bw.dynamic_times(3, 1000)
#     pp(results)
#     assert len(results) == 653

#     verify_reults(results)


# def test_run_dynamic_for_all_muladd_dynamic_times_three_bandwidth2():
#     # Enumerate all 2- and 3-matrix chains up to limit 8 and run them
#     bw2 = Bandwidth(Cache(20, Bandwidth(Cache(12, simulate.muladd))))
#     results = bw2.dynamic_times(3, 20)
#     #assert len(results) == 4394
#     pp(results)

#     verify_reults(results)
