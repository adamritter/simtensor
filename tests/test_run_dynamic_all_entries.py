import os
import sys

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import simulate
from simulator import Tensor, Cache, Bandwidth
from dynamic import run_dynamic, pp, previous_key, previous_key2, extras


def verify_result(key, results, node):
    # Key is alternating (shape, level); last pair is output dims
    pairs = [(key[i], key[i + 1]) for i in range(0, len(key), 2)]
    operand_pairs = pairs[:-1]
    out_dims, out_level = pairs[-1]

    # Build input tensors at level 0 for each operand shape
    tensors = [Tensor.zeros(shp[0], shp[1], level=lvl) for shp, lvl in operand_pairs]
    if isinstance(node, Cache) or isinstance(node, Bandwidth):
        tensors = [node.alloc(t, allow_lower_level=True) for t in tensors]

    v = results.get(key)
    # print(f"Running dynamic for {key}, value {v}, extras {extras(key, results)}")
    # pk = previous_key(key, v[0])
    # print(f"    previous_key: {pk} = {results.get(pk)}")
    # if isinstance(v[0], tuple) and v[0] and v[0][0] == "JOIN":
    #     pk2 = previous_key2(key, v[0])
    #     print(f"    previous_key2: {pk2} = {results.get(pk2)}")
    out = run_dynamic(results, node, *tensors, out_level=out_level, reset_counter=True)
    # Output shape should match trailing dims in the key
    assert out.sz == [out_dims[0], out_dims[1]]
    if isinstance(node, Cache) or isinstance(node, Bandwidth):
        node.free(out, allow_lower_level=True)
    for t in tensors:
        if isinstance(node, Cache) or isinstance(node, Bandwidth):
            node.free(t, allow_lower_level=True)

    # Additionally verify that for all non-BinOpx rows, the previous_key exists.
    if isinstance(v, list) and v:
        head = v[0]
        if not (isinstance(head, str) and head == 'BinOpx'):
            op = head if isinstance(head, tuple) else None
            pk = previous_key(key, op)
            assert pk in results, f"previous_key missing for {key} with op {head} -> {pk}"
            if isinstance(op, tuple) and op and op[0] == 'JOIN':
                pk2 = previous_key2(key, op)
                assert pk2 in results, f"previous_key2 missing for {key} with op {head} -> {pk2}"


def verify_results(results, node):
    for key in results.keys():
        verify_result(key, results, node)


def test_run_dynamic_for_all_muladd_dynamic_times_three():
    # Enumerate all 2- and 3-matrix chains up to limit 8 and run them
    results = simulate.muladd.dynamic_times(3, 100)

    verify_results(results, simulate.muladd)


def test_run_dynamic_for_all_muladd_dynamic_times_three_cache():
    # Enumerate all 2- and 3-matrix chains up to limit 8 and run them
    cache = Cache(12, simulate.muladd)
    results = cache.dynamic_times(3, 1000)
    verify_results(results, cache)


def test_run_dynamic_for_all_muladd_dynamic_times_three_bandwidth():
    # Enumerate all 2- and 3-matrix chains up to limit 8 and run them
    bw = Bandwidth(Cache(12, simulate.muladd))
    results = bw.dynamic_times(3, 30)
    #pp(results)
    #assert len(results) == 2549

    verify_results(results, bw)



def test_run_dynamic_for_all_muladd_dynamic_times_three_cache():
    # Enumerate all 2- and 3-matrix chains up to limit 8 and run them
    bw2 = Cache(100000, Bandwidth(Cache(16, simulate.muladd)))
    results = bw2.dynamic_times(3, 30)
    #pp(results)
    verify_results(results, bw2)


def test_run_dynamic_for_all_muladd_dynamic_times_three_bandwidth2():
    # Enumerate all 2- and 3-matrix chains up to limit 8 and run them
    bw2 = Bandwidth(Cache(100000, Bandwidth(Cache(16, simulate.muladd))))
    results = bw2.dynamic_times(3, 30)
    #pp(results)
    verify_results(results, bw2)
