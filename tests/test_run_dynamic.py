import pytest

from simulator import BinOpx, Tensor, Cache, Bandwidth
from simulate import muladdsimple, muladd
from dynamic import run_dynamic
import bandwidth_dynamic


def _new_node():
    return BinOpx([], 0, [], 0, [], 0, muladdsimple, 1)


def test_run_dynamic_binopx_two():
    node = _new_node()
    # Generate dynamic results for 2-matrix chains within a large budget
    results = node.dynamic_times(2, 4096)
    # 4x8 @ 8x16
    A = Tensor.zeros(4, 8, level=0)
    B = Tensor.zeros(8, 16, level=0)
    out = run_dynamic(results, node, A, B, out_level=0)
    assert out.sz == [4, 16]
    # Expected ops: 4*8*16
    assert node.time == 4 * 8 * 16


def test_run_dynamic_binopx_three():
    node = _new_node()
    results = node.dynamic_times(3, 4096)
    # (4x8 @ 8x16) @ 16x2
    A = Tensor.zeros(4, 8, level=0)
    B = Tensor.zeros(8, 16, level=0)
    C = Tensor.zeros(16, 2, level=0)
    out = run_dynamic(results, node, A, B, C, out_level=0)
    assert out.sz == [4, 2]
    # Expected ops: 4*8*16 + 4*16*2
    assert node.time == (4 * 8 * 16) + (4 * 16 * 2)


def test_run_dynamic_no_match_raises():
    node = _new_node()
    results = node.dynamic_times(2, 4096)
    # 3 is not a power of two; there should be no matching entry
    A = Tensor.zeros(3, 3, level=0)
    B = Tensor.zeros(3, 3, level=0)
    with pytest.raises(KeyError):
        run_dynamic(results, node, A, B, out_level=0)


def test_run_dynamic_ldst_two_and_counters():
    # Build a two-level hierarchy: L1 --bw--> L0 --op
    node = BinOpx([], 0, [], 0, [], 0, muladdsimple, 1)
    # Small L0 so only small shapes pass capacity filter
    L0 = Cache(24, node)
    bw = Bandwidth(L0)
    L1 = Cache(1000, bw)

    # Generate dynamic results at the bandwidth link
    results = bw.dynamic_times(2, 4096)

    # Pick shapes that fit in L0 together with output: 2x2 @ 2x2 -> 2x2
    A = L1.calloc(2, 2)
    B = L1.calloc(2, 2)

    # Run once with counter reset
    out = run_dynamic(results, L1, A, B, out_level=0, reset_counter=True)
    out = run_dynamic(results, L1, A, B, out_level=1, reset_counter=True)
    assert out.sz == [2, 2]
    # CPU ops
    assert node.time == 2 * 2 * 2
    # Bandwidth counters: inputs loaded, then output stored
    assert bw.input == A.size() + B.size()
    assert bw.output == out.size()
    # Verify bw.time matches dynamic result's bw_time for the selected key variant
    key = ((2, 2), 1, (2, 2), 1, (2, 2), 1)
    assert results[key][2] == bw.time


def test_run_dynamic_handles_join_entries():
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
        assert join_key is not None

        pairs = [(join_key[i], join_key[i + 1]) for i in range(0, len(join_key), 2)]
        ops = pairs[:-1]
        out_dims, out_level = pairs[-1]

        tensors = [Tensor.zeros(shp[0], shp[1], level=lvl) for shp, lvl in ops]
        tensors = [bw.alloc(t, allow_lower_level=True) for t in tensors]

        out = run_dynamic(results, bw, *tensors, out_level=out_level, reset_counter=True)
        assert out.sz == [out_dims[0], out_dims[1]]

        bw.free(out, allow_lower_level=True)
        for t in tensors:
            bw.free(t, allow_lower_level=True)
    finally:
        bandwidth_dynamic.ENABLE_DP_JOIN_MATMULS = flag


# def test_run_dynamic_ldst_accumulate_output():
#     # Build hierarchy
#     node = BinOpx([], 0, [], 0, [], 0, muladdsimple, 1)
#     from simulator import Cache, Bandwidth
#     L0 = Cache(24, node)
#     bw = Bandwidth(L0)
#     L1 = Cache(1000, bw)

#     results = bw.dynamic_times(2, 4096)

#     A = L1.calloc(2, 2)
#     B = L1.calloc(2, 2)
#     OUT = L1.calloc(2, 2)

#     out = run_dynamic(results, L1, A, B, accumulate_output=OUT, reset_counter=True)
#     assert out is OUT
#     assert out.sz == [2, 2]
#     # CPU ops unchanged
#     assert node.time == 2 * 2 * 2
#     # With accumulation, we also load the output before compute
#     assert bw.input == A.size() + B.size() + OUT.size()
#     assert bw.output == OUT.size()
#     # Bandwidth time equals dynamic entry's bw_time plus one extra OUT.size for the output load
#     key = ((2, 2), 1, (2, 2), 1, (2, 2), 1)
#     assert results[key][2] + OUT.size() == bw.time

#     # Run again without resetting; counters should accumulate
#     out2 = run_dynamic(results, L1, A, B, out_level=0, reset_counter=False)
#     assert out2.sz == [2, 2]
#     assert node.time == 2 * (2 * 2 * 2)
#     assert bw.input == 2 * (A.size() + B.size())
#     assert bw.output == 2 * out.size()
