import pytest

from simulator import BinOpx, Tensor
from simulate import muladdsimple
from dynamic import run_dynamic


def _new_node():
    return BinOpx([], 0, [], 0, [], 0, muladdsimple, 1)


def test_run_dynamic_binopx_two():
    node = _new_node()
    # Generate dynamic results for 2-matrix chains within a large budget
    results = node.dynamic_times(2, 4096)
    # 4x8 @ 8x16
    A = Tensor.zeros(4, 8, level=0)
    B = Tensor.zeros(8, 16, level=0)
    out = run_dynamic(results, node, A, B)
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
    out = run_dynamic(results, node, A, B, C)
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
        run_dynamic(results, node, A, B)
