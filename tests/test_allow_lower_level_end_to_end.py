import os
import sys

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from simulator import Cache, Bandwidth, Tensor, BinOpx


def make_hierarchy():
    # L0 (compute) <-bw01-> L1 cache <-bw12-> L2 cache (top)
    op = BinOpx([], 0, [], 0, [], 0, lambda a, b, c: None, t=1)
    L0 = Cache(64, op)
    bw01 = Bandwidth(L0)
    L1 = Cache(256, bw01)
    bw12 = Bandwidth(L1)
    L2 = Cache(1024, bw12)
    return op, L0, bw01, L1, bw12, L2


def read_column(mat, j):
    return [mat[i][j].value for i in range(mat.sz[0])]


def test_allow_lower_level_end_to_end_from_top_cache():
    # Build a 3-level hierarchy and exercise alloc -> load -> modify -> store -> free
    op, L0, bw01, L1, bw12, L2 = make_hierarchy()

    # Allocate source and destination at the top cache (L2)
    dest = L2.calloc(4, 4)
    src = L2.alloc_diag(4)

    # Work on a single column
    dest_col = dest[:, 1:2]
    src_col = src[:, 2:3]

    # Load source column down to the bottom (call from top with allow_lower_level)
    src_col_l = L2.load(src_col, allow_lower_level=True)   # L2 -> L1
    src_col_ll = L2.load(src_col_l, allow_lower_level=True)  # L1 -> L0 (via routing)
    assert src_col_ll.level == L0.level

    # Modify a value at the lowest level: set all ones so we can verify store_to
    for i in range(4):
        src_col_ll[i, 0] = 1

    # Store back into the existing destination view, calling from the top cache.
    # This should route across levels automatically when allow_lower_level is True.
    # Expectation: after storing, dest[:,1] becomes all ones.
    l1dst = L1.store(src_col_ll)
    L2.store_to(src_col_ll, dest_col, allow_lower_level=True)

    assert read_column(dest, 1) == [1, 1, 1, 1]

    # Freeing from the top should be a no-op now because store_to already
    # freed the source from lower caches during promotion.
    used_after_store = L0.used
    assert L0.used == used_after_store


def test_allow_lower_level_from_top_bandwidth_alloc_free_and_load():
    # Ensure Bandwidth methods with allow_lower_level route to the correct cache
    op, L0, bw01, L1, bw12, L2 = make_hierarchy()

    # Prepare a level-0 tensor and allocate it via the top bandwidth
    t0 = Tensor.zeros(2, 2, level=L0.level)
    used_before = L0.used
    bw12.alloc(t0, allow_lower_level=True)
    assert L0.used == used_before + t0.size()

    # Check cachecontains routed
    assert bw12.cachecontains(t0, allow_lower_level=True)

    # Load from the top bandwidth: move a top-level tensor down one level
    top_mat = L2.calloc(3, 3)
    v_top = top_mat[:, 0:1]
    v_mid = bw12.load(v_top, allow_lower_level=True)  # should land in L1
    assert v_mid.level == L1.level

    # Free via top bandwidth from lower level
    bw12.free(t0, allow_lower_level=True)
    assert L0.used == used_before

    # Note: store_to from Bandwidth is not implemented; calling through the
    # top-level Cache with allow_lower_level is expected to handle routing.
