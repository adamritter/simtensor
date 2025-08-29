import simulator
import simulate


def matmul_store(cache2, a, b, c, tile=4):
    """Tile-based matmul that writes results back using store_to.
    Shapes: a (N x M), b (M x K), c (N x K)
    """
    n = tile
    i = 0
    while i < c.sz[0]:
        j = 0
        while j < c.sz[1]:
            # Load output tile into L0
            c_tile_view = c[i:i + n, j:j + n]
            cc = cache2.load(c_tile_view)
            # Short-long-short across the shared dimension to keep L0 usage small
            shared = a.sz[1]
            for t in range(shared):
                aa_col = cache2.load(a[i:(i + n), t:(t + 1)])  # (n x 1)
                bb_row = cache2.load(b[t:(t + 1), j:(j + n)])  # (1 x n)
                cache2.parentcache.run(simulate.matmulsimple, aa_col, bb_row, cc)
                cache2.parentcache.free(aa_col)
                cache2.parentcache.free(bb_row)
            # Store the tile back to parent view
            cache2.store_to(cc, c_tile_view)
            j += n
        i += n


def matmul3_two(cache2, a, b, c, out):
    """Compute (a @ b) @ c using two matmul calls with an explicit temporary."""
    tmp = cache2.calloc(a.sz[0], b.sz[1])
    simulate.matmul(cache2, a, b, tmp)
    simulate.matmul(cache2, tmp, c, out)
    return out


def matmul3_fused(cache2, a, b, c, out, tile=4):
    """Fused triple product: out = (a @ b) @ c, avoiding full (a @ b).

    Reuses `simulate.matmul_short_long_short_cache` for the A @ (B[:, l]) step,
    then applies an outer product with C[l, :] per l-tile to accumulate into out.
    """
    N, M = a.sz[0], a.sz[1]
    assert M == b.sz[0]
    P = b.sz[1]
    assert P == c.sz[0]
    R = c.sz[1]
    assert out.sz == [N, R]

    i0 = 0
    while i0 < N:
        ii = min(tile, N - i0)
        k0 = 0
        while k0 < R:
            kk = min(tile, R - k0)
            # Work on a small output tile
            out_tile_view = out[i0:i0 + ii, k0:k0 + kk]
            out_tile = cache2.load(out_tile_view)

            # Accumulate contributions across P
            for l in range(P):
                # tmp = a_block @ b_column(l), kept in L0
                tmp = cache2.parentcache.calloc(ii, 1)
                simulate.matmul_short_long_short_cache(cache2, a[i0:i0 + ii, :], b[:, l:(l + 1)], tmp)

                # Multiply tmp (ii x 1) by c_row (1 x kk) into out_tile
                c_row = cache2.load(c[l:(l + 1), k0:k0 + kk])
                cache2.parentcache.run(simulate.matmulsimple, tmp, c_row, out_tile)
                cache2.parentcache.free(tmp)
                cache2.parentcache.free(c_row)

            # Store the tile back to the parent cache
            cache2.store_to(out_tile, out_tile_view)
            k0 += tile
        i0 += tile

    return out


def _new_cache(L1_size=100000, L0_size=256):
    # New BinOpx to keep timing separate per run
    op = simulator.BinOpx([], 0, [], 0, [], 0, simulate.muladdsimple, 1)
    L0 = simulator.Cache(L0_size, op)
    bw = simulator.Bandwidth(L0)
    L1 = simulator.Cache(L1_size, bw)
    return L1


def _run_example(n, m, p, r, title):
    print(f"\n=== {title}: {n}x{m} * {m}x{p} * {p}x{r} ===")
    # Two matmuls: (n x m) @ (m x p) costs n*m*p, then (n x p) @ (p x r) costs n*p*r
    expected_ops = n * m * p + n * p * r
    print(f"expected ops: {expected_ops}")

    # Two-matmul baseline
    cache_two = _new_cache()
    A = cache_two.calloc(n, m)
    B = cache_two.calloc(m, p)
    C = cache_two.calloc(p, r)
    OUT = cache_two.calloc(n, r)
    matmul3_two(cache_two, A, B, C, OUT)
    bw_two = cache_two.parent
    op_two = bw_two.cache.parent  # BinOpx at the bottom
    util_two = simulator.utilization(cache_two)
    print(
        "two-matmul: input={inp} output={out} total={tot} cpu={cpu} util={util:.3f}".format(
            inp=bw_two.input,
            out=bw_two.output,
            tot=bw_two.input + bw_two.output,
            cpu=op_two.time,
            util=util_two,
        )
    )

    # Fused implementation
    cache_fused = _new_cache()
    A2 = cache_fused.calloc(n, m)
    B2 = cache_fused.calloc(m, p)
    C2 = cache_fused.calloc(p, r)
    OUT2 = cache_fused.calloc(n, r)
    matmul3_fused(cache_fused, A2, B2, C2, OUT2)
    bw_fused = cache_fused.parent
    op_fused = bw_fused.cache.parent
    util_fused = simulator.utilization(cache_fused)
    print(
        "fused:      input={inp} output={out} total={tot} cpu={cpu} util={util:.3f}".format(
            inp=bw_fused.input,
            out=bw_fused.output,
            tot=bw_fused.input + bw_fused.output,
            cpu=op_fused.time,
            util=util_fused,
        )
    )

    if (bw_fused.input + bw_fused.output) < (bw_two.input + bw_two.output):
        print("-> fused uses less bandwidth")
    else:
        print("-> two-matmul uses less or equal bandwidth")


if __name__ == "__main__":
    # Given example: 100x1 * 1x100 * 100x1 (avoids 100x100 temporary)
    _run_example(100, 1, 100, 1, "Skinny * Wide * Skinny")

    # Another example where benefit is smaller (square-ish)
    _run_example(64, 32, 32, 64, "Square-ish triple")
