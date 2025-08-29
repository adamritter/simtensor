import simulator
import simulate

def matmul3_two(cache2, a, b, c, out):
    """Compute (a @ b) @ c using two matmul calls with an explicit temporary."""
    tmp = cache2.calloc(a.sz[0], b.sz[1])
    simulate.matmul(cache2, a, b, tmp)
    simulate.matmul(cache2, tmp, c, out)
    return out

def matmul3_fused_precompute(cache2, a, b, c, out, tile=4):
    """Fused triple product with reduced store traffic by precomputing A@B cols.

    Strategy:
    - For each row tile (ii) of A, precompute and keep in L0 the temporary
      matrix TMP (ii x P) whose l-th column is A_block @ B[:, l]. This ensures
      we compute each tmp column exactly once.
    - Then iterate over output column tiles (kk). For each tile, load the
      out-tile once, and sweep l=0..P-1, multiplying TMP[:, l] by C[l, k0:k0+kk]
      to accumulate into the out tile. Finally, store the out tile back once.

    This reduces bandwidth by avoiding repeated load/store of the same out tile
    for each l, while also avoiding recomputation of tmp across k-tiles.
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

        # Precompute TMP = A_block @ B for this row tile in a single call
        # Shapes: A_block (ii x M), B (M x P) => TMP (ii x P)
        tmpbuf = cache2.parentcache.calloc(ii, P)
        simulate.matmul_short_long_short_cache(
            cache2,
            a[i0:i0 + ii, :],
            b,
            tmpbuf,
        )

        # Now sweep output column tiles; keep each out tile resident and only
        # store once after accumulating all l contributions.
        k0 = 0
        while k0 < R:
            kk = min(tile, R - k0)
            out_tile_view = out[i0:i0 + ii, k0:k0 + kk]
            out_tile = cache2.load(out_tile_view)

            # Multiply using outer products: tmpbuf is in L0, so load only C rows
            for l in range(P):
                tmp_col = tmpbuf[:, l:(l + 1)]
                c_row = cache2.load(c[l:(l + 1), k0:k0 + kk])
                cache2.parentcache.run(simulate.matmulsimple, tmp_col, c_row, out_tile)
                cache2.parentcache.free(c_row)

            # After full accumulation, write back the completed out tile once
            cache2.store_to(out_tile, out_tile_view)
            k0 += tile

        # Free TMP buffer for this row tile
        cache2.parentcache.free(tmpbuf)

        i0 += tile

    return out


def matmul3_fused(cache2, a, b, c, out, tile=4):
    """Fused triple product: out = (a @ b) @ c without forming the (a @ b) temp.

    For each row-tile of A (size ii), we compute tmp = A_block @ B[:, l]
    once per l, then sweep across the output column-tiles k0 and accumulate
    tmp @ C[l, k0:k0+kk] into the corresponding out tiles. This avoids
    recomputing tmp for each k0 (which previously over-counted compute).
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

        # Accumulate contributions across P for this row tile
        for l in range(P):
            # Compute tmp = A_block (ii x M) @ B[:, l:(l+1)] (M x 1) once
            tmp = cache2.parentcache.calloc(ii, 1)
            simulate.matmul_short_long_short_cache(
                cache2,
                a[i0:i0 + ii, :],
                b[:, l:(l + 1)],
                tmp,
            )

            # Now sweep across column tiles and accumulate into out
            k0 = 0
            while k0 < R:
                kk = min(tile, R - k0)
                out_tile_view = out[i0:i0 + ii, k0:k0 + kk]
                out_tile = cache2.load(out_tile_view)
                c_row = cache2.load(c[l:(l + 1), k0:k0 + kk])
                cache2.parentcache.run(simulate.matmulsimple, tmp, c_row, out_tile)
                cache2.parentcache.free(c_row)
                # Write updated tile back to parent view
                cache2.store_to(out_tile, out_tile_view)
                k0 += tile

            # Done with tmp for this l
            cache2.parentcache.free(tmp)

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

    # Fused implementation (tile-wise, recompute-safe)
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

    # Fused precompute implementation (reduced store traffic)
    cache_pre = _new_cache()
    A3 = cache_pre.calloc(n, m)
    B3 = cache_pre.calloc(m, p)
    C3 = cache_pre.calloc(p, r)
    OUT3 = cache_pre.calloc(n, r)
    matmul3_fused_precompute(cache_pre, A3, B3, C3, OUT3)
    bw_pre = cache_pre.parent
    op_pre = bw_pre.cache.parent
    util_pre = simulator.utilization(cache_pre)
    print(
        "fused-pre: input={inp} output={out} total={tot} cpu={cpu} util={util:.3f}".format(
            inp=bw_pre.input,
            out=bw_pre.output,
            tot=bw_pre.input + bw_pre.output,
            cpu=op_pre.time,
            util=util_pre,
        )
    )

    # Simple bandwidth comparison among the three
    totals = [
        ("two-matmul", bw_two.input + bw_two.output),
        ("fused", bw_fused.input + bw_fused.output),
        ("fused-pre", bw_pre.input + bw_pre.output),
    ]
    best = min(totals, key=lambda x: x[1])
    print(f"-> lowest bandwidth: {best[0]}")


if __name__ == "__main__":
    # Given example: 100x1 * 1x100 * 100x1 (avoids 100x100 temporary)
    _run_example(100, 1, 100, 1, "Skinny * Wide * Skinny")

    # Another example where benefit is smaller (square-ish)
    _run_example(64, 32, 32, 64, "Square-ish triple")
