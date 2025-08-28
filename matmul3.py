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
            # Load input tiles into L0
            aa = cache2.load(a[i:(i + n), :])
            bb = cache2.load(b[:, j:(j + n)])
            # Compute tile at L0 using simple matmul
            cache2.parentcache.run(simulate.matmulsimple, aa, bb, cc)
            # Free inputs from L0
            cache2.parentcache.free(aa)
            cache2.parentcache.free(bb)
            # Store the tile back to parent view
            cache2.store_to(cc, c_tile_view)
            j += n
        i += n


def matmul3_two(cache2, a, b, c, out):
    """Compute (a @ b) @ c using two matmul calls with an explicit temporary."""
    tmp = cache2.calloc(a.sz[0], b.sz[1])
    matmul_store(cache2, a, b, tmp)
    matmul_store(cache2, tmp, c, out)
    return out


def matmul3_fused(cache2, a, b, c, out, tile=4):
    """Fused triple product producing out = (a @ b) @ c without materializing a@b.

    For each output tile, accumulates contributions from the inner dimensions by
    streaming columns of `a` and rows of `c`, scaled by scalars from `b`.
    """
    N, M = a.sz[0], a.sz[1]
    P = b.sz[1]
    R = c.sz[1]

    i0 = 0
    while i0 < N:
        k0 = 0
        while k0 < R:
            # Work on a small output tile
            out_tile_view = out[i0:i0 + tile, k0:k0 + tile]
            out_tile = cache2.load(out_tile_view)

            # Accumulate over inner dims
            for j in range(M):
                # Load a column block from a
                a_col = cache2.load(a[i0:i0 + tile, j:(j + 1)])  # (tile x 1)

                for l in range(P):
                    # Load scalar from b and row block from c
                    b_scalar = cache2.load(b[j:(j + 1), l:(l + 1)])  # (1 x 1)
                    c_row = cache2.load(c[l:(l + 1), k0:k0 + tile])  # (1 x tile)

                    # Scale the row by b_scalar into a temporary row in L0
                    scaled_row = cache2.parentcache.calloc(1, c_row.sz[1])
                    for kk in range(c_row.sz[1]):
                        simulate.muladd.run(b_scalar, c_row[0][kk], scaled_row[0][kk])

                    # Outer product add into the output tile: a_col @ scaled_row
                    cache2.parentcache.run(simulate.matmulsimple, a_col, scaled_row, out_tile)

                    # Free temporaries in L0
                    cache2.parentcache.free(b_scalar)
                    cache2.parentcache.free(c_row)
                    cache2.parentcache.free(scaled_row)

                cache2.parentcache.free(a_col)

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

    # Two-matmul baseline
    cache_two = _new_cache()
    A = cache_two.calloc(n, m)
    B = cache_two.calloc(m, p)
    C = cache_two.calloc(p, r)
    OUT = cache_two.calloc(n, r)
    matmul3_two(cache_two, A, B, C, OUT)
    bw_two = cache_two.parent
    print(f"two-matmul: input={bw_two.input} output={bw_two.output} total={bw_two.input + bw_two.output}")

    # Fused implementation
    cache_fused = _new_cache()
    A2 = cache_fused.calloc(n, m)
    B2 = cache_fused.calloc(m, p)
    C2 = cache_fused.calloc(p, r)
    OUT2 = cache_fused.calloc(n, r)
    matmul3_fused(cache_fused, A2, B2, C2, OUT2)
    bw_fused = cache_fused.parent
    print(f"fused:      input={bw_fused.input} output={bw_fused.output} total={bw_fused.input + bw_fused.output}")

    if (bw_fused.input + bw_fused.output) < (bw_two.input + bw_two.output):
        print("-> fused uses less bandwidth")
    else:
        print("-> two-matmul uses less or equal bandwidth")


if __name__ == "__main__":
    # Given example: 100x1 * 1x100 * 100x1 (avoids 100x100 temporary)
    _run_example(100, 1, 100, 1, "Skinny * Wide * Skinny")

    # Another example where benefit is smaller (square-ish)
    _run_example(64, 32, 32, 64, "Square-ish triple")
