import simulator


def muladdsimple(a, b, c):
    c.setv(a.value * b.value + c.value)


# Basic multiply-add operation used by matmul
muladd = simulator.BinOpx([], 0, [], 0, [], 0, muladdsimple, 1)


def matmulsimple(muladdop, a, b, c):
    for i in range(a.sz[0]):
        for j in range(b.sz[0]):
            for k in range(b.sz[1]):
                muladdop.run(a[i][j], b[j][k], c[i][k])
    return c


def matmul_short_long_short_cache(cache2, a, b, c):
    for i in range(b.sz[0]):
        aa = cache2.load(a[:, i:(i + 1)])
        bb = cache2.load(b[i:(i + 1)])
        cache2.parentcache.run(matmulsimple, aa, bb, c)
        cache2.parentcache.free(aa)
        cache2.parentcache.free(bb)
    return c


def matmul(cache2, a, b, c):
    n = 4
    i = 0
    while i < c.sz[0]:
        j = 0
        while j < c.sz[1]:
            cc = cache2.load(c[i:i + n, j:j + n])
            matmul_short_long_short_cache(cache2, a[i:(i + n), :], b[:, j:(j + n)], cc)
            cache2.parentcache.free(cc)  # neet to store instead (original behavior)
            j += n
        i += n


if __name__ == "__main__":
    # Basic muladd and small matmul test
    print(muladd.run(
        simulator.Tensor([2], 0, []),
        simulator.Tensor([3], 0, []),
        simulator.Tensor([1], 0, []),
    ))
    print(matmulsimple(
        muladd,
        simulator.Tensor([1, 2, 3, 4], 0, [2, 2]),
        simulator.Tensor([1, 2, 3, 4], 0, [2, 2]),
        simulator.Tensor([0, 0, 0, 0], 0, [2, 2]),
    ))

    # M4: 3 LD + 2 ST / cycle, 4 NEON, but only 2 FMA? 32 named vector registers, 300-400 physical
    muladd_local = simulator.BinOpx([], 0, [], 0, [], 0, muladdsimple, 1)
    cache2 = simulator.Cache(1000, simulator.Bandwidth(simulator.Cache(24, muladd_local)))
    a = cache2.alloc_diag(4)
    b = cache2.alloc_diag(4)
    c = cache2.parentcache.calloc(4, 4)

    matmul_short_long_short_cache(cache2, a, b, c)
    print(cache2)
    print(c.sum())

    muladd_local = simulator.BinOpx([], 0, [], 0, [], 0, muladdsimple, 1)
    cache2 = simulator.Cache(10000, simulator.Bandwidth(simulator.Cache(24, muladd_local)))
    a = cache2.calloc(4, 100)
    b = cache2.calloc(100, 4)
    c = cache2.parentcache.calloc(4, 4)

    matmul_short_long_short_cache(cache2, a, b, c)
    print(cache2)

    muladd_local = simulator.BinOpx([], 0, [], 0, [], 0, muladdsimple, 1)
    cache2 = simulator.Cache(10000, simulator.Bandwidth(simulator.Cache(24, muladd_local)))
    a = cache2.calloc(20, 20)
    b = cache2.calloc(20, 20)
    c = cache2.calloc(20, 20)

    matmul(cache2, a, b, c)
    print(cache2)

