# Dynamic programming for matmul computation with minimal time
# The trick is to use the simulator and use only powers of 2 for now.
# limit time to 100 before running the simulator for now (a*b*c < 100, a*b*c+b*c*d < 100)

# OK, now let's start with just the op (simple 2 powers):
from simulator import Cache, Bandwidth, utilization
from simulate import muladd

def pp(results):
    for k, v in results.items():
        cpu = v[0] if len(v) > 0 else 0
        bw_time = v[1] if len(v) > 1 else 0
        util = 0.0 if (cpu == 0 and bw_time == 0) else (cpu / max(cpu, bw_time))
        print(f"{k}: {v} | util={util:.3f}")


if __name__ == "__main__":
    bw = Bandwidth(Cache(12, muladd))
    results = bw.dynamic_times(2, 10000)
    pp(results)







