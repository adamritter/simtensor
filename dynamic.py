# Dynamic programming for matmul computation with minimal time
# The trick is to use the simulator and use only powers of 2 for now.
# limit time to 100 before running the simulator for now (a*b*c < 100, a*b*c+b*c*d < 100)

# OK, now let's start with just the op (simple 2 powers):
from simulator import Cache, Bandwidth
from simulate import muladd

def pp(results):
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    results = Bandwidth(Cache(12, muladd)).dynamic_times(2, 1000)
    pp(results)







