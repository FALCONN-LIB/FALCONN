import numpy as np
import ffht
import timeit

reps = 1000
n = 2**20
chunk_size = 1024

a = ffht.create_aligned(n, np.float32)
np.copyto(a, np.random.randn(n))

t1 = timeit.default_timer()
for i in range(reps):
    ffht.fht(a, chunk_size)
t2 = timeit.default_timer()
print (t2 - t1 + 0.0) / (reps + 0.0)
