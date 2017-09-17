import numpy as np
import ffht
import timeit

reps = 1000
n = 2**20
chunk_size = 1024

a = np.random.randn(n).astype(np.float32)

t1 = timeit.default_timer()
for i in range(reps):
    ffht.fht(a)
t2 = timeit.default_timer()
print (t2 - t1 + 0.0) / (reps + 0.0)
