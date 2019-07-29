import _multiprobe
import numpy as np

m = _multiprobe.Multiprobe(100, 8, 4057218)
while True:
    x = np.random.randn(100).astype(np.float32)
    x /= np.linalg.norm(x)
    print(m.query(x))
