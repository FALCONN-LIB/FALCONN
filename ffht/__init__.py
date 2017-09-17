"""Python wrapper for FFHT.

This is a Python wrapper for [FFHT](https://github.com/FALCONN-LIB/FFHT).
It exposes one function, `fht`, which performs the Fast Hadamard
Transform on a given one-dimensional NumPy array.

To the best of our knowledge, FFHT is currently the fastest open-source
implementation of *any* Fourier-like transform.
"""

from _ffht import fht
