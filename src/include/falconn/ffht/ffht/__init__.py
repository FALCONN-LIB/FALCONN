"""Python wrapper for FFHT.

This is a Python wrapper for [FFHT](https://github.com/FALCONN-LIB/FFHT).
It exposes two functions: `create_aligned`, which creates aligned
one-dimensional [NumPy](http://www.numpy.org/) arrays, and `fht`, which
performs the Fast Hadamard Transform on a given one-dimensional NumPy
array.

To the best of our knowledge, FFHT is currently the fastest open-source
implementation of *any* Fourier-like transform.
"""

from _ffht import fht
import numpy as _numpy

def create_aligned(n, dtype, alignment=32):
    """ Create an aligned one-dimensional NumPy array.

    Creates a one-dimensional NumPy array of length `n`
    with a given `dtype` that is aligned to `alignment` bytes.

    NB: `alignment` must be a multiple of the size of `dtype`.
    """
    buf = _numpy.zeros(n + alignment, dtype=dtype)
    if alignment % buf.itemsize:
        raise ValueError('alignment must be a multiple of the size of dtype')
    off = buf.ctypes.data % alignment
    shift = 0
    if off != 0:
        shift = (alignment - off) / buf.itemsize
    return buf[shift : shift + n]
