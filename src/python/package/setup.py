import sys

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    sys.stderr.write('Setuptools not found!\n')
    raise

try:
    import numpy as np
except ImportError:
    sys.stderr.write('NumPy not found!\n')
    raise

module = Extension('_falconn',
                   sources=['falconn/swig/falconn_wrap.cc'],
                   extra_compile_args=['-std=c++11', '-march=native', '-O3'],
                   include_dirs=['falconn/src/include',
                                 'falconn/external/eigen',
                                 np.get_include()])

setup(name='FALCONN',
      version='1.1',
      packages=find_packages(),
      author='Ilya Razenshteyn, Ludwig Schmidt',
      author_email='falconn.lib@gmail.com',
      url='http://falconn-lib.org/',
      include_package_data=True,
      ext_modules=[module])
