import sys

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst', format='md')
except(IOError, ImportError):
    long_description = open('README.md').read()

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
      version='1.2.1',
      author='Ilya Razenshteyn, Ludwig Schmidt',
      author_email='falconn.lib@gmail.com',
      url='http://falconn-lib.org/',
      description='A library for similarity search over high-dimensional data based on Locality-Sensitive Hashing (LSH)',
      long_description=long_description,
      license='MIT',
      keywords='nearest neighbor search similarity lsh locality-sensitive hashing cosine distance euclidean',
      packages=find_packages(),
      include_package_data=True,
      ext_modules=[module])
