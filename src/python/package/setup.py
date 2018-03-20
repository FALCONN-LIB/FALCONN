import os
import sys

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst', format='md')
except (IOError, ImportError):
    long_description = open('README.md').read()

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    sys.stderr.write('Setuptools not found!\n')
    raise

extra_args = ['-std=c++11', '-march=native', '-O3']
if sys.platform == 'darwin':
    extra_args += ['-mmacosx-version-min=10.9', '-stdlib=libc++']
    os.environ['LDFLAGS'] = '-mmacosx-version-min=10.9'

module = Extension(
    '_falconn',
    sources=['internal/python_wrapper.cc'],
    extra_compile_args=extra_args,
    include_dirs=['include', 'external/eigen', 'external/pybind11/include', 'external/simple-serializer'])

setup(
    name='FALCONN',
    version='1.4.0',
    author='Ilya Razenshteyn, Ludwig Schmidt',
    author_email='falconn.lib@gmail.com',
    url='http://falconn-lib.org/',
    description=
    'A library for similarity search over high-dimensional data based on Locality-Sensitive Hashing (LSH)',
    long_description=long_description,
    license='MIT',
    keywords=
    'nearest neighbor search similarity lsh locality-sensitive hashing cosine distance euclidean',
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[module])
