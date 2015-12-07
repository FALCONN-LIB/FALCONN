## FALCONN - FAst Lookups of COsine Nearest Neighbors

FALCONN is a library with algorithms for the nearest neighbor search problem. The algorithms in FALCONN are based on
Locality-Sensitve Hashing (LSH), which is a popular class of methods for nearest neighbor search in high-dimensional spaces.
The goal of FALCONN is to provide very efficient and well tested implementations of LSH-based data structures.

Currently, FALCONN supports two LSH families for the cosine similarity: hyperplane LSH and cross polytope LSH.
Both hash families are implemented with multi-probe LSH in order to minimize memory usage.
Moreover, FALCONN is optimized for both dense and sparse data.

FALCONN is written in C++ and consists of several modular core classes with a convenient wrapper around them.
Many mathematical operations in FALCONN are vectorized through the Eigen and FFHT libraries.
The core classes of FALCONN rely on templates in order to avoid runtime overhead.

### How to use FALCONN

For now, we provide a C++ interface for FALCONN. In the future, we plan to support more programming languages such as Python (NumPy)
and Julia. For C++, FALCONN is a header-only library and has no dependencies besides Eigen (which is also header-only),
so FALCONN is easy to set up. For further details, please see the documentation.

### How fast is FALCONN?

On data sets with about 1 million points in around 100 dimensions, FALCONN requires a few milliseconds per query with a top-10 nearest
neighbor accuracy (running on a reasonably modern desktop CPU). For more details, please see our corresponding
[research paper](http://papers.nips.cc/paper/5893-practical-and-optimal-lsh-for-angular-distance).

### Questions

If you have questions about using FALCONN, we would be happy to help. Please send an email to falconn.lib@gmail.com.

### Authors

FALCONN is mainly developed by [Ilya Razenshteyn](http://www.ilyaraz.org/) and [Ludwig Schmidt](http://people.csail.mit.edu/ludwigs/).
FALCONN has grown out of a research project with our collaborators [Alexandr Andoni](http://www.mit.edu/~andoni/), [Piotr Indyk](https://people.csail.mit.edu/indyk/), and [Thijs Laarhoven](http://thijs.com/).
Many of the ideas used in FALCONN were proposed in research papers over the past 20 years (see the documentation).

### License

FALCONN is available under the MIT license (see LICENSE.txt).

