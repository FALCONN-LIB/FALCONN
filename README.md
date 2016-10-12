### FALCONN - FAst Lookups of Cosine and Other Nearest Neighbors

FALCONN is a library with algorithms for the nearest neighbor search problem. The algorithms in FALCONN are based on
[Locality-Sensitive Hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) (LSH), which is a popular class of methods for nearest neighbor search in high-dimensional spaces.
The goal of FALCONN is to provide very efficient and well-tested implementations of LSH-based data structures.

Currently, FALCONN supports two LSH families for the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity): hyperplane LSH and cross polytope LSH.
Both hash families are implemented with multi-probe LSH in order to minimize memory usage.
Moreover, FALCONN is optimized for both dense and sparse data.
Despite being designed for the cosine similarity, FALCONN can often be used for nearest neighbor search under
the Euclidean distance or a maximum inner product search.

FALCONN is written in C++ and consists of several modular core classes with a convenient wrapper around them.
Many mathematical operations in FALCONN are vectorized through the [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) and [FFHT](https://github.com/FALCONN-LIB/FFHT) libraries.
The core classes of FALCONN rely on [templates](https://en.wikipedia.org/wiki/Andrei_Alexandrescu) in order to avoid runtime overhead.

### How to use FALCONN

We provide a C++ interface for FALCONN as well as a [Python](https://www.python.org/) wrapper (that uses [NumPy](http://www.numpy.org/)). In the future, we plan to support more programming languages such as [Julia](http://julialang.org/). For C++, FALCONN is a header-only library and has no dependencies besides Eigen (which is also header-only),
so FALCONN is easy to set up. For further details, please see our [documentation](https://github.com/falconn-lib/falconn/wiki).

### How fast is FALCONN?

On data sets with about 1 million points in around 100 dimensions, FALCONN typically
requires a few milliseconds per query (running on a reasonably modern desktop CPU).

For more detailed results, see [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) of [Erik Bernhardsson](https://erikbern.com/). Let us point out that FALCONN is especially competitive, when
the RAM budget is quite restrictive, which is not the regime the above benchmarks use.

### Questions

Maybe your question is already answered in our [Frequently Asked Questions](https://github.com/falconn-lib/falconn/wiki/FAQ).
If you have additional questions about using FALCONN, we would be happy to help. Please send an email to falconn.lib@gmail.com.

### Authors

FALCONN is mainly developed by [Ilya Razenshteyn](http://www.ilyaraz.org/) and [Ludwig Schmidt](http://people.csail.mit.edu/ludwigs/).
FALCONN has grown out of a [research project](http://papers.nips.cc/paper/5893-practical-and-optimal-lsh-for-angular-distance) with our collaborators [Alexandr Andoni](http://www.mit.edu/~andoni/), [Piotr Indyk](https://people.csail.mit.edu/indyk/), and [Thijs Laarhoven](http://thijs.com/).

Many of the ideas used in FALCONN were proposed in research papers over the past 20 years (see the [documentation](https://github.com/FALCONN-LIB/FALCONN/wiki/Bibliography)).

If you want to cite FALCONN in a publication, here is the bibliographic information of  our research paper [(bibtex)](http://papers.nips.cc/paper/5893-practical-and-optimal-lsh-for-angular-distance/bibtex):

> [Practical and Optimal LSH for Angular Distance](http://papers.nips.cc/paper/5893-practical-and-optimal-lsh-for-angular-distance)  
> Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn, Ludwig Schmidt  
> NIPS 2015

### License

FALCONN is available under the [MIT License](https://opensource.org/licenses/MIT) (see LICENSE.txt).
Note that the third-party libraries in the `external/` folder are distributed under other open source licenses.
The Eigen library is licensed under the [MPL2](https://www.mozilla.org/en-US/MPL/2.0/).
The googletest and googlemock libraries are licensed under the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
Furthermore, the file `src/python/module/numpy.i` is part of the [NumPy project](http://www.numpy.org/) and also licensed under the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
