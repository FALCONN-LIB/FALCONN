"""Python wrapper for FALCONN.

This is a Python wrapper for FALCONN (http://falconn-lib.org/) that is
meant to be easy to use. It exports two classes (LSHIndex and
LSHConstructionParameters) together with two helper functions
(get_default_parameters and compute_number_of_hash_functions).

For now, the Python wrapper supports only static dense datasets,
but more to come. Also, note that FALCONN is currently not thread-safe.

FALCONN is based on Locality-Sensitive Hashing (LSH), which is briefly
covered here: https://github.com/FALCONN-LIB/FALCONN/wiki/LSH-Primer
and https://github.com/FALCONN-LIB/FALCONN/wiki/LSH-Families .

The main class is LSHIndex, which takes a dataset and builds a Nearest
Neighbor data structure. A dataset is represented as a two-dimensional
NumPy array. Since FALCONN uses NumPy arrays only as a convenient way
to store and pass data, it does not matter if NumPy is compiled with
a fancy BLAS library: this has zero effect on the performance of FALCONN.

To construct an instance of LSHIndex, one needs to prepare an instance
of LSHConstructionParameters, which stores parameters used to build the
LSH data structure. To get a sense about the parameters used, see the
following page:
https://github.com/FALCONN-LIB/FALCONN/wiki/How-to-Use-FALCONN .
To get a reasonable setting of parameters, one can (and should!) use
two helper functions we provide: get_default_parameters and
compute_number_of_hash_functions.

Besides the documentation, we provide two examples of usage of the
wrapper:
* in benchmark/random_benchmark.py the LSH data structure for a
random dataset is built and used;
* in https://github.com/FALCONN-LIB/FALCONN/tree/master/examples/glove
we show how to use LSH to perform similarity search on a GloVe dataset.

An intended workflow is as follows:
* first, use get_default_parameters to get a reasonable setting of
parameters; tune if necessary;
* second, create an instance of LSHIndex passing the
LSHConstructionParameters object we've just built;
* third, call fit() method of the instance of LSHIndex,
passing the dataset;
* then, increase the number of table probes using the set_num_probes()
method until you get the desired accuracy on a sample set of queries;
* finally, use the other methods of LSHIndex to execute queries.

A couple of useful tricks:
* Convert your dataset to numpy.float32;
* Center your dataset (by doing a -= numpy.mean(a, axis=0)) and queries:
this has no effect on the correctness, but greatly improves the
performance of our LSH families.
"""
import numpy as _numpy
from . import internal as _internal
from .internal import LSHConstructionParameters

def get_default_parameters(num_points, dimension, distance='euclidean_squared', is_sufficiently_random=False):
    """Get parameters for the LSH for _reasonable_ datasets.

    This function returns an instance of LSHConstructionParameters that
    is good enough for reasonable datasets. Here we try to achieve fast
    preprocessing, so if you are willing to build index for longer
    (and spend more memory), you need to increase the number of tables
    (l) in the resulting object.
    
    The parameters returned are especially well-suited for _centered_
    datasets (when the mean is zero).

    Arguments:
    * num_points -- the number of points in a dataset
    * dimension -- the dimension of a dataset
    * distance -- the distance function used: can be either
    'euclidean_squared' (which corresponds to the vanilla Euclidean
    distance) and 'negative_inner_product' (which corresponds to
    maximizing the dot product); if you want to perform similarity
    search with respect to the cosine similarity, normalize your
    dataset, re-center it and then use 'euclidean_squared'
    (default 'euclidean_squared')
    * is_sufficiently_random -- set to True if only very few coordinates
    are zeros; in this case one is able to speed things up a little bit
    (default False)

    Returns: an object of type LSHConstructionParameter
    """
    return _internal.get_default_parameters(num_points, dimension, distance, is_sufficiently_random)

def compute_number_of_hash_functions(num_bits, params):
    """Set the number of hashes in params equivalent to num_bits bits.

    Computes the number of hash functions in order to get a hash with
    the given number of relevant bits. For the cross polytope hash, the
    last cross polytope dimension will also be determined. The input
    object params of type LSHConstructionParameters must contain valid
    values for the following fields:
    * lsh_family;
    * dimension (for the cross polytope hash).

    The function will then set the following fields of params:
    * k;
    * last_cp_dimension (for the cross polytope hash).

    Arguments:
    * num_bits -- the desired number of bits (the number of bins will be
    2^{num_bits})
    * params -- an object of the type LSHConstructionParameters with
    fields partially set as described above
    """
    _internal.compute_number_of_hash_functions(num_bits, params)

class LSHIndex:
    def __init__(self, params):
        self._params = params
        self._dataset = None
        self._table = None

    def fit(self, dataset):
        if not isinstance(dataset, _numpy.ndarray):
            raise TypeError('dataset must be an instance of numpy.ndarray')
        if len(dataset.shape) != 2:
            raise ValueError('dataset must be a two-dimensional array')
        if dataset.dtype != _numpy.float32 and dataset.dtype != _numpy.float64:
            raise ValueError('dataset must consist of floats or doubles')
        if dataset.shape[1] != self._params.dimension:
            raise ValueError('dataset dimension mismatch: {} expected, but {} found'.format(self._params.dimension, dataset.shape[1]))
        self._dataset = dataset
        if dataset.dtype == _numpy.float32:
            self._table = _internal.construct_table_dense_float(dataset, self._params)
        else:
            self._table = _internal.construct_table_dense_double(dataset, self._params)

    def _check_built(self):
        if self._dataset is None or self._table is None:
            raise RuntimeError('LSH table is not built (use fit())')

    def _check_query(self, query):
        if not isinstance(query, _numpy.ndarray):
            raise TypeError('query must be an instance of numpy.ndarray')
        if len(query.shape) != 1:
            raise ValueError('query must be one-dimensional')
        if self._dataset.dtype != query.dtype:
            raise ValueError('dataset and query must have the same dtype')
        if query.shape[0] != self._params.dimension:
            raise ValueError('query dimension mismatch: {} expected, but {} found'.format(self._params.dimension, query.shape[0]))

    def find_k_nearest_neighbors(self, query, k):
        self._check_built()
        self._check_query(query)
        if k <= 0:
            raise ValueError('k must be positive rather than {}'.format(k))
        return self._table.find_k_nearest_neighbors(query, k)
        
    def find_near_neighbors(self, query, threshold):
        self._check_built()
        self._check_query(query)
        if threshold < 0:
            raise ValueError('threshold must be non-negative rather than {}'.format(threshold))
        return self._table.find_near_neighbors(query, threshold)
        
    def find_nearest_neighbor(self, query):
        self._check_built()
        self._check_query(query)
        return self._table.find_nearest_neighbor(query)
        
    def get_candidates_with_duplicates(self, query):
        self._check_built()
        self._check_query(query)
        return self._table.get_candidates_with_duplicates(query)
        
    def get_max_num_candidates(self):
        self._check_built()
        return self._table.get_max_num_candidates()
        
    def get_num_probes(self):
        self._check_built()
        return self._table.get_num_probes()
        
    def get_query_statistics(self):
        self._check_built()
        return self._table.get_query_statistics()
        
    def get_unique_candidates(self, query):
        self._check_built()
        self._check_query(query)
        return self._table.get_unique_candidates(query)
        
    def get_unique_sorted_candidates(self, query):
        self._check_built()
        self._check_query(query)
        return self._table.get_unique_sorted_candidates(query)
        
    def reset_query_statistics(self):
        self._check_built()
        self._table.reset_query_statistics()
        
    def set_max_num_candidates(self, max_num_candidates):
        self._check_built()
        if max_num_candidates < -1:
            raise ValueError('invalid max_num_candidates: {}'.format(max_num_candidates))
        self._table.set_max_num_candidates(max_num_candidates)
        
    def set_num_probes(self, num_probes):
        self._check_built()
        if num_probes < self._params.l:
            raise ValueError('number of probes must be at least the number of tables ({})'.format(self._params.l))
        self._table.set_num_probes(num_probes)
