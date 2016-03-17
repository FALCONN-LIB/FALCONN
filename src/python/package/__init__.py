"""Python wrapper for FALCONN.

This is a Python wrapper for [FALCONN](http://falconn-lib.org/) that is
meant to be easy to use. It exposes classes `LSHIndex`,
`LSHConstructionParameters` and `QueryStatistics` together with
helper functions `get_default_parameters()` and
`compute_number_of_hash_functions()`.

For now, the Python wrapper supports only _static dense_ datasets,
but more to come. Also, note that FALCONN is currently not thread-safe.

FALCONN is based on Locality-Sensitive Hashing (LSH), which is briefly
covered [here](https://github.com/FALCONN-LIB/FALCONN/wiki/LSH-Primer)
and [here](https://github.com/FALCONN-LIB/FALCONN/wiki/LSH-Families).

The main class is `LSHIndex`, which takes a dataset and builds an LSH
data structure. The dataset is represented as a two-dimensional
[NumPy](http://www.numpy.org/) array with data points being _rows_.
Since FALCONN uses NumPy arrays
only as a convenient way to store and pass around data, it does not
matter if NumPy is compiled with a fancy BLAS library: this has zero
effect on the performance of FALCONN.

To construct an instance of `LSHIndex`, one needs to prepare an instance
of `LSHConstructionParameters`, which stores parameters used to build the
data structure. To get a sense about the parameters used, see
[here](https://github.com/FALCONN-LIB/FALCONN/wiki/How-to-Use-FALCONN).
To get a reasonable setting of parameters, one can (and should!) use
two helper functions we provide: `get_default_parameters()` and
`compute_number_of_hash_functions()`.

Besides the documentation, we provide two examples:

* [here](https://github.com/FALCONN-LIB/FALCONN/tree/master/src/benchmark)
the LSH data structure for a random dataset is built and used;
* [here](https://github.com/FALCONN-LIB/FALCONN/tree/master/examples/glove)
we show how to use LSH to perform similarity search on a GloVe dataset.

An intended workflow is as follows:

* first, use `get_default_parameters()` to get a reasonable setting of
parameters; (later, tune them if necessary);
* second, create an instance of `LSHIndex` passing the
`LSHConstructionParameters` object we've just built;
* third, call `setup()` method of the `LSHIndex` object,
passing the dataset;
* then, increase the number of table probes using the `set_num_probes()`
method until you get the desired accuracy on a sample set of queries;
* finally, use the other methods of `LSHIndex` to execute queries.

A couple of useful tricks:

* Cast your dataset to `numpy.float32` (by doing
`dataset = dataset.astype(numpy.float32)`);
* Center your dataset (by doing
`dataset -= numpy.mean(dataset, axis=0)`) and queries:
this has no effect on the correctness, but greatly improves the
performance of our LSH families.
"""
import numpy as _numpy
from . import internal as _internal

def get_default_parameters(num_points, dimension, distance='euclidean_squared', is_sufficiently_random=False):
    """Get parameters for `LSHIndex` for _reasonable_ datasets.

    This function returns an instance of `LSHConstructionParameters` that
    is good enough for reasonable datasets. Here we try to achieve fast
    preprocessing, so if you are willing to build index for longer
    (and spend more memory) while reducing the query time, you need to
    increase the number of tables (`l`) in the resulting object.
    
    The parameters returned are especially well-suited for _centered_
    datasets (when the mean is zero).

    Arguments:

    * `num_points`: the number of points in a dataset;
    * `dimension`: the dimension of a dataset;
    * `distance` (default `'euclidean_squared'`): the distance function
    used: can be either
    `'euclidean_squared'` (which corresponds to the vanilla Euclidean
    distance) and `'negative_inner_product'` (which corresponds to
    maximizing the dot product); if you want to perform similarity
    search with respect to the cosine similarity, normalize your
    dataset, re-center it and then use `'euclidean_squared'`;
    * `is_sufficiently_random` (default `False`): set to `True` if only
    very few coordinates are zeros; in this case one is able to speed
    things up a little bit.
    """
    return _internal.get_default_parameters(num_points, dimension, distance, is_sufficiently_random)

def compute_number_of_hash_functions(num_bits, params):
    """Modify `params` such that each hash table has `2^num_bits` bins.

    Modifies `params` so that the resulting hash functions
    effectively consist of `num_bits` bits.
    The input object `params` of type `LSHConstructionParameters` must
    contain valid values for the following fields:

    * `lsh_family`;
    * `dimension` (for the cross polytope hash).

    The function will then set the following fields of `params`:

    * `k`;
    * `last_cp_dimension` (for the cross polytope hash).

    Arguments:

    * `num_bits`: the desired number of bits (the number of bins will be
    `2^num_bits`)
    * `params`: an object of the type `LSHConstructionParameters` with
    fields partially set as described above
    """
    _internal.compute_number_of_hash_functions(num_bits, params)

class LSHConstructionParameters(_internal.LSHConstructionParameters):
    """ Construction parameters for the LSH data structure.

    Contains the parameters for constructing the LSH data structures.
    Not all fields are necessary for all types of LSH. One can
    use `get_default_parameters()` and `computer_number_of_hash_functions()`
    to build an instance of `LSHConstructionParameters`.
    
    Required parameters:

    * `dimension`: dimension of the points. Required for all the hash families;
    * `lsh_family`: hash family. Required for all the hash families. Can be either
    `'hyperplane'` or `'cross_polytope'`;
    * `distance_function`: distance function. Required for all the hash families.
    Can be either `'euclidean_squared'` or `'negative_inner_product'`;
    * `k`: number of hash functions per table. Required for all the hash families.
    * `l`: number of hash tables. Required for all the hash families;
    * `storage_hash_table`: how the low-level hash tables are stored. Required for
    all the hash families. Can be equal to: `'flat_hash_table'`,
    `'bit_packed_flat_hash_table'`, `'stl_hash_table'` or
    `'linear_probing_hash_table'`.
    * `num_setup_threads`: number of threads used to set up the hash table.
    Required for all the hash families.
    Zero indicates that FALCONN should use the maximum number of available
    hardware threads (or `1` if this number cannot be determined).
    The number of threads used is always at most the number of tables `l`.

    Optional parameters:

    * `last_cp_dimension`: dimension of the last of the `k`
    cross-polytopes. Required only for the cross-polytope hash;
    * `num_rotations`: number of pseudo-random rotations. Required
    only for the cross-polytope hash. For data with lots of zeros,
    it is recommended to set num_rotations to `2`. For sufficiently
    random data, `1` rotation usually suffices;
    * `feature_hashing_dimension`: intermediate dimension for feature
    hashing of sparse data. Ignored for the hyperplane hash. A smaller
    feature hashing dimension leads to faster hash computations, but the
    quality of the hash also degrades. The value `-1` indicates that no
    feature hashing is performed (default `-1`).
    * `seed` (default `409556018`): randomness seed.
    """
    pass

class QueryStatistics(_internal.QueryStatistics):
    """Query statistics of the LSH data structure.

    Can be retrieved using `get_query_satistics()` method
    of `LSHIndex`.

    Has the following fields:

    * `average_distance_time`: average time spent computing distances;
    * `average_hash_table_time`: average time spent probing the low-level hash tables;
    * `average_lsh_time`: average time spent evaluating LSH functions;
    * `average_num_candidates`: average number of retrieved candidates;
    * `average_num_unique_candidates`: average number of retrieved _unique_ candidates;
    * `average_total_query_time`: average overall query time.
    """
    pass

class LSHIndex:
    """The main class that represents the LSH data structure.

    To construct an instance of `LSHIndex`, one needs to pass an instance
    of `LSHConstructionParameters`. During the construction, the
    parameters are not (yet) checked for correctness.

    After creating an instance of `LSHIndex`, one needs to call
    `setup()` passing a dataset. A dataset must be a two-dimensional
    NumPy array with dtype `numpy.float32` or `numpy.float64`. Rows
    of the array are interpreted as data points. Thus, the second
    dimension of the array must match the dimension from parameters.

    We recommend converting a dataset to `numpy.float32` and centering
    it: both tricks usually improve the performance a lot.

    After building the LSH data structures, one needs to set the number
    of probes via `set_num_probes()` to achieve the desired accuracy
    (the more probes one does the better accuracy is). This can be done
    empirically on a sample set of queries using binary search.

    Then, one can use the member functions:

    * `find_k_nearest_neighbors()`
    * `find_near_neighbors()`
    * `find_nearest_neighbor()`
    * `get_candidates_with_duplicates()`
    * `get_unique_candidates()`

    to execute queries.

    In short, the LSH data structure works as follows. First, it generates
    a (hopefully, short) list of candidate points, and then filters this list
    according to the distance. `LSHIndex` provides low-level functions
    that just generate the list of candidates:

    * `get_candidates_with_duplicates()`
    * `get_unique_candidates()`

    as well as high-level functions that filter this list for you:

    * `find_nearest_neighbor()`
    * `find_k_nearest_neighbors()`
    * `find_near_neighbors()`

    In addition to this, the classes exposes the
    following helper functions:

    * `get_max_num_candidates()`
    * `get_num_probes()`
    * `get_query_statistics()`
    * `reset_query_statistics()`
    * `set_max_num_candidates()`.

    See their respective documentation for help.
    """
    
    def __init__(self, params):
        """Initialize with an instance of `LSHConstructionParameters`.

        Arguments:

        * `params`: an instance of `LSHConstructionParameters`
        """
        #TODO check params for correctness
        self._params = params
        self._dataset = None
        self._table = None

    def setup(self, dataset):
        """Build the LSH data structure from a given dataset.
        
        The method builds the LSH data structure using the parameters
        passed during the construction and a given dataset (stored as
        `self._params`).

        A dataset must be a two-dimensional
        NumPy array with `dtype` `numpy.float32` or `numpy.float64`.
        The rows
        of the array are interpreted as data points. Thus, the second
        dimension of the array must match the dimension from parameters
        (`self._params.dimension`).

        An important caveat: **DON'T DELETE THE DATASET WHILE USING
        `LSHIndex`**. This can lead to silent crashes and can be very
        confusing.

        Arguments:

        * `dataset`: a two-dimensional NumPy array with dtype
        `numpy.float32` or `numpy.float64`; the second dimension must match
        dimension from the LSH parameters (`self._params.dimension`)
        """
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
            raise RuntimeError('LSH table is not built (use setup())')

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
        """Retrieve the closest `k` neighbors to `query`.

        Find the keys of the `k` closest candidates in the probing
        sequence for query. The keys are returned in order of
        increasing distance to query.

        Arguments:

        * `query`: a query given as a one-dimension NumPy array of the same
        `dtype` as the dataset; the dimension of `query` much match
        the second dimension of the dataset;
        * `k`: the number of closest candidates to retrieve.
        """
        self._check_built()
        self._check_query(query)
        if k <= 0:
            raise ValueError('k must be positive rather than {}'.format(k))
        return self._table.find_k_nearest_neighbors(query, k)
        
    def find_near_neighbors(self, query, threshold):
        """Find all the points within `threshold` distance from `query`.

        Returns the keys corresponding to candidates in the probing
        sequence for query that have distance to `query` at most `threshold`.

        Arguments:

        * `query`: a query given as a one-dimension NumPy array of the same
        `dtype` as the dataset; the dimension of `query` much match
        the second dimension of the dataset;
        * `threshold`: a distance threshold up to which we enumerate
        the candidates; note that it can be negative, and for the distance
        function `'negative_inner_product'` it actually makes sense.
        """
        self._check_built()
        self._check_query(query)
        return self._table.find_near_neighbors(query, threshold)
        
    def find_nearest_neighbor(self, query):
        """Find the key of the closest candidate.
        
        Finds the key of the closest candidate in the probing sequence
        for a query.

        Arguments:

        * `query`: a query given as a one-dimension NumPy array of the same
        `dtype` as the dataset; the dimension of `query` much match
        the second dimension of the dataset.
        """
        self._check_built()
        self._check_query(query)
        return self._table.find_nearest_neighbor(query)
        
    def get_candidates_with_duplicates(self, query):
        """Retrieve all the candidates for a given query.

        Returns the keys of all candidates in the probing sequence for
        query. If a candidate key is found in multiple tables, it will
        appear multiple times in the result. The candidates are
        returned in the order in which they appear in the probing
        sequence.

        Arguments:

        * `query`: a query given as a one-dimension NumPy array of the same
        `dtype` as the dataset; the dimension of `query` much match
        the second dimension of the dataset.
        """
        self._check_built()
        self._check_query(query)
        return self._table.get_candidates_with_duplicates(query)
        
    def get_max_num_candidates(self):
        """Get the maximum number of candidates considered in each query."""
        self._check_built()
        return self._table.get_max_num_candidates()
        
    def get_num_probes(self):
        """Get the number of probes used for each query."""
        self._check_built()
        return self._table.get_num_probes()
        
    def get_query_statistics(self):
        """Return the query statistics.

        Returns an instance of `QueryStatistics`
        that summarizes the statistics for queries
        made so far (after the last `reset_query_statistics()`
        or the construction).
        """
        self._check_built()
        return self._table.get_query_statistics()
        
    def get_unique_candidates(self, query):
        """Retrieve all the candidates (each at most once) for a query.

        Returns the keys of all candidates in the probing sequence for
        query. If a candidate key is found in multiple tables, it will
        appear once in the result.

        Arguments:

        * `query`: a query given as a one-dimension NumPy array of the same
        `dtype` as the dataset; the dimension of `query` much match
        the second dimension of the dataset.
        """
        self._check_built()
        self._check_query(query)
        return self._table.get_unique_candidates(query)
                
    def reset_query_statistics(self):
        """Reset the query statistics."""
        self._check_built()
        self._table.reset_query_statistics()
        
    def set_max_num_candidates(self, max_num_candidates=-1):
        """Set the maximum number of candidates considered in each query.

        The value `-1` indicates that all candidates retrieved
        in the probing sequence should be considered. This is the
        default and usually a good setting. A maximum number of
        candidates is mainly useful to give worst case running time
        guarantees for every query.

        Arguments:

        * `max_num_candidates` (default `-1`): the maximum number of
        candidates.
        """
        self._check_built()
        if max_num_candidates < -1:
            raise ValueError('invalid max_num_candidates: {}'.format(max_num_candidates))
        self._table.set_max_num_candidates(max_num_candidates)
        
    def set_num_probes(self, num_probes):
        """Set the number of probes used for each query.

        The default setting is `self._params.l` (the number of hash tables),
        which
        effectively disables multiprobing (the probing sequence only
        contains a single candidate per table).

        Arguments:

        * `num_probes`: the total number of probes per query.
        """
        self._check_built()
        if num_probes < self._params.l:
            raise ValueError('number of probes must be at least the number of tables ({})'.format(self._params.l))
        self._table.set_num_probes(num_probes)
