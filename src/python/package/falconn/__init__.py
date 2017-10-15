"""Python wrapper for FALCONN.

This is a Python wrapper for [FALCONN](http://falconn-lib.org/) that is
meant to be easy to use. It exposes classes `LSHIndex`,
`LSHConstructionParameters` and `QueryStatistics` together with
helper functions `get_default_parameters()` and
`compute_number_of_hash_functions()`.

For now, the Python wrapper supports only _static dense_ datasets,
but more to come.

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
* [here](https://github.com/FALCONN-LIB/FALCONN/tree/master/src/examples/glove)
we show how to use LSH to perform similarity search on a GloVe dataset.

An intended workflow is as follows:

* first, use `get_default_parameters()` to get a reasonable setting of
parameters; (later, tune them if necessary);
* second, create an instance of `LSHIndex` passing the
`LSHConstructionParameters` object we've just built;
* third, call `setup()` method of the `LSHIndex` object,
passing the dataset;
* then, construct a query object by calling `construct_query_object()`
method of the `LSHIndex` object;
* then, increase the number of table probes using the `set_num_probes()`
method of the query object until you get the desired accuracy on
a sample set of queries;
* finally, use the other methods of the query object
to execute queries.

If you would like to query `LSHIndex` from several threads, create
a dedicated query object per thread using `construct_query_object`.
Alternatively, one can build a query pool using
`construct_query_pool` method of `LSHIndex`, which one can query
in parallel.

A couple of useful tricks:

* Cast your dataset to `numpy.float32` (by doing
`dataset = dataset.astype(numpy.float32)`);
* Center your dataset (by doing
`dataset -= numpy.mean(dataset, axis=0)`) and queries:
this has no effect on the correctness, but greatly improves the
performance of our LSH families. Don't forget to use the Euclidean
distance in this case.
"""
import numpy as _numpy
import _falconn as _internal
from _falconn import LSHConstructionParameters, QueryStatistics, DistanceFunction, LSHFamily, StorageHashTable, get_default_parameters, compute_number_of_hash_functions


class Queryable:
    """A simple wrapper for query objects and query pools.
    
    Instances of `Queryable` are returned by the methods
    `construct_query_object` and `construct_query_pool` of `LSHIndex`.
    You are not expected to construct instances of `Queryable` directly.
    """

    def __init__(self, inner_entity, parent):
        self._inner_entity = inner_entity
        self._parent = parent

    def _check_query(self, query):
        if not isinstance(query, _numpy.ndarray):
            raise TypeError('query must be an instance of numpy.ndarray')
        if len(query.shape) != 1:
            raise ValueError('query must be one-dimensional')
        if self._parent._dataset.dtype != query.dtype:
            raise TypeError('dataset and query must have the same dtype')
        if query.shape[0] != self._parent._params.dimension:
            raise ValueError(
                'query dimension mismatch: {} expected, but {} found'.format(
                    self._parent._params.dimension, query.shape[0]))

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
        self._check_query(query)
        if k <= 0:
            raise ValueError('k must be positive rather than {}'.format(k))
        return self._inner_entity.find_k_nearest_neighbors(query, k)

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
        self._check_query(query)
        if threshold < 0:
            raise ValueError('threshold must be non-negative rather than {}'.
                             format(threshold))
        return self._inner_entity.find_near_neighbors(query, threshold)

    def find_nearest_neighbor(self, query):
        """Find the key of the closest candidate.

        Finds the key of the closest candidate in the probing sequence
        for a query.

        Arguments:

        * `query`: a query given as a one-dimension NumPy array of the same
        `dtype` as the dataset; the dimension of `query` much match
        the second dimension of the dataset.
        """
        self._check_query(query)
        return self._inner_entity.find_nearest_neighbor(query)

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
        self._check_query(query)
        return self._inner_entity.get_candidates_with_duplicates(query)

    def get_max_num_candidates(self):
        """Get the maximum number of candidates considered in each query."""
        return self._inner_entity.get_max_num_candidates()

    def get_num_probes(self):
        """Get the number of probes used for each query."""
        return self._inner_entity.get_num_probes()

    def get_query_statistics(self):
        """Return the query statistics.

        Returns an instance of `QueryStatistics`
        that summarizes the statistics for queries
        made so far (after the last `reset_query_statistics()`
        or the construction).
        """
        return self._inner_entity.get_query_statistics()

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
        self._check_query(query)
        return self._inner_entity.get_unique_candidates(query)

    def reset_query_statistics(self):
        """Reset the query statistics."""
        self._inner_entity.reset_query_statistics()

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
        if max_num_candidates < -1:
            raise ValueError(
                'invalid max_num_candidates: {}'.format(max_num_candidates))
        self._inner_entity.set_max_num_candidates(max_num_candidates)

    def set_num_probes(self, num_probes):
        """Set the number of probes used for each query.

        The default setting is `self._params.l` (the number of hash tables),
        which
        effectively disables multiprobing (the probing sequence only
        contains a single candidate per table).

        Arguments:

        * `num_probes`: the total number of probes per query.
        """
        if num_probes < self._parent._params.l:
            raise ValueError(
                'number of probes must be at least the number of tables ({})'.
                format(self._parent._params.l))
        self._inner_entity.set_num_probes(num_probes)


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

    After building the LSH data structures, one needs to construct
    a query object by calling `construct_query_object()`.
    Then, one needs to set the number
    of probes via calling the `set_num_probes()` method of the query object
    to achieve the desired accuracy
    (the more probes one does the better accuracy is). This can be done
    empirically on a sample set of queries using binary search.

    Then, one can use the following member functions of the query object:

    * `find_k_nearest_neighbors()`
    * `find_near_neighbors()`
    * `find_nearest_neighbor()`
    * `get_candidates_with_duplicates()`
    * `get_unique_candidates()`

    to execute queries.

    In short, the LSH data structure works as follows. First, it generates
    a (hopefully, short) list of candidate points, and then filters this list
    according to the distance. Query object provides low-level functions
    that just generate the list of candidates:

    * `get_candidates_with_duplicates()`
    * `get_unique_candidates()`

    as well as high-level functions that filter this list for you:

    * `find_nearest_neighbor()`
    * `find_k_nearest_neighbors()`
    * `find_near_neighbors()`

    In addition to this, the query object exposes the
    following helper functions:

    * `get_max_num_candidates()`
    * `get_num_probes()`
    * `get_query_statistics()`
    * `reset_query_statistics()`
    * `set_max_num_candidates()`.

    See their respective documentation for help.

    Alternatively, one can call the method `construct_query_pool()`
    to construct a *pool* of query objects, which can be safely queried
    in parallel. The vanilla query object is not thread-safe, so each
    thread must have its own query object.
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
        if self._dataset is not None or self._table is not None:
            raise RuntimeError('setup() has already been called')
        if not isinstance(dataset, _numpy.ndarray):
            raise TypeError('dataset must be an instance of numpy.ndarray')
        if len(dataset.shape) != 2:
            raise ValueError('dataset must be a two-dimensional array')
        if dataset.dtype != _numpy.float32 and dataset.dtype != _numpy.float64:
            raise TypeError('dataset must consist of floats or doubles')
        if dataset.shape[1] != self._params.dimension:
            raise ValueError(
                'dataset dimension mismatch: {} expected, but {} found'.format(
                    self._params.dimension, dataset.shape[1]))
        self._dataset = dataset
        if dataset.dtype == _numpy.float32:
            self._table = _internal.construct_table_dense_float(
                dataset, self._params)
        else:
            self._table = _internal.construct_table_dense_double(
                dataset, self._params)

    def _check_built(self):
        if self._dataset is None or self._table is None:
            raise RuntimeError('LSH table is not built (use setup())')

    def construct_query_object(self, num_probes=-1, max_num_candidates=-1):
        """Construct a query object.

        This method constructs and returns a query object, which can be used
        to query the LSH data structure. The query object is not thread-safe,
        for a thread-safe version, see the `construct_query_pool` method. Alternatively,
        you can construct a separate query object per thread.

        Arguments:

        * `num_probes` (default `-1`): the number of buckets the query algorithm
        probes. This number can later be modified using the `set_num_probes` method of
        the query object. The higher number of probes is, the better accuracy one gets,
        but the slower queries are. If `num_probes` is equal to `-1`, then we probe
        one bucket per (each of the `params.L`) table;
        * `max_num_candidates` (default `-1`): the maximum number of candidate points
        we retrieve. The value `-1` means that the said number is unbounded.
        """
        self._check_built()
        return Queryable(
            self._table.construct_query_object(num_probes, max_num_candidates),
            self)

    def construct_query_pool(self,
                             num_probes=-1,
                             max_num_candidates=-1,
                             num_query_objects=0):
        """Construct a pool of query objects.

        This method constructs and returns a pool of query objects, which
        can be used to query the LSH data structure from several threads.

        Arguments:

        * `num_probes` (default `-1`): the number of buckets the query algorithm
        probes. This number can later be modified using the `set_num_probes` method of
        the query object. The higher number of probes is, the better accuracy one gets,
        but the slower queries are. If `num_probes` is equal to `-1`, then we probe
        one bucket per (each of the `params.L`) table;
        * `max_num_candidates` (default `-1`): the maximum number of candidate points
        we retrieve. The value `-1` means that the said number is unbounded;
        * `num_query_objects` (default `0`): the number of query objects in the pool.
        The value `0` makes the number of query objects to be twice the number of hardware
        threads on the machine.
        """
        self._check_built()
        return Queryable(
            self._table.construct_query_pool(num_probes, max_num_candidates,
                                             num_query_objects), self)
