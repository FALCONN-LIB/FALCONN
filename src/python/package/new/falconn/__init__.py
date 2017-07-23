import numpy as _numpy
import _falconn as _internal
from _falconn import LSHConstructionParameters, QueryStatistics, DistanceFunction, LSHFamily, StorageHashTable, get_default_parameters, compute_number_of_hash_functions

class Queriable:
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
        self._check_query(query)
        if k <= 0:
            raise ValueError('k must be positive rather than {}'.format(k))
        return self._inner_entity.find_k_nearest_neighbors(query, k)

    def find_near_neighbors(self, query, threshold):
        self._check_query(query)
        if threshold < 0:
            raise ValueError('threshold must be non-negative rather than {}'.format(threshold))
        return self._inner_entity.find_near_neighbors(query, threshold)

    def find_nearest_neighbor(self, query):
        self._check_query(query)
        return self._inner_entity.find_nearest_neighbor(query)

    def get_candidates_with_duplicates(self, query):
        self._check_query(query)
        return self._inner_entity.get_candidates_with_duplicates(query)

    def get_max_num_candidates(self):
        return self._inner_entity.get_max_num_candidates()

    def get_num_probes(self):
        return self._inner_entity.get_num_probes()

    def get_query_statistics(self):
        return self._inner_entity.get_query_statistics()

    def get_unique_candidates(self, query):
        self._check_query(query)
        return self._inner_entity.get_unique_candidates(query)

    def reset_query_statistics(self):
        self._inner_entity.reset_query_statistics()

    def set_max_num_candidates(self, max_num_candidates=-1):
        if max_num_candidates < -1:
            raise ValueError(
                'invalid max_num_candidates: {}'.format(max_num_candidates))
        self._inner_entity.set_max_num_candidates(max_num_candidates)

    def set_num_probes(self, num_probes):
        if num_probes < self._params.l:
            raise ValueError(
                'number of probes must be at least the number of tables ({})'.
                format(self._params.l))
        self._inner_entity.set_num_probes(num_probes)

class LSHIndex:
    def __init__(self, params):
        #TODO check params for correctness
        self._params = params
        self._dataset = None
        self._table = None

    def setup(self, dataset):
        if self._dataset is not None or self._table is not None:
            raise RuntimeError('setup() has already been called')
        if not isinstance(dataset, _numpy.ndarray):
            raise TypeError('dataset must be an instance of numpy.ndarray')
        if len(dataset.shape) != 2:
            raise ValueError('dataset must be a two-dimensional array')
        if dataset.dtype != _numpy.float32 and dataset.dtype != _numpy.float64:
            raise ValueError('dataset must consist of floats or doubles')
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
        self._check_built()
        return Queriable(self._table.construct_query_object(num_probes, max_num_candidates), self)

    def construct_query_pool(self, num_probes=-1, max_num_candidates=-1, num_query_objects=0):
        self._check_built()
        return Queriable(self._table.construct_query_object(num_probes, max_num_candidates, num_query_objects), self)
