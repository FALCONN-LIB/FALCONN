import _falconn as falconn
import numpy as np


def test_number_of_hash_functions():
    params = falconn.LSHConstructionParameters()

    params.lsh_family = falconn.LSHFamily.Hyperplane
    params.dimension = 10
    falconn.compute_number_of_hash_functions(5, params)
    assert params.k == 5

    params.lsh_family = falconn.LSHFamily.CrossPolytope
    falconn.compute_number_of_hash_functions(5, params)
    assert params.k == 1
    assert params.last_cp_dimension == 16

    params.dimension = 100
    params.lsh_family = falconn.LSHFamily.Hyperplane
    falconn.compute_number_of_hash_functions(8, params)
    assert params.k == 8

    params.lsh_family = falconn.LSHFamily.CrossPolytope
    falconn.compute_number_of_hash_functions(8, params)
    assert params.k == 1
    assert params.last_cp_dimension == 128

    falconn.compute_number_of_hash_functions(10, params)
    assert params.k == 2
    assert params.last_cp_dimension == 2


def test_get_default_parameters():
    n = 100000
    dim = 128
    dist_func = falconn.DistanceFunction.NegativeInnerProduct
    params = falconn.get_default_parameters(n, dim, dist_func, True)
    assert params.l == 10
    assert params.lsh_family == falconn.LSHFamily.CrossPolytope
    assert params.storage_hash_table == falconn.StorageHashTable.BitPackedFlatHashTable
    assert params.num_setup_threads == 0
    assert params.k == 2
    assert params.dimension == dim
    assert params.distance_function == dist_func
    assert params.num_rotations == 1
    assert params.last_cp_dimension == 64
