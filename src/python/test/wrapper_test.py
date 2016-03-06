import sys
sys.path.append('python_swig')

import falconn
import numpy as np

def test_number_of_hash_functions():
  params = falconn.LSHConstructionParameters()
  
  params.lsh_family = 'hyperplane'
  params.dimension = 10
  falconn.compute_number_of_hash_functions(5, params)
  assert params.k == 5
  
  params.lsh_family = 'cross_polytope'
  falconn.compute_number_of_hash_functions(5, params)
  assert params.k == 1
  assert params.last_cp_dimension == 16

  params.dimension = 100
  params.lsh_family = 'hyperplane'
  falconn.compute_number_of_hash_functions(8, params)
  assert params.k == 8
  
  params.lsh_family = 'cross_polytope'
  falconn.compute_number_of_hash_functions(8, params)
  assert params.k == 1
  assert params.last_cp_dimension == 128

  falconn.compute_number_of_hash_functions(10, params)
  assert params.k == 2
  assert params.last_cp_dimension == 2


def test_get_default_parameters():
  n = 100000
  dim = 128
  dist_func = 'negative_inner_product'
  params = falconn.get_default_parameters(n, dim, dist_func, True)
  assert params.l == 10
  assert params.lsh_family == 'cross_polytope'
  assert params.storage_hash_table == 'bit_packed_flat_hash_table'
  assert params.num_setup_threads == 0
  assert params.k == 2
  assert params.dimension == dim
  assert params.distance_function == dist_func
  assert params.num_rotations == 1
  assert params.last_cp_dimension == 64
