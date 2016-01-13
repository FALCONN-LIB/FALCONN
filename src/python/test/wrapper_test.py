import falconn
import numpy as np

def test_number_of_hash_functions():
  params = falconn._internal.LSHConstructionParameters()
  
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
  assert params.k == 2
  assert params.dimension == dim
  assert params.distance_function == dist_func
  assert params.num_rotations == 1
  assert params.last_cp_dimension == 64

def test_lsh_index_positive():
  n = 1000
  d = 128
  p = falconn.get_default_parameters(n, d)
  t = falconn.LSHIndex(p)
  dataset = np.random.randn(n, d).astype(np.float32)
  t.fit(dataset)
  u = np.random.randn(d).astype(np.float32)
  t.find_k_nearest_neighbors(u, 10)
  t.find_near_neighbors(u, 10.0)
  t.find_nearest_neighbor(u)
  t.get_candidates_with_duplicates(u)
  t.get_max_num_candidates()
  t.get_num_probes()
  t.get_query_statistics()
  t.get_unique_candidates(u)
  t.get_unique_sorted_candidates(u)
  t.reset_query_statistics()
  t.set_max_num_candidates(100)
  t.set_num_probes(10)
