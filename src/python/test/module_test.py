import falconn
import numpy as np

def test_lsh_index_positive():
  n = 1000
  d = 128
  p = falconn.get_default_parameters(n, d)
  t = falconn.LSHIndex(p)
  dataset = np.random.randn(n, d).astype(np.float32)
  t.setup(dataset)
  u = np.random.randn(d).astype(np.float32)
  t.find_k_nearest_neighbors(u, 10)
  t.find_near_neighbors(u, 10.0)
  t.find_nearest_neighbor(u)
  t.get_candidates_with_duplicates(u)
  t.get_max_num_candidates()
  t.get_num_probes()
  t.get_query_statistics()
  t.get_unique_candidates(u)
  #t.get_unique_sorted_candidates(u)
  t.reset_query_statistics()
  t.set_max_num_candidates(100)
  t.set_num_probes(10)

def test_lsh_index_negative():
  n = 1000
  d = 128
  p = falconn.get_default_parameters(n, d)
  t = falconn.LSHIndex(p)
  try:
    t.find_nearest_neighbor(np.random.randn(d))
    assert False
  except RuntimeError:
    pass
  try:
    dataset = [[1.0, 2.0], [3.0, 4.0]]
    t.setup(dataset)
    assert False
  except TypeError:
    pass
  try:
    dataset = np.random.randn(n, d).astype(np.int32)
    t.setup(dataset)
    assert False
  except ValueError:
    pass
  try:
    dataset = np.random.randn(10, 10, 10)
    t.setup(dataset)
    assert False
  except ValueError:
    pass
  dataset = np.random.randn(n, d).astype(np.float32)
  t.setup(dataset)
  dataset = np.random.randn(n, d).astype(np.float64)
  t.setup(dataset)
  u = np.random.randn(d).astype(np.float64)
  
  try:
    t.find_k_nearest_neighbors(u, 0.5)
    assert False
  except TypeError:
    pass

  try:
    t.find_k_nearest_neighbors(u, -1)
    assert False
  except ValueError:
    pass
  
  t.find_near_neighbors(u, -1)
  
  try:
    t.set_max_num_candidates(0.5)
    assert False
  except TypeError:
    pass
  try:
    t.set_max_num_candidates(-10)
    assert False
  except ValueError:
    pass
  t.set_num_probes(t._params.l)
  try:
    t.set_num_probes(t._params.l - 1)
    assert False
  except ValueError:
    pass
  try:
    t.set_num_probes(1000.1)
    assert False
  except TypeError:
    pass

  def check_check_query(f):
    try:
      f(u.astype(np.float32))
      assert False
    except ValueError:
      pass
    try:
      f([0.0] * d)
      assert False
    except TypeError:
      pass
    try:
      f(u[:d-1])
      assert False
    except ValueError:
      pass
    try:
      f(np.random.randn(d, d))
      assert False
    except ValueError:
      pass

  check_check_query(lambda u: t.find_k_nearest_neighbors(u, 10))
  check_check_query(lambda u: t.find_near_neighbors(u, 0.5))
  check_check_query(lambda u: t.find_nearest_neighbor(u))
  check_check_query(lambda u: t.get_candidates_with_duplicates(u))
  check_check_query(lambda u: t.get_unique_candidates(u))
  #check_check_query(lambda u: t.get_unique_sorted_candidates(u))
  t.find_near_neighbors(u, 0.0)
