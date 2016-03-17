from __future__ import print_function
from __future__ import division

import math
import sys
import timeit

import numpy as np

sys.path.append('python_swig')

import falconn

def run_experiment(table, queries, true_nns):
  average_query_time_outside = 0.0
  num_correct = 0

  for query, true_nn in zip(queries, true_nns):
    start = timeit.default_timer()
    res = table.find_nearest_neighbor(query)
    end = timeit.default_timer()
    average_query_time_outside += (end - start)
    if res == true_nn:
      num_correct += 1

  average_query_time_outside /= len(queries)
  success_probability = float(num_correct) / len(queries)
  print('Average query time (measured outside): {:e}'.format(
      average_query_time_outside))
  print('Empirical success probability: {}\n'.format(success_probability))
  print('Query statistics:')
  stats = table.get_query_statistics()
  print('Average total query time: {:e} seconds'.format(
      stats.average_total_query_time))
  print('Average LSH time:         {:e} seconds'.format(stats.average_lsh_time))
  print('Average hash table time:  {:e} seconds'.format(
      stats.average_hash_table_time))
  print('Average distance time:    {:e} seconds'.format(
      stats.average_distance_time))
  print('Average number of candidates:        {}'.format(
      stats.average_num_candidates))
  print('Average number of unique candidates: {}\n'.format(
      stats.average_num_unique_candidates))
  print('Diagnostics:')
  mismatch = average_query_time_outside - stats.average_total_query_time
  print('Outside - inside average total query time: {:e} seconds ({:%})'.format(
      mismatch, mismatch / average_query_time_outside))
  unaccounted = stats.average_total_query_time - stats.average_lsh_time \
      - stats.average_hash_table_time - stats.average_distance_time
  print('Unaccounted inside query time: {:e} seconds ({:%})'.format(unaccounted,
      unaccounted / stats.average_total_query_time))
  return average_query_time_outside, float(num_correct) / len(queries)


def gen_near_neighbor(v, r):
  rp = np.random.randn(v.size)
  rp = rp / np.linalg.norm(rp)
  rp = rp - np.dot(rp, v) * v
  rp = rp / np.linalg.norm(rp)
  alpha = 1 - r * r / 2.0
  beta = math.sqrt(1.0 - alpha * alpha)
  return alpha * v + beta * rp


def aligned(a, alignment=32):
  if (a.ctypes.data % alignment) == 0:
    return a
  extra = alignment / a.itemsize
  buf = np.empty(a.size + extra, dtype=a.dtype)
  ofs = (-buf.ctypes.data % alignment) / a.itemsize
  aa = buf[ofs:ofs+a.size].reshape(a.shape)
  np.copyto(aa, a)
  assert (aa.ctypes.data % alignment) == 0
  return aa




sepline = \
    '-----------------------------------------------------------------------'

n = 1000000
d = 128
num_queries = 1000
r = math.sqrt(2.0) / 2.0
seed = 119417657

print(sepline)
print('FALCONN Python random data benchmark')
print('Data set parameters:')
print('n = {}'.format(n))
print('d = {}'.format(d))
print('num_queries = {}'.format(num_queries))
print('r = {}'.format(r))
print('seed = {}'.format(seed))
print(sepline)

print('Generating data set ...')
np.random.seed(seed)
data = np.random.randn(n, d)
norms = np.linalg.norm(data, axis=1)
data = data / np.reshape(norms, (n, 1))
data = data.astype(np.float32)
data = aligned(data)

print('Generating queries ...\n')
queries = []
for ii in range(num_queries):
  q = gen_near_neighbor(data[np.random.randint(n)], r)
  q = aligned(q)
  queries.append(q.astype(np.float32))

print('Computing true nearest neighbors via a linear scan ...')
true_nns = []
average_scan_time = 0.0
for query in queries:
  start = timeit.default_timer()
  best_index = np.argmax(np.dot(data, query))
  stop = timeit.default_timer()
  true_nns.append(best_index)
  average_scan_time += (stop - start)
average_scan_time /= num_queries
print('Average query time: {} seconds'.format(average_scan_time))
print(sepline)

# Hyperplane hashing
params_hp = falconn.LSHConstructionParameters()
params_hp.dimension = d
params_hp.lsh_family = 'hyperplane'
params_hp.distance_function = 'negative_inner_product'
params_hp.storage_hash_table = 'flat_hash_table'
params_hp.k = 19
params_hp.l = 10
params_hp.num_setup_threads = 0
params_hp.seed = seed ^ 833840234

print('Hyperplane hash\n')

start = timeit.default_timer()
hp_table = falconn.LSHIndex(params_hp)
hp_table.setup(data)
hp_table.set_num_probes(2464)
stop = timeit.default_timer()
hp_construction_time = stop - start

print('k = {}'.format(params_hp.k))
print('l = {}'.format(params_hp.l))
print('Number of probes = {}'.format(hp_table.get_num_probes()))
print('Construction time: {} seconds\n'.format(hp_construction_time))

hp_avg_time, hp_success_prob = run_experiment(hp_table, queries, true_nns)
del hp_table
print(sepline)

# Cross polytope hashing
params_cp = falconn.LSHConstructionParameters()
params_cp.dimension = d
params_cp.lsh_family = 'cross_polytope'
params_cp.distance_function = 'negative_inner_product'
params_cp.storage_hash_table = 'flat_hash_table'
params_cp.k = 3
params_cp.l = 10
params_cp.num_setup_threads = 0
params_cp.last_cp_dimension = 16
params_cp.num_rotations = 3
params_cp.seed = seed ^ 833840234

print('Cross polytope hash\n')

start = timeit.default_timer()
cp_table = falconn.LSHIndex(params_cp)
cp_table.setup(data)
cp_table.set_num_probes(896)
stop = timeit.default_timer()
cp_construction_time = stop - start

print('k = {}'.format(params_cp.k))
print('last_cp_dim = {}'.format(params_cp.last_cp_dimension))
print('num_rotations = {}'.format(params_cp.num_rotations))
print('l = {}'.format(params_cp.l))
print('Number of probes = {}'.format(cp_table.get_num_probes()))
print('Construction time: {} seconds\n'.format(cp_construction_time))

cp_avg_time, cp_success_prob = run_experiment(cp_table, queries, true_nns)

print(sepline)
print('Summary:')
print('Success probabilities:')
print('  HP: {}'.format(hp_success_prob))
print('  CP: {}'.format(cp_success_prob))
print('Average query times (seconds):')
print('  Linear scan time: {:e}'.format(average_scan_time))
print('  HP time: {:e}'.format(hp_avg_time))
print('  CP time: {:e}'.format(cp_avg_time))
print('Speed-ups:')
print('  HP vs linear scan: {}'.format(average_scan_time / hp_avg_time))
print('  CP vs linear scan: {}'.format(average_scan_time / cp_avg_time))
print('  CP vs HP: {}'.format(hp_avg_time / cp_avg_time))
