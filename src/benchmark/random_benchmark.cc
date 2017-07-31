#include "falconn/lsh_nn_table.h"

#include <Eigen/Dense>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>

using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::fixed;
using std::mt19937_64;
using std::normal_distribution;
using std::scientific;
using std::sqrt;
using std::thread;
using std::uniform_int_distribution;
using std::unique_ptr;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborQueryPool;
using falconn::LSHNearestNeighborTable;
using falconn::QueryStatistics;
using falconn::StorageHashTable;

typedef falconn::DenseVector<float> Vec;

class Timer {
 public:
  Timer() { start_time = high_resolution_clock::now(); }

  double elapsed_seconds() {
    auto end_time = high_resolution_clock::now();
    auto elapsed = duration_cast<duration<double>>(end_time - start_time);
    return elapsed.count();
  }

 private:
  high_resolution_clock::time_point start_time;
};

template <typename PointType>
void thread_function(LSHNearestNeighborQueryPool<PointType>* query_pool,
                     const vector<PointType>& queries,
                     const vector<int>& true_nns, int query_index_start,
                     int query_index_end, int* num_correct_in_thread,
                     double* total_query_time_outside_in_thread) {
  for (int ii = query_index_start; ii < query_index_end; ++ii) {
    Timer query_time;

    int32_t res = query_pool->find_nearest_neighbor(queries[ii]);

    *total_query_time_outside_in_thread += query_time.elapsed_seconds();
    if (res == true_nns[ii]) {
      *num_correct_in_thread += 1;
    }
  }
}

template <typename PointType>
void run_experiment(LSHNearestNeighborTable<PointType>* table,
                    const vector<PointType>& queries,
                    const vector<int>& true_nns, int num_probes,
                    int num_threads, double* avg_query_time,
                    double* success_probability) {
  unique_ptr<LSHNearestNeighborQueryPool<PointType>> query_pool(
      table->construct_query_pool(num_probes));
  vector<int> num_correct_per_thread(num_threads, 0);
  vector<double> total_query_time_outside_per_thread(num_threads, 0.0);
  vector<int> index_start(num_threads, 0);
  vector<int> index_end(num_threads, 0);

  int queries_per_thread = queries.size() / num_threads;
  int remainder = queries.size() % num_threads;
  int last_end = 0;
  for (int ii = 0; ii < num_threads; ++ii) {
    index_start[ii] = last_end;
    index_end[ii] = last_end + queries_per_thread;
    if (ii < remainder) {
      index_end[ii] += 1;
    }
    last_end = index_end[ii];
  }

  vector<thread> threads;
  Timer total_time;

  for (int ii = 0; ii < num_threads; ++ii) {
    threads.push_back(thread(thread_function<PointType>, query_pool.get(),
                             cref(queries), cref(true_nns), index_start[ii],
                             index_end[ii], &(num_correct_per_thread[ii]),
                             &(total_query_time_outside_per_thread[ii])));
  }
  for (int ii = 0; ii < num_threads; ++ii) {
    threads[ii].join();
  }

  double total_computation_time = total_time.elapsed_seconds();

  double average_query_time_outside = 0.0;
  *success_probability = 0.0;
  for (int ii = 0; ii < num_threads; ++ii) {
    *success_probability += num_correct_per_thread[ii];
    average_query_time_outside += total_query_time_outside_per_thread[ii];
  }
  *success_probability /= queries.size();
  average_query_time_outside /= queries.size();
  *avg_query_time = average_query_time_outside;

  cout << "Total experiment wall clock time: " << scientific
       << total_computation_time << " seconds" << endl;
  cout << "Average query time (measured outside): " << scientific
       << average_query_time_outside << " seconds" << endl;
  cout << "Empirical success probability: " << fixed << *success_probability
       << endl
       << endl;
  cout << "Query statistics:" << endl;
  QueryStatistics stats = query_pool->get_query_statistics();
  cout << "Average total query time: " << scientific
       << stats.average_total_query_time << " seconds" << endl;
  cout << "Average LSH time:         " << stats.average_lsh_time << " seconds"
       << endl;
  cout << "Average hash table time:  " << stats.average_hash_table_time
       << " seconds" << endl;
  cout << "Average distance time:    " << stats.average_distance_time
       << " seconds" << endl;
  cout << "Average number of candidates:        " << fixed
       << stats.average_num_candidates << endl;
  cout << "Average number of unique candidates: "
       << stats.average_num_unique_candidates << endl
       << endl;
  cout << "Diagnostics:" << endl;
  double threading_imbalance =
      total_computation_time -
      average_query_time_outside * queries.size() / num_threads;
  cout << "Threading imbalance (total_wall_clock_time - sum of query times "
       << "outside / num_threads): " << threading_imbalance << " seconds ("
       << 100.0 * threading_imbalance / total_computation_time
       << " % of the total wall clock time)" << endl;
  double mismatch = average_query_time_outside - stats.average_total_query_time;
  cout << "Outside - inside average total query time: " << scientific
       << mismatch << " seconds (" << fixed
       << 100.0 * mismatch / average_query_time_outside << " %)" << endl;
  double unaccounted = stats.average_total_query_time - stats.average_lsh_time -
                       stats.average_hash_table_time -
                       stats.average_distance_time;
  cout << "Unaccounted inside query time: " << scientific << unaccounted
       << " seconds (" << fixed
       << 100.0 * unaccounted / stats.average_total_query_time << " %)" << endl;
}

int main() {
  try {
    const char* sepline =
        "----------------------------------------------------------------------"
        "-";

    // Data set parameters
    int n = 1000000;             // number of data points
    int d = 128;                 // dimension
    int num_queries = 1000;      // number of query points
    double r = sqrt(2.0) / 2.0;  // distance to planted query
    uint64_t seed = 119417657;

    // Common LSH parameters
    int num_tables = 10;
    int num_setup_threads = 0;
    // TODO: make this a program argument (should we use a parsing library?)
    int num_query_threads = 1;
    StorageHashTable storage_hash_table = StorageHashTable::FlatHashTable;
    DistanceFunction distance_function = DistanceFunction::NegativeInnerProduct;

    cout << sepline << endl;
    cout << "FALCONN C++ random data benchmark" << endl << endl;
    cout << "std::thread::hardware_concurrency(): "
         << thread::hardware_concurrency() << endl;
    cout << "num_query_threads = " << num_query_threads << endl << endl;
    cout << "Data set parameters: " << endl;
    cout << "n = " << n << endl;
    cout << "d = " << d << endl;
    cout << "num_queries = " << num_queries << endl;
    cout << "r = " << r << endl;
    cout << "seed = " << seed << endl << sepline << endl;

    mt19937_64 gen(seed);
    normal_distribution<float> dist_normal(0.0, 1.0);
    uniform_int_distribution<int> dist_uniform(0, n - 1);

    // Generate random data
    cout << "Generating data set ..." << endl;
    vector<Vec> data;
    for (int ii = 0; ii < n; ++ii) {
      Vec v(d);
      for (int jj = 0; jj < d; ++jj) {
        v[jj] = dist_normal(gen);
      }
      v.normalize();
      data.push_back(v);
    }

    // Generate queries
    cout << "Generating queries ..." << endl << endl;
    vector<Vec> queries;
    for (int ii = 0; ii < num_queries; ++ii) {
      Vec q(d);
      q = data[dist_uniform(gen)];

      Vec dir(d);
      for (int jj = 0; jj < d; ++jj) {
        dir[jj] = dist_normal(gen);
      }
      dir.normalize();
      dir = dir - dir.dot(q) * q;
      dir.normalize();
      double alpha = 1.0 - r * r / 2.0;
      double beta = sqrt(1.0 - alpha * alpha);
      q = alpha * q + beta * dir;

      queries.push_back(q);
    }

    // Compute true nearest neighbors
    cout << "Computing true nearest neighbors via a linear scan ..." << endl;
    vector<int> true_nn(num_queries);
    double average_scan_time = 0.0;
    for (int ii = 0; ii < num_queries; ++ii) {
      const Vec& q = queries[ii];

      Timer query_time;

      int best_index = 0;
      float best_ip = q.dot(data[0]);
      for (int jj = 1; jj < n; ++jj) {
        float cur_ip = q.dot(data[jj]);
        if (cur_ip > best_ip) {
          best_index = jj;
          best_ip = cur_ip;
        }
      }
      true_nn[ii] = best_index;

      average_scan_time += query_time.elapsed_seconds();
    }
    average_scan_time /= num_queries;
    cout << "Average query time: " << average_scan_time << " seconds" << endl
         << sepline << endl;

    // Hyperplane hashing
    LSHConstructionParameters params_hp;
    params_hp.dimension = d;
    params_hp.lsh_family = LSHFamily::Hyperplane;
    params_hp.distance_function = distance_function;
    params_hp.storage_hash_table = storage_hash_table;
    params_hp.k = 19;
    params_hp.l = num_tables;
    params_hp.num_setup_threads = num_setup_threads;
    params_hp.seed = seed ^ 833840234;
    int num_probes_hp = 2464;

    cout << "Hyperplane hash" << endl << endl;

    Timer hp_construction;

    unique_ptr<LSHNearestNeighborTable<Vec>> hptable(
        std::move(construct_table<Vec>(data, params_hp)));

    double hp_construction_time = hp_construction.elapsed_seconds();

    cout << "k = " << params_hp.k << endl;
    cout << "l = " << params_hp.l << endl;
    cout << "Number of probes = " << num_probes_hp << endl;
    cout << "Construction time: " << hp_construction_time << " seconds" << endl
         << endl;

    double hp_avg_time;
    double hp_success_prob;
    run_experiment(hptable.get(), queries, true_nn, num_probes_hp,
                   num_query_threads, &hp_avg_time, &hp_success_prob);
    cout << sepline << endl;
    hptable.reset(nullptr);

    // Cross polytope hashing
    LSHConstructionParameters params_cp;
    params_cp.dimension = d;
    params_cp.lsh_family = LSHFamily::CrossPolytope;
    params_cp.distance_function = distance_function;
    params_cp.storage_hash_table = storage_hash_table;
    params_cp.k = 3;
    params_cp.l = num_tables;
    params_cp.last_cp_dimension = 16;
    params_cp.num_rotations = 3;
    params_cp.num_setup_threads = num_setup_threads;
    params_cp.seed = seed ^ 833840234;
    int num_probes_cp = 896;

    cout << "Cross polytope hash" << endl << endl;

    Timer cp_construction;

    unique_ptr<LSHNearestNeighborTable<Vec>> cptable(
        move(construct_table<Vec>(data, params_cp)));

    double cp_construction_time = cp_construction.elapsed_seconds();

    cout << "k = " << params_cp.k << endl;
    cout << "last_cp_dim = " << params_cp.last_cp_dimension << endl;
    cout << "num_rotations = " << params_cp.num_rotations << endl;
    cout << "l = " << params_cp.l << endl;
    cout << "Number of probes = " << num_probes_cp << endl;
    cout << "Construction time: " << cp_construction_time << " seconds" << endl
         << endl;

    double cp_avg_time;
    double cp_success_prob;
    run_experiment(cptable.get(), queries, true_nn, num_probes_cp,
                   num_query_threads, &cp_avg_time, &cp_success_prob);

    cout << sepline << endl << "Summary:" << endl;
    cout << "Success probabilities:" << endl;
    cout << "  HP: " << fixed << hp_success_prob << endl;
    cout << "  CP: " << cp_success_prob << endl;
    cout << "Average query times (seconds):" << endl;
    cout << "  Linear scan time: " << scientific << average_scan_time << endl;
    cout << "  HP time: " << hp_avg_time << endl;
    cout << "  CP time: " << cp_avg_time << endl;
    cout << "Speed-ups:" << endl;
    cout << "  HP vs linear scan: " << fixed << average_scan_time / hp_avg_time
         << endl;
    cout << "  CP vs linear scan: " << fixed << average_scan_time / cp_avg_time
         << endl;
    cout << "  CP vs HP: " << fixed << hp_avg_time / cp_avg_time << endl;
  } catch (exception& e) {
    cerr << "exception: " << e.what() << endl;
    return 1;
  } catch (...) {
    cerr << "Unknown error" << endl;
    return 1;
  }
  return 0;
}
