#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "falconn/eigen_wrapper.h"
#include "falconn/lsh_nn_table.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::fixed;
using std::scientific;
using std::unique_ptr;
using std::vector;

using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
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
  decltype(high_resolution_clock::now()) start_time;
};

template <typename PointType>
void run_experiment(LSHNearestNeighborTable<PointType>* table,
                    const std::vector<PointType> queries,
                    const std::vector<int> true_nns, double* avg_query_time,
                    double* success_probability) {
  double average_query_time_outside = 0.0;
  int num_correct = 0;

  for (int ii = 0; ii < static_cast<int>(queries.size()); ++ii) {
    Timer query_time;

    int32_t res = table->find_nearest_neighbor(queries[ii]);

    average_query_time_outside += query_time.elapsed_seconds();
    if (res == true_nns[ii]) {
      num_correct += 1;
    }
  }

  average_query_time_outside /= queries.size();
  *avg_query_time = average_query_time_outside;
  *success_probability = static_cast<double>(num_correct) / queries.size();
  cout << "Average query time (measured outside): " << scientific
       << average_query_time_outside << " seconds" << endl;
  cout << "Empirical success probability: " << fixed << *success_probability
       << endl
       << endl;
  cout << "Query statistics:" << endl;
  QueryStatistics stats = table->get_query_statistics();
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
    int n = 1000000;                  // number of data points
    int d = 128;                      // dimension
    int num_queries = 1000;           // number of query points
    double r = std::sqrt(2.0) / 2.0;  // distance to planted query
    uint64_t seed = 119417657;

    // Common LSH parameters
    int num_tables = 10;
    int num_setup_threads = 0;
    StorageHashTable storage_hash_table = StorageHashTable::FlatHashTable;
    DistanceFunction distance_function = DistanceFunction::NegativeInnerProduct;

    cout << sepline << endl;
    cout << "FALCONN C++ random data benchmark" << endl;
    cout << "Data set parameters: " << endl;
    cout << "n = " << n << endl;
    cout << "d = " << d << endl;
    cout << "num_queries = " << num_queries << endl;
    cout << "r = " << r << endl;
    cout << "seed = " << seed << endl << sepline << endl;

    std::mt19937_64 gen(seed);
    std::normal_distribution<float> dist_normal(0.0, 1.0);
    std::uniform_int_distribution<int> dist_uniform(0, n - 1);

    // Generate random data
    cout << "Generating data set ..." << endl;
    std::vector<Vec> data;
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
    std::vector<Vec> queries;
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
      double beta = std::sqrt(1.0 - alpha * alpha);
      q = alpha * q + beta * dir;

      queries.push_back(q);
    }

    // Compute true nearest neighbors
    cout << "Computing true nearest neighbors via a linear scan ..." << endl;
    std::vector<int> true_nn(num_queries);
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

    cout << "Hyperplane hash" << endl << endl;

    Timer hp_construction;

    unique_ptr<LSHNearestNeighborTable<Vec>> hptable(
        std::move(construct_table<Vec>(data, params_hp)));
    hptable->set_num_probes(2464);

    double hp_construction_time = hp_construction.elapsed_seconds();

    cout << "k = " << params_hp.k << endl;
    cout << "l = " << params_hp.l << endl;
    cout << "Number of probes = " << hptable->get_num_probes() << endl;
    cout << "Construction time: " << hp_construction_time << " seconds" << endl
         << endl;

    double hp_avg_time;
    double hp_success_prob;
    run_experiment(hptable.get(), queries, true_nn, &hp_avg_time,
                   &hp_success_prob);
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

    cout << "Cross polytope hash" << endl << endl;

    Timer cp_construction;

    unique_ptr<LSHNearestNeighborTable<Vec>> cptable(
        std::move(construct_table<Vec>(data, params_cp)));
    cptable->set_num_probes(896);

    double cp_construction_time = cp_construction.elapsed_seconds();

    cout << "k = " << params_cp.k << endl;
    cout << "last_cp_dim = " << params_cp.last_cp_dimension << endl;
    cout << "num_rotations = " << params_cp.num_rotations << endl;
    cout << "l = " << params_cp.l << endl;
    cout << "Number of probes = " << cptable->get_num_probes() << endl;
    cout << "Construction time: " << cp_construction_time << " seconds" << endl
         << endl;

    double cp_avg_time;
    double cp_success_prob;
    run_experiment(cptable.get(), queries, true_nn, &cp_avg_time,
                   &cp_success_prob);

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
  } catch (std::exception& e) {
    cerr << "exception: " << e.what() << endl;
    return 1;
  } catch (...) {
    cerr << "Unknown error" << endl;
    return 1;
  }
  return 0;
}
