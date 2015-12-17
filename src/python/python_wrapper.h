#ifndef __PYTHON_WRAPPER_H__
#define __PYTHON_WRAPPER_H__

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "falconn_global.h"
#include "lsh_nn_table.h"

namespace falconn {
namespace python {

class PyLSHNearestNeighborTableError : public FalconnError {
 public:
  PyLSHNearestNeighborTableError(const char* msg) : FalconnError(msg) {}
};


class PyLSHNearestNeighborTableDenseDouble {
 public:
  typedef LSHNearestNeighborTable<DenseVector<double>, int32_t> InnerTable;

  PyLSHNearestNeighborTableDenseDouble(InnerTable* table)
      : table_(table) {}
  
  void set_num_probes(int num_probes) {
    table_->set_num_probes(num_probes);
  }

  int_fast64_t get_num_probes() {
    return table_->get_num_probes();
  }
  
  void set_max_num_candidates(int_fast64_t max_num_candidates) {
    table_->set_max_num_candidates(max_num_candidates);
  }

  int_fast64_t get_max_num_candidates() {
    return table_->get_max_num_candidates();
  }
  
  int_fast64_t find_closest(const double* vec, int len) {
    //return table_->find_closest(q);
  }
  
  std::vector<int_fast64_t> find_k_nearest_neighbors(const double* vec, int len,
      int_fast64_t k) {
    std::vector<int_fast64_t> result;
    //table_->find_k_nearest_neighbors(q, k, &result);
    return result;
  }

  std::vector<int_fast64_t> find_near_neighbors(const double* vec, int len,
      double threshold) {
    std::vector<int_fast64_t> result;
    //table_->find_near_neighbors(q, threshold, &result);
    return result;
  }
  
  std::vector<int_fast64_t> get_candidates_with_duplicates(const double* vec,
      int len) {
    std::vector<int_fast64_t> result;
    //table_->get_candidates_with_duplicates(q, &result);
    return result;
  }

  std::vector<int_fast64_t> get_unique_candidates(const double* vec, int len) {
    std::vector<int_fast64_t> result;
    //table_->get_unique_candidates(q, &result);
    return result;
  }

  std::vector<int_fast64_t> get_unique_sorted_candidates(const double* vec,
      int len) {
    std::vector<int_fast64_t> result;
    //table_->get_unique_sorted_candidates(q, &result);
    return result;
  }
  
  void reset_query_statistics() {
    table_->reset_query_statistics();
  }

  falconn::QueryStatistics get_query_statistics() {
    return table_->get_query_statistics();
  }

  ~PyLSHNearestNeighborTableDenseDouble() {
    delete table_;
  }

 private:
  InnerTable* table_;
};

}  // namespace python
}  // namespace falconn

#endif
