#ifndef __PYTHON_WRAPPER_H__
#define __PYTHON_WRAPPER_H__

#include <algorithm>
#include <memory>
#include <string>
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
  typedef Eigen::Map<const DenseVector<double>> ConstVectorMap;

  PyLSHNearestNeighborTableDenseDouble(InnerTable* table)
      : table_(table) {}
  
  void set_num_probes(int num_probes) {
    table_->set_num_probes(num_probes);
  }

  int32_t get_num_probes() {
    return table_->get_num_probes();
  }
  
  void set_max_num_candidates(int32_t max_num_candidates) {
    table_->set_max_num_candidates(max_num_candidates);
  }

  int32_t get_max_num_candidates() {
    return table_->get_max_num_candidates();
  }
  
  int find_closest(const double* vec, int len) {
    ConstVectorMap q(vec, len);
    return table_->find_closest(q);
  }
  
  std::vector<int32_t> find_k_nearest_neighbors(const double* vec, int len,
      int32_t k) {
    ConstVectorMap q(vec, len);
    std::vector<int32_t> result;
    table_->find_k_nearest_neighbors(q, k, &result);
    return result;
  }

  std::vector<int32_t> find_near_neighbors(const double* vec, int len,
      double threshold) {
    ConstVectorMap q(vec, len);
    std::vector<int32_t> result;
    table_->find_near_neighbors(q, threshold, &result);
    return result;
  }
  
  std::vector<int32_t> get_candidates_with_duplicates(const double* vec,
      int len) {
    ConstVectorMap q(vec, len);
    std::vector<int32_t> result;
    table_->get_candidates_with_duplicates(q, &result);
    return result;
  }

  std::vector<int32_t> get_unique_candidates(const double* vec, int len) {
    ConstVectorMap q(vec, len);
    std::vector<int32_t> result;
    table_->get_unique_candidates(q, &result);
    return result;
  }

  std::vector<int32_t> get_unique_sorted_candidates(const double* vec,
      int len) {
    ConstVectorMap q(vec, len);
    std::vector<int32_t> result;
    table_->get_unique_sorted_candidates(q, &result);
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
  InnerTable* table_ = nullptr;
};



class PyLSHNearestNeighborTableDenseFloat {
 public:
  typedef LSHNearestNeighborTable<DenseVector<float>, int32_t> InnerTable;
  typedef Eigen::Map<const DenseVector<float>> ConstVectorMap;

  PyLSHNearestNeighborTableDenseFloat(InnerTable* table)
      : table_(table) {}
  
  void set_num_probes(int num_probes) {
    table_->set_num_probes(num_probes);
  }

  int32_t get_num_probes() {
    return table_->get_num_probes();
  }
  
  void set_max_num_candidates(int32_t max_num_candidates) {
    table_->set_max_num_candidates(max_num_candidates);
  }

  int32_t get_max_num_candidates() {
    return table_->get_max_num_candidates();
  }
  
  int find_closest(const float* vec, int len) {
    ConstVectorMap q(vec, len);
    return table_->find_closest(q);
  }
  
  std::vector<int32_t> find_k_nearest_neighbors(const float* vec, int len,
      int32_t k) {
    ConstVectorMap q(vec, len);
    std::vector<int32_t> result;
    table_->find_k_nearest_neighbors(q, k, &result);
    return result;
  }

  std::vector<int32_t> find_near_neighbors(const float* vec, int len,
      float threshold) {
    ConstVectorMap q(vec, len);
    std::vector<int32_t> result;
    table_->find_near_neighbors(q, threshold, &result);
    return result;
  }
  
  std::vector<int32_t> get_candidates_with_duplicates(const float* vec,
      int len) {
    ConstVectorMap q(vec, len);
    std::vector<int32_t> result;
    table_->get_candidates_with_duplicates(q, &result);
    return result;
  }

  std::vector<int32_t> get_unique_candidates(const float* vec, int len) {
    ConstVectorMap q(vec, len);
    std::vector<int32_t> result;
    table_->get_unique_candidates(q, &result);
    return result;
  }

  std::vector<int32_t> get_unique_sorted_candidates(const float* vec,
      int len) {
    ConstVectorMap q(vec, len);
    std::vector<int32_t> result;
    table_->get_unique_sorted_candidates(q, &result);
    return result;
  }
  
  void reset_query_statistics() {
    table_->reset_query_statistics();
  }

  falconn::QueryStatistics get_query_statistics() {
    return table_->get_query_statistics();
  }

  ~PyLSHNearestNeighborTableDenseFloat() {
    delete table_;
  }

 private:
  InnerTable* table_ = nullptr;
};


struct ConstructionParameters {
  int_fast32_t dimension = -1;
  std::string lsh_family = "unknown";
  std::string distance_function = "unknown";
  int_fast32_t  k = -1;
  int_fast32_t l = -1;
  uint64_t seed = 409556018;
  int_fast32_t last_cp_dimension = -1;
  int_fast32_t num_rotations = -1;
  int_fast32_t feature_hashing_dimension = -1;
};


// %ignore'd in the swig wrapper
void python_to_cpp_construction_parameters(
    const ConstructionParameters& py_params,
    falconn::LSHConstructionParameters* cpp_params) {
  cpp_params->dimension = py_params.dimension;

  std::string tmp_lsh_family = py_params.lsh_family;
  std::transform(tmp_lsh_family.begin(), tmp_lsh_family.end(),
      tmp_lsh_family.begin(), tolower);
  if (tmp_lsh_family == "unknown") {
    cpp_params->lsh_family = LSHFamily::Unknown;
  } else if (tmp_lsh_family == "hyperplane") {
    cpp_params->lsh_family = LSHFamily::Hyperplane;
  } else if (tmp_lsh_family == "crosspolytope") {
    cpp_params->lsh_family = LSHFamily::CrossPolytope;
  } else {
    throw PyLSHNearestNeighborTableError("Unknown LSH family parameter.");
  }
 
  std::string tmp_distance_function = py_params.distance_function;
  std::transform(tmp_distance_function.begin(), tmp_distance_function.end(),
      tmp_distance_function.begin(), tolower);
  if (tmp_distance_function == "unknown") {
    cpp_params->distance_function = DistanceFunction::Unknown;
  } else if (tmp_distance_function == "negativeinnerproduct") {
    cpp_params->distance_function = DistanceFunction::NegativeInnerProduct;
  } else if (tmp_distance_function == "euclideansquared") {
    cpp_params->distance_function = DistanceFunction::EuclideanSquared;
  } else {
    throw PyLSHNearestNeighborTableError("Unknown distance_function "
        "parameter.");
  }

  cpp_params->k = py_params.k;
  cpp_params->l = py_params.l;
  cpp_params->seed = py_params.seed;
  cpp_params->last_cp_dimension = py_params.last_cp_dimension;
  cpp_params->num_rotations = py_params.num_rotations;
  cpp_params->feature_hashing_dimension = py_params.feature_hashing_dimension;
}


PyLSHNearestNeighborTableDenseDouble* construct_table_dense_double(
    const double* matrix, int num_rows, int num_columns,
    const ConstructionParameters& params) {

  falconn::LSHConstructionParameters inner_params;
  python_to_cpp_construction_parameters(params, &inner_params);

  PlainArrayPointSet<double> points;
  points.data = matrix;
  points.num_points = num_rows;
  points.dimension = num_columns;

  std::unique_ptr<LSHNearestNeighborTable<DenseVector<double>, int32_t>>
      table(std::move(construct_table<DenseVector<double>, int32_t,
          PlainArrayPointSet<double>>(points, inner_params)));

  return new PyLSHNearestNeighborTableDenseDouble(table.release());
}


PyLSHNearestNeighborTableDenseFloat* construct_table_dense_float(
    const float* matrix, int num_rows, int num_columns,
    const ConstructionParameters& params) {

  falconn::LSHConstructionParameters inner_params;
  python_to_cpp_construction_parameters(params, &inner_params);

  PlainArrayPointSet<float> points;
  points.data = matrix;
  points.num_points = num_rows;
  points.dimension = num_columns;

  std::unique_ptr<LSHNearestNeighborTable<DenseVector<float>, int32_t>>
      table(std::move(construct_table<DenseVector<float>, int32_t,
          PlainArrayPointSet<float>>(points, inner_params)));

  return new PyLSHNearestNeighborTableDenseFloat(table.release());
}

}  // namespace python
}  // namespace falconn

#endif
