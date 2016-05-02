#ifndef __PYTHON_WRAPPER_H__
#define __PYTHON_WRAPPER_H__

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "falconn/eigen_wrapper.h"
#include "falconn/falconn_global.h"
#include "falconn/lsh_nn_table.h"

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

  PyLSHNearestNeighborTableDenseDouble(InnerTable* table) : table_(table) {}

  void set_num_probes(int num_probes) { table_->set_num_probes(num_probes); }

  int32_t get_num_probes() { return table_->get_num_probes(); }

  void set_max_num_candidates(int32_t max_num_candidates) {
    table_->set_max_num_candidates(max_num_candidates);
  }

  int32_t get_max_num_candidates() { return table_->get_max_num_candidates(); }

  int find_nearest_neighbor(const double* vec, int len) {
    ConstVectorMap q(vec, len);
    return table_->find_nearest_neighbor(q);
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

  /*std::vector<int32_t> get_unique_sorted_candidates(const double* vec,
      int len) {
    ConstVectorMap q(vec, len);
    std::vector<int32_t> result;
    table_->get_unique_sorted_candidates(q, &result);
    return result;
  }*/

  void reset_query_statistics() { table_->reset_query_statistics(); }

  falconn::QueryStatistics get_query_statistics() {
    return table_->get_query_statistics();
  }

 private:
  std::shared_ptr<InnerTable> table_ = nullptr;
};

class PyLSHNearestNeighborTableDenseFloat {
 public:
  typedef LSHNearestNeighborTable<DenseVector<float>, int32_t> InnerTable;
  typedef Eigen::Map<const DenseVector<float>> ConstVectorMap;

  PyLSHNearestNeighborTableDenseFloat(InnerTable* table) : table_(table) {}

  void set_num_probes(int num_probes) { table_->set_num_probes(num_probes); }

  int32_t get_num_probes() { return table_->get_num_probes(); }

  void set_max_num_candidates(int32_t max_num_candidates) {
    table_->set_max_num_candidates(max_num_candidates);
  }

  int32_t get_max_num_candidates() { return table_->get_max_num_candidates(); }

  int find_nearest_neighbor(const float* vec, int len) {
    ConstVectorMap q(vec, len);
    return table_->find_nearest_neighbor(q);
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

  /*std::vector<int32_t> get_unique_sorted_candidates(const float* vec,
      int len) {
    ConstVectorMap q(vec, len);
    std::vector<int32_t> result;
    table_->get_unique_sorted_candidates(q, &result);
    return result;
  }*/

  void reset_query_statistics() { table_->reset_query_statistics(); }

  falconn::QueryStatistics get_query_statistics() {
    return table_->get_query_statistics();
  }

 private:
  std::shared_ptr<InnerTable> table_ = nullptr;
};

struct LSHConstructionParameters {
  int_fast32_t dimension = -1;
  std::string lsh_family = "unknown";
  std::string distance_function = "unknown";
  std::string storage_hash_table = "unknown";
  int_fast32_t k = -1;
  int_fast32_t l = -1;
  int_fast32_t num_setup_threads = -1;
  uint64_t seed = 409556018;
  int_fast32_t last_cp_dimension = -1;
  int_fast32_t num_rotations = -1;
  int_fast32_t feature_hashing_dimension = -1;
};

// %ignore'd in the swig wrapper
DistanceFunction distance_function_from_string(const std::string& str) {
  std::string tmp_distance_function = str;
  std::transform(tmp_distance_function.begin(), tmp_distance_function.end(),
                 tmp_distance_function.begin(), tolower);
  for (int ii = 0; ii < static_cast<int>(kDistanceFunctionStrings.size());
       ++ii) {
    if (tmp_distance_function == kDistanceFunctionStrings[ii]) {
      return DistanceFunction(ii);
    }
  }
  throw PyLSHNearestNeighborTableError("Unknown distance_function parameter.");
}

// %ignore'd in the swig wrapper
LSHFamily lsh_family_from_string(const std::string& str) {
  std::string tmp_lsh_family = str;
  std::transform(tmp_lsh_family.begin(), tmp_lsh_family.end(),
                 tmp_lsh_family.begin(), tolower);
  for (int ii = 0; ii < static_cast<int>(kLSHFamilyStrings.size()); ++ii) {
    if (tmp_lsh_family == kLSHFamilyStrings[ii]) {
      return LSHFamily(ii);
    }
  }
  throw PyLSHNearestNeighborTableError("Unknown LSH family parameter.");
}

// %ignore'd in the swig wrapper
StorageHashTable storage_hash_table_from_string(const std::string& str) {
  std::string tmp_storage_hash_table = str;
  std::transform(tmp_storage_hash_table.begin(), tmp_storage_hash_table.end(),
                 tmp_storage_hash_table.begin(), tolower);
  for (int ii = 0; ii < static_cast<int>(kStorageHashTableStrings.size());
       ++ii) {
    if (tmp_storage_hash_table == kStorageHashTableStrings[ii]) {
      return StorageHashTable(ii);
    }
  }
  throw PyLSHNearestNeighborTableError("Unknown storage hash table parameter.");
}

// %ignore'd in the swig wrapper
void python_to_cpp_construction_parameters(
    const LSHConstructionParameters& py_params,
    falconn::LSHConstructionParameters* cpp_params) {
  cpp_params->dimension = py_params.dimension;
  cpp_params->lsh_family = lsh_family_from_string(py_params.lsh_family);
  cpp_params->distance_function =
      distance_function_from_string(py_params.distance_function);
  cpp_params->storage_hash_table =
      storage_hash_table_from_string(py_params.storage_hash_table);
  cpp_params->k = py_params.k;
  cpp_params->l = py_params.l;
  cpp_params->num_setup_threads = py_params.num_setup_threads;
  cpp_params->seed = py_params.seed;
  cpp_params->last_cp_dimension = py_params.last_cp_dimension;
  cpp_params->num_rotations = py_params.num_rotations;
  cpp_params->feature_hashing_dimension = py_params.feature_hashing_dimension;
}

// %ignore'd in the swig wrapper
void cpp_to_python_construction_parameters(
    const falconn::LSHConstructionParameters& cpp_params,
    LSHConstructionParameters* py_params) {
  py_params->dimension = cpp_params.dimension;

  int_fast32_t lsh_family_int =
      static_cast<int_fast32_t>(cpp_params.lsh_family);
  if (lsh_family_int < 0 ||
      lsh_family_int >= static_cast<int_fast32_t>(kLSHFamilyStrings.size())) {
    throw PyLSHNearestNeighborTableError("Unknown LSH family value.");
  }
  py_params->lsh_family = kLSHFamilyStrings[lsh_family_int];

  int_fast32_t distance_function_int =
      static_cast<int_fast32_t>(cpp_params.distance_function);
  if (distance_function_int < 0 ||
      distance_function_int >=
          static_cast<int_fast32_t>(kDistanceFunctionStrings.size())) {
    throw PyLSHNearestNeighborTableError("Unknown distance function value.");
  }
  py_params->distance_function =
      kDistanceFunctionStrings[distance_function_int];

  int_fast32_t storage_hash_table_int =
      static_cast<int_fast32_t>(cpp_params.storage_hash_table);
  if (storage_hash_table_int < 0 ||
      storage_hash_table_int >=
          static_cast<int_fast32_t>(kStorageHashTableStrings.size())) {
    throw PyLSHNearestNeighborTableError("Unknown storage hash table value.");
  }
  py_params->storage_hash_table =
      kStorageHashTableStrings[storage_hash_table_int];

  py_params->k = cpp_params.k;
  py_params->l = cpp_params.l;
  py_params->num_setup_threads = cpp_params.num_setup_threads;
  py_params->seed = cpp_params.seed;
  py_params->last_cp_dimension = cpp_params.last_cp_dimension;
  py_params->num_rotations = cpp_params.num_rotations;
  py_params->feature_hashing_dimension = cpp_params.feature_hashing_dimension;
}

void compute_number_of_hash_functions(int_fast32_t number_of_hash_bits,
                                      LSHConstructionParameters* params) {
  falconn::LSHConstructionParameters inner_params;
  python_to_cpp_construction_parameters(*params, &inner_params);

  falconn::compute_number_of_hash_functions<DenseVector<float>>(
      number_of_hash_bits, &inner_params);

  cpp_to_python_construction_parameters(inner_params, params);
}

LSHConstructionParameters get_default_parameters(
    int_fast64_t dataset_size, int_fast32_t dimension,
    const std::string& distance_function, bool is_sufficiently_dense) {
  falconn::LSHConstructionParameters inner_params =
      falconn::get_default_parameters<DenseVector<float>>(
          dataset_size, dimension,
          distance_function_from_string(distance_function),
          is_sufficiently_dense);
  LSHConstructionParameters params;
  cpp_to_python_construction_parameters(inner_params, &params);
  return params;
}

PyLSHNearestNeighborTableDenseDouble construct_table_dense_double(
    const double* matrix, int num_rows, int num_columns,
    const LSHConstructionParameters& params) {
  falconn::LSHConstructionParameters inner_params;
  python_to_cpp_construction_parameters(params, &inner_params);

  PlainArrayPointSet<double> points;
  points.data = matrix;
  points.num_points = num_rows;
  points.dimension = num_columns;

  std::unique_ptr<LSHNearestNeighborTable<DenseVector<double>, int32_t>>
  table(std::move(
      construct_table<DenseVector<double>, int32_t, PlainArrayPointSet<double>>(
          points, inner_params)));

  return PyLSHNearestNeighborTableDenseDouble(table.release());
}

PyLSHNearestNeighborTableDenseFloat construct_table_dense_float(
    const float* matrix, int num_rows, int num_columns,
    const LSHConstructionParameters& params) {
  falconn::LSHConstructionParameters inner_params;
  python_to_cpp_construction_parameters(params, &inner_params);

  PlainArrayPointSet<float> points;
  points.data = matrix;
  points.num_points = num_rows;
  points.dimension = num_columns;

  std::unique_ptr<LSHNearestNeighborTable<DenseVector<float>, int32_t>>
  table(std::move(
      construct_table<DenseVector<float>, int32_t, PlainArrayPointSet<float>>(
          points, inner_params)));

  return PyLSHNearestNeighborTableDenseFloat(table.release());
}

}  // namespace python
}  // namespace falconn

#endif
