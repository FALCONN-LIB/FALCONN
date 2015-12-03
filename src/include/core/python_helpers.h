#ifndef __LSH_PYTHON_HELPERS_H__
#define __LSH_PYTHON_HELPERS_H__

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "polytope_hash.h"

Eigen::VectorXf FloatListToEigenVector(const std::vector<float>& v) {
  Eigen::VectorXf res(v.size());
  for (size_t ii = 0; ii < v.size(); ++ii) {
    res[ii] = v[ii];
  }
  return res;
}

std::vector<float> EigenVectorToFloatList(const Eigen::VectorXf& v) {
  std::vector<float> res(v.size());
  for (int ii = 0; ii < v.size(); ++ii) {
    res[ii] = v[ii];
  }
  return res;
}


// TODO: this is sort of a hack because our swig Python wrapper doesn't know
// the definition of Eigen::VectorXf .
const lsh::CrossPolytopeHashDense<float, uint32_t>::VectorType&
    EigenVectorToCPHashVector(const Eigen::VectorXf& v) {
  return v;
}

const lsh::HyperplaneHashDense<float, uint32_t>::VectorType&
    EigenVectorToHPHashVector(const Eigen::VectorXf& v) {
  return v;
}


void GenRandomDenseUnitVectors(int_fast32_t n,
                               int_fast32_t dim,
                               int_fast64_t seed,
                               std::vector<Eigen::VectorXf>* res) {
  std::vector<Eigen::VectorXf>& result = *res;
  result.resize(n);
  std::mt19937_64 gen(seed);
  std::normal_distribution<float> normal_dist(0, 1);
  for (int_fast32_t ii = 0; ii < n; ++ii) {
    result[ii].setZero(dim);
    for (int_fast32_t jj = 0; jj < dim; ++jj) {
      result[ii][jj] = normal_dist(gen);
    }
    result[ii].normalize();
  }
}


bool ReadSparseData(const std::string& filename,
                    std::vector<std::vector<std::pair<int, float>>>* data,
                    int *d,
                    int to_read) {
  int32_t n_in;
  int32_t dim_in;
  data->clear();
  FILE *input = fopen(filename.c_str(), "rb");
  if (input == NULL) {
    return false;
  }

  if (fread(&n_in, sizeof(n_in), 1, input) != 1) {
    return false;
  }
  if (n_in <= 0) {
    return false;
  }
  if (fread(&dim_in, sizeof(dim_in), 1, input) != 1) {
    return false;
  }
  if (dim_in <= 0) {
    return false;
  }
  *d = dim_in;

  int num_to_read = to_read;
  if (num_to_read < 0) {
    num_to_read = n_in;
  }
  data->resize(num_to_read);

  for (int i = 0; i < num_to_read; ++i) {
    int32_t aux;
    if (fread(&aux, sizeof(aux), 1, input) != 1) {
      return false;
    }
    if (aux < 0 || aux > *d) {
      return false;
    }
    (*data)[i].resize(aux);
    if (fread((*data)[i].data(), sizeof((*data)[i][0]), aux, input)
        != static_cast<size_t>(aux)) {
      return false;
    }
  }
  fclose(input);

  return true;
}


std::pair<bool, int> ReadSparseDataPython(
    const std::string& filename,
    std::vector<std::vector<std::pair<int, float>>>* data,
    int to_read) {
  int d;
  bool success = ReadSparseData(filename, data, &d, to_read);
  return std::make_pair(success, d);
}


bool ReadSparseDataAndSplit(
    const std::string& filename,
    const std::vector<int>& query_indices,
    std::vector<std::vector<std::pair<int, float>>>* data,
    std::vector<std::vector<std::pair<int, float>>>* queries,
    int *d,
    int to_read) {
  int32_t n_in;
  int32_t dim_in;
  data->clear();
  FILE *input = fopen(filename.c_str(), "rb");
  if (input == NULL) {
    return false;
  }

  if (fread(&n_in, sizeof(n_in), 1, input) != 1) {
    return false;
  }
  if (n_in <= 0) {
    return false;
  }
  if (fread(&dim_in, sizeof(dim_in), 1, input) != 1) {
    return false;
  }
  if (dim_in <= 0) {
    return false;
  }
  *d = dim_in;

  int num_to_read = to_read;
  if (num_to_read < 0) {
    num_to_read = n_in;
  }
  int num_queries = static_cast<int>(query_indices.size());
  data->resize(num_to_read - num_queries);
  queries->resize(num_queries);

  int data_index = 0;
  int cur_query = 0;

  for (int ii = 0; ii < num_to_read; ++ii) {
    int32_t aux;
    if (fread(&aux, sizeof(aux), 1, input) != 1) {
      return false;
    }
    if (aux < 0 || aux > *d) {
      return false;
    }

    std::vector<std::pair<int, float>>* dest = nullptr;
    if (cur_query < num_queries && ii == query_indices[cur_query]) {
      dest = &((*queries)[cur_query]);
      cur_query += 1;

      if (cur_query < num_queries) {
        if (query_indices[cur_query - 1] >= query_indices[cur_query]) {
          return false;
        }
      }
    } else {
      if (data_index >= static_cast<int>(data->size())) {
        return false;
      }
      dest = &((*data)[data_index]);
      data_index += 1;
    }
    dest->resize(aux);
    if (fread(dest->data(), sizeof((*dest)[0]), aux, input)
        != static_cast<size_t>(aux)) {
      return false;
    }
  }
  fclose(input);

  if (cur_query != num_queries) {
    return false;
  }

  return true;
}


std::pair<bool, int> ReadSparseDataAndSplitPython(
    const std::string& filename,
    const std::vector<int>& query_indices,
    std::vector<std::vector<std::pair<int, float>>>* data,
    std::vector<std::vector<std::pair<int, float>>>* queries,
    int to_read) {
  int d;
  bool success = ReadSparseDataAndSplit(filename, query_indices, data, queries,
      &d, to_read);
  return std::make_pair(success, d);
}

#endif
