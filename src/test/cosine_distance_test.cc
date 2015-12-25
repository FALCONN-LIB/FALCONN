#include "falconn/core/cosine_distance.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace fc = falconn::core;

using fc::CosineDistanceDense;
using fc::CosineDistanceSparse;
using std::make_pair;
using std::vector;

typedef CosineDistanceDense<float>::VectorType DenseVector;
typedef CosineDistanceSparse<float>::VectorType SparseVector;

const float eps = 0.00001;

TEST(CosineDistanceTest, SparseDistanceFunctionTest1) {
  SparseVector v1;
  v1.push_back(make_pair(1, 2.0));
  v1.push_back(make_pair(2, 3.0));
  v1.push_back(make_pair(4, -1.0));
  SparseVector v2;
  v2.push_back(make_pair(1, 2.0));
  v2.push_back(make_pair(3, 3.0));
  v2.push_back(make_pair(4, 0.5));
  CosineDistanceSparse<float> distance_function;
  float distance = distance_function(v1, v2);
  ASSERT_NEAR(distance, -3.5, eps);
}

TEST(CosineDistanceTest, SparseDistanceFunctionTest2) {
  SparseVector v1;
  v1.push_back(make_pair(1, 2.0));
  SparseVector v2;
  CosineDistanceSparse<float> distance_function;
  float distance = distance_function(v1, v2);
  ASSERT_NEAR(distance, 0.0, eps);
}

TEST(CosineDistanceTest, SparseDistanceFunctionTest3) {
  SparseVector v1;
  SparseVector v2;
  v2.push_back(make_pair(1, 2.0));
  CosineDistanceSparse<float> distance_function;
  float distance = distance_function(v1, v2);
  ASSERT_NEAR(distance, 0.0, eps);
}

TEST(CosineDistanceTest, DenseDistanceFunctionTest1) {
  DenseVector v1(4);
  v1[0] = 0.0;
  v1[1] = 1.0;
  v1[2] = 2.0;
  v1[3] = 0.5;
  DenseVector v2(4);
  v2[0] = 8.0;
  v2[1] = 1.0;
  v2[2] = -3.0;
  v2[3] = 4.0;
  CosineDistanceDense<float> distance_function;
  float distance = distance_function(v1, v2);
  ASSERT_NEAR(distance, 3.0, eps);
}

TEST(CosineDistanceTest, DenseDistanceFunctionTest2) {
  DenseVector v1(4);
  v1[0] = 0.0;
  v1[1] = 1.0;
  v1[2] = 2.0;
  v1[3] = 0.5;
  float v2_raw[4] = {8.0, 1.0, -3.0, 4.0};
  Eigen::Map<DenseVector> v2(v2_raw, 4);
  CosineDistanceDense<float> distance_function;
  float distance = distance_function(v1, v2);
  ASSERT_NEAR(distance, 3.0, eps);
}
