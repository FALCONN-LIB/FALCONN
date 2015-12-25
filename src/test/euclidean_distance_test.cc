#include "falconn/core/euclidean_distance.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace fc = falconn::core;

using fc::EuclideanDistanceDense;
using fc::EuclideanDistanceSparse;
using std::make_pair;
using std::vector;

typedef EuclideanDistanceDense<float>::VectorType DenseVector;
typedef EuclideanDistanceSparse<float>::VectorType SparseVector;

const float eps = 0.00001;

TEST(EuclideanDistanceTest, SparseDistanceFunctionTest1) {
  SparseVector v1;
  v1.push_back(make_pair(1, 2.0));
  v1.push_back(make_pair(2, 3.0));
  v1.push_back(make_pair(4, -1.0));
  SparseVector v2;
  v2.push_back(make_pair(1, 2.0));
  v2.push_back(make_pair(3, 3.0));
  v2.push_back(make_pair(4, 0.5));
  EuclideanDistanceSparse<float> distance_function;
  float distance = distance_function(v1, v2);
  ASSERT_NEAR(distance, 20.25, eps);
}

TEST(EuclideanDistanceTest, SparseDistanceFunctionTest2) {
  SparseVector v1;
  v1.push_back(make_pair(1, 2.0));
  SparseVector v2;
  EuclideanDistanceSparse<float> distance_function;
  float distance = distance_function(v1, v2);
  ASSERT_NEAR(distance, 4.0, eps);
}

TEST(EuclideanDistanceTest, SparseDistanceFunctionTest3) {
  SparseVector v1;
  SparseVector v2;
  v2.push_back(make_pair(1, 2.0));
  EuclideanDistanceSparse<float> distance_function;
  float distance = distance_function(v1, v2);
  ASSERT_NEAR(distance, 4.0, eps);
}

TEST(EuclideanDistanceTest, DenseDistanceFunctionTest1) {
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
  EuclideanDistanceDense<float> distance_function;
  float distance = distance_function(v1, v2);
  ASSERT_NEAR(distance, 101.25, eps);
}

TEST(EuclideanDistanceTest, DenseDistanceFunctionTest2) {
  DenseVector v1(4);
  v1[0] = 0.0;
  v1[1] = 1.0;
  v1[2] = 2.0;
  v1[3] = 0.5;
  float v2_raw[4] = {8.0, 1.0, -3.0, 4.0};
  Eigen::Map<DenseVector> v2(v2_raw, 4);
  EuclideanDistanceDense<float> distance_function;
  float distance = distance_function(v1, v2);
  ASSERT_NEAR(distance, 101.25, eps);
}
