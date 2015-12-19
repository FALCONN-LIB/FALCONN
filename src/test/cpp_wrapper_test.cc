#include "falconn/lsh_nn_table.h"

#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

using falconn::compute_number_of_hash_functions;
using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborTable;
using falconn::set_up_parameters;
using falconn::SparseVector;
using std::make_pair;
using std::unique_ptr;
using std::vector;

TEST(WrapperTest, DenseHPTest1) {
  int dim = 4;
  typedef DenseVector<float> Point;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.k = 2;
  params.l = 4;

  Point p1(dim);
  p1[0] = 1.0;
  p1[1] = 0.0;
  p1[2] = 0.0;
  p1[3] = 0.0;
  Point p2(dim);
  p2[0] = 0.6;
  p2[1] = 0.8;
  p2[2] = 0.0;
  p2[3] = 0.0;
  Point p3(dim);
  p3[0] = 0.0;
  p3[1] = 0.0;
  p3[2] = 1.0;
  p3[3] = 0.0;
  vector<Point> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);
  
  unique_ptr<LSHNearestNeighborTable<Point>> table(std::move(
      construct_table<Point>(points, params)));

  int32_t res1 = table->find_closest(p1);
  EXPECT_EQ(0, res1);
  int32_t res2 = table->find_closest(p2);
  EXPECT_EQ(1, res2);
  int32_t res3 = table->find_closest(p3);
  EXPECT_EQ(2, res3);

  Point p4(dim);
  p4[0] = 0.0;
  p4[1] = 1.0;
  p4[2] = 0.0;
  p4[3] = 0.0;
  int32_t res4 = table->find_closest(p4);
  EXPECT_EQ(1, res4);
}


TEST(WrapperTest, DenseCPTest1) {
  int dim = 4;
  typedef DenseVector<float> Point;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::CrossPolytope;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.k = 2;
  params.l = 8;
  params.last_cp_dimension = dim;
  params.num_rotations = 3;

  Point p1(dim);
  p1[0] = 1.0;
  p1[1] = 0.0;
  p1[2] = 0.0;
  p1[3] = 0.0;
  Point p2(dim);
  p2[0] = 0.6;
  p2[1] = 0.8;
  p2[2] = 0.0;
  p2[3] = 0.0;
  Point p3(dim);
  p3[0] = 0.0;
  p3[1] = 0.0;
  p3[2] = 1.0;
  p3[3] = 0.0;
  vector<Point> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);
  
  unique_ptr<LSHNearestNeighborTable<Point>> table(std::move(
      construct_table<Point>(points, params)));

  int32_t res1 = table->find_closest(p1);
  EXPECT_EQ(0, res1);
  int32_t res2 = table->find_closest(p2);
  EXPECT_EQ(1, res2);
  int32_t res3 = table->find_closest(p3);
  EXPECT_EQ(2, res3);

  Point p4(dim);
  p4[0] = 0.0;
  p4[1] = 1.0;
  p4[2] = 0.0;
  p4[3] = 0.0;
  int32_t res4 = table->find_closest(p4);
  EXPECT_EQ(1, res4);
}


TEST(WrapperTest, SparseHPTest1) {
  int dim = 100;
  typedef SparseVector<float> Point;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.k = 2;
  params.l = 4;

  Point p1;
  p1.push_back(make_pair(24, 1.0));
  Point p2;
  p2.push_back(make_pair(7, 0.8));
  p2.push_back(make_pair(24, 0.6));
  Point p3;
  p3.push_back(make_pair(50, 1.0));
  vector<Point> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);
  
  unique_ptr<LSHNearestNeighborTable<Point>> table(std::move(
      construct_table<Point>(points, params)));

  int32_t res1 = table->find_closest(p1);
  EXPECT_EQ(0, res1);
  int32_t res2 = table->find_closest(p2);
  EXPECT_EQ(1, res2);
  int32_t res3 = table->find_closest(p3);
  EXPECT_EQ(2, res3);

  Point p4;
  p4.push_back(make_pair(7, 1.0));
  int32_t res4 = table->find_closest(p4);
  EXPECT_EQ(1, res4);
}


TEST(WrapperTest, SparseCPTest1) {
  int dim = 100;
  typedef SparseVector<float> Point;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::CrossPolytope;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.k = 2;
  params.l = 4;
  params.feature_hashing_dimension = 8;
  params.last_cp_dimension = 8;
  params.num_rotations = 3;

  Point p1;
  p1.push_back(make_pair(24, 1.0));
  Point p2;
  p2.push_back(make_pair(7, 0.8));
  p2.push_back(make_pair(24, 0.6));
  Point p3;
  p3.push_back(make_pair(50, 1.0));
  vector<Point> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);
  
  unique_ptr<LSHNearestNeighborTable<Point>> table(std::move(
      construct_table<Point>(points, params)));

  int32_t res1 = table->find_closest(p1);
  EXPECT_EQ(0, res1);
  int32_t res2 = table->find_closest(p2);
  EXPECT_EQ(1, res2);
  int32_t res3 = table->find_closest(p3);
  EXPECT_EQ(2, res3);

  Point p4;
  p4.push_back(make_pair(7, 1.0));
  int32_t res4 = table->find_closest(p4);
  EXPECT_EQ(1, res4);
}

TEST(WrapperTest, ComputeNumberOfHashFunctionsTest) {
  typedef DenseVector<float> VecDense;
  typedef SparseVector<float> VecSparse;

  LSHConstructionParameters params;
  params.dimension = 10;
  params.lsh_family = LSHFamily::Hyperplane;

  compute_number_of_hash_functions<VecDense>(5, &params);
  EXPECT_EQ(5, params.k);

  params.lsh_family = LSHFamily::CrossPolytope;
  compute_number_of_hash_functions<VecDense>(5, &params);
  EXPECT_EQ(1, params.k);
  EXPECT_EQ(10, params.last_cp_dimension);
  
  params.dimension = 100;
  params.lsh_family = LSHFamily::Hyperplane;
  compute_number_of_hash_functions<VecSparse>(8, &params);
  EXPECT_EQ(8, params.k);

  params.lsh_family = LSHFamily::CrossPolytope;
  params.feature_hashing_dimension = 30;
  compute_number_of_hash_functions<VecSparse>(9, &params);
  EXPECT_EQ(2, params.k);
  EXPECT_EQ(4, params.last_cp_dimension);
}

TEST(WrapperTest, SetUpParametersTest1) {
  typedef DenseVector<float> Vec;

  LSHConstructionParameters params = set_up_parameters<Vec>(1000000, 128,
      DistanceFunction::NegativeInnerProduct, true);

  EXPECT_EQ(1, params.num_rotations);
  EXPECT_EQ(-1, params.feature_hashing_dimension);
}
  
TEST(WrapperTest, SetUpParametersTest2) {
  typedef SparseVector<float> Vec;
  
  LSHConstructionParameters params = set_up_parameters<Vec>(1000000, 100000,
      DistanceFunction::NegativeInnerProduct, true);
  
  EXPECT_EQ(2, params.num_rotations);
  EXPECT_EQ(1024, params.feature_hashing_dimension);
}
