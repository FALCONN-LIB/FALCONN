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
using falconn::LSHNearestNeighborQuery;
using falconn::LSHNearestNeighborQueryPool;
using falconn::get_default_parameters;
using falconn::SparseVector;
using falconn::StorageHashTable;
using std::make_pair;
using std::unique_ptr;
using std::vector;

// Point dimension is 4
void basic_test_dense_1(const LSHConstructionParameters& params) {
  typedef DenseVector<float> Point;
  int dim = 4;

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

  unique_ptr<LSHNearestNeighborTable<Point>> table(
      construct_table<Point>(points, params));
  unique_ptr<LSHNearestNeighborQuery<Point>> query(
      table->construct_query_object());

  int32_t res1 = query->find_nearest_neighbor(p1);
  EXPECT_EQ(0, res1);
  int32_t res2 = query->find_nearest_neighbor(p2);
  EXPECT_EQ(1, res2);
  int32_t res3 = query->find_nearest_neighbor(p3);
  EXPECT_EQ(2, res3);

  Point p4(dim);
  p4[0] = 0.0;
  p4[1] = 1.0;
  p4[2] = 0.0;
  p4[3] = 0.0;
  int32_t res4 = query->find_nearest_neighbor(p4);
  EXPECT_EQ(1, res4);

  unique_ptr<LSHNearestNeighborQueryPool<Point>> query_pool(
      table->construct_query_pool());

  // Same queries as above but now through a query pool
  res1 = query_pool->find_nearest_neighbor(p1);
  EXPECT_EQ(0, res1);
  res2 = query_pool->find_nearest_neighbor(p2);
  EXPECT_EQ(1, res2);
  res3 = query_pool->find_nearest_neighbor(p3);
  EXPECT_EQ(2, res3);

  res4 = query_pool->find_nearest_neighbor(p4);
  EXPECT_EQ(1, res4);
}

void basic_test_sparse_1(const LSHConstructionParameters& params) {
  typedef SparseVector<float> Point;
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

  unique_ptr<LSHNearestNeighborTable<Point>> table(
      construct_table<Point>(points, params));
  unique_ptr<LSHNearestNeighborQuery<Point>> query(
      table->construct_query_object());

  int32_t res1 = query->find_nearest_neighbor(p1);
  EXPECT_EQ(0, res1);
  int32_t res2 = query->find_nearest_neighbor(p2);
  EXPECT_EQ(1, res2);
  int32_t res3 = query->find_nearest_neighbor(p3);
  EXPECT_EQ(2, res3);

  Point p4;
  p4.push_back(make_pair(7, 1.0));
  int32_t res4 = query->find_nearest_neighbor(p4);
  EXPECT_EQ(1, res4);
}

TEST(WrapperTest, DenseHPTest1) {
  int dim = 4;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 4;
  params.num_setup_threads = 0;

  basic_test_dense_1(params);
}

TEST(WrapperTest, DenseCPTest1) {
  int dim = 4;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::CrossPolytope;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 8;
  params.last_cp_dimension = dim;
  params.num_rotations = 3;
  params.num_setup_threads = 0;

  basic_test_dense_1(params);
}

TEST(WrapperTest, SparseHPTest1) {
  int dim = 100;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 4;
  params.num_setup_threads = 0;

  basic_test_sparse_1(params);
}

TEST(WrapperTest, SparseCPTest1) {
  int dim = 100;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::CrossPolytope;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 4;
  params.feature_hashing_dimension = 8;
  params.last_cp_dimension = 8;
  params.num_rotations = 3;
  params.num_setup_threads = 0;

  basic_test_sparse_1(params);
}

TEST(WrapperTest, FlatHashTableTest1) {
  int dim = 4;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::FlatHashTable;
  params.k = 2;
  params.l = 4;
  params.num_setup_threads = 0;

  basic_test_dense_1(params);
}

TEST(WrapperTest, BitPackedFlatHashTableTest1) {
  int dim = 4;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 4;
  params.num_setup_threads = 0;

  basic_test_dense_1(params);
}

TEST(WrapperTest, STLHashTableTest1) {
  int dim = 4;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::STLHashTable;
  params.k = 2;
  params.l = 4;
  params.num_setup_threads = 0;

  basic_test_dense_1(params);
}

TEST(WrapperTest, LinearProbingHashTableTest1) {
  int dim = 4;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::LinearProbingHashTable;
  params.k = 2;
  params.l = 4;
  params.num_setup_threads = 0;

  basic_test_dense_1(params);
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
  EXPECT_EQ(16, params.last_cp_dimension);

  params.dimension = 100;
  params.lsh_family = LSHFamily::Hyperplane;
  compute_number_of_hash_functions<VecSparse>(8, &params);
  EXPECT_EQ(8, params.k);

  params.lsh_family = LSHFamily::CrossPolytope;
  params.feature_hashing_dimension = 32;
  compute_number_of_hash_functions<VecSparse>(9, &params);
  EXPECT_EQ(2, params.k);
  EXPECT_EQ(4, params.last_cp_dimension);
}

TEST(WrapperTest, GetDefaultParametersTest1) {
  typedef DenseVector<float> Vec;

  LSHConstructionParameters params = get_default_parameters<Vec>(
      1000000, 128, DistanceFunction::NegativeInnerProduct, true);

  EXPECT_EQ(1, params.num_rotations);
  EXPECT_EQ(-1, params.feature_hashing_dimension);
  EXPECT_EQ(10, params.l);
  EXPECT_EQ(128, params.dimension);
  EXPECT_EQ(DistanceFunction::NegativeInnerProduct, params.distance_function);
  EXPECT_EQ(LSHFamily::CrossPolytope, params.lsh_family);
  EXPECT_EQ(3, params.k);
  EXPECT_EQ(2, params.last_cp_dimension);
  EXPECT_EQ(StorageHashTable::BitPackedFlatHashTable,
            params.storage_hash_table);
  EXPECT_EQ(0, params.num_setup_threads);
}

TEST(WrapperTest, GetDefaultParametersTest2) {
  typedef SparseVector<float> Vec;

  LSHConstructionParameters params = get_default_parameters<Vec>(
      1000000, 100000, DistanceFunction::NegativeInnerProduct, true);

  EXPECT_EQ(2, params.num_rotations);
  EXPECT_EQ(1024, params.feature_hashing_dimension);
  EXPECT_EQ(0, params.num_setup_threads);
  EXPECT_EQ(StorageHashTable::BitPackedFlatHashTable,
            params.storage_hash_table);
}
