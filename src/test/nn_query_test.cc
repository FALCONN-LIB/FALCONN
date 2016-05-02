#include "falconn/core/nn_query.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "falconn/core/composite_hash_table.h"
#include "falconn/core/cosine_distance.h"
#include "falconn/core/data_storage.h"
#include "falconn/core/hyperplane_hash.h"
#include "falconn/core/lsh_table.h"
#include "falconn/core/probing_hash_table.h"
#include "test_utils.h"

namespace fc = falconn::core;
namespace ft = falconn::test;

using fc::CosineDistanceDense;
using fc::CosineDistanceSparse;
using fc::ArrayDataStorage;
using fc::DynamicCompositeHashTable;
using fc::DynamicLinearProbingHashTable;
// using lsh::DynamicLSHTable;
using fc::HyperplaneHashDense;
using fc::HyperplaneHashSparse;
using fc::NearestNeighborQuery;
using fc::StaticLSHTable;
using fc::StaticLinearProbingHashTable;
using fc::StaticCompositeHashTable;
using ft::check_result;
using std::make_pair;
using std::vector;

typedef HyperplaneHashDense<float>::VectorType DenseVector;
typedef HyperplaneHashSparse<float>::VectorType SparseVector;

TEST(NNQueryTest, DenseTest1) {
  const int dim = 5;
  int k = 2;
  int l = 2;
  int seed = 52671998;
  int table_size = 10;

  DenseVector p1(dim);
  p1[0] = 5.0;
  p1[1] = 0.0;
  p1[2] = -7.0;
  p1[3] = 0.0;
  p1[4] = 3.0;
  p1.normalize();
  DenseVector p2(dim);
  p2[0] = 0.0;
  p2[1] = 4.0;
  p2[2] = -6.0;
  p2[3] = 0.0;
  p2[4] = 3.0;
  p2.normalize();
  DenseVector p3(dim);
  p3[0] = 0.0;
  p3[1] = 0.0;
  p3[2] = 0.0;
  p3[3] = -1.0;
  p3[4] = 0.0;
  p3.normalize();
  vector<DenseVector> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);

  typedef ArrayDataStorage<DenseVector> ArrayDataStorageType;
  ArrayDataStorageType data_storage(points);

  HyperplaneHashDense<float> lsh_object(dim, k, l, seed);
  StaticLinearProbingHashTable<uint32_t>::Factory table_factory(table_size);
  typedef StaticCompositeHashTable<uint32_t, int32_t,
                                   StaticLinearProbingHashTable<uint32_t>>
      CompositeTableType;
  CompositeTableType hash_table(l, &table_factory);
  typedef StaticLSHTable<DenseVector, int32_t, HyperplaneHashDense<float>,
                         uint32_t, CompositeTableType>
      LSHTableType;
  LSHTableType lsh_table(&lsh_object, &hash_table, points, 1);
  LSHTableType::Query query(lsh_table);
  NearestNeighborQuery<LSHTableType::Query, DenseVector, int32_t, DenseVector,
                       float, CosineDistanceDense<float>, ArrayDataStorageType>
      nn_query(&query, data_storage);

  int res1 = nn_query.find_nearest_neighbor(p1, p1, l, -1);
  EXPECT_EQ(0, res1);
  int res2 = nn_query.find_nearest_neighbor(p2, p2, l, -1);
  EXPECT_EQ(1, res2);
  int res3 = nn_query.find_nearest_neighbor(p3, p3, l, -1);
  EXPECT_EQ(2, res3);

  DenseVector p2query(dim);
  p2query[0] = 0.0;
  p2query[1] = 4.0;
  p2query[2] = -5.5;
  p2query[3] = 0.0;
  p2query[4] = 3.0;
  p2query.normalize();
  int res4 = nn_query.find_nearest_neighbor(p2query, p2query, l, -1);
  EXPECT_EQ(1, res4);
}

TEST(NNQueryTest, SparseTest1) {
  const int dim = 100;
  int k = 3;
  int l = 2;
  int seed = 89021344;
  int table_size = 10;

  // TODO: normalize these vectors (write a sparse_vector_utils.h for this)
  SparseVector p1;
  p1.push_back(make_pair(1, 10.0));
  p1.push_back(make_pair(5, -7.0));
  p1.push_back(make_pair(60, 3.0));

  SparseVector p2;
  p2.push_back(make_pair(2, 4.0));
  p2.push_back(make_pair(5, -6.0));
  p2.push_back(make_pair(60, 3.0));

  SparseVector p3;
  p3.push_back(make_pair(3, -1.0));
  p3.push_back(make_pair(20, 3.0));
  p3.push_back(make_pair(72, -5.0));

  vector<SparseVector> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);

  typedef ArrayDataStorage<SparseVector> ArrayDataStorageType;
  ArrayDataStorageType data_storage(points);

  HyperplaneHashSparse<float> lsh_object(dim, k, l, seed);
  StaticLinearProbingHashTable<uint32_t>::Factory table_factory(table_size);
  typedef StaticCompositeHashTable<uint32_t, int32_t,
                                   StaticLinearProbingHashTable<uint32_t>>
      CompositeTableType;
  CompositeTableType hash_table(l, &table_factory);
  typedef StaticLSHTable<SparseVector, int32_t, HyperplaneHashSparse<float>,
                         uint32_t, CompositeTableType>
      LSHTableType;
  LSHTableType lsh_table(&lsh_object, &hash_table, points, 1);
  LSHTableType::Query query(lsh_table);
  NearestNeighborQuery<LSHTableType::Query, SparseVector, int32_t, SparseVector,
                       float, CosineDistanceSparse<float>, ArrayDataStorageType>
      nn_query(&query, data_storage);

  int res1 = nn_query.find_nearest_neighbor(p1, p1, l, -1);
  EXPECT_EQ(0, res1);
  int res2 = nn_query.find_nearest_neighbor(p2, p2, l, -1);
  EXPECT_EQ(1, res2);
  int res3 = nn_query.find_nearest_neighbor(p3, p3, l, -1);
  EXPECT_EQ(2, res3);

  SparseVector p2query;
  p2query.push_back(make_pair(2, 4.0));
  p2query.push_back(make_pair(5, -5.5));
  p2query.push_back(make_pair(60, 3.0));
  int res4 = nn_query.find_nearest_neighbor(p2query, p2query, l, -1);
  EXPECT_EQ(1, res4);
}

TEST(NNQueryTest, MultiprobeTest1) {
  const int dim = 4;
  int k = 3;
  int l = 1;
  int seed = 6584012;
  int table_size = 10;

  DenseVector p1(dim);
  p1[0] = 5.0;
  p1[1] = 0.0;
  p1[2] = -7.0;
  p1[3] = 0.0;
  p1.normalize();
  vector<DenseVector> points;
  points.push_back(p1);

  typedef ArrayDataStorage<DenseVector> ArrayDataStorageType;
  ArrayDataStorageType data_storage(points);

  HyperplaneHashDense<float> lsh_object(dim, k, l, seed);
  StaticLinearProbingHashTable<uint32_t>::Factory table_factory(table_size);
  typedef StaticCompositeHashTable<uint32_t, int32_t,
                                   StaticLinearProbingHashTable<uint32_t>>
      CompositeTableType;
  CompositeTableType hash_table(l, &table_factory);
  typedef StaticLSHTable<DenseVector, int32_t, HyperplaneHashDense<float>,
                         uint32_t, CompositeTableType>
      LSHTableType;
  LSHTableType lsh_table(&lsh_object, &hash_table, points, 1);
  LSHTableType::Query query(lsh_table);
  NearestNeighborQuery<LSHTableType::Query, DenseVector, int32_t, DenseVector,
                       float, CosineDistanceDense<float>, ArrayDataStorageType>
      nn_query(&query, data_storage);

  int res1 = nn_query.find_nearest_neighbor(p1, p1, 2 << k, -1);
  EXPECT_EQ(0, res1);

  DenseVector pquery(dim);
  pquery[0] = 0.0;
  pquery[1] = 1.0;
  pquery[2] = 0.0;
  pquery[3] = 0.0;
  int res2 = nn_query.find_nearest_neighbor(pquery, pquery, 2 << k, -1);
  EXPECT_EQ(0, res2);
}

TEST(NNQueryTest, FindNearNeighborsTest1) {
  const int dim = 4;
  int k = 2;
  int l = 2;
  int seed = 6584012;
  int table_size = 10;

  DenseVector p1(dim);
  p1[0] = 1.0;
  p1[1] = 0.0;
  p1[2] = 0.0;
  p1[3] = 0.0;
  DenseVector p2(dim);
  p2[0] = 0.8;
  p2[1] = 0.6;
  p2[2] = 0.0;
  p2[3] = 0.0;
  DenseVector p3(dim);
  p3[0] = 0.0;
  p3[1] = 0.0;
  p3[2] = 1.0;
  p3[3] = 0.0;
  vector<DenseVector> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);

  typedef ArrayDataStorage<DenseVector> ArrayDataStorageType;
  ArrayDataStorageType data_storage(points);

  HyperplaneHashDense<float> lsh_object(dim, k, l, seed);
  StaticLinearProbingHashTable<uint32_t>::Factory table_factory(table_size);
  typedef StaticCompositeHashTable<uint32_t, int32_t,
                                   StaticLinearProbingHashTable<uint32_t>>
      CompositeTableType;
  CompositeTableType hash_table(l, &table_factory);
  typedef StaticLSHTable<DenseVector, int32_t, HyperplaneHashDense<float>,
                         uint32_t, CompositeTableType>
      LSHTableType;
  LSHTableType lsh_table(&lsh_object, &hash_table, points, 1);
  LSHTableType::Query query(lsh_table);
  NearestNeighborQuery<LSHTableType::Query, DenseVector, int32_t, DenseVector,
                       float, CosineDistanceDense<float>, ArrayDataStorageType>
      nn_query(&query, data_storage);

  double threshold = -0.5;

  vector<int32_t> res1;
  nn_query.find_near_neighbors(p1, p1, threshold, l, -1, &res1);
  vector<int32_t> expected1 = {0, 1};
  check_result(make_pair(res1.begin(), res1.end()), expected1);

  vector<int32_t> res2;
  nn_query.find_near_neighbors(p2, p2, threshold, l, -1, &res2);
  vector<int32_t> expected2 = {0, 1};
  check_result(make_pair(res2.begin(), res2.end()), expected2);

  vector<int32_t> res3;
  nn_query.find_near_neighbors(p3, p3, threshold, l, -1, &res3);
  vector<int32_t> expected3 = {2};
  check_result(make_pair(res3.begin(), res3.end()), expected3);

  DenseVector p4(dim);
  p4[0] = 0.0;
  p4[1] = 0.0;
  p4[2] = 0.0;
  p4[3] = 1.0;
  vector<int32_t> res4;
  nn_query.find_near_neighbors(p4, p4, threshold, l, -1, &res4);
  vector<int32_t> expected4 = {};
  check_result(make_pair(res4.begin(), res4.end()), expected4);
}

TEST(NNQueryTest, KNNTest1) {
  const int dim = 4;
  int k = 2;
  int l = 2;
  int seed = 6584012;
  int table_size = 10;

  DenseVector p1(dim);
  p1[0] = 1.0;
  p1[1] = 0.0;
  p1[2] = 0.0;
  p1[3] = 0.0;
  DenseVector p2(dim);
  p2[0] = 0.8;
  p2[1] = std::sqrt(1.0 - p2[0] * p2[0]);
  p2[2] = 0.0;
  p2[3] = 0.0;
  DenseVector p3(dim);
  p3[0] = 0.9;
  p3[1] = 0.0;
  p3[2] = std::sqrt(1.0 - p3[0] * p3[0]);
  p3[3] = 0.0;
  DenseVector p4(dim);
  p4[0] = 0.85;
  p4[1] = 0.0;
  p4[2] = 0.0;
  p4[3] = std::sqrt(1.0 - p4[0] * p4[0]);
  vector<DenseVector> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);
  points.push_back(p4);

  typedef ArrayDataStorage<DenseVector> ArrayDataStorageType;
  ArrayDataStorageType data_storage(points);

  HyperplaneHashDense<float> lsh_object(dim, k, l, seed);
  StaticLinearProbingHashTable<uint32_t>::Factory table_factory(table_size);
  typedef StaticCompositeHashTable<uint32_t, int32_t,
                                   StaticLinearProbingHashTable<uint32_t>>
      CompositeTableType;
  CompositeTableType hash_table(l, &table_factory);
  typedef StaticLSHTable<DenseVector, int32_t, HyperplaneHashDense<float>,
                         uint32_t, CompositeTableType>
      LSHTableType;
  LSHTableType lsh_table(&lsh_object, &hash_table, points, 1);
  LSHTableType::Query query(lsh_table);
  NearestNeighborQuery<LSHTableType::Query, DenseVector, int32_t, DenseVector,
                       float, CosineDistanceDense<float>, ArrayDataStorageType>
      nn_query(&query, data_storage);

  vector<int32_t> res1;
  nn_query.find_k_nearest_neighbors(p1, p1, 2, l, -1, &res1);
  ASSERT_EQ(2u, res1.size());
  EXPECT_EQ(0, res1[0]);
  EXPECT_EQ(2, res1[1]);
}

/*

TEST(LSHTableTest, DynamicLSHTableDenseTest1) {
  const int dim = 5;
  int k = 3;
  int l = 2;
  int seed = 6584012;

  DenseVector p1(dim);
  p1[0] = 5.0;
  p1[1] = 0.0;
  p1[2] = -7.0;
  p1[3] = 0.0;
  p1[4] = 3.0;
  DenseVector p2(dim);
  p2[0] = 0.0;
  p2[1] = 4.0;
  p2[2] = -6.0;
  p2[3] = 0.0;
  p2[4] = 3.0;
  DenseVector p3(dim);
  p3[0] = 0.0;
  p3[1] = 0.0;
  p3[2] = 0.0;
  p3[3] = -1.0;
  p3[4] = 0.0;
  vector<DenseVector> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);

  HyperplaneHashDense<float> lsh_object(dim, k, l, seed);
  DynamicLinearProbingHashTableFactory<uint32_t> table_factory(
      0.5, 0.25, 3.0, 1);
  typedef DynamicCompositeHashTable<uint32_t,
                                    DynamicLinearProbingHashTable<uint32_t>,
                                    DynamicLinearProbingHashTableFactory<
                                        uint32_t>>
          CompositeTableType;
  CompositeTableType hash_table(l, &table_factory);
  DynamicLSHTable<DenseVector,
                  float,
                  CosineDistanceDense<float>,
                  HyperplaneHashDense<float>,
                  uint32_t,
                  CompositeTableType> lsh_table(&lsh_object,
                                                &hash_table,
                                                points);

  lsh_table.insert(0);
  lsh_table.insert(1);
  lsh_table.insert(2);
  //printf("after insert 2\n");

  int res1 = lsh_table.find_nearest_neighbor(p1);
  EXPECT_EQ(0, res1);
  //printf("after find_nearest_neighbor p1\n");
  int res2 = lsh_table.find_nearest_neighbor(p2);
  EXPECT_EQ(1, res2);
  int res3 = lsh_table.find_nearest_neighbor(p3);
  EXPECT_EQ(2, res3);

  DenseVector p2query(dim);
  p2query[0] = 0.0;
  p2query[1] = 4.0;
  p2query[2] = -5.5;
  p2query[3] = 0.0;
  p2query[4] = 3.0;
  int res4 = lsh_table.find_nearest_neighbor(p2query);
  EXPECT_EQ(1, res4);

  lsh_table.remove(0);
  int res5 = lsh_table.find_nearest_neighbor(p1);
  EXPECT_EQ(1, res5);
}

*/
