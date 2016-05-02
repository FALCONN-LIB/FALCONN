#include "falconn/core/lsh_table.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "falconn/core/composite_hash_table.h"
#include "falconn/core/data_storage.h"
#include "falconn/core/hyperplane_hash.h"
#include "falconn/core/probing_hash_table.h"
#include "test_utils.h"

namespace fc = falconn::core;
namespace ft = falconn::test;

using fc::DynamicCompositeHashTable;
using fc::DynamicLinearProbingHashTable;
// using lsh::DynamicLSHTable;
using fc::HyperplaneHashDense;
using fc::HyperplaneHashSparse;
using fc::PlainArrayDataStorage;
using fc::StaticLSHTable;
using fc::StaticLinearProbingHashTable;
using fc::StaticCompositeHashTable;
using ft::check_result;
using std::make_pair;
using std::vector;

typedef HyperplaneHashDense<float>::VectorType DenseVector;
// typedef HyperplaneHashSparse<float>::VectorType SparseVector;

int default_num_threads = 1;

// TODO: test for sparse vectors
// TODO: tests for dynamic tables

// TODO: make this test robust
TEST(LSHTableTest, LSHTableGetCandidatesTest1) {
  const int dim = 5;
  int k = 5;
  int l = 2;
  int seed = 65840120;
  int table_size = 10;

  DenseVector p1(dim);
  p1[0] = 1.0;
  p1[1] = 0.0;
  p1[2] = 0.0;
  p1[3] = 0.0;
  p1[4] = 0.0;
  DenseVector p2(dim);
  p2[0] = 0.8;
  p2[1] = 0.2;
  p2[2] = 0.0;
  p2[3] = 0.0;
  p2[4] = 0.0;
  DenseVector p3(dim);
  p3[0] = 0.0;
  p3[1] = 0.0;
  p3[2] = 1.0;
  p3[3] = 0.0;
  p3[4] = 0.0;
  vector<DenseVector> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);

  HyperplaneHashDense<float> lsh_object(dim, k, l, seed);
  StaticLinearProbingHashTable<uint32_t>::Factory table_factory(table_size);
  typedef StaticCompositeHashTable<uint32_t, int32_t,
                                   StaticLinearProbingHashTable<uint32_t>>
      CompositeTableType;
  CompositeTableType hash_table(l, &table_factory);
  typedef StaticLSHTable<DenseVector, int32_t, HyperplaneHashDense<float>,
                         uint32_t, CompositeTableType>
      LSHTableType;
  LSHTableType lsh_table(&lsh_object, &hash_table, points, default_num_threads);
  LSHTableType::Query query(lsh_table);

  vector<int32_t> res1;
  query.get_unique_candidates(p1, l, -1, &res1);
  vector<int32_t> expected1 = {0, 1};
  check_result(make_pair(res1.begin(), res1.end()), expected1);

  vector<int32_t> res2;
  query.get_unique_candidates(p2, l, -1, &res2);
  vector<int32_t> expected2 = {0, 1};
  check_result(make_pair(res2.begin(), res2.end()), expected2);

  vector<int32_t> res3;
  query.get_unique_candidates(p3, l, -1, &res3);
  vector<int32_t> expected3 = {2};
  check_result(make_pair(res3.begin(), res3.end()), expected3);

  DenseVector p4(dim);
  p4[0] = 0.0;
  p4[1] = 0.0;
  p4[2] = 0.0;
  p4[3] = 0.0;
  p4[4] = 1.0;
  vector<int32_t> res4;
  query.get_unique_candidates(p4, l, -1, &res4);
  vector<int32_t> expected4 = {};
  check_result(make_pair(res4.begin(), res4.end()), expected4);
}

TEST(LSHTableTest, LSHTableGetCandidatesTest2) {
  const int dim = 5;
  int k = 2;
  int l = 2;
  int seed = 6584012;
  int table_size = 10;

  DenseVector p1(dim);
  p1[0] = 1.0;
  p1[1] = 0.0;
  p1[2] = 0.0;
  p1[3] = 0.0;
  p1[4] = 0.0;
  DenseVector p2(dim);
  p2[0] = 0.8;
  p2[1] = 0.2;
  p2[2] = 0.0;
  p2[3] = 0.0;
  p2[4] = 0.0;
  DenseVector p3(dim);
  p3[0] = 0.0;
  p3[1] = 0.0;
  p3[2] = 1.0;
  p3[3] = 0.0;
  p3[4] = 0.0;
  vector<DenseVector> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);

  HyperplaneHashDense<float> lsh_object(dim, k, l, seed);
  StaticLinearProbingHashTable<uint32_t>::Factory table_factory(table_size);
  typedef StaticCompositeHashTable<uint32_t, int32_t,
                                   StaticLinearProbingHashTable<uint32_t>>
      CompositeTableType;
  CompositeTableType hash_table(l, &table_factory);
  typedef StaticLSHTable<DenseVector, int32_t, HyperplaneHashDense<float>,
                         uint32_t, CompositeTableType>
      LSHTableType;
  LSHTableType lsh_table(&lsh_object, &hash_table, points, default_num_threads);
  LSHTableType::Query query(lsh_table);

  vector<int32_t> res1;
  query.get_unique_candidates(p1, l, -1, &res1);
  vector<int32_t> expected1 = {0, 1};
  check_result(make_pair(res1.begin(), res1.end()), expected1);

  vector<int32_t> res2;
  query.get_unique_candidates(p2, l, -1, &res2);
  vector<int32_t> expected2 = {0, 1};
  check_result(make_pair(res2.begin(), res2.end()), expected2);

  vector<int32_t> res3;
  query.get_unique_candidates(p3, l, -1, &res3);
  vector<int32_t> expected3 = {2};
  check_result(make_pair(res3.begin(), res3.end()), expected3);

  DenseVector p4(dim);
  p4[0] = 0.0;
  p4[1] = 0.0;
  p4[2] = 0.0;
  p4[3] = 0.0;
  p4[4] = 1.0;
  vector<int32_t> res4;
  query.get_unique_candidates(p4, l, -1, &res4);
  vector<int32_t> expected4 = {0, 1};
  check_result(make_pair(res4.begin(), res4.end()), expected4);
}

TEST(LSHTableTest, LSHTableGetCandidatesTest3) {
  const int dim = 5;
  int k = 2;
  int l = 2;
  int seed = 6584012;
  int table_size = 10;

  DenseVector p1(dim);
  p1[0] = 1.0;
  p1[1] = 0.0;
  p1[2] = 0.0;
  p1[3] = 0.0;
  p1[4] = 0.0;
  DenseVector p2(dim);
  p2[0] = 0.8;
  p2[1] = 0.2;
  p2[2] = 0.0;
  p2[3] = 0.0;
  p2[4] = 0.0;
  DenseVector p3(dim);
  p3[0] = 0.0;
  p3[1] = 0.0;
  p3[2] = 1.0;
  p3[3] = 0.0;
  p3[4] = 0.0;
  vector<DenseVector> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);

  HyperplaneHashDense<float> lsh_object(dim, k, l, seed);
  StaticLinearProbingHashTable<uint32_t>::Factory table_factory(table_size);
  typedef StaticCompositeHashTable<uint32_t, int32_t,
                                   StaticLinearProbingHashTable<uint32_t>>
      CompositeTableType;
  CompositeTableType hash_table(l, &table_factory);
  typedef StaticLSHTable<DenseVector, int32_t, HyperplaneHashDense<float>,
                         uint32_t, CompositeTableType>
      LSHTableType;
  LSHTableType lsh_table(&lsh_object, &hash_table, points, default_num_threads);
  LSHTableType::Query query(lsh_table);

  int max_num_candidates = 1;

  vector<int32_t> res1;
  query.get_unique_candidates(p1, l, max_num_candidates, &res1);
  vector<int32_t> expected1 = {0};
  check_result(make_pair(res1.begin(), res1.end()), expected1);

  vector<int32_t> res2;
  query.get_unique_candidates(p2, l, max_num_candidates, &res2);
  vector<int32_t> expected2 = {0};
  check_result(make_pair(res2.begin(), res2.end()), expected2);

  vector<int32_t> res3;
  query.get_unique_candidates(p3, l, max_num_candidates, &res3);
  vector<int32_t> expected3 = {2};
  check_result(make_pair(res3.begin(), res3.end()), expected3);

  DenseVector p4(dim);
  p4[0] = 0.0;
  p4[1] = 0.0;
  p4[2] = 0.0;
  p4[3] = 0.0;
  p4[4] = 1.0;
  vector<int32_t> res4;
  query.get_unique_candidates(p4, l, max_num_candidates, &res4);
  vector<int32_t> expected4 = {0};
  check_result(make_pair(res4.begin(), res4.end()), expected4);
}

TEST(LSHTableTest, LSHTableGetCandidatesTest4) {
  const int dim = 4;
  int k = 1;
  int l = 1;
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
  DenseVector p4(dim);
  p4[0] = 0.0;
  p4[1] = 0.0;
  p4[2] = 0.0;
  p4[3] = 1.0;
  vector<DenseVector> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);

  int max_num_candidates = 1;
  int num_probes = 2;

  int num_trials = 20000;
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<> dist(1, 1000000000);

  typedef StaticCompositeHashTable<uint32_t, int32_t,
                                   StaticLinearProbingHashTable<uint32_t>>
      CompositeTableType;
  typedef StaticLSHTable<DenseVector, int32_t, HyperplaneHashDense<float>,
                         uint32_t, CompositeTableType>
      LSHTableType;

  std::vector<double> dist1(3, 0.0), dist2(3, 0.0), dist3(3, 0.0),
      dist4(3, 0.0);

  for (int trial = 0; trial < num_trials; ++trial) {
    int trial_seed = dist(gen);

    std::vector<int> point_permutation = {0, 1, 2};
    std::shuffle(point_permutation.begin(), point_permutation.end(), gen);

    vector<DenseVector> cur_points;
    for (int ii = 0; ii < 3; ++ii) {
      cur_points.push_back(points[point_permutation[ii]]);
    }

    HyperplaneHashDense<float> lsh_object(dim, k, l, trial_seed);
    StaticLinearProbingHashTable<uint32_t>::Factory table_factory(table_size);
    CompositeTableType hash_table(l, &table_factory);
    LSHTableType lsh_table(&lsh_object, &hash_table, cur_points,
                           default_num_threads);
    LSHTableType::Query query(lsh_table);

    vector<int32_t> res1;
    query.get_unique_candidates(p1, num_probes, max_num_candidates, &res1);
    ASSERT_EQ(1u, res1.size());
    ASSERT_LT(res1[0], 3);
    ASSERT_GE(res1[0], 0);
    dist1[point_permutation[res1[0]]] += 1.0 / num_trials;

    vector<int32_t> res2;
    query.get_unique_candidates(p2, num_probes, max_num_candidates, &res2);
    ASSERT_EQ(1u, res2.size());
    ASSERT_LT(res2[0], 3);
    ASSERT_GE(res2[0], 0);
    dist2[point_permutation[res2[0]]] += 1.0 / num_trials;

    vector<int32_t> res3;
    query.get_unique_candidates(p3, num_probes, max_num_candidates, &res3);
    ASSERT_EQ(1u, res3.size());
    ASSERT_LT(res3[0], 3);
    ASSERT_GE(res3[0], 0);
    dist3[point_permutation[res3[0]]] += 1.0 / num_trials;

    vector<int32_t> res4;
    query.get_unique_candidates(p4, num_probes, max_num_candidates, &res4);
    ASSERT_EQ(1u, res4.size());
    ASSERT_LT(res4[0], 3);
    ASSERT_GE(res4[0], 0);
    dist4[point_permutation[res4[0]]] += 1.0 / num_trials;
  }

  EXPECT_GT(dist1[0], dist1[1]);
  EXPECT_GT(dist1[0], dist1[2]);
  EXPECT_GT(dist1[1], dist1[2]);

  EXPECT_GT(dist2[1], dist2[0]);
  EXPECT_GT(dist2[1], dist2[2]);
  EXPECT_GT(dist2[0], dist2[2]);

  EXPECT_GT(dist3[2], dist3[0]);
  EXPECT_GT(dist3[2], dist3[1]);
  EXPECT_NEAR(dist3[0], dist3[1], 0.01);

  EXPECT_NEAR(dist4[0], dist4[1], 0.01);
  EXPECT_GT(dist4[2], dist4[0]);
  EXPECT_GT(dist4[2], dist4[1]);
}

TEST(LSHTableTest, LSHTableGetCandidatesTest5) {
  const int dim = 4;
  int k = 1;
  int l = 2;
  int seed = 34562798;
  int table_size = 10;

  DenseVector p1(dim);
  p1[0] = 1.0;
  p1[1] = 0.0;
  p1[2] = 0.0;
  p1[3] = 0.0;
  vector<DenseVector> points;
  points.push_back(p1);

  HyperplaneHashDense<float> lsh_object(dim, k, l, seed);
  StaticLinearProbingHashTable<uint32_t>::Factory table_factory(table_size);
  typedef StaticCompositeHashTable<uint32_t, int32_t,
                                   StaticLinearProbingHashTable<uint32_t>>
      CompositeTableType;
  CompositeTableType hash_table(l, &table_factory);
  typedef StaticLSHTable<DenseVector, int32_t, HyperplaneHashDense<float>,
                         uint32_t, CompositeTableType>
      LSHTableType;
  LSHTableType lsh_table(&lsh_object, &hash_table, points, default_num_threads);
  LSHTableType::Query query(lsh_table);

  vector<int32_t> res1;
  query.get_unique_candidates(p1, 2 * l, -1, &res1);
  vector<int32_t> expected1 = {0};
  check_result(make_pair(res1.begin(), res1.end()), expected1);

  vector<int32_t> res2;
  query.get_candidates_with_duplicates(p1, l, -1, &res2);
  vector<int32_t> expected2 = {0, 0};
  check_result(make_pair(res2.begin(), res2.end()), expected2);
}

// Using a different DataStorage.
TEST(LSHTableTest, LSHTableGetCandidatesTest6) {
  typedef Eigen::Map<const DenseVector> ConstVectorMap;
  const int dim = 4;
  int k = 5;
  int l = 2;
  int seed = 65840120;
  int table_size = 10;

  float data[] = {1.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  int num_points = 3;

  PlainArrayDataStorage<DenseVector, int32_t> ds(data, num_points, dim);

  HyperplaneHashDense<float> lsh_object(dim, k, l, seed);
  StaticLinearProbingHashTable<uint32_t>::Factory table_factory(table_size);
  typedef StaticCompositeHashTable<uint32_t, int32_t,
                                   StaticLinearProbingHashTable<uint32_t>>
      CompositeTableType;
  CompositeTableType hash_table(l, &table_factory);
  typedef StaticLSHTable<DenseVector, int32_t, HyperplaneHashDense<float>,
                         uint32_t, CompositeTableType,
                         PlainArrayDataStorage<DenseVector, int32_t>>
      LSHTableType;
  LSHTableType lsh_table(&lsh_object, &hash_table, ds, default_num_threads);
  LSHTableType::Query query(lsh_table);

  vector<int32_t> res1;
  ConstVectorMap p1(data, dim);
  query.get_unique_candidates(p1, l, -1, &res1);
  vector<int32_t> expected1 = {0, 1};
  check_result(make_pair(res1.begin(), res1.end()), expected1);

  vector<int32_t> res2;
  ConstVectorMap p2(data + dim, dim);
  query.get_unique_candidates(p2, l, -1, &res2);
  vector<int32_t> expected2 = {0, 1};
  check_result(make_pair(res2.begin(), res2.end()), expected2);

  vector<int32_t> res3;
  ConstVectorMap p3(data + 2 * dim, dim);
  query.get_unique_candidates(p3, l, -1, &res3);
  vector<int32_t> expected3 = {2};
  check_result(make_pair(res3.begin(), res3.end()), expected3);

  DenseVector p4(dim);
  p4[0] = 0.0;
  p4[1] = 0.0;
  p4[2] = 0.0;
  p4[3] = 1.0;
  vector<int32_t> res4;
  query.get_unique_candidates(p4, l, -1, &res4);
  vector<int32_t> expected4 = {};
  check_result(make_pair(res4.begin(), res4.end()), expected4);
}

TEST(LSHTableTest, LSHTableMultithreadedTest1) {
  const int num_threads = 2;
  const int dim = 5;
  int k = 2;
  int l = 2;
  int seed = 6584012;
  int table_size = 10;

  DenseVector p1(dim);
  p1[0] = 1.0;
  p1[1] = 0.0;
  p1[2] = 0.0;
  p1[3] = 0.0;
  p1[4] = 0.0;
  DenseVector p2(dim);
  p2[0] = 0.8;
  p2[1] = 0.2;
  p2[2] = 0.0;
  p2[3] = 0.0;
  p2[4] = 0.0;
  DenseVector p3(dim);
  p3[0] = 0.0;
  p3[1] = 0.0;
  p3[2] = 1.0;
  p3[3] = 0.0;
  p3[4] = 0.0;
  vector<DenseVector> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);

  HyperplaneHashDense<float> lsh_object(dim, k, l, seed);
  StaticLinearProbingHashTable<uint32_t>::Factory table_factory(table_size);
  typedef StaticCompositeHashTable<uint32_t, int32_t,
                                   StaticLinearProbingHashTable<uint32_t>>
      CompositeTableType;
  CompositeTableType hash_table(l, &table_factory);
  typedef StaticLSHTable<DenseVector, int32_t, HyperplaneHashDense<float>,
                         uint32_t, CompositeTableType>
      LSHTableType;
  LSHTableType lsh_table(&lsh_object, &hash_table, points, num_threads);
  LSHTableType::Query query(lsh_table);

  vector<int32_t> res1;
  query.get_unique_candidates(p1, l, -1, &res1);
  vector<int32_t> expected1 = {0, 1};
  check_result(make_pair(res1.begin(), res1.end()), expected1);

  vector<int32_t> res2;
  query.get_unique_candidates(p2, l, -1, &res2);
  vector<int32_t> expected2 = {0, 1};
  check_result(make_pair(res2.begin(), res2.end()), expected2);

  vector<int32_t> res3;
  query.get_unique_candidates(p3, l, -1, &res3);
  vector<int32_t> expected3 = {2};
  check_result(make_pair(res3.begin(), res3.end()), expected3);

  DenseVector p4(dim);
  p4[0] = 0.0;
  p4[1] = 0.0;
  p4[2] = 0.0;
  p4[3] = 0.0;
  p4[4] = 1.0;
  vector<int32_t> res4;
  query.get_unique_candidates(p4, l, -1, &res4);
  vector<int32_t> expected4 = {0, 1};
  check_result(make_pair(res4.begin(), res4.end()), expected4);
}
