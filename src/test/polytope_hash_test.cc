#include "falconn/core/polytope_hash.h"

#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "falconn/core/data_storage.h"
#include "test_utils.h"

namespace fc = falconn::core;
namespace ft = falconn::test;

using fc::ArrayDataStorage;
using fc::CrossPolytopeHashDense;
using fc::CrossPolytopeHashSparse;
using fc::cp_hash_helpers::FHTHelper;
using fc::cp_hash_helpers::compute_k_parameters_for_bits;
using fc::log2ceil;
using ft::count_bits;
using std::make_pair;
using std::pair;
using std::vector;

typedef CrossPolytopeHashDense<float, uint32_t> CPHD;
typedef CPHD::VectorType DenseVector;
typedef CPHD::Query DenseQuery;
typedef CrossPolytopeHashSparse<float>::VectorType SparseVector;
typedef CrossPolytopeHashSparse<float> SparseCPHash;
typedef CrossPolytopeHashSparse<float>::Query SparseQuery;

TEST(PolytopeHashTest, Log2CeilTest) {
  EXPECT_EQ(0, log2ceil(1));
  EXPECT_EQ(1, log2ceil(2));
  EXPECT_EQ(2, log2ceil(3));
  EXPECT_EQ(2, log2ceil(4));
  EXPECT_EQ(3, log2ceil(5));
  EXPECT_EQ(3, log2ceil(6));
  EXPECT_EQ(3, log2ceil(7));
  EXPECT_EQ(3, log2ceil(8));
  EXPECT_EQ(4, log2ceil(9));
  EXPECT_EQ(7, log2ceil(127));
  EXPECT_EQ(7, log2ceil(128));
  EXPECT_EQ(8, log2ceil(129));
  EXPECT_EQ(8, log2ceil(256));
  EXPECT_EQ(9, log2ceil(257));
  EXPECT_EQ(10, log2ceil(513));
  EXPECT_EQ(10, log2ceil(1024));
  EXPECT_EQ(11, log2ceil(1025));
  EXPECT_EQ(11, log2ceil(2048));
}

TEST(PolytopeHashTest, FHTHelperTest1) {
  // Repeating multiple times to make it more likely for potential alignment
  // issues to occur.
  for (int trial = 0; trial < 100; ++trial) {
    int dim = 16;
    FHTHelper<float> fht(dim);
    DenseVector v(dim);
    v.setZero();
    v[0] = 1.0;
    fht.apply(v.data());

    float expected_coordinate = 1.0;
    for (int ii = 0; ii < dim; ++ii) {
      EXPECT_NEAR(expected_coordinate, v[ii], 0.00001);
    }
  }
}

TEST(PolytopeHashTest, FHTHelperTest2) {
  // Repeating multiple times to make it more likely for potential alignment
  // issues to occur.
  for (int trial = 0; trial < 100; ++trial) {
    int dim = 128;
    FHTHelper<float> fht(dim);
    DenseVector v(dim);
    v.setZero();
    v[0] = 1.0;
    fht.apply(v.data());

    float expected_coordinate = 1.0;
    for (int ii = 0; ii < dim; ++ii) {
      EXPECT_NEAR(expected_coordinate, v[ii], 0.00001);
    }
  }
}

TEST(PolytopeHashTest, FHTHelperTest3) {
  // Repeating multiple times to make it more likely for potential alignment
  // issues to occur.
  for (int trial = 0; trial < 100; ++trial) {
    int dim = 128;
    FHTHelper<double> fht(dim);
    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor> v(dim);
    v.setZero();
    v[0] = 1.0;
    fht.apply(v.data());

    double expected_coordinate = 1.0;
    for (int ii = 0; ii < dim; ++ii) {
      EXPECT_NEAR(expected_coordinate, v[ii], 0.00001);
    }
  }
}

TEST(PolytopeHashTest, KParametersForBitsTest) {
  int dim = 8;
  int bits = 8;
  int_fast32_t k;
  int_fast32_t last_cp_dim;
  compute_k_parameters_for_bits(dim, bits, &k, &last_cp_dim);
  EXPECT_EQ(2, k);
  EXPECT_EQ(8, last_cp_dim);

  dim = 9;
  bits = 8;
  compute_k_parameters_for_bits(dim, bits, &k, &last_cp_dim);
  EXPECT_EQ(2, k);
  EXPECT_EQ(4, last_cp_dim);

  dim = 128;
  bits = 20;
  compute_k_parameters_for_bits(dim, bits, &k, &last_cp_dim);
  EXPECT_EQ(3, k);
  EXPECT_EQ(8, last_cp_dim);
}

// TODO: change (most) ASSERT to EXPECT below

TEST(PolytopeHashTest, DecodeCPTest1) {
  int dim = 4;
  DenseVector data1(dim);
  data1 << 0.0, 3.0, 4.0, -2.0;
  uint32_t res1 = CPHD::decodeCP(data1, dim);
  ASSERT_EQ(2u, res1);

  DenseVector data2(dim);
  data2 << -9.0, 3.0, -4.0, 2.0;
  uint32_t res2 = CPHD::decodeCP(data2, dim);
  ASSERT_EQ(4u, res2);

  DenseVector data3(dim);
  data3 << 2.99, -3.0, 3.0, 2.0;
  uint32_t res3 = CPHD::decodeCP(data3, dim);
  ASSERT_EQ(5u, res3);

  uint32_t res4 = CPHD::decodeCP(data1, 2);
  ASSERT_EQ(1u, res4);
}

TEST(PolytopeHashTest, HashTest1) {
  DenseVector v1(4);
  v1[0] = 1.0;
  v1[1] = 0.5;
  v1[2] = 0.0;
  v1[3] = 0.0;
  DenseVector v2(4);
  v2[0] = 1.0;
  v2[1] = 0.501;
  v2[2] = 0.0;
  v2[3] = 0.0;
  DenseVector v3(4);
  v3[0] = 0.001;
  v3[1] = 0.0;
  v3[2] = 1.0;
  v3[3] = 0.5;

  int dim = 4;
  int k = 3;
  int l = 2;
  int num_rotations = 3;
  uint64_t seed = 52341829;
  CrossPolytopeHashDense<float> hash(dim, k, l, num_rotations, dim, seed);
  vector<uint32_t> result1(l), result2(l), result3(l);
  hash.hash(v1, &result1);
  hash.hash(v2, &result2);
  hash.hash(v3, &result3);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(result1[ii], result2[ii]);
    ASSERT_NE(result1[ii], result3[ii]);
  }
}

TEST(PolytopeHashTest, HashTest2) {
  DenseVector v1(3);
  v1[0] = 1.0;
  v1[1] = 0.5;
  v1[2] = 0.0;
  DenseVector v2(3);
  v2[0] = 1.0;
  v2[1] = 0.501;
  v2[2] = 0.0;
  DenseVector v3(3);
  v3[0] = 0.001;
  v3[1] = 0.0;
  v3[2] = 1.0;

  int dim = 3;
  int k = 3;
  int l = 2;
  int num_rotations = 3;
  uint64_t seed = 52341829;
  CrossPolytopeHashDense<float> hash(dim, k, l, num_rotations, 4, seed);
  vector<uint32_t> result1(l), result2(l), result3(l);
  hash.hash(v1, &result1);
  hash.hash(v2, &result2);
  hash.hash(v3, &result3);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(result1[ii], result2[ii]);
    ASSERT_NE(result1[ii], result3[ii]);
  }
}

TEST(PolytopeHashTest, OddLastCPDim) {
  DenseVector v1(4);
  v1[0] = 1.0;
  v1[1] = 0.5;
  v1[2] = 0.0;
  v1[3] = 0.0;
  DenseVector v2(4);
  v2[0] = 1.0;
  v2[1] = 0.501;
  v2[2] = 0.0;
  v2[3] = 0.0;
  DenseVector v3(4);
  v3[0] = 0.001;
  v3[1] = 0.0;
  v3[2] = 1.0;
  v3[3] = 0.5;

  int dim = 4;
  int k = 3;
  int l = 2;
  int num_rotations = 3;
  uint64_t seed = 52341829;
  CrossPolytopeHashDense<float> hash(dim, k, l, num_rotations, dim - 1, seed);
  vector<uint32_t> result1(l), result2(l), result3(l);
  hash.hash(v1, &result1);
  hash.hash(v2, &result2);
  hash.hash(v3, &result3);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(result1[ii], result2[ii]);
    ASSERT_NE(result1[ii], result3[ii]);
  }
}

TEST(PolytopeHashTest, SparseHashTest1) {
  SparseVector v1;
  v1.push_back(make_pair(0, 1.0));
  v1.push_back(make_pair(1, 0.5));
  SparseVector v2;
  v2.push_back(make_pair(0, 1.0));
  v2.push_back(make_pair(1, 0.501));
  SparseVector v3;
  v3.push_back(make_pair(0, 0.001));
  v3.push_back(make_pair(2, 1.0));
  v3.push_back(make_pair(3, 0.5));

  int dim = 16;
  int k = 3;
  int l = 2;
  int num_rotations = 3;
  int feature_hashing_dim = 4;
  uint64_t seed = 14032009;
  CrossPolytopeHashSparse<float> hash(
      dim, k, l, num_rotations, feature_hashing_dim, feature_hashing_dim, seed);
  vector<uint32_t> result1(l), result2(l), result3(l);
  hash.hash(v1, &result1);
  hash.hash(v2, &result2);
  hash.hash(v3, &result3);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(result1[ii], result2[ii]);
    ASSERT_NE(result1[ii], result3[ii]);
  }
}

TEST(PolytopeHashTest, DenseMultiprobeTest1) {
  DenseVector v1(4);
  v1[0] = 1.0;
  v1[1] = 0.5;
  v1[2] = 0.0;
  v1[3] = 0.0;
  DenseVector v2(4);
  v2[0] = 1.0;
  v2[1] = 0.501;
  v2[2] = 0.0;
  v2[3] = 0.0;
  DenseVector v3(4);
  v3[0] = 0.001;
  v3[1] = 0.0;
  v3[2] = 1.0;
  v3[3] = 0.5;

  int dim = 4;
  int k = 3;
  int l = 2;
  int num_rotations = 3;
  uint64_t seed = 52341829;
  CrossPolytopeHashDense<float> hash(dim, k, l, num_rotations, dim, seed);
  vector<uint32_t> hashes1(l), hashes2(l), hashes3(l);
  hash.hash(v1, &hashes1);
  hash.hash(v2, &hashes2);
  hash.hash(v3, &hashes3);

  DenseQuery query(hash);

  vector<vector<uint32_t>> probes_by_table1, probes_by_table2, probes_by_table3;
  query.get_probes_by_table(v1, &probes_by_table1, l);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(1u, probes_by_table1[ii].size());
    ASSERT_EQ(probes_by_table1[ii][0], hashes1[ii]);
  }
  query.get_probes_by_table(v2, &probes_by_table2, l);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(1u, probes_by_table2[ii].size());
    ASSERT_EQ(probes_by_table2[ii][0], hashes2[ii]);
  }
  query.get_probes_by_table(v3, &probes_by_table3, l);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(1u, probes_by_table3[ii].size());
    ASSERT_EQ(probes_by_table3[ii][0], hashes3[ii]);
  }
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(probes_by_table1[ii][0], probes_by_table2[ii][0]);
    ASSERT_NE(probes_by_table1[ii][0], probes_by_table3[ii][0]);
  }
}

TEST(PolytopeHashTest, DenseMultiprobeTest2) {
  DenseVector v1(4);
  v1[0] = 1.0;
  v1[1] = 0.2;
  v1[2] = 0.1;
  v1[3] = 0.05;

  int dim = 4;
  int k = 3;
  int l = 1;
  int num_rotations = 3;
  uint64_t seed = 54320123;
  CrossPolytopeHashDense<float> hash(dim, k, l, num_rotations, dim - 2, seed);
  DenseQuery query(hash);

  vector<uint32_t> hashes(l);
  hash.hash(v1, &hashes);
  ASSERT_EQ(1u, hashes.size());

  vector<vector<uint32_t>> probes_by_table;
  query.get_probes_by_table(v1, &probes_by_table, 2);

  ASSERT_EQ(1u, probes_by_table.size());
  ASSERT_EQ(2u, probes_by_table[0].size());
  ASSERT_EQ(hashes[0], probes_by_table[0][0]);

  uint32_t bitdiff = probes_by_table[0][0] ^ probes_by_table[0][1];
  ASSERT_GE(count_bits(bitdiff), 1);
  ASSERT_LE(count_bits(bitdiff), 3);
}

TEST(PolytopeHashTest, DenseMultiprobeTest3) {
  DenseVector v1(4);
  v1[0] = 1.0;
  v1[1] = 0.2;
  v1[2] = 0.1;
  v1[3] = 0.05;

  int dim = 4;
  int k = 3;
  int l = 1;
  int num_rotations = 3;
  uint64_t seed = 54320123;
  CrossPolytopeHashDense<float> hash(dim, k, l, num_rotations, dim - 2, seed);
  DenseQuery query(hash);

  vector<uint32_t> hashes(l);
  hash.hash(v1, &hashes);
  ASSERT_EQ(1u, hashes.size());

  // 256 instead of 512 because the last CP has dim 2, not 4.
  vector<vector<uint32_t>> probes_by_table;
  query.get_probes_by_table(v1, &probes_by_table, 256);

  ASSERT_EQ(1u, probes_by_table.size());
  ASSERT_EQ(256u, probes_by_table[0].size());
  ASSERT_EQ(hashes[0], probes_by_table[0][0]);

  uint32_t bitdiff = probes_by_table[0][0] ^ probes_by_table[0][1];
  ASSERT_GE(count_bits(bitdiff), 1);
  ASSERT_LE(count_bits(bitdiff), 3);

  sort(probes_by_table[0].begin(), probes_by_table[0].end());
  for (unsigned int ii = 0; ii < 256; ++ii) {
    ASSERT_EQ(ii, probes_by_table[0][ii]);
  }
}

TEST(PolytopeHashTest, DenseMultiprobeTest4) {
  int num_trials = 1000;
  uint64_t seed = 541873389;
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<int> dist(0, 1000000000);
  std::normal_distribution<float> normal_dist(0, 1);
  int dim = 128;
  int log_dim = static_cast<int>(std::round(log2(dim)));
  DenseVector v1(dim);
  float length = 0.0;
  for (int ii = 0; ii < dim; ++ii) {
    v1[ii] = normal_dist(gen);
    length += v1[ii] * v1[ii];
  }
  for (int ii = 0; ii < dim; ++ii) {
    v1[ii] /= length;
  }

  unsigned int num_probes = 200;

  int k = 3;
  int l = 1;
  int num_rotations = 3;

  for (int ii = 0; ii < num_trials; ++ii) {
    CrossPolytopeHashDense<float> hash(dim, k, l, num_rotations, 8, dist(gen));
    DenseQuery query(hash);

    vector<uint32_t> hashes(l);
    hash.hash(v1, &hashes);
    ASSERT_EQ(1u, hashes.size());

    vector<vector<uint32_t>> probes_by_table;
    query.get_probes_by_table(v1, &probes_by_table, num_probes);

    ASSERT_EQ(1u, probes_by_table.size());
    ASSERT_EQ(num_probes, probes_by_table[0].size());
    ASSERT_EQ(hashes[0], probes_by_table[0][0]);

    for (int ii = 1; ii <= 2; ++ii) {
      uint32_t bitdiff = probes_by_table[0][0] ^ probes_by_table[0][ii];
      ASSERT_GE(count_bits(bitdiff), 1);
      ASSERT_LE(count_bits(bitdiff), log_dim + 1);
    }

    uint32_t bitdiff = probes_by_table[0][0] ^ probes_by_table[0][3];
    ASSERT_GE(count_bits(bitdiff), 1);
    ASSERT_LE(count_bits(bitdiff), 2 * (log_dim + 1));
  }
}

TEST(PolytopeHashTest, DenseMultiprobeTest5) {
  int num_trials = 1000;
  uint64_t seed = 2442989;
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<int> dist(0, 1000000000);
  std::normal_distribution<float> normal_dist(0, 1);
  int dim = 512;
  int log_dim = static_cast<int>(std::round(log2(dim)));
  DenseVector v1(dim);
  float length = 0.0;
  for (int ii = 0; ii < dim; ++ii) {
    v1[ii] = normal_dist(gen);
    length += v1[ii] * v1[ii];
  }
  for (int ii = 0; ii < dim; ++ii) {
    v1[ii] /= length;
  }

  unsigned int num_probes = 200;

  int k = 2;
  int l = 1;
  int num_rotations = 1;

  for (int ii = 0; ii < num_trials; ++ii) {
    CrossPolytopeHashDense<float> hash(dim, k, l, num_rotations, 8, dist(gen));
    DenseQuery query(hash);

    vector<uint32_t> hashes(l);
    hash.hash(v1, &hashes);
    ASSERT_EQ(1u, hashes.size());

    vector<vector<uint32_t>> probes_by_table;
    query.get_probes_by_table(v1, &probes_by_table, num_probes);

    ASSERT_EQ(1u, probes_by_table.size());
    ASSERT_EQ(num_probes, probes_by_table[0].size());
    ASSERT_EQ(hashes[0], probes_by_table[0][0]);

    uint32_t bitdiff1 = probes_by_table[0][0] ^ probes_by_table[0][1];
    ASSERT_GE(count_bits(bitdiff1), 1);
    ASSERT_LE(count_bits(bitdiff1), log_dim + 1);

    uint32_t bitdiff2 = probes_by_table[0][0] ^ probes_by_table[0][2];
    ASSERT_GE(count_bits(bitdiff2), 1);
    ASSERT_LE(count_bits(bitdiff2), 2 * (log_dim + 1));
  }
}

// Same as DenseMultiprobeTest1 but with non-power-of-two vector dimension
TEST(PolytopeHashTest, DenseMultiprobeTest6) {
  DenseVector v1(3);
  v1[0] = 1.0;
  v1[1] = 0.5;
  v1[2] = 0.0;
  DenseVector v2(3);
  v2[0] = 1.0;
  v2[1] = 0.501;
  v2[2] = 0.0;
  DenseVector v3(3);
  v3[0] = 0.001;
  v3[1] = 0.0;
  v3[2] = 1.0;

  int dim = 3;
  int k = 3;
  int l = 2;
  int num_rotations = 3;
  uint64_t seed = 52341829;
  CrossPolytopeHashDense<float> hash(dim, k, l, num_rotations, 4, seed);
  vector<uint32_t> hashes1(l), hashes2(l), hashes3(l);
  hash.hash(v1, &hashes1);
  hash.hash(v2, &hashes2);
  hash.hash(v3, &hashes3);

  DenseQuery query(hash);

  vector<vector<uint32_t>> probes_by_table1, probes_by_table2, probes_by_table3;
  query.get_probes_by_table(v1, &probes_by_table1, l);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(1u, probes_by_table1[ii].size());
    ASSERT_EQ(probes_by_table1[ii][0], hashes1[ii]);
  }
  query.get_probes_by_table(v2, &probes_by_table2, l);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(1u, probes_by_table2[ii].size());
    ASSERT_EQ(probes_by_table2[ii][0], hashes2[ii]);
  }
  query.get_probes_by_table(v3, &probes_by_table3, l);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(1u, probes_by_table3[ii].size());
    ASSERT_EQ(probes_by_table3[ii][0], hashes3[ii]);
  }
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(probes_by_table1[ii][0], probes_by_table2[ii][0]);
    ASSERT_NE(probes_by_table1[ii][0], probes_by_table3[ii][0]);
  }
}

// Same as DenseMultiprobeTest3 but with non-power-of-two vector dimension
TEST(PolytopeHashTest, DenseMultiprobeTest7) {
  DenseVector v1(3);
  v1[0] = 1.0;
  v1[1] = 0.2;
  v1[2] = 0.1;

  int dim = 3;
  int k = 3;
  int l = 1;
  int num_rotations = 3;
  uint64_t seed = 54320123;
  CrossPolytopeHashDense<float> hash(dim, k, l, num_rotations, 2, seed);
  DenseQuery query(hash);

  vector<uint32_t> hashes(l);
  hash.hash(v1, &hashes);
  ASSERT_EQ(1u, hashes.size());

  // 256 instead of 512 because the last CP has dim 2, not 4.
  vector<vector<uint32_t>> probes_by_table;
  query.get_probes_by_table(v1, &probes_by_table, 256);

  ASSERT_EQ(1u, probes_by_table.size());
  ASSERT_EQ(256u, probes_by_table[0].size());
  ASSERT_EQ(hashes[0], probes_by_table[0][0]);

  uint32_t bitdiff = probes_by_table[0][0] ^ probes_by_table[0][1];
  ASSERT_GE(count_bits(bitdiff), 1);
  ASSERT_LE(count_bits(bitdiff), 3);

  sort(probes_by_table[0].begin(), probes_by_table[0].end());
  for (unsigned int ii = 0; ii < 256; ++ii) {
    ASSERT_EQ(ii, probes_by_table[0][ii]);
  }
}

// Same as DenseMultiprobeTest7 but with last_CP_dim > dim (dim is not a power
// of two)
TEST(PolytopeHashTest, DenseMultiprobeTest8) {
  DenseVector v1(3);
  v1[0] = 1.0;
  v1[1] = 0.2;
  v1[2] = 0.1;

  int dim = 3;
  int k = 3;
  int l = 1;
  int num_rotations = 3;
  uint64_t seed = 54320123;
  CrossPolytopeHashDense<float> hash(dim, k, l, num_rotations, 4, seed);
  DenseQuery query(hash);

  vector<uint32_t> hashes(l);
  hash.hash(v1, &hashes);
  ASSERT_EQ(1u, hashes.size());

  // Now 512 because the last CP has dim 4.
  vector<vector<uint32_t>> probes_by_table;
  query.get_probes_by_table(v1, &probes_by_table, 512);

  ASSERT_EQ(1u, probes_by_table.size());
  ASSERT_EQ(512u, probes_by_table[0].size());
  ASSERT_EQ(hashes[0], probes_by_table[0][0]);

  uint32_t bitdiff = probes_by_table[0][0] ^ probes_by_table[0][1];
  ASSERT_GE(count_bits(bitdiff), 1);
  ASSERT_LE(count_bits(bitdiff), 3);

  sort(probes_by_table[0].begin(), probes_by_table[0].end());
  for (unsigned int ii = 0; ii < 512; ++ii) {
    ASSERT_EQ(ii, probes_by_table[0][ii]);
  }
}

TEST(PolytopeHashTest, SparseMultiprobeTest1) {
  SparseVector v1;
  v1.push_back(make_pair(0, 1.0));
  v1.push_back(make_pair(1, 0.5));
  SparseVector v2;
  v2.push_back(make_pair(0, 1.0));
  v2.push_back(make_pair(1, 0.501));
  SparseVector v3;
  v3.push_back(make_pair(0, 0.001));
  v3.push_back(make_pair(2, 1.0));
  v3.push_back(make_pair(3, 0.5));

  int dim = 16;
  int k = 3;
  int l = 2;
  int num_rotations = 3;
  int feature_hashing_dim = 4;
  uint64_t seed = 14032009;
  CrossPolytopeHashSparse<float> hash(
      dim, k, l, num_rotations, feature_hashing_dim, feature_hashing_dim, seed);
  vector<uint32_t> hashes1(l), hashes2(l), hashes3(l);
  hash.hash(v1, &hashes1);
  hash.hash(v2, &hashes2);
  hash.hash(v3, &hashes3);

  SparseQuery query(hash);

  vector<vector<uint32_t>> probes_by_table1, probes_by_table2, probes_by_table3;
  query.get_probes_by_table(v1, &probes_by_table1, l);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(1u, probes_by_table1[ii].size());
    ASSERT_EQ(probes_by_table1[ii][0], hashes1[ii]);
  }
  query.get_probes_by_table(v2, &probes_by_table2, l);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(1u, probes_by_table2[ii].size());
    ASSERT_EQ(probes_by_table2[ii][0], hashes2[ii]);
  }
  query.get_probes_by_table(v3, &probes_by_table3, l);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(1u, probes_by_table3[ii].size());
    ASSERT_EQ(probes_by_table3[ii][0], hashes3[ii]);
  }
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(probes_by_table1[ii][0], probes_by_table2[ii][0]);
    ASSERT_NE(probes_by_table1[ii][0], probes_by_table3[ii][0]);
  }
}

TEST(PolytopeHashTest, SparseMultiprobeTest2) {
  SparseVector v1;
  v1.push_back(make_pair(0, 1.0));
  v1.push_back(make_pair(1, 0.5));
  for (int ii = 2; ii < 16; ++ii) {
    v1.push_back(make_pair(ii, 1.0 / (100.0 * ii)));
  }

  int dim = 16;
  int k = 3;
  int l = 1;
  int num_rotations = 3;
  int feature_hashing_dim = 4;
  uint64_t seed = 323309423;
  CrossPolytopeHashSparse<float> hash(dim, k, l, num_rotations,
                                      feature_hashing_dim,
                                      feature_hashing_dim - 2, seed);

  SparseQuery query(hash);

  vector<uint32_t> hashes(l);
  hash.hash(v1, &hashes);
  ASSERT_EQ(1u, hashes.size());

  vector<vector<uint32_t>> probes_by_table;
  query.get_probes_by_table(v1, &probes_by_table, 2);

  ASSERT_EQ(1u, probes_by_table.size());
  ASSERT_EQ(2u, probes_by_table[0].size());
  ASSERT_EQ(hashes[0], probes_by_table[0][0]);

  uint32_t bitdiff = probes_by_table[0][0] ^ probes_by_table[0][1];
  ASSERT_GE(count_bits(bitdiff), 1);
  ASSERT_LE(count_bits(bitdiff), 3);
}

TEST(PolytopeHashTest, SparseMultiprobeTest3) {
  SparseVector v1;
  v1.push_back(make_pair(0, 1.0));
  v1.push_back(make_pair(1, 0.5));
  for (int ii = 2; ii < 16; ++ii) {
    v1.push_back(make_pair(ii, 1.0 / (100.0 * ii)));
  }

  int dim = 16;
  int k = 3;
  int l = 1;
  int num_rotations = 3;
  int feature_hashing_dim = 4;
  uint64_t seed = 323309423;
  CrossPolytopeHashSparse<float> hash(dim, k, l, num_rotations,
                                      feature_hashing_dim,
                                      feature_hashing_dim - 2, seed);

  SparseQuery query(hash);

  vector<uint32_t> hashes(l);
  hash.hash(v1, &hashes);
  ASSERT_EQ(1u, hashes.size());

  // 256 instead of 512 because the last CP has dim 2, not 4.
  vector<vector<uint32_t>> probes_by_table;
  query.get_probes_by_table(v1, &probes_by_table, 256);

  ASSERT_EQ(1u, probes_by_table.size());
  ASSERT_EQ(256u, probes_by_table[0].size());
  ASSERT_EQ(hashes[0], probes_by_table[0][0]);

  uint32_t bitdiff = probes_by_table[0][0] ^ probes_by_table[0][1];
  ASSERT_GE(count_bits(bitdiff), 1);
  ASSERT_LE(count_bits(bitdiff), 3);

  sort(probes_by_table[0].begin(), probes_by_table[0].end());
  for (unsigned int ii = 0; ii < 256; ++ii) {
    ASSERT_EQ(ii, probes_by_table[0][ii]);
  }
}

TEST(PolytopeHashTest, DenseBatchHashTest1) {
  DenseVector v1(4);
  v1[0] = 1.0;
  v1[1] = 0.5;
  v1[2] = 0.0;
  v1[3] = 0.0;
  DenseVector v2(4);
  v2[0] = 1.0;
  v2[1] = 0.501;
  v2[2] = 0.0;
  v2[3] = 0.0;
  DenseVector v3(4);
  v3[0] = 0.001;
  v3[1] = 0.0;
  v3[2] = 1.0;
  v3[3] = 0.5;

  int dim = 4;
  int k = 3;
  int l = 2;
  int num_rotations = 3;
  uint64_t seed = 52341829;
  CrossPolytopeHashDense<float> hash(dim, k, l, num_rotations, dim, seed);
  vector<uint32_t> result1(l), result2(l), result3(l);
  hash.hash(v1, &result1);
  hash.hash(v2, &result2);
  hash.hash(v3, &result3);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(result1[ii], result2[ii]);
    ASSERT_NE(result1[ii], result3[ii]);
  }

  vector<DenseVector> vs = {v1, v2, v3};
  typedef ArrayDataStorage<DenseVector> BatchVectorType;
  BatchVectorType batch_data(vs);
  CPHD::BatchHash<BatchVectorType> bh(hash);
  vector<uint32_t> hashes;
  for (int ii = 0; ii < l; ++ii) {
    bh.batch_hash_single_table(batch_data, ii, &hashes);
    ASSERT_EQ(3u, hashes.size());
    ASSERT_EQ(result1[ii], hashes[0]);
    ASSERT_EQ(result2[ii], hashes[1]);
    ASSERT_EQ(result3[ii], hashes[2]);
  }
}

TEST(PolytopeHashTest, SparseBatchHashTest1) {
  SparseVector v1;
  v1.push_back(make_pair(0, 1.0));
  v1.push_back(make_pair(1, 0.5));
  SparseVector v2;
  v2.push_back(make_pair(0, 1.0));
  v2.push_back(make_pair(1, 0.501));
  SparseVector v3;
  v3.push_back(make_pair(0, 0.001));
  v3.push_back(make_pair(2, 1.0));
  v3.push_back(make_pair(3, 0.5));

  int dim = 16;
  int k = 3;
  int l = 2;
  int num_rotations = 3;
  int feature_hashing_dim = 4;
  uint64_t seed = 14032009;
  CrossPolytopeHashSparse<float> hash(
      dim, k, l, num_rotations, feature_hashing_dim, feature_hashing_dim, seed);
  vector<uint32_t> result1(l), result2(l), result3(l);
  hash.hash(v1, &result1);
  hash.hash(v2, &result2);
  hash.hash(v3, &result3);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(result1[ii], result2[ii]);
    ASSERT_NE(result1[ii], result3[ii]);
  }

  vector<SparseVector> vs = {v1, v2, v3};
  typedef ArrayDataStorage<SparseVector> BatchVectorType;
  BatchVectorType batch_data(vs);
  SparseCPHash::BatchHash<BatchVectorType> bh(hash);
  vector<uint32_t> hashes;
  for (int ii = 0; ii < l; ++ii) {
    bh.batch_hash_single_table(batch_data, ii, &hashes);
    ASSERT_EQ(3u, hashes.size());
    ASSERT_EQ(result1[ii], hashes[0]);
    ASSERT_EQ(result2[ii], hashes[1]);
    ASSERT_EQ(result3[ii], hashes[2]);
  }
}
