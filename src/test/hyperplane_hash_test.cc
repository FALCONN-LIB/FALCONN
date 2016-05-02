#include "falconn/core/hyperplane_hash.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "falconn/core/data_storage.h"
#include "test_utils.h"

namespace fc = falconn::core;
namespace ft = falconn::test;

using ft::count_bits;
using fc::ArrayDataStorage;
using fc::HyperplaneHashDense;
using fc::HyperplaneHashSparse;
using std::make_pair;
using std::sort;
using std::vector;

typedef HyperplaneHashDense<float>::VectorType DenseVector;
typedef HyperplaneHashDense<float>::MatrixType MatrixType;
typedef HyperplaneHashDense<float>::Query DenseQuery;
typedef HyperplaneHashDense<float> HPHashDense;
typedef HyperplaneHashSparse<float>::VectorType SparseVector;
typedef HyperplaneHashSparse<float>::Query SparseQuery;
typedef HyperplaneHashSparse<float> HPHashSparse;

TEST(HyperplaneHashTest, SparseHyperplaneHashTest1) {
  SparseVector v1;
  v1.push_back(make_pair(0, 1.0));
  SparseVector v2;
  v2.push_back(make_pair(0, 1.0));
  v2.push_back(make_pair(1, 0.001));
  SparseVector v3;
  v3.push_back(make_pair(0, 0.001));
  v3.push_back(make_pair(1, 1.0));

  int dim = 8;
  int k = 3;
  int l = 2;
  uint64_t seed = 3425890;
  HyperplaneHashSparse<float> hash(dim, k, l, seed);
  vector<uint32_t> result1(l), result2(l), result3(l);
  hash.hash(v1, &result1);
  hash.hash(v2, &result2);
  hash.hash(v3, &result3);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(result1[ii], result2[ii]);
    ASSERT_NE(result1[ii], result3[ii]);
  }
}

TEST(HyperplaneHashTest, DenseHyperplaneHashTest1) {
  DenseVector v1(4);
  v1[0] = 1.0;
  v1[1] = 0.0;
  v1[2] = 0.0;
  v1[3] = 0.0;
  DenseVector v2(4);
  v2[0] = 1.0;
  v2[1] = 0.001;
  v2[2] = 0.0;
  v2[3] = 0.0;
  DenseVector v3(4);
  v3[0] = 0.001;
  v3[1] = 1.0;
  v3[2] = 0.0;
  v3[3] = 0.0;

  int dim = 4;
  int k = 3;
  int l = 2;
  uint64_t seed = 45234528;
  HyperplaneHashDense<float> hash(dim, k, l, seed);
  vector<uint32_t> result1(l), result2(l), result3(l);
  hash.hash(v1, &result1);
  hash.hash(v2, &result2);
  hash.hash(v3, &result3);
  for (int ii = 0; ii < l; ++ii) {
    ASSERT_EQ(result1[ii], result2[ii]);
    ASSERT_NE(result1[ii], result3[ii]);
  }
}

TEST(HyperplaneHashTest, DenseHyperplaneMultiProbeTest1) {
  DenseVector v1(4);
  v1[0] = 1.0;
  v1[1] = 0.0;
  v1[2] = 0.0;
  v1[3] = 0.0;
  DenseVector v2(4);
  v2[0] = 1.0;
  v2[1] = 0.001;
  v2[2] = 0.0;
  v2[3] = 0.0;
  DenseVector v3(4);
  v3[0] = 0.001;
  v3[1] = 1.0;
  v3[2] = 0.0;
  v3[3] = 0.0;

  int dim = 4;
  int k = 3;
  int l = 2;
  uint64_t seed = 236718389;
  HyperplaneHashDense<float> hash(dim, k, l, seed);
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

float compute_score(const DenseVector& ips, uint32_t bitmask, int k) {
  float score = 0.0;
  for (int ii = 0; ii < k; ++ii) {
    if ((bitmask & (1 << ii)) != 0) {
      score += ips[k - ii - 1] * ips[k - ii - 1];
    }
  }
  return score;
}

TEST(HyperplaneHashTest, DenseHyperplaneMultiProbeTest2) {
  DenseVector v1(4);
  v1[0] = 1.0;
  v1[1] = 0.0;
  v1[2] = 0.0;
  v1[3] = 0.0;

  int dim = 4;
  int k = 3;
  int l = 1;
  uint64_t seed = 84529034;
  HyperplaneHashDense<float> hash(dim, k, l, seed);
  DenseQuery query(hash);
  vector<vector<uint32_t>> probes_by_table;

  query.get_probes_by_table(v1, &probes_by_table, 8);
  ASSERT_EQ(1u, probes_by_table.size());
  ASSERT_EQ(8u, probes_by_table[0].size());

  vector<uint32_t> hash_val;
  hash.hash(v1, &hash_val);
  ASSERT_EQ(1u, hash_val.size());
  ASSERT_EQ(hash_val[0], probes_by_table[0][0]);

  uint32_t bitdiff = probes_by_table[0][0] ^ probes_by_table[0][1];
  ASSERT_EQ(1, count_bits(bitdiff));

  const MatrixType& hyperplanes = hash.get_hyperplanes();
  DenseVector ips = hyperplanes * v1;
  uint32_t hash0 = probes_by_table[0][0];
  for (unsigned int ii = 0; ii < 8; ++ii) {
    if (ii > 0) {
      ASSERT_LE(compute_score(ips, hash0 ^ probes_by_table[0][ii - 1], k),
                compute_score(ips, hash0 ^ probes_by_table[0][ii], k));
    }
  }

  sort(probes_by_table[0].begin(), probes_by_table[0].end());
  for (unsigned int ii = 0; ii < 8; ++ii) {
    ASSERT_EQ(ii, probes_by_table[0][ii]);
  }
}

TEST(HyperplaneHashTest, DenseHyperplaneMultiProbeTest3) {
  int dim = 16;
  DenseVector v1;
  v1.setZero(dim);
  v1[0] = 1.0;

  int k = 8;
  int l = 1;
  unsigned int num_probes = 1 << k;
  uint64_t seed = 572893248;
  HyperplaneHashDense<float> hash(dim, k, l, seed);
  DenseQuery query(hash);
  vector<vector<uint32_t>> probes_by_table;

  query.get_probes_by_table(v1, &probes_by_table, num_probes);
  ASSERT_EQ(1u, probes_by_table.size());
  ASSERT_EQ(num_probes, probes_by_table[0].size());

  vector<uint32_t> hash_val;
  hash.hash(v1, &hash_val);
  ASSERT_EQ(1u, hash_val.size());
  ASSERT_EQ(hash_val[0], probes_by_table[0][0]);

  uint32_t bitdiff = probes_by_table[0][0] ^ probes_by_table[0][1];
  ASSERT_EQ(1, count_bits(bitdiff));

  const MatrixType& hyperplanes = hash.get_hyperplanes();
  DenseVector ips = hyperplanes * v1;
  uint32_t hash0 = probes_by_table[0][0];
  for (unsigned int ii = 0; ii < num_probes; ++ii) {
    if (ii > 0) {
      ASSERT_LE(compute_score(ips, hash0 ^ probes_by_table[0][ii - 1], k),
                compute_score(ips, hash0 ^ probes_by_table[0][ii], k));
    }
    /*printf("ii = %u  hash = %u   xor = %u  score = %f  num_flipped = %d\n",
        ii, result[ii], hash0 ^ result[ii],
        compute_score(ips, hash0 ^ result[ii], k),
        count_bits(hash0 ^ result[ii]));*/
  }

  sort(probes_by_table[0].begin(), probes_by_table[0].end());
  for (unsigned int ii = 0; ii < num_probes; ++ii) {
    ASSERT_EQ(ii, probes_by_table[0][ii]);
  }
}

TEST(HyperplaneHashTest, DenseHyperplaneMultiProbeTest4) {
  int dim = 128;
  DenseVector v1;
  v1.setZero(dim);
  v1[0] = 1.0;

  DenseVector v2;
  float R = std::sqrt(2.0) / 2.0;
  v2.setZero(dim);
  v2[1] = 1.0;
  float alpha = 1.0 - R * R / 2.0;
  float beta = std::sqrt(1.0 - alpha * alpha);
  v2 = alpha * v1 + beta * v2;

  int k = 10;
  int l = 1;
  unsigned int num_probes = 1 << k;
  uint64_t seed = 572893248;
  // uint64_t seed = time(nullptr);
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<int_fast64_t> seed_gen(0, 1000000000);
  float eps = 0.000001;

  int num_trials = 1000;
  std::vector<int> positions1;
  std::vector<int> positions2;

  for (int trial = 0; trial < num_trials; ++trial) {
    HyperplaneHashDense<float> hash(dim, k, l, seed_gen(gen));
    DenseQuery query(hash);
    vector<vector<uint32_t>> probes_by_table;

    query.get_probes_by_table(v1, &probes_by_table, num_probes);
    ASSERT_EQ(1u, probes_by_table.size());
    ASSERT_EQ(num_probes, probes_by_table[0].size());

    vector<uint32_t> hash_val;
    hash.hash(v1, &hash_val);
    ASSERT_EQ(1u, hash_val.size());
    ASSERT_EQ(hash_val[0], probes_by_table[0][0]);

    vector<uint32_t> hash_val2;
    vector<vector<uint32_t>> probes_by_table2;
    hash.hash(v2, &hash_val2);
    ASSERT_EQ(1u, hash_val2.size());
    uint32_t hash_v2 = hash_val2[0];
    query.get_probes_by_table(v2, &probes_by_table2, num_probes);
    ASSERT_EQ(1u, probes_by_table2.size());
    ASSERT_EQ(num_probes, probes_by_table2[0].size());
    ASSERT_EQ(hash_val2[0], probes_by_table2[0][0]);

    uint32_t bitdiff = probes_by_table[0][0] ^ probes_by_table[0][1];
    ASSERT_EQ(1, count_bits(bitdiff));

    bitdiff = probes_by_table2[0][0] ^ probes_by_table2[0][1];
    ASSERT_EQ(1, count_bits(bitdiff));

    const MatrixType& hyperplanes = hash.get_hyperplanes();

    DenseVector ips = hyperplanes * v1;
    uint32_t hash0 = probes_by_table[0][0];
    for (unsigned int ii = 0; ii < num_probes; ++ii) {
      if (ii > 0) {
        ASSERT_LE(
            compute_score(ips, hash0 ^ probes_by_table[0][ii - 1], k) - eps,
            compute_score(ips, hash0 ^ probes_by_table[0][ii], k));
      }
      if (probes_by_table[0][ii] == hash_v2) {
        /*printf("ii = %u  hash = %u  xor = %u  score = %f  num_flipped = %d\n",
            ii, result[ii], hash0 ^ result[ii],
            compute_score(ips, hash0 ^ result[ii], k),
            count_bits(hash0 ^ result[ii]));*/
        /*printf("  Hash match occurs at position %d (fraction %f)\n", ii,
            static_cast<float>(ii) / num_probes);*/
        positions1.push_back(ii);
      }
    }

    DenseVector ips2 = hyperplanes * v2;
    hash0 = probes_by_table2[0][0];
    for (unsigned int ii = 0; ii < num_probes; ++ii) {
      if (ii > 0) {
        ASSERT_LE(
            compute_score(ips2, hash0 ^ probes_by_table2[0][ii - 1], k) - eps,
            compute_score(ips2, hash0 ^ probes_by_table2[0][ii], k));
      }
      if (probes_by_table2[0][ii] == hash_val[0]) {
        positions2.push_back(ii);
      }
    }

    sort(probes_by_table[0].begin(), probes_by_table[0].end());
    for (unsigned int ii = 0; ii < num_probes; ++ii) {
      ASSERT_EQ(ii, probes_by_table[0][ii]);
    }
    sort(probes_by_table2[0].begin(), probes_by_table2[0].end());
    for (unsigned int ii = 0; ii < num_probes; ++ii) {
      ASSERT_EQ(ii, probes_by_table2[0][ii]);
    }
  }

  sort(positions1.begin(), positions1.end());
  int percentile_pos = positions1[0.9 * positions1.size()];
  // printf("90th percentile position: %d, which is %f of all probes\n",
  //    percentile_pos, static_cast<float>(percentile_pos) / num_probes);
  ASSERT_LT(static_cast<float>(percentile_pos) / num_probes, 0.2);

  sort(positions2.begin(), positions2.end());
  int percentile_pos2 = positions2[0.9 * positions2.size()];
  // printf("90th percentile position: %d, which is %f of all probes\n",
  //    percentile_pos, static_cast<float>(percentile_pos) / num_probes);
  ASSERT_LT(static_cast<float>(percentile_pos2) / num_probes, 0.2);
}

TEST(HyperplaneHashTest, DenseHyperplaneMultiProbeTest5) {
  int dim = 16;
  DenseVector v1;
  v1.setZero(dim);
  v1[0] = 1.0;

  int k = 8;
  int l = 1;
  unsigned int num_probes = 1 << k;
  uint64_t seed = 572893248;
  HyperplaneHashDense<float> hash(dim, k, l, seed);
  DenseQuery query(hash);
  vector<uint32_t> probes;

  std::pair<DenseQuery::ProbingSequenceIterator,
            DenseQuery::ProbingSequenceIterator>
      iters = query.get_probing_sequence(v1);
  for (; iters.first != iters.second; ++iters.first) {
    ASSERT_EQ(0, iters.first->second);
    probes.push_back(iters.first->first);
  }
  ASSERT_EQ(num_probes, probes.size());

  vector<uint32_t> hash_val;
  hash.hash(v1, &hash_val);
  ASSERT_EQ(1u, hash_val.size());
  ASSERT_EQ(hash_val[0], probes[0]);

  uint32_t bitdiff = probes[0] ^ probes[1];
  ASSERT_EQ(1, count_bits(bitdiff));

  const MatrixType& hyperplanes = hash.get_hyperplanes();
  DenseVector ips = hyperplanes * v1;
  uint32_t hash0 = probes[0];
  for (unsigned int ii = 0; ii < num_probes; ++ii) {
    if (ii > 0) {
      ASSERT_LE(compute_score(ips, hash0 ^ probes[ii - 1], k),
                compute_score(ips, hash0 ^ probes[ii], k));
    }
    /*printf("ii = %u  hash = %u   xor = %u  score = %f  num_flipped = %d\n",
        ii, result[ii], hash0 ^ result[ii],
        compute_score(ips, hash0 ^ result[ii], k),
        count_bits(hash0 ^ result[ii]));*/
  }

  sort(probes.begin(), probes.end());
  for (unsigned int ii = 0; ii < num_probes; ++ii) {
    ASSERT_EQ(ii, probes[ii]);
  }
}

TEST(HyperplaneHashTest, SparseHyperplaneMultiProbeTest1) {
  SparseVector v1;
  v1.push_back(make_pair(0, 1.0));
  SparseVector v2;
  v2.push_back(make_pair(0, 1.0));
  v2.push_back(make_pair(1, 0.001));
  SparseVector v3;
  v3.push_back(make_pair(0, 0.001));
  v3.push_back(make_pair(1, 1.0));

  int dim = 8;
  int k = 3;
  int l = 2;
  uint64_t seed = 890124523;

  HyperplaneHashSparse<float> hash(dim, k, l, seed);
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

TEST(HyperplaneHashTest, SparseHyperplaneMultiProbeTest2) {
  SparseVector v1;
  v1.push_back(make_pair(0, 1.0));

  int dim = 8;
  int k = 3;
  int l = 1;
  uint64_t seed = 1294087;
  HyperplaneHashSparse<float> hash(dim, k, l, seed);
  SparseQuery query(hash);
  vector<vector<uint32_t>> probes_by_table;

  query.get_probes_by_table(v1, &probes_by_table, 8);
  ASSERT_EQ(1u, probes_by_table.size());
  ASSERT_EQ(8u, probes_by_table[0].size());

  vector<uint32_t> hash_val;
  hash.hash(v1, &hash_val);
  ASSERT_EQ(1u, hash_val.size());
  ASSERT_EQ(hash_val[0], probes_by_table[0][0]);

  uint32_t bitdiff = probes_by_table[0][0] ^ probes_by_table[0][1];
  ASSERT_EQ(1, count_bits(bitdiff));

  sort(probes_by_table[0].begin(), probes_by_table[0].end());
  for (unsigned int ii = 0; ii < 8; ++ii) {
    ASSERT_EQ(ii, probes_by_table[0][ii]);
  }
}

TEST(HyperplaneHashTest, DenseHyperplaneBatchHashTest1) {
  DenseVector v1(4);
  v1[0] = 1.0;
  v1[1] = 0.0;
  v1[2] = 0.0;
  v1[3] = 0.0;
  DenseVector v2(4);
  v2[0] = 1.0;
  v2[1] = 0.001;
  v2[2] = 0.0;
  v2[3] = 0.0;
  DenseVector v3(4);
  v3[0] = 0.001;
  v3[1] = 1.0;
  v3[2] = 0.0;
  v3[3] = 0.0;

  int dim = 4;
  int k = 3;
  int l = 2;
  uint64_t seed = 45234528;
  HyperplaneHashDense<float> hash(dim, k, l, seed);
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
  HPHashDense::BatchHash<BatchVectorType> bh(hash);
  vector<uint32_t> hashes;
  for (int ii = 0; ii < l; ++ii) {
    bh.batch_hash_single_table(batch_data, ii, &hashes);
    ASSERT_EQ(3u, hashes.size());
    ASSERT_EQ(result1[ii], hashes[0]);
    ASSERT_EQ(result2[ii], hashes[1]);
    ASSERT_EQ(result3[ii], hashes[2]);
  }
}

TEST(HyperplaneHashTest, SparseHyperplaneBatchHashTest1) {
  SparseVector v1;
  v1.push_back(make_pair(0, 1.0));
  SparseVector v2;
  v2.push_back(make_pair(0, 1.0));
  v2.push_back(make_pair(1, 0.001));
  SparseVector v3;
  v3.push_back(make_pair(0, 0.001));
  v3.push_back(make_pair(1, 1.0));

  int dim = 8;
  int k = 3;
  int l = 2;
  uint64_t seed = 3425890;
  HyperplaneHashSparse<float> hash(dim, k, l, seed);
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
  HPHashSparse::BatchHash<BatchVectorType> bh(hash);
  vector<uint32_t> hashes;
  for (int ii = 0; ii < l; ++ii) {
    bh.batch_hash_single_table(batch_data, ii, &hashes);
    ASSERT_EQ(3u, hashes.size());
    ASSERT_EQ(result1[ii], hashes[0]);
    ASSERT_EQ(result2[ii], hashes[1]);
    ASSERT_EQ(result3[ii], hashes[2]);
  }
}
