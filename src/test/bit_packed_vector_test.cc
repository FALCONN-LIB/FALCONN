#include "falconn/core/bit_packed_vector.h"

#include <cstdint>
#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace fc = falconn::core;

using fc::BitPackedVector;

TEST(BitPackedVectorTest, SimpleTest1) {
  int size = 4;
  BitPackedVector<int_fast64_t> v(size, 3);

  for (int ii = 0; ii < size; ++ii) {
    v.set(ii, ii);
  }
  for (int ii = 0; ii < size; ++ii) {
    EXPECT_EQ(ii, v.get(ii));
  }
}

TEST(BitPackedVectorTest, SimpleTest2) {
  int num_bits = 10;
  int size = 1 << num_bits;
  BitPackedVector<int_fast64_t> v(size, num_bits);

  for (int ii = 0; ii < size; ++ii) {
    v.set(ii, ii);
  }
  for (int ii = 0; ii < size; ++ii) {
    EXPECT_EQ(ii, v.get(ii));
  }
}

TEST(BitPackedVectorTest, RandomTest1) {
  int_fast64_t num_bits = 30;
  int size = 1000000;
  BitPackedVector<int_fast64_t> v(size, num_bits);
  std::vector<int_fast64_t> ref(size);

  int_fast64_t maxint = 1 << num_bits;
  std::mt19937_64 gen(4565729829);
  std::uniform_int_distribution<int_fast64_t> dis(0, maxint - 1);

  for (int ii = 0; ii < size; ++ii) {
    int_fast64_t cur = dis(gen);
    v.set(ii, cur);
    ref[ii] = cur;
  }

  for (int ii = 0; ii < size; ++ii) {
    EXPECT_EQ(ref[ii], v.get(ii));
  }
}

TEST(BitPackedVectorTest, ExhaustiveTest1) {
  int_fast64_t num_bits = 2;
  int_fast64_t maxint = 1 << num_bits;
  int_fast64_t size = 4;

  BitPackedVector<int_fast64_t> v(size, num_bits);

  for (int ii = 0; ii < maxint; ++ii) {
    for (int jj = 0; jj < maxint; ++jj) {
      for (int kk = 0; kk < maxint; ++kk) {
        for (int ll = 0; ll < maxint; ++ll) {
          v.set(0, ii);
          v.set(1, jj);
          v.set(2, kk);
          v.set(3, ll);

          // printf("%d %d %d %d\n", ii, jj, kk, ll);

          EXPECT_EQ(ii, v.get(0));
          EXPECT_EQ(jj, v.get(1));
          EXPECT_EQ(kk, v.get(2));
          EXPECT_EQ(ll, v.get(3));
        }
      }
    }
  }
}
