#include "falconn/core/bit_packed_flat_hash_table.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "test_utils.h"

namespace fc = falconn::core;
namespace ft = falconn::test;

using fc::BitPackedFlatHashTable;

TEST(BitPackedFlatHashTableTest, RetrieveTest1) {
  int num_buckets = 10;
  int num_items = 8;
  BitPackedFlatHashTable<uint32_t> table(num_buckets, num_items);
  ft::run_retrieve_test_1(&table);
}

// test 2 does not apply because the test uses large key ranges

TEST(BitPackedFlatHashTableTest, RetrieveTest2) {
  int num_buckets = 8;
  int num_items = 9;
  BitPackedFlatHashTable<uint32_t> table(num_buckets, num_items);
  ft::run_retrieve_test_3(&table);
}

TEST(BitPackedFlatHashTableTest, RetrieveTest3) {
  int num_buckets = 64;
  int num_items = 1000;
  int num_trials = 100;
  uint64_t seed = 302342321;
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<uint64_t> dis(
      0, std::numeric_limits<uint64_t>::max());

  for (int ii = 0; ii < num_trials; ++ii) {
    BitPackedFlatHashTable<uint32_t> table(num_buckets, num_items);
    uint64_t cur_seed = dis(gen);
    ft::run_retrieve_test_4(&table, cur_seed);
  }
}

TEST(BitPackedFlatHashTableTest, RetrieveTest4) {
  int num_buckets = 10;
  int num_items = 3;
  BitPackedFlatHashTable<uint32_t> table(num_buckets, num_items);
  ft::run_retrieve_test_5(&table);
}

TEST(BitPackedFlatHashTableTest, RetrieveTest5) {
  int num_buckets = 8;
  int num_items = 4;
  BitPackedFlatHashTable<uint32_t> table(num_buckets, num_items);
  ft::run_retrieve_test_6(&table);
}
