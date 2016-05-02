#include "falconn/core/flat_hash_table.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "test_utils.h"

namespace fc = falconn::core;
namespace ft = falconn::test;

using fc::FlatHashTable;
using std::pair;
using std::vector;

TEST(FlatHashTableTest, RetrieveTest1) {
  int num_buckets = 10;
  FlatHashTable<uint32_t> table(num_buckets);
  ft::run_retrieve_test_1(&table);
}

// test 2 does not apply because the test uses large key ranges

TEST(FlatHashTableTest, RetrieveTest2) {
  int num_buckets = 8;
  FlatHashTable<uint32_t> table(num_buckets);
  ft::run_retrieve_test_3(&table);
}

TEST(FlatHashTableTest, RetrieveTest3) {
  int num_buckets = 64;
  int num_trials = 100;
  uint64_t seed = 302342321;
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<uint64_t> dis(
      0, std::numeric_limits<uint64_t>::max());

  for (int ii = 0; ii < num_trials; ++ii) {
    FlatHashTable<uint32_t> table(num_buckets);
    uint64_t cur_seed = dis(gen);
    ft::run_retrieve_test_4(&table, cur_seed);
  }
}

TEST(FlatHashTableTest, RetrieveTest4) {
  int num_buckets = 10;
  FlatHashTable<uint32_t> table(num_buckets);
  ft::run_retrieve_test_5(&table);
}
