#include "falconn/core/stl_hash_table.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "test_utils.h"

namespace fc = falconn::core;
namespace ft = falconn::test;

using fc::STLHashTable;
using std::pair;
using std::vector;

TEST(STLHashTableTest, RetrieveTest1) {
  STLHashTable<uint32_t> table;
  ft::run_retrieve_test_1(&table);
}

TEST(STLHashTableTest, RetrieveTest2) {
  STLHashTable<uint64_t> table;
  ft::run_retrieve_test_2(&table);
}

TEST(STLHashTableTest, RetrieveTest3) {
  STLHashTable<uint32_t> table;
  ft::run_retrieve_test_3(&table);
}

TEST(STLHashTableTest, RetrieveTest4) {
  int num_trials = 100;
  uint64_t seed = 302342321;
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<uint64_t> dis(
      0, std::numeric_limits<uint64_t>::max());

  for (int ii = 0; ii < num_trials; ++ii) {
    STLHashTable<uint32_t> table;
    uint64_t cur_seed = dis(gen);
    ft::run_retrieve_test_4(&table, cur_seed);
  }
}

TEST(STLHashTableTest, RetrieveTest5) {
  STLHashTable<uint32_t> table;
  ft::run_retrieve_test_5(&table);
}
