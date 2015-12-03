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
