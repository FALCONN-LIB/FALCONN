#include "stl_hash_table.h"

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
