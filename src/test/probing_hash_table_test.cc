#include "probing_hash_table.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "test_utils.h"

namespace fc = falconn::core;
namespace ft = falconn::test;

using fc::DynamicLinearProbingHashTable;
using fc::StaticLinearProbingHashTable;
using std::pair;
using std::vector;

TEST(ProbingHashTableTest, StaticLinearProbingRetrieveTest1) {
  int table_size = 6;
  StaticLinearProbingHashTable<uint32_t> table(table_size);
  ft::run_retrieve_test_1(&table);
}

// Same hash table queries as above, but this time the table has free buckets.
TEST(ProbingHashTableTest, StaticLinearProbingRetrieveTest2) {
  int table_size = 12;
  StaticLinearProbingHashTable<uint32_t> table(table_size);
  ft::run_retrieve_test_1(&table);
}

TEST(ProbingHashTableTest, StaticLinearProbingRetrieveTest3) {
  int table_size = 6;
  StaticLinearProbingHashTable<uint64_t> table(table_size);
  ft::run_retrieve_test_2(&table);
}

// Same hash table queries as above, but this time the table has free buckets.
TEST(ProbingHashTableTest, StaticLinearProbingRetrieveTest4) {
  int table_size = 12;
  StaticLinearProbingHashTable<uint64_t> table(table_size);
  ft::run_retrieve_test_2(&table);
}

TEST(ProbingHashTableTest, DynamicLinearProbingRetrieveTest1) {
  DynamicLinearProbingHashTable<uint32_t> table(0.5, 0.25, 3.0, 1);
  ft::run_dynamic_retrieve_test_1(&table);
}

TEST(ProbingHashTableTest, DynamicLinearProbingRetrieveTest2) {
  DynamicLinearProbingHashTable<uint32_t> table(0.5, 0.25, 3.0, 1);
  ft::run_dynamic_retrieve_test_2(&table);
}

TEST(ProbingHashTableTest, DynamicLinearProbingRetrieveTest3) {
  DynamicLinearProbingHashTable<uint32_t> table(0.5, 0.25, 3.0, 1);
  ft::run_dynamic_retrieve_test_3(&table);
}

TEST(ProbingHashTableTest, DynamicLinearProbingRetrieveTest4) {
  DynamicLinearProbingHashTable<uint32_t> table(0.5, 0.25, 3.0, 1);
  ft::run_dynamic_retrieve_test_4(&table);
}

TEST(ProbingHashTableTest, DynamicLinearProbingRetrieveTest5) {
  DynamicLinearProbingHashTable<uint32_t> table(0.5, 0.25, 3.0, 1);
  ft::run_dynamic_retrieve_test_5(&table);
}

TEST(ProbingHashTableTest, DynamicLinearProbingRetrieveTest6) {
  DynamicLinearProbingHashTable<uint64_t> table(0.5, 0.25, 3.0, 1);
  ft::run_dynamic_retrieve_test_6(&table);
}
