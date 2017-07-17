#ifndef __TEST_UTILS_H__
#define __TEST_UTILS_H__

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace falconn {
namespace test {

void compare_vectors(const std::vector<float>& expected,
                     const std::vector<float>& result, float eps) {
  ASSERT_EQ(expected.size(), result.size());
  for (size_t ii = 0; ii < expected.size(); ++ii) {
    ASSERT_NEAR(expected[ii], result[ii], eps);
  }
}

// TODO: use Google mock instead
// http://stackoverflow.com/questions/1460703/comparison-of-arrays-in-google-test
template <typename IteratorType, typename ValueType>
void check_result(std::pair<IteratorType, IteratorType> result,
                  const std::vector<ValueType>& expected_result) {
  std::vector<ValueType> sorted_result;
  while (result.first != result.second) {
    sorted_result.push_back(*(result.first));
    ++(result.first);
  }
  ASSERT_EQ(expected_result.size(), sorted_result.size());
  std::sort(sorted_result.begin(), sorted_result.end());
  for (size_t ii = 0; ii < expected_result.size(); ++ii) {
    ASSERT_EQ(expected_result[ii], sorted_result[ii]);
  }
}

template <typename HashTable>
void run_retrieve_test_1(HashTable* table) {
  typedef std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
      IteratorPair;
  std::vector<uint32_t> entries = {1, 0, 1, 8, 5, 2, 5, 9};
  table->add_entries(entries);

  std::vector<int32_t> expected_result1 = {};
  IteratorPair result1 = table->retrieve(3);
  check_result(result1, expected_result1);

  std::vector<int32_t> expected_result2 = {3};
  IteratorPair result2 = table->retrieve(8);
  check_result(result2, expected_result2);

  std::vector<int32_t> expected_result3 = {0, 2};
  IteratorPair result3 = table->retrieve(1);
  check_result(result3, expected_result3);

  std::vector<int32_t> expected_result4 = {4, 6};
  IteratorPair result4 = table->retrieve(5);
  check_result(result4, expected_result4);

  std::vector<int32_t> expected_result5 = {7};
  IteratorPair result5 = table->retrieve(9);
  check_result(result5, expected_result5);
}

// uses large hash value to check for overflows.
// actual test sequence is similar to the test above.
template <typename HashTable>
void run_retrieve_test_2(HashTable* table) {
  typedef std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
      IteratorPair;
  uint64_t hv1 = 10000000000000000;
  uint64_t hv0 = 0;
  uint64_t hv8 = 80000000000000000;
  uint64_t hv5 = 50000000000000000;
  uint64_t hv2 = 20000000000000000;
  uint64_t hv9 = 90000000000000000;
  uint64_t hv3 = 30000000000000000;
  std::vector<uint64_t> entries = {hv1, hv0, hv1, hv8, hv5, hv2, hv5, hv9};
  table->add_entries(entries);

  std::vector<int32_t> expected_result1 = {};
  IteratorPair result1 = table->retrieve(hv3);
  check_result(result1, expected_result1);

  std::vector<int32_t> expected_result2 = {3};
  IteratorPair result2 = table->retrieve(hv8);
  check_result(result2, expected_result2);

  std::vector<int32_t> expected_result3 = {0, 2};
  IteratorPair result3 = table->retrieve(hv1);
  check_result(result3, expected_result3);

  std::vector<int32_t> expected_result4 = {4, 6};
  IteratorPair result4 = table->retrieve(hv5);
  check_result(result4, expected_result4);

  std::vector<int32_t> expected_result5 = {7};
  IteratorPair result5 = table->retrieve(hv9);
  check_result(result5, expected_result5);
}

// num_buckets = 8
// num_items = 9
template <typename HashTable>
void run_retrieve_test_3(HashTable* table) {
  typedef std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
      IteratorPair;
  std::vector<uint32_t> entries = {3, 7, 1, 3, 2, 0, 5, 7, 6};
  table->add_entries(entries);

  std::vector<int32_t> expected_result1 = {0, 3};
  IteratorPair result1 = table->retrieve(3);
  check_result(result1, expected_result1);

  std::vector<int32_t> expected_result2 = {1, 7};
  IteratorPair result2 = table->retrieve(7);
  check_result(result2, expected_result2);

  std::vector<int32_t> expected_result3 = {2};
  IteratorPair result3 = table->retrieve(1);
  check_result(result3, expected_result3);

  std::vector<int32_t> expected_result4 = {6};
  IteratorPair result4 = table->retrieve(5);
  check_result(result4, expected_result4);

  std::vector<int32_t> expected_result5 = {5};
  IteratorPair result5 = table->retrieve(0);
  check_result(result5, expected_result5);

  std::vector<int32_t> expected_result6 = {4};
  IteratorPair result6 = table->retrieve(2);
  check_result(result6, expected_result6);

  std::vector<int32_t> expected_result7 = {};
  IteratorPair result7 = table->retrieve(4);
  check_result(result7, expected_result7);
}

// num_buckets = 64
// num_items = 1000
template <typename HashTable>
void run_retrieve_test_4(HashTable* table, uint64_t seed) {
  typedef std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
      IteratorPair;
  int_fast64_t num_buckets = 64;
  int_fast64_t num_items = 1000;

  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<> dis(0, num_buckets - 1);

  std::vector<uint32_t> entries;
  std::vector<std::vector<int_fast64_t>> expected_results(num_buckets);
  for (int_fast64_t ii = 0; ii < num_items; ++ii) {
    uint32_t key = dis(gen);
    entries.push_back(key);
    expected_results[key].push_back(ii);
  }

  table->add_entries(entries);

  for (uint32_t ii = 0; ii < static_cast<uint32_t>(num_buckets); ++ii) {
    IteratorPair result = table->retrieve(ii);
    check_result(result, expected_results[ii]);
  }
}

// Written to catch an earlier bug in the bit-packed flat hash table
// num_buckets = 10
// num_items = 3
template <typename HashTable>
void run_retrieve_test_5(HashTable* table) {
  typedef std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
      IteratorPair;
  std::vector<uint32_t> entries = {7, 5, 7};
  table->add_entries(entries);

  std::vector<int32_t> expected_result1 = {};
  IteratorPair result1 = table->retrieve(0);
  check_result(result1, expected_result1);

  std::vector<int32_t> expected_result2 = {};
  IteratorPair result2 = table->retrieve(1);
  check_result(result2, expected_result2);

  std::vector<int32_t> expected_result3 = {};
  IteratorPair result3 = table->retrieve(2);
  check_result(result3, expected_result3);

  std::vector<int32_t> expected_result4 = {};
  IteratorPair result4 = table->retrieve(3);
  check_result(result4, expected_result4);

  std::vector<int32_t> expected_result5 = {};
  IteratorPair result5 = table->retrieve(4);
  check_result(result5, expected_result5);

  std::vector<int32_t> expected_result6 = {1};
  IteratorPair result6 = table->retrieve(5);
  check_result(result6, expected_result6);

  std::vector<int32_t> expected_result7 = {};
  IteratorPair result7 = table->retrieve(6);
  check_result(result7, expected_result7);

  std::vector<int32_t> expected_result8 = {0, 2};
  IteratorPair result8 = table->retrieve(7);
  check_result(result8, expected_result8);

  std::vector<int32_t> expected_result9 = {};
  IteratorPair result9 = table->retrieve(8);
  check_result(result9, expected_result9);

  std::vector<int32_t> expected_result10 = {};
  IteratorPair result10 = table->retrieve(9);
  check_result(result10, expected_result10);
}

// Written to catch an earlier bug in the bit-packed flat hash table
// num_buckets = 8
// num_items = 4
template <typename HashTable>
void run_retrieve_test_6(HashTable* table) {
  typedef std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
      IteratorPair;
  std::vector<uint32_t> entries = {3, 5, 5, 1};
  table->add_entries(entries);

  std::vector<int32_t> expected_result1 = {3};
  IteratorPair result1 = table->retrieve(1);
  check_result(result1, expected_result1);

  std::vector<int32_t> expected_result2 = {0};
  IteratorPair result2 = table->retrieve(3);
  check_result(result2, expected_result2);

  std::vector<int32_t> expected_result3 = {1, 2};
  IteratorPair result3 = table->retrieve(5);
  check_result(result3, expected_result3);

  std::vector<int32_t> expected_result4 = {};
  IteratorPair result4 = table->retrieve(6);
  check_result(result4, expected_result4);

  std::vector<int32_t> expected_result5 = {};
  IteratorPair result5 = table->retrieve(7);
  check_result(result5, expected_result5);
}

template <typename HashTable>
void run_dynamic_retrieve_test_1(HashTable* table) {
  typedef std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
      IteratorPair;
  std::vector<uint32_t> entries = {1, 0, 1, 8, 5, 2, 5, 9};
  for (size_t ii = 0; ii < entries.size(); ++ii) {
    table->insert(entries[ii], ii);
  }

  std::vector<int32_t> expected_result1 = {};
  IteratorPair result1 = table->retrieve(3);
  check_result(result1, expected_result1);

  std::vector<int32_t> expected_result2 = {3};
  IteratorPair result2 = table->retrieve(8);
  check_result(result2, expected_result2);

  std::vector<int32_t> expected_result3 = {0, 2};
  IteratorPair result3 = table->retrieve(1);
  check_result(result3, expected_result3);

  std::vector<int32_t> expected_result4 = {4, 6};
  IteratorPair result4 = table->retrieve(5);
  check_result(result4, expected_result4);

  std::vector<int32_t> expected_result5 = {7};
  IteratorPair result5 = table->retrieve(9);
  check_result(result5, expected_result5);
}

template <typename HashTable>
void run_dynamic_retrieve_test_2(HashTable* table) {
  typedef std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
      IteratorPair;
  std::vector<uint32_t> entries = {1, 0, 1, 8, 5, 2, 5};
  for (size_t ii = 0; ii < entries.size(); ++ii) {
    table->insert(entries[ii], ii);
  }

  std::vector<int32_t> expected_result1 = {3};
  IteratorPair result1 = table->retrieve(8);
  check_result(result1, expected_result1);

  table->remove(8, 3);

  std::vector<int32_t> expected_result2 = {};
  IteratorPair result2 = table->retrieve(8);
  check_result(result2, expected_result2);
}

template <typename HashTable>
void run_dynamic_retrieve_test_3(HashTable* table) {
  typedef std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
      IteratorPair;
  std::vector<uint32_t> entries = {1, 0, 1, 8, 5, 2, 5};
  for (size_t ii = 0; ii < entries.size(); ++ii) {
    table->insert(entries[ii], ii);
  }

  std::vector<int32_t> expected_result1 = {0, 2};
  IteratorPair result1 = table->retrieve(1);
  check_result(result1, expected_result1);

  table->remove(1, 2);

  std::vector<int32_t> expected_result2 = {0};
  IteratorPair result2 = table->retrieve(1);
  check_result(result2, expected_result2);
}

template <typename HashTable>
void run_dynamic_retrieve_test_4(HashTable* table) {
  typedef std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
      IteratorPair;
  std::vector<uint32_t> entries = {1, 0, 1, 8, 5, 2, 5};
  for (size_t ii = 0; ii < entries.size(); ++ii) {
    table->insert(entries[ii], ii);
  }

  std::vector<int32_t> expected_result1 = {0, 2};
  IteratorPair result1 = table->retrieve(1);
  check_result(result1, expected_result1);

  table->remove(1, 2);

  std::vector<int32_t> expected_result2 = {0};
  IteratorPair result2 = table->retrieve(1);
  check_result(result2, expected_result2);

  table->insert(1, 2);

  std::vector<int32_t> expected_result3 = {0, 2};
  IteratorPair result3 = table->retrieve(1);
  check_result(result3, expected_result3);
}

template <typename HashTable>
void run_dynamic_retrieve_test_5(HashTable* table) {
  typedef std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
      IteratorPair;
  std::vector<uint32_t> entries = {7, 3, 3, 8, 1, 0, 1, 2};
  for (size_t ii = 0; ii < entries.size(); ++ii) {
    table->insert(entries[ii], ii);
  }

  for (size_t ii = 0; ii < entries.size(); ++ii) {
    table->remove(entries[ii], ii);
  }

  for (size_t ii = 0; ii < entries.size() - 1; ++ii) {
    table->insert(entries[ii], ii);
  }

  std::vector<int32_t> expected_result1 = {};
  IteratorPair result1 = table->retrieve(4);
  check_result(result1, expected_result1);

  std::vector<int32_t> expected_result2 = {1, 2};
  IteratorPair result2 = table->retrieve(3);
  check_result(result2, expected_result2);

  std::vector<int32_t> expected_result3 = {5};
  IteratorPair result3 = table->retrieve(0);
  check_result(result3, expected_result3);

  std::vector<int32_t> expected_result4 = {};
  IteratorPair result4 = table->retrieve(2);
  check_result(result4, expected_result4);
}

// Similar test sequence as in the dynamic test 1, but with large hash values.
template <typename HashTable>
void run_dynamic_retrieve_test_6(HashTable* table) {
  typedef std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
      IteratorPair;
  uint64_t hv1 = 10000000000000000;
  uint64_t hv0 = 0;
  uint64_t hv8 = 80000000000000000;
  uint64_t hv5 = 50000000000000000;
  uint64_t hv2 = 20000000000000000;
  uint64_t hv9 = 90000000000000000;
  uint64_t hv3 = 30000000000000000;
  std::vector<uint64_t> entries = {hv1, hv0, hv1, hv8, hv5, hv2, hv5, hv9};
  for (size_t ii = 0; ii < entries.size(); ++ii) {
    table->insert(entries[ii], ii);
  }

  std::vector<int32_t> expected_result1 = {};
  IteratorPair result1 = table->retrieve(hv3);
  check_result(result1, expected_result1);

  std::vector<int32_t> expected_result2 = {3};
  IteratorPair result2 = table->retrieve(hv8);
  check_result(result2, expected_result2);

  std::vector<int32_t> expected_result3 = {0, 2};
  IteratorPair result3 = table->retrieve(hv1);
  check_result(result3, expected_result3);

  std::vector<int32_t> expected_result4 = {4, 6};
  IteratorPair result4 = table->retrieve(hv5);
  check_result(result4, expected_result4);

  std::vector<int32_t> expected_result5 = {7};
  IteratorPair result5 = table->retrieve(hv9);
  check_result(result5, expected_result5);
}

template <typename T>
int_fast32_t count_bits(T value) {
  int_fast32_t count = 0;
  while (value > 0) {
    if ((value & 1) == 1) {
      count++;
    }
    value = value >> 1;
  }
  return count;
}

}  // namespace test
}  // namespace falconn

#endif
