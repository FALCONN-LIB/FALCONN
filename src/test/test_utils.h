#ifndef __TEST_UTILS_H__
#define __TEST_UTILS_H__

#include <algorithm>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace falconn {
namespace test {

void compare_vectors(const std::vector<float>& expected,
                     const std::vector<float>& result,
                     float eps) {
  ASSERT_EQ(expected.size(), result.size());
  for (size_t ii = 0; ii < expected.size(); ++ii) {
    ASSERT_NEAR(expected[ii], result[ii], eps);
  }
}

// TODO: use Google mock instead
//http://stackoverflow.com/questions/1460703/comparison-of-arrays-in-google-test
template<
typename IteratorType,
typename ValueType>
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

template<typename HashTable>
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
template<typename HashTable>
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

template<typename HashTable>
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

template<typename HashTable>
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

template<typename HashTable>
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

template<typename HashTable>
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

template<typename HashTable>
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
template<typename HashTable>
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
