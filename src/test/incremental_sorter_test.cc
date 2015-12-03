#include "falconn/core/incremental_sorter.h"

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace fc = falconn::core;

using fc::IncrementalSorter;
using std::sort;
using std::vector;

template <typename T>
void check_sorter(const vector<T>& vec, int block_size) {
  vector<T> vec2(vec);
  vector<T> vec3(vec);
  IncrementalSorter<T> sorter;
  sorter.reset(&vec2, block_size);
  sort(vec3.begin(), vec3.end());
  for (size_t ii = 0; ii < vec.size(); ++ii) {
    ASSERT_EQ(vec3[ii], sorter.get(ii));
  }
}

TEST(IncrementalSorterTest, SorterTest1) {
  vector<int> v = {7, 2, 3, 1, 8};
  vector<int> v2(v);
  IncrementalSorter<int> sorter;
  sorter.reset(&v, 2);
  ASSERT_EQ(sorter.get(0), 1);
  ASSERT_EQ(sorter.get(1), 2);
  ASSERT_EQ(sorter.get(2), 3);
  ASSERT_EQ(sorter.get(3), 7);
  ASSERT_EQ(sorter.get(4), 8);
}

TEST(IncrementalSorterTest, SorterTest2) {
  int size = 256;
  vector<float> v(size);
  int seed = 45234859;

  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (size_t ii = 0; ii < v.size(); ++ii) {
    v[ii] = dist(gen);
  }

  check_sorter<float>(v, 10);
}

TEST(IncrementalSorterTest, SorterTest3) {
  int size = 1024;
  vector<std::pair<float, int>> v(size);
  int seed = 45234859;

  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (size_t ii = 0; ii < v.size(); ++ii) {
    v[ii].first = dist(gen);
    v[ii].second = ii;
  }

  check_sorter<std::pair<float, int>>(v, 10);
}
