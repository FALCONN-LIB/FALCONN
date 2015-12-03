#include "fht.h"

#include <vector>

#include "gtest/gtest.h"

#include "test_utils.h"

using lsh::compare_vectors;
using lsh::fht;
using std::vector;

const float eps = 0.0001;

TEST(PolytopeHashTest, FHTTest1) {
  vector<float> data1 = {0.0, 1.0, 0.0, 0.0};
  vector<float> expected_result1 = {1.0, -1.0, 1.0, -1.0};
  int log_dim1 = std::log2(data1.size());
  fht(data1.data(), data1.size(), log_dim1);
  compare_vectors(expected_result1, data1, eps);
}
