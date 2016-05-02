#include "falconn/core/heap.h"

#include <vector>

#include "gtest/gtest.h"

namespace fc = falconn::core;

using fc::AugmentedHeap;
using fc::SimpleHeap;
using std::vector;

TEST(HeapTest, SimpleHeapTest1) {
  SimpleHeap<float, int> h;
  h.resize(10);
  h.insert_unsorted(2.0, 2);
  h.insert_unsorted(1.0, 1);
  h.insert_unsorted(5.0, 5);
  h.insert_unsorted(3.0, 3);
  h.heapify();

  float k;
  int d;
  h.extract_min(&k, &d);
  ASSERT_EQ(1.0, k);
  ASSERT_EQ(1, d);
  h.extract_min(&k, &d);
  ASSERT_EQ(2.0, k);
  ASSERT_EQ(2, d);

  h.insert(4.0, 4);
  h.extract_min(&k, &d);
  ASSERT_EQ(3.0, k);
  ASSERT_EQ(3, d);
  h.extract_min(&k, &d);
  ASSERT_EQ(4.0, k);
  ASSERT_EQ(4, d);
  h.extract_min(&k, &d);
  ASSERT_EQ(5.0, k);
  ASSERT_EQ(5, d);

  h.reset();
  h.insert_unsorted(2.0, 2);
  h.insert_unsorted(10.0, 10);
  h.insert_unsorted(8.0, 8);
  h.heapify();
  h.extract_min(&k, &d);
  ASSERT_EQ(2.0, k);
  ASSERT_EQ(2, d);
  h.extract_min(&k, &d);
  ASSERT_EQ(8.0, k);
  ASSERT_EQ(8, d);

  h.insert(9.5, 9);
  h.extract_min(&k, &d);
  ASSERT_EQ(9.5, k);
  ASSERT_EQ(9, d);

  h.extract_min(&k, &d);
  ASSERT_EQ(10.0, k);
  ASSERT_EQ(10, d);
}

TEST(HeapTest, SimpleHeapTest2) {
  // Same as above, but without initial resize
  SimpleHeap<float, int> h;
  h.insert_unsorted(2.0, 2);
  h.insert_unsorted(1.0, 1);
  h.insert_unsorted(5.0, 5);
  h.insert_unsorted(3.0, 3);
  h.heapify();

  float k;
  int d;
  h.extract_min(&k, &d);
  ASSERT_EQ(1.0, k);
  ASSERT_EQ(1, d);
  h.extract_min(&k, &d);
  ASSERT_EQ(2.0, k);
  ASSERT_EQ(2, d);

  h.insert(4.0, 4);
  h.extract_min(&k, &d);
  ASSERT_EQ(3.0, k);
  ASSERT_EQ(3, d);
  h.extract_min(&k, &d);
  ASSERT_EQ(4.0, k);
  ASSERT_EQ(4, d);
  h.extract_min(&k, &d);
  ASSERT_EQ(5.0, k);
  ASSERT_EQ(5, d);

  h.reset();
  h.insert_unsorted(2.0, 2);
  h.insert_unsorted(10.0, 10);
  h.insert_unsorted(8.0, 8);
  h.heapify();
  h.extract_min(&k, &d);
  ASSERT_EQ(2.0, k);
  ASSERT_EQ(2, d);
  h.extract_min(&k, &d);
  ASSERT_EQ(8.0, k);
  ASSERT_EQ(8, d);

  h.insert(9.5, 9);
  h.extract_min(&k, &d);
  ASSERT_EQ(9.5, k);
  ASSERT_EQ(9, d);

  h.extract_min(&k, &d);
  ASSERT_EQ(10.0, k);
  ASSERT_EQ(10, d);
}

TEST(HeapTest, SimpleHeapTest3) {
  SimpleHeap<float, int> h;
  h.insert_unsorted(2.0, 2);
  h.insert_unsorted(1.0, 1);
  h.insert_unsorted(5.0, 5);
  h.insert_unsorted(3.0, 3);
  h.heapify();

  EXPECT_EQ(1.0, h.min_key());

  h.replace_top(0.5, 0);
  float k;
  int d;
  h.extract_min(&k, &d);
  EXPECT_EQ(0.5, k);
  EXPECT_EQ(0, d);

  h.extract_min(&k, &d);
  EXPECT_EQ(2.0, k);
  EXPECT_EQ(2, d);
}

TEST(HeapTest, AugmentedHeapTest1) {
  // Same as above, but without initial resize
  AugmentedHeap<float, int> h;
  h.insert_unsorted(2.0, 2);
  h.insert_unsorted(1.0, 1);
  h.insert_unsorted(5.0, 5);
  h.insert_unsorted(3.0, 3);
  h.heapify();

  float k;
  int d;
  h.extract_min(&k, &d);
  ASSERT_EQ(1.0, k);
  ASSERT_EQ(1, d);

  h.insert_guaranteed_top(1.0, 10);
  h.extract_min(&k, &d);
  ASSERT_EQ(1.0, k);
  ASSERT_EQ(10, d);

  h.extract_min(&k, &d);
  ASSERT_EQ(2.0, k);
  ASSERT_EQ(2, d);

  h.insert(4.0, 4);

  h.extract_min(&k, &d);
  ASSERT_EQ(3.0, k);
  ASSERT_EQ(3, d);

  h.extract_min(&k, &d);
  ASSERT_EQ(4.0, k);
  ASSERT_EQ(4, d);

  h.extract_min(&k, &d);
  ASSERT_EQ(5.0, k);
  ASSERT_EQ(5, d);
}
