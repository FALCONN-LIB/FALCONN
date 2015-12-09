#include "falconn/core/data_storage.h"

#include <vector>

#include "gtest/gtest.h"

namespace fc = falconn::core;

#include "falconn/core/data_transformation.h"

using falconn::DenseVector;
using fc::ArrayDataStorage;
using fc::NormalizingTransformation;
using fc::TransformedDataStorage;
using std::vector;


TEST(DataStorageTest, TransformedTest1) {
  typedef DenseVector<float> Vec;
  int dim = 4;

  Vec p1(dim);
  p1 << 1.0, 0.0, 0.0, 0.0;
  Vec p2(dim);
  p2 << 0.0, 2.0, 0.0, 0.0;
  Vec p3(dim);
  p3 << 0.0, 0.0, 0.0, 3.0;
  vector<Vec> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);
  
  ArrayDataStorage<Vec> storage(points);
  NormalizingTransformation<Vec> transformation;
  TransformedDataStorage<Vec, NormalizingTransformation<Vec>,
      ArrayDataStorage<Vec>> transformed_storage(transformation, storage);


  auto iter = transformed_storage.get_full_sequence();

  float eps = 0.0001;

  ASSERT_TRUE(iter.is_valid());
  EXPECT_NEAR(iter.get_point()[0], 1.0, eps);
  EXPECT_NEAR(iter.get_point()[1], 0.0, eps);
  EXPECT_NEAR(iter.get_point()[2], 0.0, eps);
  EXPECT_NEAR(iter.get_point()[3], 0.0, eps);

  ++iter;
  ASSERT_TRUE(iter.is_valid());
  EXPECT_NEAR(iter.get_point()[0], 0.0, eps);
  EXPECT_NEAR(iter.get_point()[1], 1.0, eps);
  EXPECT_NEAR(iter.get_point()[2], 0.0, eps);
  EXPECT_NEAR(iter.get_point()[3], 0.0, eps);
  
  ++iter;
  ASSERT_TRUE(iter.is_valid());
  EXPECT_NEAR(iter.get_point()[0], 0.0, eps);
  EXPECT_NEAR(iter.get_point()[1], 0.0, eps);
  EXPECT_NEAR(iter.get_point()[2], 0.0, eps);
  EXPECT_NEAR(iter.get_point()[3], 1.0, eps);

  ++iter;
  ASSERT_FALSE(iter.is_valid());

  EXPECT_NEAR(points[0][0], 1.0, eps);
  EXPECT_NEAR(points[0][1], 0.0, eps);
  EXPECT_NEAR(points[0][2], 0.0, eps);
  EXPECT_NEAR(points[0][3], 0.0, eps);

  EXPECT_NEAR(points[1][0], 0.0, eps);
  EXPECT_NEAR(points[1][1], 2.0, eps);
  EXPECT_NEAR(points[1][2], 0.0, eps);
  EXPECT_NEAR(points[1][3], 0.0, eps);
  
  EXPECT_NEAR(points[2][0], 0.0, eps);
  EXPECT_NEAR(points[2][1], 0.0, eps);
  EXPECT_NEAR(points[2][2], 0.0, eps);
  EXPECT_NEAR(points[2][3], 3.0, eps);
}
