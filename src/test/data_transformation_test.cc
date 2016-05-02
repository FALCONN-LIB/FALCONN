#include "falconn/core/data_transformation.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "falconn/core/data_storage.h"

namespace fc = falconn::core;

using falconn::DenseVector;
using fc::ArrayDataStorage;
using fc::CenteringTransformation;
using fc::ComposedTransformation;
using fc::IdentityTransformation;
using fc::NormalizingTransformation;
using std::vector;

TEST(DataTransformationTest, IdentityTest1) {
  typedef DenseVector<float> Vec;
  int dim = 4;

  Vec p1(dim);
  p1[0] = 5.0;
  p1[1] = 0.0;
  p1[2] = -7.0;
  p1[3] = 0.0;
  Vec p1copy = p1;

  IdentityTransformation<Vec> transformation;
  transformation.apply(&p1);

  EXPECT_EQ(5.0, p1[0]);
  for (int ii = 0; ii < dim; ++ii) {
    EXPECT_EQ(p1copy[ii], p1[ii]);
  }
}

TEST(DataTransformationTest, NormalizingTest1) {
  typedef DenseVector<float> Vec;
  int dim = 4;

  Vec p1(dim);
  p1[0] = 0.8;
  p1[1] = 0.0;
  p1[2] = 0.6;
  p1[3] = 0.0;
  Vec p1copy = p1;
  p1 *= 3.0;

  NormalizingTransformation<Vec> transformation;
  transformation.apply(&p1);

  float eps = 0.00001;
  EXPECT_NEAR(0.8, p1[0], eps);
  for (int ii = 0; ii < dim; ++ii) {
    EXPECT_NEAR(p1copy[ii], p1[ii], eps);
  }
}

TEST(DataTransformationTest, CenteringTest1) {
  typedef DenseVector<float> Vec;
  int dim = 4;

  Vec p1(dim);
  p1 << 0.5, 0.2, 0.0, 0.9;
  Vec p2(dim);
  p2 << 0.0, 4.0, -1.0, 0.0;
  Vec p3(dim);
  p3 << 2.5, 1.8, 0.1, 0.0;
  vector<Vec> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);
  ArrayDataStorage<Vec> storage(points);

  CenteringTransformation<Vec, ArrayDataStorage<Vec>> transformation(storage);

  Vec p4(dim);
  p4 << 1.0, 2.0, -0.3, 0.30;
  transformation.apply(&p4);

  float eps = 0.00001;
  for (int ii = 0; ii < dim; ++ii) {
    EXPECT_NEAR(0.0, p4[ii], eps);
  }
}

TEST(DataTransformationTest, ComposedTest1) {
  typedef DenseVector<float> Vec;
  int dim = 4;

  Vec p1(dim);
  p1 << 1.0, 2.0, 0.0, 2.0;
  Vec p2(dim);
  p2 << 2.0, 4.0, -1.0, 0.0;
  vector<Vec> points;
  points.push_back(p1);
  points.push_back(p2);
  ArrayDataStorage<Vec> storage(points);

  typedef CenteringTransformation<Vec, ArrayDataStorage<Vec>> CenteringType;
  typedef NormalizingTransformation<Vec> NormalizingType;

  std::unique_ptr<NormalizingType> normalizing(new NormalizingType());
  std::unique_ptr<CenteringType> centering(new CenteringType(storage));

  ComposedTransformation<Vec, NormalizingType, CenteringType> composed(
      std::move(normalizing), std::move(centering));

  Vec p3(dim);
  p3 << 3.5, 3.0, -0.5, 1.0;
  composed.apply(&p3);

  float eps = 0.00001;
  EXPECT_NEAR(1.0, p3[0], eps);
  for (int ii = 1; ii < dim; ++ii) {
    EXPECT_NEAR(0.0, p3[ii], eps);
  }
}
