#include <falconn/core/data_storage.h>
#include <falconn/core/random_projection_sketches.h>
#include <falconn/falconn_global.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include <cmath>

using std::max;
using std::mt19937_64;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::vector;

using falconn::DenseVector;

using falconn::core::ArrayDataStorage;
using falconn::core::PlainArrayDataStorage;
using falconn::core::RandomProjectionSketches;
using falconn::core::SketchesError;

TEST(SketchesTest, EmptyDataset) {
  const uint_fast64_t seed = 4057218;
  try {
    vector<DenseVector<float>> dataset;
    ArrayDataStorage<DenseVector<float>> ads(dataset);
    RandomProjectionSketches<DenseVector<float>,
                             ArrayDataStorage<DenseVector<float>>>
        rps(1, ads, 2, seed);
    FAIL();
  } catch (SketchesError &e) {
  } catch (...) {
    FAIL();
  }
  try {
    vector<DenseVector<float>> dataset;
    dataset.push_back(DenseVector<float>(100));
    ArrayDataStorage<DenseVector<float>> ads(dataset);
    RandomProjectionSketches<DenseVector<float>,
                             ArrayDataStorage<DenseVector<float>>>
        rps(1, ads, 2, seed);
  } catch (SketchesError &e) {
    FAIL();
  } catch (...) {
    FAIL();
  }
  try {
    vector<DenseVector<double>> dataset;
    ArrayDataStorage<DenseVector<double>> ads(dataset);
    RandomProjectionSketches<DenseVector<double>,
                             ArrayDataStorage<DenseVector<double>>>
        rps(1, ads, 2, seed);
    FAIL();
  } catch (SketchesError &e) {
  } catch (...) {
    FAIL();
  }
  try {
    vector<DenseVector<double>> dataset;
    dataset.push_back(DenseVector<double>(100));
    ArrayDataStorage<DenseVector<double>> ads(dataset);
    RandomProjectionSketches<DenseVector<double>,
                             ArrayDataStorage<DenseVector<double>>>
        rps(1, ads, 2, seed);
  } catch (SketchesError &e) {
    FAIL();
  } catch (...) {
    FAIL();
  }
  try {
    vector<float> dataset(100);
    PlainArrayDataStorage<DenseVector<float>> pads(&dataset[0], 0, 100);
    RandomProjectionSketches<DenseVector<float>, int32_t,
                             PlainArrayDataStorage<DenseVector<float>>>
        rps(1, pads, 2, seed);
    FAIL();
  } catch (SketchesError &e) {
  } catch (...) {
    FAIL();
  }
  try {
    vector<float> dataset(100);
    PlainArrayDataStorage<DenseVector<float>> pads(&dataset[0], 1, 100);
    RandomProjectionSketches<DenseVector<float>, int32_t,
                             PlainArrayDataStorage<DenseVector<float>>>
        rps(1, pads, 2, seed);
  } catch (SketchesError &e) {
    FAIL();
  } catch (...) {
    FAIL();
  }
  try {
    vector<double> dataset(100);
    PlainArrayDataStorage<DenseVector<double>> pads(&dataset[0], 0, 100);
    RandomProjectionSketches<DenseVector<double>, int32_t,
                             PlainArrayDataStorage<DenseVector<double>>>
        rps(1, pads, 2, seed);
    FAIL();
  } catch (SketchesError &e) {
  } catch (...) {
    FAIL();
  }
  try {
    vector<double> dataset(100);
    PlainArrayDataStorage<DenseVector<double>> pads(&dataset[0], 1, 100);
    RandomProjectionSketches<DenseVector<double>, int32_t,
                             PlainArrayDataStorage<DenseVector<double>>>
        rps(1, pads, 2, seed);
  } catch (SketchesError &e) {
    FAIL();
  } catch (...) {
    FAIL();
  }
}

TEST(SketchesTest, DimensionMismatchTest) {
  const uint_fast64_t seed = 4057218;
  vector<DenseVector<float>> dataset(1, DenseVector<float>(128));
  ArrayDataStorage<DenseVector<float>> ads(dataset);
  RandomProjectionSketches<DenseVector<float>> rps(1, ads, 2, seed);
  try {
    rps.load_query(0, DenseVector<float>(129));
    FAIL();
  } catch (SketchesError &e) {
  } catch (...) {
    FAIL();
  }
}

TEST(SketchesTest, InvalidWorkerIDTest) {
  const uint_fast64_t seed = 4057218;
  vector<DenseVector<float>> dataset(1, DenseVector<float>(128));
  ArrayDataStorage<DenseVector<float>> ads(dataset);
  RandomProjectionSketches<DenseVector<float>> rps(1, ads, 2, seed);
  rps.load_query(0, DenseVector<float>(128));
  ASSERT_THROW(rps.load_query(1, DenseVector<float>(128)), SketchesError);
  ASSERT_THROW(rps.get_score(1, 0), SketchesError);
}

TEST(SketchesTest, StatisticalTest) {
  const int n = 1000;
  const int d = 100;
  const int num_it = 1000;
  const int threshold = 50;
  const int num_cand = 100;
  const float r = sqrt(2.0) / 2.0;
  const float alpha = 1.0 - r * r / 2.0;
  const float beta = sqrt(1.0 - alpha * alpha);
  const uint_fast64_t seed = 4057218;
  mt19937_64 gen(4057218);
  normal_distribution<float> g(0.0, 1.0);
  vector<DenseVector<float>> dataset;
  for (int i = 0; i < n; ++i) {
    DenseVector<float> p(d);
    for (int j = 0; j < d; ++j) {
      p[j] = g(gen);
    }
    p.normalize();
    dataset.push_back(p);
  }
  ArrayDataStorage<DenseVector<float>> ads(dataset);
  RandomProjectionSketches<DenseVector<float>> rps(1, ads, 2, seed);
  uniform_int_distribution<int> u(0, n - 1);
  int worst = 0;
  int worst_th = 0;
  vector<int32_t> filtered;
  for (int it = 0; it < num_it; ++it) {
    int nn_id = u(gen);
    DenseVector<float> v(d);
    for (int i = 0; i < d; ++i) {
      v[i] = g(gen);
    }
    v -= v.dot(dataset[nn_id]) * dataset[nn_id];
    v.normalize();
    DenseVector<float> q(alpha * dataset[nn_id] + beta * v);
    rps.load_query(0, q);
    int th = rps.get_score(0, nn_id);
    ASSERT_TRUE(rps.get_score(0, nn_id) <= threshold);
    worst_th = max(worst_th, th);
    int cnt = 0;
    for (int i = 0; i < n; ++i) {
      if (rps.get_score(0, i) <= th) {
        ++cnt;
      }
    }
    worst = max(worst, cnt);
  }
  ASSERT_LE(worst, num_cand);
  ASSERT_LE(worst_th, threshold);
}
