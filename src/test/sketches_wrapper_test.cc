#include <falconn/falconn_global.h>
#include <falconn/sketches.h>

#include <gtest/gtest.h>

#include <random>
#include <vector>

using std::mt19937_64;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::vector;

using falconn::DenseVector;
using falconn::PlainArrayPointSet;
using falconn::construct_random_projection_sketches;

using Eigen::Map;

TEST(SketchesWrapperTest, StatisticalTestVector) {
  const int n = 1000;
  const int d = 100;
  const int num_it = 1000;
  const float r = sqrt(2.0) / 2.0;
  const float alpha = 1.0 - r * r / 2.0;
  const float beta = sqrt(1.0 - alpha * alpha);
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
  auto sketches = construct_random_projection_sketches<DenseVector<float>>(
      dataset, 128, gen);
  auto query_object = sketches->construct_query_object(35);
  uniform_int_distribution<int> u(0, n - 1);
  vector<int32_t> all;
  for (int i = 0; i < n; ++i) {
    all.push_back(i);
  }
  vector<int32_t> filtered;
  int64_t num_found = 0;
  int64_t num_candidates = 0;
  for (int it = 0; it < num_it; ++it) {
    int nn_id = u(gen);
    DenseVector<float> v(d);
    for (int i = 0; i < d; ++i) {
      v[i] = g(gen);
    }
    v -= v.dot(dataset[nn_id]) * dataset[nn_id];
    v.normalize();
    DenseVector<float> q(alpha * dataset[nn_id] + beta * v);
    query_object->filter_close(q, all, &filtered);
    bool found = false;
    for (auto x : filtered) {
      if (x == nn_id) {
        found = true;
        break;
      }
    }
    if (found) {
      ++num_found;
    }
    num_candidates += filtered.size();
  }
  ASSERT_GE(num_found, 0.9 * num_it + 1);
  ASSERT_LE(num_candidates, num_it);
}

TEST(SketchesWrapperTest, StatisticalTestVectorDouble) {
  const int n = 1000;
  const int d = 100;
  const int num_it = 1000;
  const double r = sqrt(2.0) / 2.0;
  const double alpha = 1.0 - r * r / 2.0;
  const double beta = sqrt(1.0 - alpha * alpha);
  mt19937_64 gen(4057218);
  normal_distribution<double> g(0.0, 1.0);
  vector<DenseVector<double>> dataset;
  for (int i = 0; i < n; ++i) {
    DenseVector<double> p(d);
    for (int j = 0; j < d; ++j) {
      p[j] = g(gen);
    }
    p.normalize();
    dataset.push_back(p);
  }
  auto sketches = construct_random_projection_sketches<DenseVector<double>>(
      dataset, 128, gen);
  auto query_object = sketches->construct_query_object(35);
  uniform_int_distribution<int> u(0, n - 1);
  vector<int32_t> all;
  for (int i = 0; i < n; ++i) {
    all.push_back(i);
  }
  vector<int32_t> filtered;
  int64_t num_found = 0;
  int64_t num_candidates = 0;
  for (int it = 0; it < num_it; ++it) {
    int nn_id = u(gen);
    DenseVector<double> v(d);
    for (int i = 0; i < d; ++i) {
      v[i] = g(gen);
    }
    v -= v.dot(dataset[nn_id]) * dataset[nn_id];
    v.normalize();
    DenseVector<double> q(alpha * dataset[nn_id] + beta * v);
    query_object->filter_close(q, all, &filtered);
    bool found = false;
    for (auto x : filtered) {
      if (x == nn_id) {
        found = true;
        break;
      }
    }
    if (found) {
      ++num_found;
    }
    num_candidates += filtered.size();
  }
  ASSERT_GE(num_found, 0.9 * num_it + 1);
  ASSERT_LE(num_candidates, num_it);
}

TEST(SketchesWrapperTest, StatisticalTestPointer) {
  const int n = 1000;
  const int d = 100;
  const int num_it = 1000;
  const float r = sqrt(2.0) / 2.0;
  const float alpha = 1.0 - r * r / 2.0;
  const float beta = sqrt(1.0 - alpha * alpha);
  mt19937_64 gen(4057218);
  normal_distribution<float> g(0.0, 1.0);
  float *dataset = new float[n * d];
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      dataset[i * d + j] = g(gen);
    }
    Map<DenseVector<float>>(dataset + i * d, d).normalize();
  }
  PlainArrayPointSet<float> paps;
  paps.data = dataset;
  paps.num_points = n;
  paps.dimension = d;
  auto sketches =
      construct_random_projection_sketches<DenseVector<float>>(paps, 128, gen);
  auto query_object = sketches->construct_query_object(35);
  uniform_int_distribution<int> u(0, n - 1);
  vector<int32_t> all;
  for (int i = 0; i < n; ++i) {
    all.push_back(i);
  }
  vector<int32_t> filtered;
  int64_t num_found = 0;
  int64_t num_candidates = 0;
  float *v = new float[d];
  float *q = new float[d];
  Map<DenseVector<float>> vv(v, d);
  Map<DenseVector<float>> qq(q, d);
  for (int it = 0; it < num_it; ++it) {
    int nn_id = u(gen);
    Map<DenseVector<float>> dd(dataset + nn_id * d, d);
    for (int i = 0; i < d; ++i) {
      v[i] = g(gen);
    }
    vv -= vv.dot(dd) * dd;
    vv.normalize();
    DenseVector<float> qq(alpha * dd + beta * vv);
    query_object->filter_close(qq, all, &filtered);
    bool found = false;
    for (auto x : filtered) {
      if (x == nn_id) {
        found = true;
        break;
      }
    }
    if (found) {
      ++num_found;
    }
    num_candidates += filtered.size();
  }
  ASSERT_GE(num_found, 0.9 * num_it + 1);
  ASSERT_LE(num_candidates, num_it);
  delete[] dataset;
  delete[] v;
  delete[] q;
}

TEST(SketchesWrapperTest, StatisticalTestPointerDouble) {
  const int n = 1000;
  const int d = 100;
  const int num_it = 1000;
  const double r = sqrt(2.0) / 2.0;
  const double alpha = 1.0 - r * r / 2.0;
  const double beta = sqrt(1.0 - alpha * alpha);
  mt19937_64 gen(4057218);
  normal_distribution<double> g(0.0, 1.0);
  double *dataset = new double[n * d];
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      dataset[i * d + j] = g(gen);
    }
    Map<DenseVector<double>>(dataset + i * d, d).normalize();
  }
  PlainArrayPointSet<double> paps;
  paps.data = dataset;
  paps.num_points = n;
  paps.dimension = d;
  auto sketches =
      construct_random_projection_sketches<DenseVector<double>>(paps, 128, gen);
  auto query_object = sketches->construct_query_object(35);
  uniform_int_distribution<int> u(0, n - 1);
  vector<int32_t> all;
  for (int i = 0; i < n; ++i) {
    all.push_back(i);
  }
  vector<int32_t> filtered;
  int64_t num_found = 0;
  int64_t num_candidates = 0;
  double *v = new double[d];
  double *q = new double[d];
  Map<DenseVector<double>> vv(v, d);
  Map<DenseVector<double>> qq(q, d);
  for (int it = 0; it < num_it; ++it) {
    int nn_id = u(gen);
    Map<DenseVector<double>> dd(dataset + nn_id * d, d);
    for (int i = 0; i < d; ++i) {
      v[i] = g(gen);
    }
    vv -= vv.dot(dd) * dd;
    vv.normalize();
    DenseVector<double> qq(alpha * dd + beta * vv);
    query_object->filter_close(qq, all, &filtered);
    bool found = false;
    for (auto x : filtered) {
      if (x == nn_id) {
        found = true;
        break;
      }
    }
    if (found) {
      ++num_found;
    }
    num_candidates += filtered.size();
  }
  ASSERT_GE(num_found, 0.9 * num_it + 1);
  ASSERT_LE(num_candidates, num_it);
  delete[] dataset;
  delete[] v;
  delete[] q;
}
