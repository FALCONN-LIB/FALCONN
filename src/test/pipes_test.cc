#include <falconn/experimental/pipes.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>
#include <vector>

using falconn::core::RandomProjectionSketches;
using falconn::experimental::DeduplicationPipe;
using falconn::experimental::DistanceScorer;
using falconn::experimental::ExhaustiveProducer;
using falconn::experimental::TopKPipe;

using falconn::DenseVector;

std::vector<DenseVector<float>> get_dummy_dataset(int32_t n) {
  std::vector<DenseVector<float>> dataset;
  for (int i = 0; i < n; ++i) {
    DenseVector<float> p(1);
    p[0] = i;
    dataset.push_back(p);
  }
  return dataset;
}

TEST(TopKPipeTest, TestLookaheads) {
  const int32_t n = 100;
  const int32_t k = 10;
  std::vector<DenseVector<float>> dataset = get_dummy_dataset(n);
  DenseVector<float> query(1);
  query[0] = 0;

  for (int32_t lookahead = 0; lookahead <= k; lookahead++) {
    ExhaustiveProducer producer(1, n);
    DistanceScorer<DenseVector<float>> scorer(1, dataset);
    TopKPipe<DistanceScorer<DenseVector<float>>> top_k(1, k, true, lookahead);

    scorer.load_query(0, query);
    auto it0 = producer.run(0);
    auto it1 = top_k.run(0, it0, scorer);
    std::vector<int> ans;
    while (it1.is_valid()) {
      ans.push_back(it1.get());
      ++it1;
    }
    ASSERT_THAT(ans, ::testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
  }
}

TEST(TopKPipeTest, TestSetK) {
  const int n = 100;
  std::vector<DenseVector<float>> dataset = get_dummy_dataset(n);
  DenseVector<float> query(1);
  query[0] = 0;
  ExhaustiveProducer producer(1, n);
  DistanceScorer<DenseVector<float>> scorer(1, dataset);
  TopKPipe<DistanceScorer<DenseVector<float>>> top_k(1, 1, true);

  for (int32_t k = 1; k <= 20; k++) {
    top_k.set_k(k);
    scorer.load_query(0, query);
    auto it0 = producer.run(0);
    auto it1 = top_k.run(0, it0, scorer);
    std::vector<int32_t> ans;
    while (it1.is_valid()) {
      ans.push_back(it1.get());
      ++it1;
    }
    ASSERT_EQ(static_cast<int32_t>(ans.size()), k);
  }
}

TEST(TopKPipeTest, InvalidWorkerIDTest) {
  int n = 2;
  auto dataset = get_dummy_dataset(n);
  ExhaustiveProducer producer(1, n);
  DistanceScorer<DenseVector<float>> scorer(1, dataset);
  TopKPipe<DistanceScorer<DenseVector<float>>> top_k(1, 1, true);
  auto it0 = producer.run(0);

  ASSERT_THROW(top_k.run(1, it0, scorer), falconn::experimental::TopKPipeError);
  ASSERT_THROW(top_k.run(-1, it0, scorer),
               falconn::experimental::TopKPipeError);
}

TEST(DeduplicationPipeTest, TestNoDups) {
  const int32_t n = 10;
  auto dataset = get_dummy_dataset(n);
  ExhaustiveProducer producer(1, n);
  DeduplicationPipe<DenseVector<float>> dedup(1, n);
  auto it0 = producer.run(0);
  auto it1 = dedup.run(0, it0);

  std::vector<int32_t> ans;
  while (it1.is_valid()) {
    ans.push_back(it1.get());
    ++it1;
  }
  ASSERT_THAT(ans, ::testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

template <typename PointType>
class MockIterator {
 public:
  MockIterator(const std::vector<PointType> &data)
      : data_(data), n_(static_cast<int32_t>(data.size())), i_(0) {}

  bool is_valid() const { return i_ < n_; }

  PointType get() const { return data_[i_]; }

  void operator++() { ++i_; }

 private:
  const std::vector<PointType> &data_;
  int32_t n_, i_;
};

TEST(DeduplicationPipeTest, TestRemoveDuplicates) {
  const int32_t n = 10;
  const int32_t times = 5;
  std::vector<int32_t> dataset;
  for (int32_t i = 0; i < times; i++) {
    for (int32_t j = 0; j < n; j++) {
      dataset.push_back(j);
    }
  }
  std::random_shuffle(dataset.begin(), dataset.end());
  MockIterator<int32_t> it0(dataset);
  DeduplicationPipe<int32_t> dedup(1, n * times);
  auto it1 = dedup.run(0, it0);

  std::vector<int32_t> ans;
  while (it1.is_valid()) {
    ans.push_back(it1.get());
    ++it1;
  }

  std::sort(ans.begin(), ans.end());
  ASSERT_THAT(ans, ::testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

TEST(DeduplicationPipeTest, InvalidWorkerIDTest) {
  const int n = 10;
  ExhaustiveProducer producer(1, n);
  DeduplicationPipe<int32_t> dedup(2, n);
  auto it0 = producer.run(0);
  dedup.run(1, it0);
  ASSERT_THROW(dedup.run(2, it0),
               falconn::experimental::DeduplicationPipeError);
}
