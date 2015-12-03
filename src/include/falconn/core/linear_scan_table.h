#ifndef __LINEAR_SCAN_TABLE_H__
#define __LINEAR_SCAN_TABLE_H__

#include <algorithm>
#include <chrono>
#include <queue>
#include <vector>

namespace lsh {

template<
typename PointType,
typename DistanceType,
typename DistanceFunction,
typename DataStorage = std::vector<PointType>>
class LinearScanTable {
 public:
  LinearScanTable(const DataStorage& points) : points_(points) {}

  int find_closest(const PointType& q) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;

    if (points_.size() == 0) {
      return -1;
    }

    auto start = high_resolution_clock::now();

    int best_index = 0;
    DistanceType best_distance = dst_(q, points_[best_index]);

    for (size_t ii = 1; ii < points_.size(); ++ii) {
      DistanceType cur_distance = dst_(q, points_[ii]);
      if (cur_distance < best_distance) {
        best_distance = cur_distance;
        best_index = ii;
      }
    }

    auto end = high_resolution_clock::now();
    auto elapsed  = duration_cast<duration<double>>(end - start);
    total_query_time_ += elapsed.count();
    num_queries_ += 1;

    return best_index;
  }

  void find_k_closest(const PointType& q, int k, std::vector<int>* result) {
    std::priority_queue<std::pair<DistanceType, int>> candidates;
    
    if (k <= 0) {
      result->clear();
      return;
    }

    for (size_t ii = 0; ii < static_cast<size_t>(k); ++ii) {
      candidates.push(std::make_pair(dst_(q, points_[ii]), ii));
    }

    for (size_t ii = k; ii < points_.size(); ++ii) {
      DistanceType cur_dst = dst_(q, points_[ii]);
      if (cur_dst < candidates.top().first) {
        candidates.pop();
        candidates.push(std::make_pair(cur_dst, ii));
      }
    }
    
    result->resize(k);
    for (size_t ii = 0; ii < static_cast<size_t>(k); ++ii) {
      (*result)[k - ii - 1] = candidates.top().second;
      candidates.pop();
    }
  }
  
  void reset_query_statistics() {
    num_queries_ = 0;
    total_query_time_ = 0.0;
  }

  double get_average_query_time() {
    if (num_queries_ == 0) {
      return 0.0;
    } else {
      return total_query_time_ / num_queries_;
    }
  }

 private:
  const DataStorage& points_;
  DistanceFunction dst_;
  int num_queries_ = 0;
  double total_query_time_ = 0.0;
};

}  // namespace lsh

#endif
