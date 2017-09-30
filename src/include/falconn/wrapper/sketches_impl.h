#ifndef __SKETCHES_IMPL_H__
#define __SKETCHES_IMPL_H__

#include "../core/random_projection_sketches.h"
#include "../sketches.h"
#include "data_storage_adapter.h"

#include <algorithm>
#include <memory>
#include <thread>

#include <cstdint>

namespace falconn {
namespace wrapper {
template <typename PointType, typename KeyType, typename DataStorageType>
class RandomProjectionSketchesQueryWrapper
    : public SketchesQueryable<PointType, KeyType> {
 public:
  RandomProjectionSketchesQueryWrapper(
      const core::RandomProjectionSketches<PointType, DataStorageType>& rps,
      int32_t distance_threshold)
      : rpsq_(rps, distance_threshold) {}
  void filter_close(const PointType& query,
                    const std::vector<KeyType>& candidates,
                    std::vector<KeyType>* filtered_candidates) {
    rpsq_.load_query(query);
    rpsq_.filter_close(candidates, filtered_candidates);
  }

 private:
  core::RandomProjectionSketchesQuery<PointType, KeyType, DataStorageType>
      rpsq_;
};

template <typename PointType, typename KeyType, typename DataStorageType,
          typename RNGType>
class RandomProjectionSketchesWrapper
    : public Sketches<PointType, int32_t, KeyType> {
 public:
  RandomProjectionSketchesWrapper(const DataStorageType& points,
                                  int32_t num_chunks, RNGType& rng)
      : rps_(points, num_chunks, rng) {}

  std::unique_ptr<SketchesQueryable<PointType, KeyType>> construct_query_object(
      int32_t distance_threshold) {
    return std::unique_ptr<SketchesQueryable<PointType, KeyType>>(
        new RandomProjectionSketchesQueryWrapper<PointType, KeyType,
                                                 DataStorageType>(
            rps_, distance_threshold));
  }

 private:
  core::RandomProjectionSketches<PointType, DataStorageType> rps_;
};

template <typename PointType, typename DistanceType, typename KeyType>
class SketchesQueryPoolGeneric : public SketchesQueryable<PointType, KeyType> {
 public:
  SketchesQueryPoolGeneric(Sketches<PointType, DistanceType, KeyType>& parent,
                           DistanceType threshold,
                           int_fast32_t num_query_objects)
      : num_query_objects_(num_query_objects),
        internal_locks_(num_query_objects) {
    for (int_fast32_t i = 0; i < num_query_objects; ++i) {
      internal_query_objects_.push_back(
          parent.construct_query_object(threshold));
      internal_locks_[i].clear(std::memory_order_release);
    }
  }

  void filter_close(const PointType& query,
                    const std::vector<KeyType>& candidates,
                    std::vector<KeyType>* filtered_candidates) {
    int_fast32_t k = get_index_and_lock();
    internal_query_objects_[k]->filter_close(query, candidates,
                                             filtered_candidates);
    internal_locks_[k].clear(std::memory_order_release);
  }

 private:
  int_fast32_t num_query_objects_;
  std::vector<std::unique_ptr<SketchesQueryable<PointType, KeyType>>>
      internal_query_objects_;
  std::vector<std::atomic_flag> internal_locks_;

  // TODO: reduce copy/paste
  int_fast32_t get_index_and_lock() {
    static thread_local std::minstd_rand gen((std::random_device())());
    std::uniform_int_distribution<int_fast32_t> dist(0, num_query_objects_ - 1);
    int_fast32_t cur_index = dist(gen);
    while (true) {
      if (!internal_locks_[cur_index].test_and_set(std::memory_order_acquire)) {
        return cur_index;
      }
      if (cur_index == num_query_objects_ - 1) {
        cur_index = 0;
      } else {
        cur_index += 1;
      }
    }
  }
};

}  // namespace wrapper

template <typename PointType, typename DistanceType, typename KeyType>
std::unique_ptr<SketchesQueryable<PointType, KeyType>>
Sketches<PointType, DistanceType, KeyType>::construct_query_pool(
    DistanceType distance_threshold, int_fast32_t num_query_objects) {
  if (num_query_objects <= 0) {
    num_query_objects = std::max(1u, 2 * std::thread::hardware_concurrency());
  }
  return std::unique_ptr<SketchesQueryable<PointType, KeyType>>(
      new wrapper::SketchesQueryPoolGeneric<PointType, DistanceType, KeyType>(
          *this, distance_threshold, num_query_objects));
}

template <typename PointType, typename KeyType, typename PointSet,
          typename RNGType>
std::unique_ptr<Sketches<PointType, int32_t, KeyType>>
construct_random_projection_sketches(const PointSet& points,
                                     int_fast32_t num_bits, RNGType& rng) {
  if (num_bits <= 0) {
    throw SketchesSetupError("number of bits must be positive");
  }
  if (num_bits % 64) {
    throw SketchesSetupError("number of bits must be a multiple of 64");
  }
  int32_t num_chunks = num_bits / 64;
  typedef typename wrapper::DataStorageAdapter<PointSet>::template DataStorage<
      KeyType>
      DataStorageType;
  auto data_storage_ptr = wrapper::DataStorageAdapter<
      PointSet>::template construct_data_storage<KeyType>(points);
  return std::unique_ptr<Sketches<PointType, int32_t, KeyType>>(
      new wrapper::RandomProjectionSketchesWrapper<PointType, KeyType,
                                                   DataStorageType, RNGType>(
          *data_storage_ptr.get(), num_chunks, rng));
}

}  // namespace falconn

#endif /* __SKETCHES_IMPL_H__ */
