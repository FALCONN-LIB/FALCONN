#ifndef __RANDOM_PROJECTION_SKETCHES_H__
#define __RANDOM_PROJECTION_SKETCHES_H__

#define UNUSED(x) (void)(x)

#include <memory>
#include <random>
#include <vector>
#include "../falconn_global.h"
#include "data_storage.h"
#include "polytope_hash.h"

#include <cstdlib>

namespace falconn {
namespace core {

class SketchesError : public FalconnError {
 public:
  SketchesError(const char *msg) : FalconnError(msg) {}
};

namespace sketches_helpers {

template <typename PointType,
          typename DataStorageType = ArrayDataStorage<PointType>>
class RandomProjectionSketchesWorker {
 public:
  typedef typename PointTypeTraits<PointType>::ScalarType ScalarType;

  RandomProjectionSketchesWorker(int32_t dimension, int32_t num_rotations,
                                 int32_t num_chunks,
                                 const std::vector<ScalarType> &random_signs)
      : dimension_(dimension),
        num_rotations_(num_rotations),
        num_chunks_(num_chunks),
        random_signs_(random_signs) {
    padded_dimension_ = dimension;
    while (padded_dimension_ & (padded_dimension_ - 1)) {
      ++padded_dimension_;
    }

    buffer_.resize(padded_dimension_);

    if (padded_dimension_ != dimension_) {
      padding_ = PointType::Zero(padded_dimension_ - dimension_);
    }
  }

  void compute_sketch(const PointType &point, uint64_t *result) {
    if (point.size() != dimension_) {
      throw SketchesError("dimension mismatch");
    }
    for (int32_t i = 0; i < num_rotations_; ++i) {
      Eigen::Map<PointType>(&buffer_[0], dimension_) =
          point.cwiseProduct(Eigen::Map<const PointType>(
              &random_signs_[i * dimension_], dimension_));
      if (dimension_ != padded_dimension_) {
        Eigen::Map<PointType>(&buffer_[dimension_],
                              padded_dimension_ - dimension_) = padding_;
      }

      cp_hash_helpers::FHTFunction<ScalarType>::apply(&buffer_[0],
                                                      padded_dimension_);

      for (int32_t j = 0; j < padded_dimension_; ++j) {
        int32_t pos = i * padded_dimension_ + j;

        int32_t chunk_id = pos / 64;
        if (chunk_id >= num_chunks_) {
          break;
        }
        uint64_t bit = 0;
        if (buffer_[j] > 0.0) {
          bit = 1;
        }

        int32_t chunk_off = pos % 64;

        result[chunk_id] = (result[chunk_id] & ~(uint64_t(1) << chunk_off)) |
                           (bit << chunk_off);
      }
    }
  }

 private:
  int32_t dimension_;
  int32_t padded_dimension_;
  int32_t num_rotations_;
  int32_t num_chunks_;
  std::vector<ScalarType> random_signs_;
  std::vector<ScalarType> buffer_;
  PointType padding_;
};
}  // namespace sketches_helpers

template <typename PointType, typename KeyType = int32_t,
          typename DataStorageType = ArrayDataStorage<PointType>>
class RandomProjectionSketches {
 public:
  typedef typename PointTypeTraits<PointType>::ScalarType ScalarType;
  using ScoreType = int32_t;
  RandomProjectionSketches(int32_t num_workers, const DataStorageType &points,
                           int32_t num_chunks, uint_fast64_t seed)
      : num_chunks_(num_chunks),
        num_workers_(num_workers),
        sketches_(points.size() * num_chunks) {
    if (points.size() == 0) {
      throw SketchesError("empty dataset");
    }

    if (num_chunks < 1) {
      throw SketchesError("there must be at least one chunk");
    }

    typename DataStorageType::FullSequenceIterator iter =
        points.get_full_sequence();
    dimension_ = iter.get_point().size();

    int32_t num_bits = 64 * num_chunks_;

    int32_t padded_dimension = dimension_;
    while (padded_dimension & (padded_dimension - 1)) {
      ++padded_dimension;
    }

    num_rotations_ = num_bits / padded_dimension;
    if (num_bits % padded_dimension) {
      ++num_rotations_;
    }

    random_signs_.resize(num_rotations_ * dimension_);

    std::uniform_int_distribution<> random_bit(0, 1);
    std::mt19937_64 gen(seed);

    for (int32_t i = 0; i < num_rotations_ * dimension_; ++i) {
      random_signs_[i] = 1.0 - 2.0 * random_bit(gen);
    }

    workers_.resize(
        num_workers_,
        sketches_helpers::RandomProjectionSketchesWorker<PointType,
                                                         DataStorageType>(
            dimension_, num_rotations_, num_chunks_, random_signs_));
    query_sketches_.resize(num_workers_, std::vector<uint64_t>(num_chunks_));

    size_t counter = 0;
    while (iter.is_valid()) {
      workers_[0].compute_sketch(iter.get_point(),
                                 &sketches_[counter * num_chunks_]);
      ++counter;
      ++iter;
    }
  }

  void load_query(int32_t worker_id, const PointType &query) {
    if (worker_id < 0 || worker_id >= num_workers_) {
      throw SketchesError(
          "worker id is not in the range 0 to num_workers_ - 1");
    }
    workers_[worker_id].compute_sketch(query, &query_sketches_[worker_id][0]);
  }

  void prepare(int32_t worker_id, int32_t dataset_point_id) {
    UNUSED(worker_id);
    size_t ind = static_cast<size_t>(dataset_point_id) * num_chunks_;
    __builtin_prefetch(sketches_.data() + ind);
  }

  inline int32_t get_score(int32_t worker_id, KeyType dataset_point_id) {
    if (worker_id < 0 || worker_id >= num_workers_) {
      throw SketchesError(
          "worker id is not in the range 0 to num_workers_ - 1");
    }
    int32_t hamming_distance = 0;
    size_t ind = dataset_point_id * num_chunks_;

    for (int32_t i = 0; i < num_chunks_; ++i) {
      hamming_distance +=
          __builtin_popcountll(sketches_[ind] ^ query_sketches_[worker_id][i]);
      ++ind;
    }
    return hamming_distance;
  }

 private:
  int32_t num_chunks_;
  int32_t num_workers_;
  std::vector<uint64_t> sketches_;
  std::vector<ScalarType> random_signs_;

  int32_t dimension_;
  int32_t num_rotations_;

  std::vector<sketches_helpers::RandomProjectionSketchesWorker<PointType,
                                                               DataStorageType>>
      workers_;
  std::vector<std::vector<uint64_t>> query_sketches_;
};

}  // namespace core
}  // namespace falconn

#endif /* __RANDOM_PROJECTION_SKETCHES_H__ */