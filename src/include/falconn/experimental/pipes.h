#ifndef __PIPES_H__
#define __PIPES_H__

#define UNUSED(x) (void)(x)

#include "../core/composite_hash_table.h"
#include "../core/data_storage.h"
#include "../core/flat_hash_table.h"
#include "../core/heap.h"
#include "../core/polytope_hash.h"
#include "../core/random_projection_sketches.h"

#include "../falconn_global.h"

#include <serialize.h>

#include <future>
#include <memory>
#include <vector>

#include <cstdint>

namespace falconn {
namespace experimental {

using Point = DenseVector<float>;

using core::ArrayDataStorage;
using core::CrossPolytopeHashDense;
using core::FlatHashTable;
using core::StaticCompositeHashTable;

class PipelineError : public FalconnError {
 public:
  PipelineError(const char *msg) : FalconnError(msg) {}
};

class ExhaustiveProducer {
 public:
  class Iterator {
   public:
    Iterator(int32_t n) : n_(n), i_(0) {}

    bool is_valid() const { return i_ < n_; }

    int32_t get() const { return i_; }

    void operator++() { ++i_; }

   private:
    int32_t n_, i_;
  };

  ExhaustiveProducer(int32_t num_workers, int32_t n) : n_(n) {
    UNUSED(num_workers);
  }

  Iterator run(int32_t worker_id) const {
    UNUSED(worker_id);
    return Iterator(n_);
  }

 private:
  int32_t n_;
};

template <typename PointType>
class TablePipe;

class HashProducerError : public FalconnError {
 public:
  HashProducerError(const char *msg) : FalconnError(msg) {}
};

template <typename PointType>
class HashProducer {
 public:
  using HashType = CrossPolytopeHashDense<>;

  class Iterator {
   public:
    Iterator(HashProducer &parent, int32_t worker_id)
        : parent_(parent), worker_id_(worker_id) {
      valid_ =
          parent_.mp_[worker_id_]->get_next_probe(&cur_probe_, &cur_table_);
    }

    bool is_valid() const { return valid_; }

    std::pair<uint32_t, int_fast32_t> get() const {
      return std::make_pair(cur_probe_, cur_table_);
    }

    void operator++() {
      valid_ =
          parent_.mp_[worker_id_]->get_next_probe(&cur_probe_, &cur_table_);
    }

   private:
    HashProducer &parent_;
    int32_t worker_id_;
    bool valid_;
    uint32_t cur_probe_;
    int_fast32_t cur_table_;
  };

  HashProducer(int32_t num_workers, int32_t dimension, int32_t num_hash_bits,
               int32_t num_tables, int32_t num_probes = -1,
               int32_t num_rotations = 2, uint_fast64_t seed = 4057218)
      : num_hash_bits_(num_hash_bits),
        num_tables_(num_tables),
        num_probes_(num_probes),
        num_workers_(num_workers) {
    int32_t b = 0;
    while ((1 << b) < dimension) {
      ++b;
    }
    int32_t k = num_hash_bits / (b + 1);
    int32_t last_cp_dimension;
    if (num_hash_bits % (b + 1)) {
      ++k;
      last_cp_dimension = 1 << ((num_hash_bits % (b + 1)) - 1);
    } else {
      last_cp_dimension = 1 << b;
    }
    hasher_ = std::make_shared<HashType>(
        dimension, k, num_tables, num_rotations, last_cp_dimension, seed);
    ht_.clear();
    tv_.resize(num_workers);
    for (int32_t i = 0; i < num_workers; ++i) {
      ht_.push_back(std::make_shared<HashType::HashTransformation>(*hasher_));
      mp_.push_back(std::make_shared<HashType::MultiProbeLookup>(*hasher_));
      hasher_->reserve_transformed_vector_memory(&tv_[i]);
    }
  }

  void load_query(int32_t worker_id, const Point &query) {
    if (worker_id < 0 || worker_id >= num_workers_) {
      throw HashProducerError(
          "worker id is not in the range 0 to num_workers_ - 1");
    }
    ht_[worker_id]->apply(query, &tv_[worker_id]);
    mp_[worker_id]->setup_probing(tv_[worker_id], num_probes_);
  }

  Iterator run(int32_t worker_id) {
    if (worker_id < 0 || worker_id >= num_workers_) {
      throw HashProducerError(
          "worker id is not in the range 0 to num_workers_ - 1");
    }
    return Iterator(*this, worker_id);
  }

  void add_table() {
    hasher_->add_table();
    ++num_tables_;
    for (int32_t i = 0; i < num_workers_; ++i) {
      ht_[i] = std::make_shared<HashType::HashTransformation>(*hasher_);
      mp_[i] = std::make_shared<HashType::MultiProbeLookup>(*hasher_);
      hasher_->reserve_transformed_vector_memory(&tv_[i]);
    }
  }

  void set_num_probes(int32_t num_probes) { num_probes_ = num_probes; }

 private:
  int32_t num_hash_bits_;
  int32_t num_tables_;
  int32_t num_probes_;
  int32_t num_workers_;
  std::shared_ptr<HashType> hasher_;
  std::vector<std::shared_ptr<HashType::HashTransformation>> ht_;
  std::vector<std::shared_ptr<HashType::MultiProbeLookup>> mp_;
  std::vector<HashType::TransformedVectorType> tv_;

  friend class TablePipe<PointType>;
};

class TablePipeError : public FalconnError {
 public:
  TablePipeError(const char *msg) : FalconnError(msg) {}
};

template <typename PointType>
class TablePipe {
 public:
  using DataStorageType = ArrayDataStorage<Point>;
  using InnerHashTableType = FlatHashTable<uint32_t>;
  using FactoryType = InnerHashTableType::Factory;
  using HashTableType =
      StaticCompositeHashTable<uint32_t, int32_t, InnerHashTableType>;

  template <typename IteratorType>
  class Iterator {
   public:
    Iterator(TablePipe &parent, IteratorType &it)
        : parent_(parent),
          it_(it),
          it_inner_(std::make_pair(nullptr, nullptr)) {
      advance();
    }

    bool is_valid() const { return it_inner_.first != it_inner_.second; }

    int32_t get() const { return *(it_inner_.first); }

    void operator++() {
      ++it_inner_.first;
      advance();
    }

   private:
    TablePipe &parent_;
    IteratorType &it_;
    std::pair<InnerHashTableType::Iterator, InnerHashTableType::Iterator>
        it_inner_;

    void advance() {
      while (it_inner_.first == it_inner_.second && it_.is_valid()) {
        auto x = it_.get();
        it_inner_ = parent_.table_->retrieve_individual(x.first, x.second);
        ++it_;
      }
    }
  };

  TablePipe(int32_t num_workers, const std::vector<Point> &dataset,
            const HashProducer<PointType> &hash,
            int_fast32_t num_setup_threads = 0,
            const std::string &file_name = "")
      : dataset_(dataset), hash_(hash) {
    UNUSED(num_workers);
    factory_ = std::make_shared<FactoryType>(1 << hash.num_hash_bits_);
    table_ = std::make_shared<HashTableType>(hash.num_tables_, factory_.get());
    if (file_name == "") {
      if (num_setup_threads < 0) {
        throw TablePipeError("Number of setup threads cannot be negative.");
      }
      if (num_setup_threads == 0) {
        num_setup_threads = std::max(1u, std::thread::hardware_concurrency());
      }
      int_fast32_t l = hash.num_tables_;

      num_setup_threads = std::min(l, num_setup_threads);
      int_fast32_t num_tables_per_thread = l / num_setup_threads;
      int_fast32_t num_leftover_tables = l % num_setup_threads;

      std::vector<std::future<void>> thread_results;
      int_fast32_t next_table_range_start = 0;

      for (int_fast32_t ii = 0; ii < num_setup_threads; ++ii) {
        int_fast32_t next_table_range_end =
            next_table_range_start + num_tables_per_thread - 1;
        if (ii < num_leftover_tables) {
          next_table_range_end += 1;
        }
        thread_results.push_back(
            std::async(std::launch::async, &TablePipe::setup_table_range, this,
                       next_table_range_start, next_table_range_end));
        next_table_range_start = next_table_range_end + 1;
      }

      for (int_fast32_t ii = 0; ii < num_setup_threads; ++ii) {
        thread_results[ii].get();
      }
    } else {
      FILE *input = fopen(file_name.c_str(), "rb");
      if (!input) {
        throw TablePipeError("can't open file for reading");
      }
      std::vector<uint32_t> table_hashes;
      for (int_fast32_t ii = 0; ii < hash.num_tables_; ++ii) {
        ir::deserialize(input, &table_hashes);
        table_->add_entries_for_table(table_hashes, ii);
      }
      if (fclose(input)) {
        throw TablePipeError("can't close file after reading");
      }
    }
  }

  template <typename IteratorType>
  Iterator<IteratorType> run(int32_t worker_id, IteratorType &it) {
    UNUSED(worker_id);
    return Iterator<IteratorType>(*this, it);
  }

  void add_table() {
    table_->add_table();
    if (table_->get_l() != hash_.num_tables_) {
      throw TablePipeError("invalid number of tables");
    }
    typename HashProducer<PointType>::HashType::template BatchHash<
        DataStorageType>
        bh(*hash_.hasher_);
    std::vector<uint32_t> table_hashes;
    bh.batch_hash_single_table(dataset_, hash_.num_tables_ - 1, &table_hashes);
    table_->add_entries_for_table(table_hashes, hash_.num_tables_ - 1);
  }

  void serialize(FILE *output) { table_->serialize(output); }

  void serialize(const std::string &file_name) { table_->serialize(file_name); }

 private:
  const std::vector<Point> &dataset_;
  const HashProducer<PointType> &hash_;
  std::shared_ptr<FactoryType> factory_;
  std::shared_ptr<HashTableType> table_;

  void setup_table_range(int_fast32_t from, int_fast32_t to) {
    typename HashProducer<PointType>::HashType::template BatchHash<
        DataStorageType>
        bh(*hash_.hasher_);
    std::vector<uint32_t> table_hashes;
    for (int_fast32_t ii = from; ii <= to; ++ii) {
      bh.batch_hash_single_table(dataset_, ii, &table_hashes);
      table_->add_entries_for_table(table_hashes, ii);
    }
  }
};

template <typename PointType>
class DeduplicationPipeThreadUnsafe {
 public:
  template <typename IteratorType>
  class Iterator {
   public:
    Iterator(DeduplicationPipeThreadUnsafe &filter, IteratorType &it)
        : filter_(filter), it_(it) {
      advance();
    }
    bool is_valid() const { return it_.is_valid(); }
    int32_t get() const { return it_.get(); }
    void operator++() {
      ++it_;
      advance();
    }

   private:
    DeduplicationPipeThreadUnsafe &filter_;
    IteratorType &it_;

    void advance() {
      while (it_.is_valid()) {
        int32_t x = it_.get();
        if (filter_.used_[x] != filter_.query_id_) {
          filter_.used_[x] = filter_.query_id_;
          break;
        }
        ++it_;
      }
    }
  };

  DeduplicationPipeThreadUnsafe(int32_t num_workers, int32_t n)
      : n_(n), used_(n, -1), query_id_(0) {
    UNUSED(num_workers);
  }

  template <typename IteratorType>
  Iterator<IteratorType> run(IteratorType &g) {
    ++query_id_;
    return Iterator<IteratorType>(*this, g);
  }

 private:
  int32_t n_;
  std::vector<int32_t> used_;
  int32_t query_id_;
};

template <typename PointType, typename PointSet = std::vector<PointType>>
class DistanceScorerThreadUnsafe {
 public:
  using ScoreType = float;

  DistanceScorerThreadUnsafe(int32_t num_workers, const PointSet &dataset)
      : dataset_(dataset), query_(nullptr) {
    UNUSED(num_workers);
  }

  void load_query(int32_t worker_id, const PointType &query) {
    UNUSED(worker_id);
    query_ = &query;
  }

  void prepare(int32_t worker_id, const int32_t point_id) {
    UNUSED(worker_id);
    __builtin_prefetch((dataset_.data() + point_id), 0, 1);
  }

  ScoreType get_score(int32_t worker_id, int32_t point_id) {
    UNUSED(worker_id);
    return (*query_ - dataset_[point_id]).squaredNorm();
  }

 private:
  const PointSet &dataset_;
  const PointType *query_;
};

class TopKPipeError : public FalconnError {
 public:
  TopKPipeError(const char *msg) : FalconnError(msg) {}
};

template <typename ScorerType>
class TopKPipeThreadUnsafe {
 public:
  class Iterator {
   public:
    Iterator(TopKPipeThreadUnsafe &parent, int32_t len)
        : parent_(parent), i_(0), len_(len) {}

    bool is_valid() const { return i_ < len_; }

    int32_t get() const { return parent_.h_.get_data()[i_].data; }

    void operator++() { ++i_; }

   private:
    TopKPipeThreadUnsafe &parent_;
    int32_t i_, len_;
  };

  TopKPipeThreadUnsafe(int32_t num_workers, int32_t k, bool sort = false,
                       int32_t look_ahead = 1)
      : k_(k), sort_(sort), look_ahead_(look_ahead), arr_(look_ahead) {
    UNUSED(num_workers);
    if (k <= 0) {
      throw TopKPipeError("k must be positive");
    }

    if (look_ahead < 0) {
      throw TopKPipeError("look_ahead must be non-negative");
    }
  }

  void set_k(int32_t k) {
    if (k <= 0) {
      throw TopKPipeError("k must be positive");
    }
    k_ = k;
  }

  template <typename IteratorType>
  Iterator run(int32_t worker_id, IteratorType &g, ScorerType &s) {
    h_.reset();
    h_.resize(k_);
    int32_t initially_inserted = 0;
    if (g.is_valid()) {
      s.prepare(worker_id, g.get());
    }
    for (; initially_inserted < k_; ++initially_inserted) {
      if (g.is_valid()) {
        auto val = g.get();
        auto score = s.get_score(worker_id, val);
        h_.insert_unsorted(-score, val);
        ++g;
        if (g.is_valid()) {
          s.prepare(worker_id, g.get());
        }
      } else {
        break;
      }
    }

    if (initially_inserted >= k_) {
      h_.heapify();
      if (look_ahead_ == 1) {
        while (g.is_valid()) {
          auto val = g.get();
          ++g;

          if (g.is_valid()) {
            s.prepare(worker_id, g.get());
          }

          auto score = s.get_score(worker_id, val);
          if (score < -h_.min_key()) {
            h_.replace_top(-score, val);
          }
        }
      }

      if (look_ahead_ > 1) {
        int int_count = 0;
        while (g.is_valid()) {
          auto val = g.get();
          s.prepare(worker_id, val);
          ++g;
          if (int_count >= look_ahead_) {
            auto now_val = arr_[int_count % look_ahead_];
            auto score = s.get_score(worker_id, now_val);
            if (score < -h_.min_key()) {
              h_.replace_top(-score, now_val);
            }
          }
          arr_[int_count % look_ahead_] = val;
          int_count++;
        }
      }

      if (look_ahead_ == 0) {
        if (g.is_valid()) {
          s.prepare(worker_id, g.get());
        }
        while (g.is_valid()) {
          auto val = g.get();
          ++g;
          if (g.is_valid()) {
            s.prepare(worker_id, g.get());
          }

          auto score = s.get_score(worker_id, val);
          if (score < -h_.min_key()) {
            h_.replace_top(-score, val);
          }
        }
      }
    }

    if (sort_) {
      std::sort(h_.get_data().begin(),
                h_.get_data().begin() + initially_inserted,
                [](auto x, auto y) { return x.key > y.key; });
    }
    Iterator res(*this, initially_inserted);
    return res;
  }

 private:
  using ScoreType = typename ScorerType::ScoreType;
  core::SimpleHeap<ScoreType, int32_t> h_;
  int32_t k_;
  bool sort_;
  int32_t look_ahead_;
  std::vector<int32_t> arr_;
};

class DistanceScorerError : public FalconnError {
 public:
  DistanceScorerError(const char *msg) : FalconnError(msg) {}
};

template <typename PointType>
class DistanceScorer {
 public:
  using ScoreType = typename DistanceScorerThreadUnsafe<PointType>::ScoreType;

  DistanceScorer(int32_t num_workers, const std::vector<PointType> &dataset)
      : workers_(num_workers,
                 DistanceScorerThreadUnsafe<PointType>(num_workers, dataset)) {}

  void load_query(int32_t worker_id, const PointType &query) {
    if (worker_id < 0 || worker_id >= static_cast<int32_t>(workers_.size())) {
      throw DistanceScorerError(
          "worker id is not in the range 0 to num_workers_ - 1");
    }
    workers_[worker_id].load_query(worker_id, query);
  }

  void prepare(int32_t worker_id, int32_t point_id) {
    if (worker_id < 0 || worker_id >= static_cast<int32_t>(workers_.size())) {
      throw DistanceScorerError(
          "worker id is not in the range 0 to num_workers_ - 1");
    }
    workers_[worker_id].prepare(worker_id, point_id);
  }

  ScoreType get_score(int32_t worker_id, int32_t point_id) {
    if (worker_id < 0 || worker_id >= static_cast<int32_t>(workers_.size())) {
      throw DistanceScorerError(
          "worker id is not in the range 0 to num_workers_ - 1");
    }
    return workers_[worker_id].get_score(worker_id, point_id);
  }

 private:
  std::vector<DistanceScorerThreadUnsafe<PointType>> workers_;
};

class DeduplicationPipeError : public FalconnError {
 public:
  DeduplicationPipeError(const char *msg) : FalconnError(msg) {}
};

template <typename PointType>
class DeduplicationPipe {
 public:
  DeduplicationPipe(int32_t num_workers, int32_t num_points)
      : workers_(num_workers,
                 DeduplicationPipeThreadUnsafe<PointType>(1, num_points)) {}

  template <typename IteratorType>
  typename DeduplicationPipeThreadUnsafe<PointType>::template Iterator<
      IteratorType>
  run(int32_t worker_id, IteratorType &it) {
    if (worker_id < 0 || worker_id >= static_cast<int32_t>(workers_.size())) {
      throw DeduplicationPipeError(
          "worker id is not in the range 0 to num_workers_ - 1");
    }
    return workers_[worker_id].run(it);
  }

 private:
  std::vector<DeduplicationPipeThreadUnsafe<PointType>> workers_;
};

template <typename ScorerType>
class TopKPipe {
 public:
  TopKPipe(int32_t num_workers, int32_t k, bool sort = false,
           int32_t look_ahead = 1)
      : workers_(num_workers, TopKPipeThreadUnsafe<ScorerType>(
                                  num_workers, k, sort, look_ahead)) {}

  void set_k(int32_t k) {
    for (auto &worker : workers_) {
      worker.set_k(k);
    }
  }

  template <typename IteratorType>
  typename TopKPipeThreadUnsafe<ScorerType>::Iterator run(int32_t worker_id,
                                                          IteratorType &it,
                                                          ScorerType &s) {
    if (worker_id < 0 || worker_id >= static_cast<int32_t>(workers_.size())) {
      throw TopKPipeError(
          "worker id is not in the range 0 to num_workers_ - 1");
    }
    return workers_[worker_id].run(worker_id, it, s);
  }

 private:
  std::vector<TopKPipeThreadUnsafe<ScorerType>> workers_;
};

}  // namespace experimental
}  // namespace falconn

#endif  // __PIPES_H__
