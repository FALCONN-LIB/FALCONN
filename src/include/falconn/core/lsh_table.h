#ifndef __LSH_TABLE_H__
#define __LSH_TABLE_H__

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "data_storage.h"
#include "../falconn_global.h"

namespace falconn {
namespace core {

class LSHTableError : public FalconnError {
 public:
  LSHTableError(const char* msg) : FalconnError(msg) {}
};


// An LSH implementation for a single set of LSH functions.
// The actual LSH (and low-level hashing) implementations can be added via
// template parameters.

template<
typename LSH,                  // the LSH family
typename HashTable,            // the low-level hash tables
typename Derived>
class BasicLSHTable {
 public:
  BasicLSHTable(LSH* lsh,
                HashTable* hash_table) : lsh_(lsh),
                                         hash_table_(hash_table) {
    if (lsh_ == nullptr) {
      throw LSHTableError("The LSH object cannot be a nullptr.");
    }
    if (hash_table_ == nullptr) {
      throw LSHTableError("The low-level hash table cannot be a nullptr.");
    }
    if (lsh_->get_l() != hash_table_->get_l()) {
      throw LSHTableError("Number of tables in LSH and low level hash table"
                          " objects does not match.");
    }
  }

  LSH* LSH_object() {
    return lsh_;
  }

  HashTable* low_level_hash_table() {
    return hash_table_;
  }
 
 protected:
  LSH* lsh_;
  HashTable* hash_table_;
};


template<
typename PointType,            // the type of the data points to be stored
typename KeyType,              // must be integral for a static table
typename LSH,                  // the LSH family
typename HashType,             // type returned by a set of k LSH functions
typename HashTable,            // the low-level hash tables
typename DataStorageType = ArrayDataStorage<PointType, KeyType>>
class StaticLSHTable : public BasicLSHTable<LSH, HashTable, StaticLSHTable<
    PointType, KeyType, LSH, HashType, HashTable, DataStorageType>> {
 public:
  StaticLSHTable(LSH* lsh,
                 HashTable* hash_table,
                 const DataStorageType& points)
      : BasicLSHTable<LSH, HashTable, StaticLSHTable<PointType, KeyType, LSH,
            HashType, HashTable, DataStorageType>>(lsh, hash_table),
        n_(points.size()) {
    typename LSH::template BatchHash<DataStorageType> bh(*(this->lsh_));
    std::vector<HashType> table_hashes;

    for (int_fast32_t ii = 0; ii < this->lsh_->get_l(); ++ii) {
      bh.batch_hash_single_table(points, ii, &table_hashes);
      this->hash_table_->add_entries_for_table(table_hashes, ii);
    }
  }
  
  // TODO: add query statistics back in
  class Query {
   public:
    Query(const StaticLSHTable& parent)
      : parent_(parent),
        is_candidate_(parent.n_),
        lsh_query_(*(parent.lsh_)) {}


    void get_candidates_with_duplicates(const PointType& p,
                                        int_fast64_t num_probes,
                                        int_fast64_t max_num_candidates,
                                        std::vector<KeyType>* result) {
      auto start_time = std::chrono::high_resolution_clock::now();
      stats_num_queries_ += 1; 
      
      lsh_query_.get_probes_by_table(p, &tmp_probes_by_table_, num_probes);

      auto lsh_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_lsh = std::chrono::duration_cast<
          std::chrono::duration<double>>(lsh_end_time - start_time);
      stats_.average_lsh_time += elapsed_lsh.count();

      hash_table_iterators_ = parent_.hash_table_->retrieve_bulk(
          tmp_probes_by_table_);
      
      int_fast64_t num_candidates = 0;
      result->clear();
      if (max_num_candidates < 0) {
        max_num_candidates = std::numeric_limits<int_fast64_t>::max();
      }
      while (num_candidates < max_num_candidates
             && hash_table_iterators_.first != hash_table_iterators_.second) {
        num_candidates += 1;
        result->push_back(*(hash_table_iterators_.first));
        ++hash_table_iterators_.first;
      }
      
      auto hashing_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_hashing = std::chrono::duration_cast<
          std::chrono::duration<double>>(hashing_end_time - lsh_end_time);
      stats_.average_hash_table_time += elapsed_hashing.count();

      stats_.average_num_candidates += num_candidates;
      
      auto end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_total = std::chrono::duration_cast<
          std::chrono::duration<double>>(end_time - start_time);
      stats_.average_total_query_time += elapsed_total.count();
    }

    void get_unique_candidates(const PointType& p,
                               int_fast64_t num_probes,
                               int_fast64_t max_num_candidates,
                               std::vector<KeyType>* result) {
      auto start_time = std::chrono::high_resolution_clock::now();
      stats_num_queries_ += 1; 
      
      get_unique_candidates_internal(p, num_probes, max_num_candidates, result);
      
      auto end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_total = std::chrono::duration_cast<
          std::chrono::duration<double>>(end_time - start_time);
      stats_.average_total_query_time += elapsed_total.count();
    }

    void get_unique_sorted_candidates(const PointType& p,
                                      int_fast64_t num_probes,
                                      int_fast64_t max_num_candidates,
                                      std::vector<KeyType>* result) {
      auto start_time = std::chrono::high_resolution_clock::now();
      stats_num_queries_ += 1; 
      
      get_unique_candidates_internal(p, num_probes, max_num_candidates, result);
      std::sort(result->begin(), result->end());
      
      auto end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_total = std::chrono::duration_cast<
          std::chrono::duration<double>>(end_time - start_time);
      stats_.average_total_query_time += elapsed_total.count();
    }

    void reset_query_statistics() {
      stats_num_queries_ = 0;
      stats_.average_total_query_time = 0.0;
      stats_.average_lsh_time = 0.0;
      stats_.average_hash_table_time = 0.0;
      stats_.average_distance_time = 0.0;
      stats_.average_num_candidates = 0.0;
      stats_.average_num_unique_candidates = 0.0;
    }


    QueryStatistics get_query_statistics() {
      QueryStatistics res = stats_;
      if (stats_num_queries_ > 0) {
        res.average_total_query_time /= stats_num_queries_;
        res.average_lsh_time /= stats_num_queries_;
        res.average_hash_table_time /= stats_num_queries_;
        res.average_distance_time /= stats_num_queries_;
        res.average_num_candidates /= stats_num_queries_;
        res.average_num_unique_candidates /= stats_num_queries_;
      }
      return res;
    }

    // TODO: add void get_candidate_sequence(const PointType& p)
    // TODO: add void get_unique_candidate_sequence(const PointType& p)

   private:
    const StaticLSHTable& parent_;
    int_fast32_t query_counter_ = 0;
    std::vector<int32_t> is_candidate_;
    typename LSH::Query lsh_query_;
    std::vector<std::vector<HashType>> tmp_probes_by_table_;
    std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
        hash_table_iterators_;
    
    QueryStatistics stats_;
    int_fast64_t stats_num_queries_ = 0;
    
    void get_unique_candidates_internal(const PointType& p,
                                        int_fast64_t num_probes,
                                        int_fast64_t max_num_candidates,
                                        std::vector<KeyType>* result) {
      auto start_time = std::chrono::high_resolution_clock::now();
      
      lsh_query_.get_probes_by_table(p, &tmp_probes_by_table_, num_probes);
      
      auto lsh_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_lsh = std::chrono::duration_cast<
          std::chrono::duration<double>>(lsh_end_time - start_time);
      stats_.average_lsh_time += elapsed_lsh.count();
      
      hash_table_iterators_ = parent_.hash_table_->retrieve_bulk(
          tmp_probes_by_table_);
      query_counter_ += 1;

      int_fast64_t num_candidates = 0;
      result->clear();
      if (max_num_candidates < 0) {
        max_num_candidates = std::numeric_limits<int_fast64_t>::max();
      }
      while (num_candidates < max_num_candidates
             && hash_table_iterators_.first != hash_table_iterators_.second) {
        num_candidates += 1;
        int_fast64_t cur = *(hash_table_iterators_.first);
        if (is_candidate_[cur] != query_counter_) {
          is_candidate_[cur] = query_counter_;
          result->push_back(cur);
        }

        ++hash_table_iterators_.first;
      }
      
      auto hashing_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_hashing = std::chrono::duration_cast<
          std::chrono::duration<double>>(hashing_end_time - lsh_end_time);
      stats_.average_hash_table_time += elapsed_hashing.count();
      
      stats_.average_num_candidates += num_candidates;
      stats_.average_num_unique_candidates += result->size();
    }
  };
 
 private:
  int_fast64_t n_;
};


/*
template<
typename PointType,            // the type of the data points to be stored
typename DistanceType,         // type of distances between points, e.g., float
typename DistanceFunction,     // distance function used to compare points
typename LSH,                  // the LSH family
typename HashType,             // type returned by a set of k LSH functions
typename HashTable,            // the low-level hash tables
typename Prefetcher = DefaultPrefetcher<PointType>,  // to prefetch candidates
typename DataStorage = std::vector<PointType>>   // for storing the point data
class BasicLSHTable {
 public:
  BasicLSHTable(LSH* lsh,
                HashTable* hash_table,
                const DataStorage& points) : lsh_(lsh),
                                             hash_table_(hash_table),
                                             points_(points) {
    if (lsh_ == nullptr) {
      throw LSHTableInitError("The LSH object cannot be a nullptr.");
    }
    if (hash_table_ == nullptr) {
      throw LSHTableInitError("The low-level hash table cannot be a nullptr.");
    }
    if (lsh_->get_l() != hash_table_->get_l()) {
      throw LSHTableInitError("Number of tables in LSH and low level hash table"
                              " objects does not match.");
    }

    l_ = lsh_->get_l();
    lsh_->set_num_extra_probes(0);
    tmp_probes_per_table_.resize(l_);
    tmp_probes_.resize(l_);
    tmp_hashes_.resize(l_);
  }

  LSH* LSH_object() {
    return lsh_;
  }

  HashTable* low_level_hash_table() {
    return hash_table_;
  }

  const DataStorage& data_storage() {
    return points_;
  }

  void set_num_extra_probes(int_fast32_t num_extra_probes) {
    lsh_->set_num_extra_probes(num_extra_probes);
    tmp_probes_.resize(l_ + num_extra_probes);
  }

  int_fast64_t find_closest(const PointType& q,
                            int_fast64_t max_num_candidates = -1) {
    return find_closest(q, q, max_num_candidates);
  }

  int_fast64_t find_closest(const PointType& q_hash,
                            const PointType& q_comparison,
                            int_fast64_t max_num_candidates = -1) {
    query_counter_ += 1;
    stats_num_queries_ += 1;

    // Steps 1 and 2: compute LSH function and retrieve candidates
    retrieve_unique_candidates(q_hash, &unique_candidates_, max_num_candidates);

    auto time2 = std::chrono::high_resolution_clock::now();

    // Step 3: compute distances for candidates

    int_fast64_t best_index = -1;
    if (unique_candidates_.size() > 0) {
      std::sort(unique_candidates_.begin(),
                unique_candidates_.end());
      best_index = unique_candidates_[0];
      DistanceType best_distance = dst_(q_comparison, points_[best_index]);
      
      // TODO: find better prefetch strategy?
      if (unique_candidates_.size() >= 2) {
        prefetcher_.prefetch(points_, unique_candidates_[1]);

        if (unique_candidates_.size() >= 3) {
          prefetcher_.prefetch(points_, unique_candidates_[2]);
        }
      }

      for (int_fast64_t ii = 1; ii < unique_candidates_.size(); ++ii) {
        int_fast64_t cur = unique_candidates_[ii];

        if (ii + 2 < unique_candidates_.size()) {
          prefetcher_.prefetch(points_, unique_candidates_[ii + 2]);
        }

        DistanceType cur_distance = dst_(q_comparison, points_[cur]);
        if (cur_distance < best_distance) {
          best_distance = cur_distance;
          best_index = cur;
        }
      }
    }

    auto time3 = std::chrono::high_resolution_clock::now();

    auto elapsed_dst  = std::chrono::duration_cast<
        std::chrono::duration<double>>(time3 - time2);
    stats_.average_distance_time += elapsed_dst.count();
    
    return best_index;
  }


  void find_all_near_neighbors(const PointType& q,
                               DistanceType threshold,
                               std::vector<int_fast64_t>* result,
                               int_fast64_t max_num_candidates = -1) {
    if (result == nullptr) {
      throw LSHTableLookupError("Results vector pointer is nullptr.");
    }

    std::vector<int_fast64_t>& res = *result;
    res.clear();

    query_counter_ += 1;
    stats_num_queries_ += 1;

    // Steps 1 and 2: compute LSH function and retrieve candidates
    retrieve_unique_candidates(q, &unique_candidates_, max_num_candidates);
    
    auto time2 = std::chrono::high_resolution_clock::now();

    // Step 3: compute distances for candidates

    if (unique_candidates_.size() > 0) {
      std::sort(unique_candidates_.begin(),
                unique_candidates_.end());
      // TODO: find better prefetch strategy?
      if (unique_candidates_.size() >= 1) {
        prefetcher_.prefetch(points_, unique_candidates_[0]);

        if (unique_candidates_.size() >= 2) {
          prefetcher_.prefetch(points_, unique_candidates_[1]);
        }
      }

      for (int_fast64_t ii = 0; ii < unique_candidates_.size(); ++ii) {
        int_fast64_t cur = unique_candidates_[ii];

        if (ii + 2 < unique_candidates_.size()) {
          prefetcher_.prefetch(points_, unique_candidates_[ii + 2]);
        }

        DistanceType cur_distance = dst_(q, points_[cur]);
        if (cur_distance <= threshold) {
          res.push_back(cur);
        }
      }
    }

    auto time3 = std::chrono::high_resolution_clock::now();
    
    auto elapsed_dst  = std::chrono::duration_cast<
        std::chrono::duration<double>>(time3 - time2);
    stats_.average_distance_time += elapsed_dst.count();
  }


  // Canidates will not appear in sorted index order.
  void get_all_candidates(const PointType& q,
                          std::vector<int_fast64_t>* result,
                          int_fast64_t max_num_candidates = -1) {
    if (result == nullptr) {
      throw LSHTableLookupError("Results vector pointer is nullptr.");
    }

    result->clear();

    query_counter_ += 1;
    stats_num_queries_ += 1;

    // Steps 1 and 2: compute LSH function and retrieve candidates
    retrieve_unique_candidates(q, result, max_num_candidates);
    
    // No Step 3 (distance computation)
  }


  void reset_query_statistics() {
    stats_num_queries_ = 0;
    stats_.average_lsh_time = 0.0;
    stats_.average_hash_table_time = 0.0;
    stats_.average_distance_time = 0.0;
    stats_.average_num_candidates = 0.0;
    stats_.average_num_unique_candidates = 0.0;
  }


  QueryStatistics get_query_statistics() {
    QueryStatistics res = stats_;
    if (stats_num_queries_ > 0) {
      res.average_lsh_time /= stats_num_queries_;
      res.average_hash_table_time /= stats_num_queries_;
      res.average_distance_time /= stats_num_queries_;
      res.average_num_candidates /= stats_num_queries_;
      res.average_num_unique_candidates /= stats_num_queries_;
    }
    return res;
  }


 protected:
  LSH* lsh_;
  HashTable* hash_table_;
  const DataStorage& points_;  // essentially a vector of data points

  int_fast32_t l_;

  QueryStatistics stats_;
  int_fast64_t stats_num_queries_ = 0;

  int_fast32_t query_counter_ = 0;
  std::vector<int32_t> is_candidate_;
  std::vector<int_fast64_t> unique_candidates_;

  DistanceFunction dst_;
  std::vector<HashType> tmp_probes_;  // for storing the result of one LSH call
  std::vector<HashType> tmp_hashes_;  // for storing the result of one LSH call
  std::vector<int_fast32_t> tmp_probes_per_table_;
  Prefetcher prefetcher_;


  void retrieve_unique_candidates(const PointType& q,
                                  std::vector<int_fast64_t>* unique_candidates,
                                  int_fast64_t max_num_candidates) {
    auto time0 = std::chrono::high_resolution_clock::now();

    // Step 1: compute LSH functions for the query point
   
    lsh_->get_probes(q, &tmp_probes_, &tmp_probes_per_table_);

    auto time1 = std::chrono::high_resolution_clock::now();

    // Step 2: retrieve the candidates from the hash tables
    
    auto iterators = hash_table_->retrieve(tmp_probes_, tmp_probes_per_table_);
    int_fast64_t num_candidates = 0;
    unique_candidates->clear();
    if (max_num_candidates < 0) {
      max_num_candidates = std::numeric_limits<int_fast64_t>::max();
    }
    while (num_candidates < max_num_candidates
           && iterators.first != iterators.second) {
      num_candidates += 1;
      int_fast64_t cur = *(iterators.first);

      if (is_candidate_[cur] != query_counter_) {
        is_candidate_[cur] = query_counter_;
        unique_candidates->push_back(cur);
      }

      ++iterators.first;
    }

    auto time2 = std::chrono::high_resolution_clock::now();

    auto elapsed_lsh  = std::chrono::duration_cast<
        std::chrono::duration<double>>(time1 - time0);
    auto elapsed_hash = std::chrono::duration_cast<
        std::chrono::duration<double>>(time2 - time1);

    stats_.average_lsh_time += elapsed_lsh.count();
    stats_.average_hash_table_time += elapsed_hash.count();
    stats_.average_num_candidates += num_candidates;
    stats_.average_num_unique_candidates += unique_candidates_.size();
  }
};



template<
typename PointType,            // the type of the data points to be stored
typename DistanceType,         // type of distances between points, e.g., float
typename DistanceFunction,     // distance function used to compare points
typename LSH,                  // the LSH family
typename HashType,             // type returned by a set of k LSH functions
typename HashTable,            // the low-level hash tables
typename Prefetcher = DefaultPrefetcher<PointType>,  // to prefetch candidates
typename DataStorage = std::vector<PointType>>   // for storing the point data
class DynamicLSHTable : public BasicLSHTable<PointType,
                                             DistanceType,
                                             DistanceFunction,
                                             LSH,
                                             HashType,
                                             HashTable,
                                             Prefetcher,
                                             DataStorage> {
public:
  DynamicLSHTable(LSH* lsh,
                  HashTable* hash_table,
                  const DataStorage& points) : BasicLSHTable<PointType,
                                                             DistanceType,
                                                             DistanceFunction,
                                                             LSH,
                                                             HashType,
                                                             HashTable,
                                                             Prefetcher,
                                                             DataStorage>(
                                                                 lsh,
                                                                 hash_table,
                                                                 points) {}

  void insert(int_fast64_t index) {
    (this->lsh_)->hash((this->points_)[index], &(this->tmp_hashes_));
    (this->hash_table_)->insert((this->tmp_hashes_), index);

    if (index >= (this->is_candidate_).size()) {
      size_t old_size = (this->is_candidate_).size();
      (this->is_candidate_).resize(std::max(old_size * 2,
                                            static_cast<size_t>(1)));
      for (size_t ii = old_size; ii < (this->is_candidate_).size(); ++ii) {
        (this->is_candidate_)[ii] = 0;
      }
    }
  }

  void remove(int_fast64_t index) {
    (this->lsh_)->hash((this->points_)[index], &(this->tmp_hashes_));
    (this->hash_table_)->remove((this->tmp_hashes_), index);
  }
};*/

}  // namespace core
}  // namespace falconn


#endif
