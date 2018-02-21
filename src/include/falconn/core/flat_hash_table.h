#ifndef __FLAT_HASH_TABLE_H__
#define __FLAT_HASH_TABLE_H__

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hash_table_helpers.h"

namespace falconn {
namespace core {

class FlatHashTableError : public HashTableError {
 public:
  FlatHashTableError(const char* msg) : HashTableError(msg) {}
};

template <typename KeyType, typename ValueType = int32_t,
          typename IndexType = int32_t>
class FlatHashTable {
 public:
  class Factory {
   public:
    Factory(IndexType num_buckets) : num_buckets_(num_buckets) {
      if (num_buckets_ < 1) {
        throw FlatHashTableError("Number of buckets must be at least 1.");
      }
    }

    FlatHashTable<KeyType, ValueType, IndexType>* new_hash_table() {
      return new FlatHashTable<KeyType, ValueType, IndexType>(num_buckets_);
    }

   private:
    IndexType num_buckets_ = 0;
  };

  typedef IndexType* Iterator;

  FlatHashTable(IndexType num_buckets) : num_buckets_(num_buckets) {}

  // TODO: add version with explicit values array? (maybe not because the flat
  // hash table is arguably most useful for the static table setting?)
  void add_entries(const std::vector<KeyType>& keys) {
    if (num_buckets_ <= 0) {
      throw FlatHashTableError("Non-positive number of buckets");
    }
    if (entries_added_) {
      throw FlatHashTableError("Entries were already added.");
    }

    entries_added_ = true;

    std::vector<IndexType> bucket_counts(num_buckets_, 0);
    indices_.resize(keys.size());

    for (IndexType ii = 0; static_cast<size_t>(ii) < indices_.size(); ++ii) {
      if (keys[ii] >= static_cast<KeyType>(num_buckets_) || keys[ii] < 0) {
        throw FlatHashTableError("Key value out of range.");
      }
      indices_[ii] = ii;
      bucket_counts[keys[ii]]++;
    }

    KeyComparator comp(keys);
    std::sort(indices_.begin(), indices_.end(), comp);

    bucket_start_.resize(num_buckets_, 0);
    for (IndexType ii = 1; ii < num_buckets_; ++ii) {
      bucket_start_[ii] = bucket_start_[ii - 1] + bucket_counts[ii - 1];
    }
  }

  std::pair<Iterator, Iterator> retrieve(const KeyType& key) {
    IndexType start = bucket_start_[key];
    IndexType end = static_cast<IndexType>(indices_.size());
    if (static_cast<IndexType>(key) < num_buckets_ - 1) {
      end = bucket_start_[key + 1];
    }
    assert(start <= end);
    // printf("retrieve for key %u\n", key);
    // printf("  start: %lld  end %lld\n", start, end);
    return std::make_pair(&(indices_[start]), &(indices_[end]));
  }

 private:
  IndexType num_buckets_ = -1;
  bool entries_added_ = false;
  // start index of the respective hash bucket
  std::vector<IndexType> bucket_start_;
  // point indices
  std::vector<ValueType> indices_;

  class KeyComparator {
   public:
    KeyComparator(const std::vector<KeyType>& keys) : keys_(keys) {}

    bool operator()(IndexType ii, IndexType jj) {
      return keys_[ii] < keys_[jj];
    }

    const std::vector<KeyType>& keys_;
  };
};

}  // namespace core
}  // namespace falconn

#endif
