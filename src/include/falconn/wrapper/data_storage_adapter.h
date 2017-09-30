#ifndef __DATA_STORAGE_ADAPTER_H__
#define __DATA_STORAGE_ADAPTER_H__

#include "../core/data_storage.h"
#include "../falconn_global.h"

#include <memory>
#include <type_traits>
#include <vector>

namespace falconn {
namespace wrapper {

template <typename PointSet>
class DataStorageAdapter {
 public:
  DataStorageAdapter() {
    static_assert(FalseStruct<PointSet>::value,
                  "Point set type not supported.");
  }

  template <typename PS>
  struct FalseStruct : std::false_type {};
};

template <typename PointType>
class DataStorageAdapter<std::vector<PointType>> {
 public:
  template <typename KeyType>
  using DataStorage = core::ArrayDataStorage<PointType, KeyType>;

  template <typename KeyType>
  static std::unique_ptr<DataStorage<KeyType>> construct_data_storage(
      const std::vector<PointType>& points) {
    std::unique_ptr<DataStorage<KeyType>> res(new DataStorage<KeyType>(points));
    return std::move(res);
  }
};

template <typename CoordinateType>
class DataStorageAdapter<PlainArrayPointSet<CoordinateType>> {
 public:
  template <typename KeyType>
  using DataStorage =
      core::PlainArrayDataStorage<DenseVector<CoordinateType>, KeyType>;

  template <typename KeyType>
  static std::unique_ptr<DataStorage<KeyType>> construct_data_storage(
      const PlainArrayPointSet<CoordinateType>& points) {
    std::unique_ptr<DataStorage<KeyType>> res(new DataStorage<KeyType>(
        points.data, points.num_points, points.dimension));
    return std::move(res);
  }
};

}  // namespace wrapper
}  // namespace falconn

#endif /* __DATA_STORAGE_ADAPTER_H__ */
