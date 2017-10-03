namespace falconn {

template <typename PointType, typename KeyType = int32_t>
class LSHNearestNeighborQuery {
 public:
  virtual void set_num_probes(int_fast64_t num_probes) = 0;

  virtual int_fast64_t get_num_probes() = 0;

  virtual void set_max_num_candidates(int_fast64_t max_num_candidates) = 0;

  virtual int_fast64_t get_max_num_candidates() = 0;

  virtual KeyType find_nearest_neighbor(
      const PointType& q,
      SketchesQueryable<PointType, KeyType>* sketches = nullptr) = 0;

  virtual void find_k_nearest_neighbors(
      const PointType& q, int_fast64_t k, std::vector<KeyType>* result,
      SketchesQueryable<PointType, KeyType>* sketches = nullptr) = 0;

  virtual void find_near_neighbors(
      const PointType& q,
      typename PointTypeTraits<PointType>::ScalarType threshold,
      std::vector<KeyType>* result,
      SketchesQueryable<PointType, KeyType>* sketches = nullptr) = 0;

  virtual void get_unique_candidates(
      const PointType& q, std::vector<KeyType>* result,
      SketchesQueryable<PointType, KeyType>* sketches = nullptr) = 0;

  virtual void get_candidates_with_duplicates(
      const PointType& q, std::vector<KeyType>* result,
      SketchesQueryable<PointType, KeyType>* sketches = nullptr) = 0;

  virtual void reset_query_statistics() = 0;

  virtual QueryStatistics get_query_statistics() = 0;

  virtual ~LSHNearestNeighborQuery() {}
};

template <typename PointType, typename KeyType = int32_t>
class LSHNearestNeighborQueryPool {
 public:
  virtual void set_num_probes(int_fast64_t num_probes) = 0;

  virtual int_fast64_t get_num_probes() = 0;

  virtual void set_max_num_candidates(int_fast64_t max_num_candidates) = 0;

  virtual int_fast64_t get_max_num_candidates() = 0;

  virtual KeyType find_nearest_neighbor(
      const PointType& q,
      SketchesQueryable<PointType, KeyType>* sketches = nullptr) = 0;

  virtual void find_k_nearest_neighbors(
      const PointType& q, int_fast64_t k, std::vector<KeyType>* result,
      SketchesQueryable<PointType, KeyType>* sketches = nullptr) = 0;

  virtual void find_near_neighbors(
      const PointType& q,
      typename PointTypeTraits<PointType>::ScalarType threshold,
      std::vector<KeyType>* result,
      SketchesQueryable<PointType, KeyType>* sketches = nullptr) = 0;

  virtual void get_unique_candidates(
      const PointType& q, std::vector<KeyType>* result,
      SketchesQueryable<PointType, KeyType>* sketches = nullptr) = 0;

  virtual void get_candidates_with_duplicates(
      const PointType& q, std::vector<KeyType>* result,
      SketchesQueryable<PointType, KeyType>* sketches = nullptr) = 0;

  virtual void reset_query_statistics() = 0;

  virtual QueryStatistics get_query_statistics() = 0;

  virtual ~LSHNearestNeighborQueryPool() {}
};

template <typename PointType, typename KeyType = int32_t>
class LSHNearestNeighborTable {
 public:
  static const int_fast64_t kNoMaxNumCandidates = -1;

  virtual std::unique_ptr<LSHNearestNeighborQuery<PointType, KeyType>>

  construct_query_object(int_fast64_t num_probes = -1,
                         int_fast64_t max_num_candidates = -1) const = 0;

  virtual std::unique_ptr<LSHNearestNeighborQueryPool<PointType, KeyType>>

  construct_query_pool(int_fast64_t num_probes = -1,
                       int_fast64_t max_num_candidates = -1,
                       int_fast64_t num_query_objects = 0) const = 0;

  virtual ~LSHNearestNeighborTable() {}
};

enum class LSHFamily { Unknown = 0, Hyperplane = 1, CrossPolytope = 2 };

enum class DistanceFunction {
  Unknown = 0,
  NegativeInnerProduct = 1,
  EuclideanSquared = 2
};

enum class StorageHashTable {
  Unknown = 0,
  FlatHashTable = 1,
  BitPackedFlatHashTable = 2,
  STLHashTable = 3,
  LinearProbingHashTable = 4
};

struct LSHConstructionParameters {
  int_fast32_t dimension = -1;
  LSHFamily lsh_family = LSHFamily::Unknown;
  DistanceFunction distance_function = DistanceFunction::Unknown;
  int_fast32_t k = -1;
  int_fast32_t l = -1;
  StorageHashTable storage_hash_table = StorageHashTable::Unknown;
  int_fast32_t num_setup_threads = -1;
  uint64_t seed = 409556018;
  int_fast32_t last_cp_dimension = -1;
  int_fast32_t num_rotations = -1;
  int_fast32_t feature_hashing_dimension = -1;
};

template <typename PointType>
void compute_number_of_hash_functions(int_fast32_t number_of_hash_bits,
                                      LSHConstructionParameters* params);

template <typename PointType>
LSHConstructionParameters get_default_parameters(
    int_fast64_t dataset_size, int_fast32_t dimension,
    DistanceFunction distance_function, bool is_sufficiently_dense);

template <typename PointType, typename KeyType = int32_t,
          typename PointSet = std::vector<PointType>>
std::unique_ptr<LSHNearestNeighborTable<PointType, KeyType>> construct_table(
    const PointSet& points, const LSHConstructionParameters& params);

}  // namespace falconn
