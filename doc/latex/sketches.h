namespace falconn {

template <typename PointType, typename KeyType = int32_t>
class SketchesQueryable {
 public:
  virtual void filter_close(const PointType& query,
                            const std::vector<KeyType>& candidates,
                            std::vector<KeyType>* filtered_candidates) = 0;

  virtual ~SketchesQueryable() {}
};

template <typename PointType, typename DistanceType, typename KeyType = int32_t>
class Sketches {
 public:
  virtual std::unique_ptr<SketchesQueryable<PointType, KeyType>>
  construct_query_object(DistanceType distance_threshold) = 0;

  std::unique_ptr<SketchesQueryable<PointType, KeyType>> construct_query_pool(
      DistanceType distance_threshold, int_fast32_t num_query_objects = -1);

  virtual ~Sketches() {}
};

template <typename PointType, typename KeyType = int32_t,
          typename PointSet = std::vector<PointType>, typename RNGType>
std::unique_ptr<Sketches<PointType, int32_t, KeyType>>
construct_random_projection_sketches(const PointSet& points,
                                     int_fast32_t num_bits, RNGType& rng);

}  // namespace falconn
