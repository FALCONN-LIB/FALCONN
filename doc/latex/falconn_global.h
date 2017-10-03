namespace falconn {

template <typename CoordinateType>
using DenseVector =
    Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>;

template <typename CoordinateType, typename IndexType = int32_t>
using SparseVector = std::vector<std::pair<IndexType, CoordinateType>>;

struct QueryStatistics {
  double average_total_query_time;
  double average_lsh_time;
  double average_hash_table_time;
  double average_sketches_time;
  double average_distance_time;
  double average_num_candidates;
  double average_num_unique_candidates;
  double average_num_filtered_candidates;
  int_fast64_t num_queries;
};

template <typename CoordinateType>
struct PlainArrayPointSet {
  const CoordinateType* data;
  int_fast32_t num_points;
  int_fast32_t dimension;
};

}  // namespace falconn
