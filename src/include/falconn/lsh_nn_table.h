#ifndef __WRAPPER_H__
#define __WRAPPER_H__

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

#include "falconn_global.h"

///
/// The main namespace.
///
namespace falconn {


// An exception class for errors occuring in the wrapper classes. Errors from
// the internal classes will throw other errors that also derive from
// FalconError.
class LSHNearestNeighborTableError : public FalconnError {
 public:
  LSHNearestNeighborTableError(const char* msg) : FalconnError(msg) {}
};


///
/// The common interface shared by all LSH table wrappers.
///
/// The template parameter PointType should be one of the two point types
/// introduced above (DenseVector and SparseVector), e.g., DenseVector<float>.
///
/// The KeyType template parameter is optional and the default int32_t is
/// sufficient for up to 10^9 points.
///
template<typename PointType, typename KeyType = int32_t>
class LSHNearestNeighborTable {
 public:
  ///
  /// Sets the number of probes used for each query.
  /// The default setting is l (number of tables), which effectively disables
  /// multiprobing (the probing sequence only contains a single candidate per
  /// table).
  ///
  virtual void set_num_probes(int_fast64_t num_probes) = 0;
  ///
  /// Gets the number of probes used for each query.
  ///
  virtual int_fast64_t get_num_probes() = 0;
 
  ///
  /// Sets the maximum number of candidates considered in each query.
  /// The constant kNoMaxNumCandidates indicates that all candidates retrieved
  /// in the probing sequence should be considered. This is the default and
  /// usually a good setting. A maximum number of candidates is mainly useful
  /// to give worst case running time guarantees for every query.
  ///
  virtual void set_max_num_candidates(int_fast64_t max_num_candidates) = 0;
  ///
  /// Gets the maximum number of candidates considered in each query.
  ///
  virtual int_fast64_t get_max_num_candidates() = 0;
  ///
  /// A special constant for set_max_num_candidates which is effectively
  /// equivalent to the infinity.
  ///
  static const int_fast64_t kNoMaxNumCandidates = -1;

  ///
  /// Finds the key of the closest candidate in the probing sequence for q.
  ///
  virtual KeyType find_closest(const PointType& q) = 0;

  ///
  /// Find the keys of the k closest candidates in the probing sequence for q.
  /// The keys are returned in order of increasing distance to q.
  ///
  virtual void find_k_nearest_neighbors(
      const PointType& q,
      int_fast64_t k,
      std::vector<KeyType>* result) = 0;

  ///
  /// Returns the keys corresponding to candidates in the probing sequence for q
  /// that have distance at most threshold.
  ///
  virtual void find_near_neighbors(
      const PointType& q,
      typename PointTypeTraits<PointType>::ScalarType threshold,
      std::vector<KeyType>* result) = 0;
 
  ///
  /// Returns the keys of all candidates in the probing sequence for q. If a
  /// candidate key is found in multiple tables, it will appear multiple times
  /// in the result. The candidates are returned in the order in which they
  /// appear in the probing sequence.
  ///
  virtual void get_candidates_with_duplicates(
      const PointType& q,
      std::vector<KeyType>* result) = 0;

  ///
  /// Returns the keys of all candidates in the probing sequence for q.
  /// Every candidate key occurs only once in the result. The
  /// candidates are returned in the order of their first occurrence in the
  /// probing sequence.
  ///
  virtual void get_unique_candidates(
      const PointType& q,
      std::vector<KeyType>* result) = 0;
  
  ///
  /// Returns the keys of all candidates in the probing sequence for q.
  /// Every candidate key occurs only once in the result,
  /// and the candidate keys are also sorted by their key. This
  /// can be good for memory locality when the next processing step is a linear
  /// scan over the resulting candidates.
  ///
  virtual void get_unique_sorted_candidates(
      const PointType& q,
      std::vector<KeyType>* result) = 0;
 
  ///
  /// Resets the query statistics.
  ///
  virtual void reset_query_statistics() = 0;

  ///
  /// Returns the current query statistics.
  ///
  /// TODO: figure out the right semantics here: should the average distance
  /// time be averaged over all queries or only the near(est) neighbor queries?
  ///
  virtual QueryStatistics get_query_statistics() = 0;

  ///
  /// Virtual destructor.
  ///
  virtual ~LSHNearestNeighborTable() {}
};


///
/// The supported LSH families.
///
enum class LSHFamily {
  Unknown = 0,
  
  ///
  /// The hyperplane hash proposed in
  ///
  /// "Similarity estimation techniques from rounding algorithms"
  /// Moses S. Charikar
  /// STOC 2002
  ///
  Hyperplane = 1,

  ///
  /// The cross polytope hash first proposed in
  ///
  /// "Spherical LSH for Approximate Nearest Neighbor Search on Unit Hypersphere",
  /// Kengo Terasawa, Yuzuru Tanaka,
  /// WADS 2007
  ///
  /// Our implementation utilizes the improvements described in
  ///
  /// "Practical and Optimal LSH for Angular Distance",
  /// Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn, Ludwig
  /// Schmidt,
  /// NIPS 2015
  ///
  CrossPolytope = 2
};


///
/// The supported distance functions.
///
/// Note that we use distance functions only to filter the candidates in
/// find_closest, find_k_nearest_neighbors and find_near_neighbors. For just
/// returning all the candidates, the distance function is irrelevant.
///
enum class DistanceFunction {
  Unknown = 0,

  ///
  /// The distance between p and q is -<p, q>. For unit vectors p and q,
  /// this means that the nearest neighbor to q has the smallest angle with q.
  ///
  NegativeInnerProduct = 1,  
};


///
/// Contains the parameters for constructing a LSH table wrapper. Not all fields
/// are necessary for all types of LSH tables.
///
struct LSHConstructionParameters {
  // Required parameters
  ///
  /// Dimension of the points. Required for all the hash families.
  ///
  int_fast32_t dimension = -1;
  ///
  /// Hash family. Required for all the hash families.
  ///
  LSHFamily lsh_family = LSHFamily::Unknown;
  ///
  /// Distance function. Required for all the hash families.
  ///
  DistanceFunction distance_function = DistanceFunction::Unknown;
  ///
  /// Number of hash function per table. Required for all the hash families.
  ///
  int_fast32_t k = -1;
  ///
  /// Number of hash tables. Required for all the hash families.
  ///
  int_fast32_t l = -1;
  ///
  /// Randomness seed.
  ///
  uint64_t seed = 409556018;

  // Optional parameters
  ///
  /// Dimension of the last of the k cross-polytopes. Required
  /// only for the cross-polytope hash.
  ///
  int_fast32_t last_cp_dimension = -1;    // only necessary for CP hash.
  ///
  /// Number of pseudo-random rotations. Required only for the
  /// cross-polytope hash.
  /// 
  /// For sparse data, it is recommended to set it to 2, for sufficiently
  /// dense data, 1 is enough as well.
  ///
  int_fast32_t num_rotations = -1;  // only necessary for CP hash.
  ///
  /// Intermediate dimension for the feature hashing. Ignored for the hyperplane
  /// hash. The smaller it is, the faster hashing becomes, but the worse the hash
  /// quality becomes. The value -1 means no feature hashing is being performed.
  ///
  int_fast32_t feature_hashing_dimension = -1;   
};


///
/// Computes the number of hash functions in order to get a hash with the given
/// number of relevant bits. For the cross polytope hash, the last cross polytope
/// dimension will also be determined. The input struct params must contain valid
/// values for the following fields:
///   - lsh_family
///   - dimension (for the cross polytope hash)
///   - feature_hashing_dimension (for the cross polytope hash with sparse
///     vectors)
/// The function will then set the following fields of params:
///   - k
///   - last_cp_dim (for the cross polytope hash, both dense and sparse)
///
template<typename PointType>
void compute_number_of_hash_functions(int_fast32_t number_of_hash_bits,
                                      LSHConstructionParameters* params);


///
/// An exception class for errors occuring while setting up the LSH table
/// wrapper.
///
class LSHNNTableSetupError : public FalconnError {
 public:
  LSHNNTableSetupError(const char* msg) : FalconnError(msg) {}
};


///
/// Function for constructing an LSH table wrapper. The template parameters
/// PointType and KeyType are as in LSHNearestNeighborTable above. The
/// PointSet template parameter default is set so that a std::vector<PointType>
/// can be passed as the set of points for which a LSH table should be
/// constructed.
///
/// The points object *must* stay valid for the lifetime of the LSH table.
///
/// The caller assumes ownership of the returned pointer.
///
template<
typename PointType,
typename KeyType = int32_t,
typename PointSet = std::vector<PointType>>
std::unique_ptr<LSHNearestNeighborTable<PointType, KeyType>> construct_table(
    const PointSet& points,
    const LSHConstructionParameters& params);

}  // namespace falconn

#include "wrapper/cpp_wrapper_impl.h"

#endif
