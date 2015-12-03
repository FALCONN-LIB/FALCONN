#ifndef __FALCONN_GLOBAL_H__
#define __FALCONN_GLOBAL_H__

#include <stdexcept>

namespace falconn {

// Common exception base class
class FalconnError: public std::logic_error {
 public:
  FalconnError(const char* msg) : logic_error(msg) {}
};


// Data structure for point query statistics
struct QueryStatistics {
  double average_total_query_time = 0.0; // total query time
  double average_lsh_time = 0.0;         // computing LSH functions
  double average_hash_table_time = 0.0;  // retrieving from the hash tables
  double average_distance_time = 0.0;    // computing the candidate distances
  double average_num_candidates = 0;
  double average_num_unique_candidates = 0;
};


// Workaround for the CYGWIN bug described in
// http://stackoverflow.com/questions/28997206/cygwin-support-for-c11-in-g4-9-2

#ifdef __CYGWIN__

#include <cmath>

namespace std {
using ::log2;
using ::round;
};

#endif

}  // namespace falconn

#endif
