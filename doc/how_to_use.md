# How to Use FALCONN

As of now, FALCONN supports static datasets. We plan to add the support for dynamic datasets shortly.

## Installation and Usage

FALCONN is a header-only library. It means that you simply need to add the folder `src/include`
to the folders searched for the header files (`-I` switch for `gcc`).

FALCONN uses [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) as a dependency.
Eigen is header-only as well, so you just need to make your compiler search in `external/eigen`
for the header files.

## The Main Class

To use FALCONN, one should include the main header file `lsh_nn_table.h` and use the namespace `falconn`
provided there.
The main wrapper class one should use is `LSHNearestNeighborTable`.
It should be constructed using a function `construct_table`.

### Constructing a Table

The function `construct_table` has one _required_ template parameter `PointType`, which is a type that
stores a single point. It can be equal to one of the following: `DenseVector<float>`, `DenseVector<double>`,
`SparseVector<float>`, `SparseVector<double>`. One needs to pass two things to `construct_table`:

- A dataset `points` whose type is `std::vector<PointType>`
- LSH construction parameters `params`, who type is `LSHConstructionParameters`.

The function `construct_table` returns a `unique_ptr` to the actual LSH table.

### Construction Parameters

A structure `LSHConstructionParameters` has quite a few fields. Some of them are necessary only for the
cross-polytope hash family.

- Parameters that are *always* required:
  - `dimension` is the dimension of the dataset
  - `lsh_family` can either be `LSHFamily::Hyperplane` or `LSHFamily::Crosspolytope`
  - `distance_function` is the distance function used for selecting the near neighbors among
  the filtered points, currently it can only be `DistanceFunction::NegativeInnerProduct`,
  which for vectors on the sphere is equivalent to the angle
  - `k` is the number of hash functions per hash table (see [LSH 101](lsh.md) on how to choose it)
  - `l` is the number of hash table (see [LSH 101](lsh.md) on how to choose it)
- (Required) parameters specific to the cross-polytope hash:
  - `last_cp_dimension` is the dimension of the last of the `k` cross-polytopes. This field
  is there to enable better granularity. `last_cp_dimension` must be between `1` and the smallest
  power of two that is not less than `dimension`. To simplify the tuning of `k` and
  `last_cp_dimension` for the cross-polytope hash, we provide a helper function
  `compute_number_of_hash_functions` that does the following. It takes `number_of_hash_bits`
  and the partially set construction parameters (in particular, `dimension` and `lsh_family` must be set),
  the function sets `k` and `last_cp_dimension` automatically such that the number of buckets
  is equal to `2^number_hash_bits`; we recommend using this subroutine
  - `num_rotations` is the number of pseudo-random rotations (see
  [Hyperplane and Cross-polytope](hyper_cp.md)
  for details); for dense enough data, set it to 1, for sparse data set it to 2
- (Optional) parameters:
  - `seed` is the randomness seed
- (Optional) parameters specific to the cross-polytope hash:
  - `feature_hashing_dimension` is the intermediate dimension (that must be a power of two):
  we first perform feature hashing to it,
  and then invoke the vanilla cross-polytope hash; the smaller `feature_hashing_dimension` is, the faster
  hash becomes, but the quality becomes worse as well; don't set it for medium-dimensional data,
  and set it to 512, 1024 or 2048 for high-dimensional sparse data (such as bag of words etc)

### Using the Table

The first important thing to do after setting up a table is to set the number of probes
which one makes during the query (see [LSH 101](lsh.md) for the discussion). This
can be done using a member function `set_num_probes`. By default, we query one bucket
per hash table, which is often suboptimal. Note that `set_num_probes` sets the number
of probes we make for _all_ the hash tables together.

Finally, we are ready to answer queries. On a very high level, LSH table answers queries as follows.
First, it retrieves a list of candidate data points using the multiprobe LSH procedure, then it scans
this list and chooses the best points according to the distance.

FALCONN provides low-level routines for generating the list candidates, as well as more high-level
functions that filter this list.

- High-level methods:
  - `find_closest` returns the closest data point among the candidates generated
  by the LSH look-up
  - `find_k_nearest_neighbors` returns the k closest data points among the candidates generated
  by the LSH look-up
  - `find_near_neighbors` returns all the candidates within a given distance threshold
- Low-level methods:
  - `get_candidates_with_duplicates` returns all the candidates with possible duplicates; this is
  the fastest and the _lowest level_ method
  - `get_unique_candidates` the same as above, but now the duplicates are eliminated
  - `get_unique_sorted_candidates`: the same as above, but the candidates are sorted according
  to the key

## Headers and Dependencies