# Hyperplane and Cross-polytope

Currently, FALCONN supports two LSH families for Cosine similarity:
Hyperplane and Cross-polytope. In general, Cross-polytope LSH is better, but
Hyperplane is simpler to tune, since it has fewer parameters.

Let us point out that, in practice, both Hyperplane and Cross-polytope LSH can
be useful to handle the Euclidean distance in the whole R<sup>d</sup>, despite
being designed for cosine similarity.

Note that Cosine similarity corresponds to the Euclidean distance on
a sphere. So, to design an LSH family for Cosine similarity, we need to
build a good random space partition of a unit sphere in a d-dimensional
Euclidean space R<sup>d</sup> centered in the origin.

## Hyperplane LSH

The Hyperplane LSH is very simple. To partition a sphere, we sample a _random
hyperplane_ through the center of the sphere and cut the sphere across it in two
equal parts. An alternative way of thinking about it is as follows: we sample
a random d-dimensional vector r whose coordinates are i.i.d. standard Gaussians,
and, to hash a point v, we compute the sign of the dot product &lt;r, v&gt;.

For two points with angle &alpha; between then, the probability of collision is
exactly equal to 1 - &alpha;/&pi;.

## Cross-polytope LSH

First, we describe a version of the Cross-polytope LSH, which is simple but
impractical. We start with sampling a random rotation S. Then, for a point v, we
set its hash to the plus-minus standard basis vector that is the closest to Sv.
That is, we partition the sphere into 2d buckets. Another way of saying the same
is: we inscribe a randomly rotated
[cross-polytope](https://en.wikipedia.org/wiki/Cross-polytope) into the sphere,
and then partition the sphere according to the Voronoi diagram of the vertices
of the cross-polytope.

The reason why the above simple version is impractical is that truly random
rotations are expensive: computing Sv given v takes time O(d<sup>2</sup>). To
deal with this issue, we use _pseudo-random_ rotations instead. The underlying
primitive here is the
[Fast Hadamard Transform](https://github.com/)

The reason why the above simple version is impractical is that truly random
rotations are expensive: computing Sv given v takes time O(d<sup>2</sup>). To
deal with this issue, we use _pseudo-random_ rotations instead. The underlying
primitive here is the
[Fast Hadamard Transform](https://github.com/FALCONN-LIB/FFHT). Unlike truly
random rotations, we might need _more than one_ pseudo-random rotation to get
enough randomness. This brings the evaluation time down to O(d log d).

For extremely high-dimensional sparse data (think bag of words), we
apply feature hashing to some intermediate dimension d', and then apply the
cross-polytope LSH, which speeds-up the hash computation. The smaller d' is,
the faster hashing becomes, but at a cost of worsening the quality.

Finally, instead of inscribing a fully-dimensional cross-polytope we can
inscribe a d''-dimensional cross-polytope with 2d'' vertices. This allows for
better granularity. For instance, when d'' = 1, this simply becomes the
Hyperplane LSH family.
