# Locality-Sensitive Hashing: a Primer

Locality-Sensitive Hashing (LSH) is a class of methods for the nearest neighbor search problem, which is defined as follows:
given a dataset of points in a metric space (e.g., R<sup>d</sup> with the Euclidean distance),
our goal is to preprocess the data set so that we can quickly answer _nearest neighbor queries_:
given a previously unseen query point, we want to find one or several points in our dataset that
are closest to the query point.
LSH is one of the main techniques for nearest neighbor search in high dimensions (but there are also many others, e.g., see the
corresponding [Wikipedia article](https://en.wikipedia.org/wiki/Nearest_neighbor_search)).

In a nutshell, LSH is a way to randomly partition the ambient space into cells that respect the desired similarity metric.
The core building block of LSH are _locality-sensitive hash functions_.
We say that a hash function (or more precisely, a family of hash functions) is locality sensitive if the following property holds:
pairs of points that are close together are more likely to collide than pairs of points that are far apart.
LSH data structures use locality-sensitive hash functions in order to partition the space in wich the dataset lives: every possible hash value essentially corresponds to its own cell.

## An example: Hyperplane LSH

We first consider a simple example hash function that is locality sensitive. Suppose the data points lie on a _unit sphere_
in a d-dimensional space (Euclidean distance on a sphere corresponds to the
_cosine similarity_). In order to partition the sphere, we then sample a _random
hyperplane_ through the center of the sphere and cut the sphere across it, which gives two
equal cells. Intuitively, this approach gives a locality-sensitive hash function because any two close points will almost
always be on the same side of a hyperplane, while pairs of points that are far apart tend to be separated by the random
hyperplane (in the most extreme case of points that are opposite, any hyperplane will separate the points).

Let us now show how to use LSH for similarity search. A naive idea would be to
sample a partition, create a bucket for every cell, and group the data points
into these buckets. Given a query, we would then try all the data points that
are in the same bucket as the query point. This procedure can be implemented
efficiently using a hash table. But the example of hyperplane hash function outlined above shows that
this may not always be good enough. Indeed, with a single hyperplane we only have two cells, so, we would
need to test roughly half of the data points when answering a single query!

A better idea is to use many partitions at once. Instead of sampling one
partition, we sample K of them. Now buckets correspond to size-K tuples of
cells in the corresponding partitions. For instance, for the case of Hyperplane
LSH we have 2^K buckets, which means that, for a given query, we might hope to
enumerate a 1/2^K fraction of the data points (caveat: this is a bit of an
oversimplification, and the reality is slightly more complicated). So, choosing
K wisely, we achieve a good query time.

Are we done? Not quite. The reason is that we forgot to look at the probability
of success (finding the nearest neighbor). If we choose K to be large enough,
the probability of the query and the corresponding nearest neighbor falling
into the same bucket will be tiny. To cope with it, we repeat the whole
process several (say, L) times. Now, instead of having a single hash table,
we have L of them. And, given a query, we try all the data points in each of
the L buckets. The overall number of partitions we sample is KL.

How should one choose K and L? If K is too small, we have too many data points
per bucket, and the query time is high. If, on the other hand, K is too large,
we need L to be large to get a good probability of success. This leads to high
space consumption and slow queries. But, between these extremes, there is a
sweet spot!

That said, as it turns out, in most of the practical scenarios, to get good
query time, we need the number of tables L to be pretty large: somewhere
between 100 and 1000. This makes the space consumption prohibitive. Luckily,
there is a way to reduce the number of tables. The idea is to query more than
one bucket in each table. For a parameter T, for each of the L tables, we try T
buckets that are _most likely_ to contain the nearest neighbor. So, by
increasing T, we can decrease L and vice versa.

All in all, we have three parameters: the number of partitions per hash table K,
the number of hash tables L, and the number of probes per hash table T. One
usually first chooses L according to the memory budget. Then, one gets a
trade-off between K and T: the larger K is, the more probes we need to achieve a
given probability of success, and vice versa. The best way to choose K and T is
often to try several values of K and for each K find T that gives the desired
accuracy on a set of sample queries using binary search.
