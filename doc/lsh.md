# Locality-Sensitive Hashing: a Primer

The similarity search problem is as follows: given a dataset of points in a
_geometric space_, preprocess it, so that, given a query point, retrieve the
nearest data point (or several closest ones).

Locality-Sensitive Hashing (LSH, in short) is a way to randomly partition the
geometric space, where one's dataset lies, into cells so that the closer two
points are the more likely they end up in the same cell. It is one of the main
techniques for similarity search in high dimensions.

Let us consider a simple example. Suppose the data points lie on a _unit sphere_
in a d-dimensional space (Euclidean distance on a sphere corresponds to the
_cosine similarity_). Then, to partition the sphere, we sample a _random
hyperplane_ through the center of the sphere and cut the sphere across it in two
equal parts. Intuitively, it works, since any two close points will almost
always be on the same side of a hyperplane, while any pair of the opposite
points will always be separated.

Let us show how to use LSH for similarity search. A naive idea would be to
sample a partition, create a bucket for every cell, and group the data points
into these buckets. Then, given a query, we would try all the data points that
are in the same bucket as the query point. This procedure can be implemented
efficiently using a hash table. But the example of Hyperplane LSH shows that
this may not be good enough. Indeed, we merely have two buckets, so, we would
need to try around half of the data points when answering a query!

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
