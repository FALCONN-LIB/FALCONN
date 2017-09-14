import falconn
import numpy as np

n = 1000
d = 128


def test_lsh_index_positive():
    p = falconn.get_default_parameters(n, d)
    t = falconn.LSHIndex(p)
    dataset = np.random.randn(n, d).astype(np.float32)
    t.setup(dataset)

    def is_int(x):
        try:
            res = isinstance(x, (int, long))
            return res
        except NameError:
            res = isinstance(x, int)
            return res

    def test_positive(q):
        u = np.random.randn(d).astype(np.float32)
        assert isinstance(q.find_k_nearest_neighbors(u, 10), list)
        assert isinstance(q.find_near_neighbors(u, 10.0), list)
        assert is_int(q.find_nearest_neighbor(u))
        assert isinstance(q.get_candidates_with_duplicates(u), list)
        assert is_int(q.get_max_num_candidates())
        assert is_int(q.get_num_probes())
        assert isinstance(q.get_query_statistics(), falconn.QueryStatistics)
        assert isinstance(q.get_unique_candidates(u), list)
        assert q.reset_query_statistics() is None
        assert q.set_max_num_candidates(100) is None
        assert q.set_num_probes(10) is None

    q = t.construct_query_object()
    test_positive(q)
    q = t.construct_query_pool()
    test_positive(q)


def test_lsh_index_negative():
    p = falconn.get_default_parameters(n, d)
    try:
        t = falconn.LSHIndex(p)
        t.construct_query_object()
        assert False
    except RuntimeError:
        pass
    try:
        t = falconn.LSHIndex(p)
        t.setup([[1.0, 2.0], [3.0, 4.0]])
        assert False
    except TypeError:
        pass
    try:
        t = falconn.LSHIndex(p)
        t.setup(np.random.randn(n, d).astype(np.int32))
        assert False
    except TypeError:
        pass
    try:
        t = falconn.LSHIndex(p)
        t.setup(np.random.randn(10, 10, 10))
        assert False
    except ValueError:
        pass
    try:
        t = falconn.LSHIndex(p)
        t.setup(np.random.randn(n, d))
        t.setup(np.random.randn(n, d))
        assert False
    except RuntimeError:
        pass
    for (t1, t2) in [(np.float32, np.float64), (np.float64, np.float32)]:
        for g in [
                lambda t: t.construct_query_object(),
                lambda t: t.construct_query_pool()
        ]:
            t = falconn.LSHIndex(p)
            t.setup(np.random.randn(n, d).astype(t1))
            q = g(t)
            u = np.random.randn(d).astype(t1)

            try:
                q.find_k_nearest_neighbors(u, 0.5)
                assert False
            except TypeError:
                pass

            try:
                q.find_k_nearest_neighbors(u, -1)
                assert False
            except ValueError:
                pass

            try:
                q.find_near_neighbors(u, -1)
                assert False
            except ValueError:
                pass

            try:
                q.set_max_num_candidates(0.5)
                assert False
            except TypeError:
                pass
            try:
                q.set_max_num_candidates(-10)
                assert False
            except ValueError:
                pass
            q.set_num_probes(t._params.l)
            try:
                q.set_num_probes(t._params.l - 1)
                assert False
            except ValueError:
                pass
            try:
                q.set_num_probes(1000.1)
                assert False
            except TypeError:
                pass

            def check_check_query(f):
                try:
                    f(u.astype(t2))
                    assert False
                except TypeError:
                    pass
                try:
                    f([0.0] * d)
                    assert False
                except TypeError:
                    pass
                try:
                    f(u[:d - 1])
                    assert False
                except ValueError:
                    pass
                try:
                    f(np.random.randn(d, d))
                    assert False
                except ValueError:
                    pass

            check_check_query(lambda u: q.find_k_nearest_neighbors(u, 10))
            check_check_query(lambda u: q.find_near_neighbors(u, 0.5))
            check_check_query(lambda u: q.find_nearest_neighbor(u))
            check_check_query(lambda u: q.get_candidates_with_duplicates(u))
            check_check_query(lambda u: q.get_unique_candidates(u))
