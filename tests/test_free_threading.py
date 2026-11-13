import sys
import sysconfig
import threading

import numpy as np
import pytest

import ecpfs

from .conftest import build_two_clusters_index

FREE_THREADED = sysconfig.get_config_var("Py_GIL_DISABLED") == 1


@pytest.mark.skipif(not FREE_THREADED, reason="needs a free-threaded Python build")
def test_importing_ecpfs_keeps_the_gil_disabled():
    assert not sys._is_gil_enabled()


def test_threads_searching_one_index_get_the_single_thread_results(tmp_path):
    index_path = build_two_clusters_index(tmp_path)
    queries = [np.array([x, x], dtype=np.float32) for x in (0.0, 1.0, 10.0, 11.0)]

    def search(index, query):
        items, query_id = index.new_search(query=query, k=4, search_exp=2, max_increments=-1, exclude_vec=[])
        index.end_search(query_id)
        return [item_id for _, item_id in items]

    with ecpfs.Index(index_path) as index:
        expected = [search(index, query) for query in queries]
        start = threading.Barrier(8)
        mismatches = []

        def search_repeatedly():
            start.wait()
            for i in range(200):
                found = search(index, queries[i % len(queries)])
                if found != expected[i % len(queries)]:
                    mismatches.append(found)

        threads = [threading.Thread(target=search_repeatedly) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert mismatches == []
