import numpy as np
import pytest

import ecpfs

from .conftest import TWO_CLUSTERS, write_h5_embeddings


def assert_is_panic_exception(exc):
    # Rust panics surface to Python as pyo3_runtime.PanicException, a
    # BaseException subclass, not a plain Exception. pyo3_runtime isn't
    # independently importable (it's a synthetic module PyO3 attaches to
    # the exception type, not a real sys.modules entry), so an explicit
    # type-name/module check is the reliable way to assert on it from
    # outside the extension.
    assert type(exc).__name__ == "PanicException"
    assert type(exc).__module__ == "pyo3_runtime"
    assert not issubclass(type(exc), Exception), "PanicException must not be catchable as a plain Exception"


def test_select_representatives_rejects_an_unsupported_embeddings_file_extension(tmp_path):
    index_path = tmp_path / "index.zarr"
    builder = ecpfs.Builder(index_path, levels=2, metric=ecpfs.Metric.L2)

    bogus_path = tmp_path / "embeddings.txt"
    bogus_path.write_text("not an embeddings file")

    with pytest.raises(BaseException, match="unsupported embeddings file format") as exc_info:
        builder.select_representatives(bogus_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)

    assert_is_panic_exception(exc_info.value)


def test_build_rejects_an_unsupported_embeddings_file_extension(tmp_path):
    # build() opens its own embeddings_file independently of whatever file
    # select_representatives used, so this is a separate call site from the
    # one above.
    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings(h5_path, TWO_CLUSTERS)
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(index_path, levels=2, metric=ecpfs.Metric.L2)
    builder.select_representatives(h5_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)

    bogus_path = tmp_path / "embeddings.txt"
    bogus_path.write_text("not an embeddings file")

    with pytest.raises(BaseException, match="unsupported embeddings file format") as exc_info:
        builder.build(bogus_path, fallback_batch_rows=100)

    assert_is_panic_exception(exc_info.value)


def test_select_representatives_custom_rejects_mismatched_ids_and_embeddings_lengths(tmp_path):
    index_path = tmp_path / "index.zarr"
    builder = ecpfs.Builder(index_path, levels=2, metric=ecpfs.Metric.L2)

    ids = np.array([0, 1, 2], dtype=np.uint32)
    embeddings = TWO_CLUSTERS[:2]  # 2 rows, but 3 ids - a real mismatch.

    with pytest.raises(BaseException, match="ids and embeddings must have the same length") as exc_info:
        builder.select_representatives_custom(ids, embeddings)

    assert_is_panic_exception(exc_info.value)
