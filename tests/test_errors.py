import pytest

import ecpfs


def assert_is_panic_exception(exc):
    # Rust panics reach Python as pyo3_runtime.PanicException, a BaseException
    # subclass rather than an Exception. pyo3_runtime cannot be imported, since
    # PyO3 attaches it to the exception type instead of adding it to sys.modules,
    # so checking the type's name and module is the way to assert on it here.
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
