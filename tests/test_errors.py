import pytest

import ecpfs


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
