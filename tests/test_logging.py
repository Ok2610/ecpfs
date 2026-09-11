import json
from pathlib import Path

import ecpfs


def test_init_logging_creates_a_real_jsonl_file_and_returns_its_path(tmp_path):
    log_path = ecpfs.init_logging(log_dir=str(tmp_path), level="info")

    log_path = Path(log_path)
    assert log_path.exists()
    assert log_path.parent == tmp_path


def test_init_logging_off_still_creates_the_file_but_logs_nothing(tmp_path):
    log_path = Path(ecpfs.init_logging(log_dir=str(tmp_path), level="off"))

    assert log_path.exists()
    assert log_path.read_text() == ""


def test_init_logging_writes_valid_json_lines(tmp_path):
    log_path = Path(ecpfs.init_logging(log_dir=str(tmp_path), level="debug"))

    # Trigger some real log output through the extension.
    index_path = tmp_path / "index.zarr"
    builder = ecpfs.Builder(index_path, levels=1, metric=ecpfs.Metric.L2)

    lines = [line for line in log_path.read_text().splitlines() if line.strip()]
    for line in lines:
        json.loads(line)
