import json
import subprocess
import sys
from pathlib import Path

import pytest

import ecpfs


def test_init_logging_creates_a_jsonl_file_with_valid_log_lines(tmp_path):
    # Only the first init_logging call in a process sets logging up; a later
    # one returns the same path and ignores its own log_dir and level. This
    # runs in a subprocess so it is that first call, whatever else the test
    # session did.
    script = f"""
import json
import ecpfs

log_path = ecpfs.init_logging(log_dir={str(tmp_path)!r}, level="debug")
print(log_path)

builder = ecpfs.Builder({str(tmp_path / "index.zarr")!r}, levels=1, metric=ecpfs.Metric.L2)

with open(log_path) as f:
    lines = [line for line in f.read().splitlines() if line.strip()]
assert lines, "expected at least one log line from constructing the Builder"
for line in lines:
    json.loads(line)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    log_path = Path(result.stdout.strip())
    assert log_path.exists()
    assert log_path.parent == tmp_path


def test_init_logging_rejects_an_unknown_level(tmp_path):
    # The level is parsed before logging is set up, so this fails the same way
    # whatever earlier tests did. No subprocess needed.
    with pytest.raises(ValueError, match="unknown log level"):
        ecpfs.init_logging(log_dir=str(tmp_path), level="bogus")
