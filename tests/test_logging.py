import json
import subprocess
import sys
from pathlib import Path

import pytest

import ecpfs


def test_init_logging_creates_a_real_jsonl_file_with_valid_log_lines(tmp_path):
    # ecp_core::logging::init's global logger can only be set once per
    # process (log::set_boxed_logger), so init_logging is idempotent -
    # a later call in the same process silently returns the first call's
    # path, ignoring its own log_dir/level. A plain in-process call here
    # would be at the mercy of whatever other test ran first in this
    # pytest session, so this runs in its own subprocess to guarantee it's
    # really the first (and only) call.
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
    # parse_level runs before logging::init touches the process-global
    # logger, so this fails the same way regardless of what earlier tests
    # already called init_logging with - no subprocess needed.
    with pytest.raises(ValueError, match="unknown log level"):
        ecpfs.init_logging(log_dir=str(tmp_path), level="bogus")
