import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest

from magezero.policy.engine import PersistentEngine

FAKE = '''
import sys
assert sys.argv[1:] == ["--serve"]
print("engine log line", flush=True)
print("MZ-POLICY READY", flush=True)
for line in sys.stdin:
    path = line.strip()
    print("game log: playing", path, flush=True)
    if path.endswith("bad.yml"):
        print("MZ-POLICY ERROR java.lang.IllegalStateException: no games terminated", flush=True)
    elif path.endswith("outcomes.yml"):
        print("MZ-POLICY RESULT 5 8 1 1 1", flush=True)
    elif path.endswith("hang.yml"):
        import time
        time.sleep(600)
    else:
        print("MZ-POLICY RESULT 3 4", flush=True)
'''


@pytest.fixture
def launcher(tmp_path):
    (tmp_path / "fake_engine.py").write_text(FAKE, encoding="utf-8")
    if sys.platform == "win32":
        script = tmp_path / "engine.bat"
        script.write_text(f'@echo off\r\n"{sys.executable}" "{tmp_path / "fake_engine.py"}" %*\r\n')
    else:
        script = tmp_path / "engine.sh"
        script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path / "fake_engine.py"}" "$@"\n')
        script.chmod(0o755)
    return script


def test_persistent_engine_runs_batches_and_reports_failures(launcher, tmp_path):
    engine = PersistentEngine(launcher, tmp_path / "engine.log", ready_timeout=60)
    try:
        # Engines built before per-game outcome counts: missing games are timeouts.
        assert engine.run(tmp_path / "first.yml") == (3, 4, 0, 0, 1)
        with pytest.raises(RuntimeError, match="no games terminated"):
            engine.run(tmp_path / "bad.yml")
        assert engine.run(tmp_path / "outcomes.yml") == (5, 8, 1, 1, 1)  # survives a failed batch
    finally:
        engine.close()
    assert engine.process.poll() is not None
    log = (tmp_path / "engine.log").read_text(encoding="utf-8")
    assert "engine log line" in log and log.count("game log: playing") == 3


def test_hung_batch_times_out_and_kill_stops_the_engine(launcher, tmp_path):
    engine = PersistentEngine(launcher, tmp_path / "engine.log", ready_timeout=60)
    try:
        with pytest.raises(TimeoutError):
            engine.run(tmp_path / "hang.yml", timeout=2)
    finally:
        engine.kill()
    assert engine.process.poll() is not None
    engine.kill()  # idempotent
