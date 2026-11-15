"""Shared fixtures: the engine and streamer as real long-running services.

`services` starts both processes ONCE per test session against a temporary database
built by init_db (db/nids.db is never touched), on free loopback ports, and every
integration test drives them through the control channel only. Because the same two
processes serve every run in the session, the suite also shows that runs repeat
without restarting anything (plan §7.1).
"""
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from nids.control.client import ControlClient
from nids.prep.convert_xlsx import ROOT

ART = ROOT / "artefacts"
TEST_RAW = ROOT / "data" / "processed" / "test_raw.parquet"
LABEL_MAP = ART / "label_map.json"
HAVE_DATA = TEST_RAW.exists() and (ART / "rf_binary_seed42.joblib").exists()


@dataclass
class Services:
    db: Path
    client: ControlClient
    engine: subprocess.Popen
    streamer: subprocess.Popen
    data_port: int
    logs: Path


def _launch(args, log_path: Path) -> tuple[subprocess.Popen, dict]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
    err = open(log_path, "w")
    p = subprocess.Popen([sys.executable, "-m", *args], cwd=ROOT, env=env, text=True,
                         stdout=subprocess.PIPE, stderr=err)
    for line in p.stdout:
        if line.startswith("{"):
            ev = json.loads(line)
            if ev.get("event") == "ready":
                return p, ev
    p.wait()
    raise RuntimeError(f"{args[0]} did not start:\n{log_path.read_text()[-3000:]}")


def _terminate(p: subprocess.Popen) -> int:
    if p.poll() is None:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    return p.returncode


@pytest.fixture(scope="session")
def services(tmp_path_factory):
    if not HAVE_DATA:
        pytest.skip("test partition / artefacts missing")
    from nids.storage.init_db import init_db
    d = tmp_path_factory.mktemp("services")
    db = d / "nids.db"
    init_db(db, TEST_RAW, LABEL_MAP)
    eng, e = _launch(["nids.engine.engine", "--db", str(db), "--data-port", "0",
                      "--ctl-port", "0"], d / "engine.log")
    try:
        stm, s = _launch(["nids.stream.streamer", "--data-port", str(e["data_port"]),
                          "--ctl-port", "0"], d / "streamer.log")
    except Exception:
        _terminate(eng)
        raise
    client = ControlClient(e["ctl_port"], s["ctl_port"])
    client.wait_ready(60)
    yield Services(db, client, eng, stm, e["data_port"], d)
    rc = (_terminate(stm), _terminate(eng))
    assert rc == (0, 0), (f"services exited with {rc}\n"
                          + (d / "engine.log").read_text()[-2000:]
                          + (d / "streamer.log").read_text()[-2000:])
