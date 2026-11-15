"""Tests for the control channel (M7; plan §7.1, IT1/IT2 of plan §9; FR2, FR8, NFR5, NFR6).

Unit: framing and command validation, ControlServer behaviour, the streamer's Pacer.
Integration (tests/conftest.py `services`, one engine + one streamer for the session):
  - prepare -> start -> eos completes a 200-record run for each model (IT1)
  - further runs work without restarting either process
  - stop mid-run ends cleanly (eos "stopped", t4 flushed, ended_at set); stop on a
    prepared run cancels it
  - set_rate is demo-only: logged, re-paces the stream, marks MODEL_RUN.rate_changes
  - busy / invalid commands are refused without stopping a service
  - every listening socket is bound to 127.0.0.1 (ST2, NFR6)
  - run.py starts both services and shuts down cleanly on Ctrl+C (NFR5)
"""
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import pytest

from nids.control import protocol
from nids.control.client import ControlClient, ControlError
from nids.prep.convert_xlsx import ROOT
from nids.stream.streamer import Pacer
from tests.conftest import HAVE_DATA

needs_data = pytest.mark.skipif(not HAVE_DATA, reason="test partition / artefacts missing")
COMPONENTS = ("t_tx_ms", "t_pre_ms", "t_inf_ms", "t_ser_ms", "L_ms")


# ---------------------------------------------------------------- protocol units
def test_encode_decode_round_trip():
    raw = protocol.encode({"cmd": "status"})
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    assert protocol.decode(raw) == {"cmd": "status"}
    for bad in (b"not json\n", b"[1,2]\n", b"\"x\"\n"):
        with pytest.raises(protocol.ProtocolError):
            protocol.decode(bad)


def test_validate_commands():
    ok = protocol.validate({"cmd": "prepare", "model": "rf", "seed": 42, "dt_ms": 100,
                            "mode": "steady", "scenario": "S1", "n": 5000},
                           protocol.ENGINE_CMDS)
    assert ok["dt_ms"] == 100.0 and ok["n"] == 5000
    bad = [
        ({"cmd": "prepare", "model": "rf", "dt_ms": 100}, "seed"),            # missing
        ({"cmd": "prepare", "model": "rf", "seed": True, "dt_ms": 1}, "seed"),  # bool
        ({"cmd": "prepare", "model": "rf", "seed": 42, "dt_ms": -1}, "dt_ms"),
        ({"cmd": "prepare", "model": "rf", "seed": 42, "dt_ms": 1, "mode": "x"}, "mode"),
        ({"cmd": "prepare", "model": "rf", "seed": 42, "dt_ms": 1, "evil": 1}, "unknown"),
        ({"cmd": "start", "run_id": 1, "dt_ms": 1, "n": 5}, "unknown cmd"),   # wrong service
        ({"cmd": "shutdown"}, "unknown cmd"),
    ]
    for msg, word in bad:
        with pytest.raises(protocol.ProtocolError, match=word):
            protocol.validate(msg, protocol.ENGINE_CMDS)
    with pytest.raises(protocol.ProtocolError, match="n"):
        protocol.validate({"cmd": "start", "run_id": 1, "dt_ms": 1, "n": 0},
                          protocol.STREAMER_CMDS)


def test_control_server_loopback_errors_and_recovery():
    with pytest.raises(ValueError):
        protocol.ControlServer(0, lambda c: {}, protocol.ENGINE_CMDS, host="0.0.0.0")

    def handler(cmd):
        if cmd["cmd"] == "stop":
            raise RuntimeError("boom")
        return {"state": "idle"}

    srv = protocol.ControlServer(0, handler, protocol.ENGINE_CMDS)
    srv.start()
    try:
        assert srv.sock.getsockname()[0] == "127.0.0.1"
        with socket.create_connection(("127.0.0.1", srv.port), timeout=5) as s:
            f = s.makefile("rb")
            for line, expect_ok in ((b"garbage\n", False), (b'{"cmd":"stop"}\n', False),
                                    (b'{"cmd":"status"}\n', True)):
                s.sendall(line)
                r = json.loads(f.readline())
                assert r["ok"] is expect_ok, r
            assert r == {"ok": True, "state": "idle"}
            # a second client is served while the first connection is still open
            assert protocol.request(srv.port, {"cmd": "status"}, timeout=2)["ok"]
            f.close()
    finally:
        srv.close()
    with pytest.raises(ValueError):
        protocol.request(srv.port, {"cmd": "status"}, host="0.0.0.0")
    with pytest.raises(ValueError):
        ControlClient(host="192.168.1.1")


# ---------------------------------------------------------------- Pacer units
def test_pacer_matches_absolute_schedule_without_changes():
    offs = np.arange(10, dtype=np.int64) * 10_000_000
    p = Pacer(offs)
    p.begin(1_000)
    assert [p.due(i) for i in range(10)] == (1_000 + offs).tolist()


def test_pacer_set_rate_reanchors_exactly():
    p = Pacer(np.arange(100, dtype=np.int64) * 20_000_000)
    with pytest.raises(ValueError):
        p.set_rate(5)                                  # not started
    p.begin(0)
    for i in range(10):
        p.due(i)                                       # seq 0..9 fixed, last due 180 ms
    assert p.set_rate(5) == 10
    assert p.due(10) == 185_000_000 and p.due(11) == 190_000_000
    assert p.set_rate(50) == 12
    assert p.due(12) == 240_000_000
    assert p.changes == [{"from_seq": 10, "dt_ms": 5}, {"from_seq": 12, "dt_ms": 50}]
    q = Pacer(np.zeros(5, dtype=np.int64), mode="burst")
    q.begin(0)
    with pytest.raises(ValueError, match="steady"):
        q.set_rate(5)
    r = Pacer(np.arange(5, dtype=np.int64))
    r.begin(1_000_000)
    assert r.set_rate(2) == 0 and r.due(0) == 1_000_000 and r.due(1) == 3_000_000


# ---------------------------------------------------------------- integration helpers
def ro(db):
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def model_run(db, run_id):
    con = ro(db)
    con.row_factory = sqlite3.Row
    try:
        mr = dict(con.execute("SELECT * FROM model_run WHERE run_id=?", (run_id,)).fetchone())
        mr["n_pred"] = con.execute("SELECT COUNT(*) FROM prediction WHERE run_id=?",
                                   (run_id,)).fetchone()[0]
        mr["null_t4"] = con.execute("SELECT COUNT(*) FROM prediction WHERE run_id=? AND "
                                    "t4_store_ns IS NULL", (run_id,)).fetchone()[0]
        mr["negative"] = con.execute(
            "SELECT COUNT(*) FROM v_latency WHERE run_id=? AND (t_tx_ms < 0 OR t_pre_ms < 0 "
            "OR t_inf_ms < 0 OR t_ser_ms < 0)", (run_id,)).fetchone()[0]
        mr["t0"] = [r[0] for r in con.execute(
            "SELECT t0_emit_ns FROM prediction WHERE run_id=? ORDER BY seq_no", (run_id,))]
    finally:
        con.close()
    return mr


def wait_sent(client, at_least, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.streamer("status")["run"]
        if run and run["sent"] >= at_least:
            return run
        time.sleep(0.05)
    raise TimeoutError(f"streamer did not reach {at_least} records")


def assert_idle(client):
    st = client.status()
    assert st["engine"]["state"] == "idle" and st["streamer"]["state"] == "idle", st


# ---------------------------------------------------------------- IT1 via control
@needs_data
@pytest.mark.parametrize("model", ["iforest", "rf", "lstm"])
def test_200_record_run_via_control(services, model):
    c = services.client
    out = c.run(model, "binary", 42, dt_ms=20, mode="steady", scenario="IT1", n=200,
                timeout=300)
    run_id, eng, stm = out["run_id"], out["engine"], out["streamer"]
    assert eng == {"run_id": run_id, "processed": 200, "malformed": 0, "eos": True,
                   "end_reason": "complete", "rate_changes": None}
    assert stm["run_id"] == run_id and stm["n_sent"] == 200 and not stm["stopped"]
    mr = model_run(services.db, run_id)
    assert (mr["model_name"], mr["task"], mr["random_seed"], mr["emission_interval_ms"],
            mr["load_condition"], mr["scenario"], mr["n_records"], mr["malformed_count"]) \
        == (model, "binary", 42, 20.0, "steady", "IT1", 200, 0)
    assert mr["env_id"] is not None and json.loads(mr["hyperparameters"])["config"]
    assert mr["ended_at"] and mr["end_reason"] == "complete"
    assert mr["rate_changes"] is None                     # an experiment-valid run
    assert mr["n_pred"] == 200 and mr["null_t4"] == 0 and mr["negative"] == 0
    assert_idle(c)
    print(f"\n{model}: run {run_id} via control channel: 200 sent, 200 predictions, "
          f"pacing late median={stm['lateness_ms']['median']:.3f} ms")


@needs_data
def test_second_run_without_restart(services):
    c = services.client
    pids = (c.engine("status")["pid"], c.streamer("status")["pid"])
    ids = []
    for model in ("iforest", "rf"):
        out = c.run(model, dt_ms=10, scenario="repeat", n=100, timeout=120)
        assert out["engine"]["processed"] == 100 and out["engine"]["end_reason"] == "complete"
        ids.append(out["run_id"])
    assert ids[1] > ids[0]
    assert (c.engine("status")["pid"], c.streamer("status")["pid"]) == pids
    assert pids == (services.engine.pid, services.streamer.pid)
    assert services.engine.poll() is None and services.streamer.poll() is None


# ---------------------------------------------------------------- stop / cancel
@needs_data
def test_stop_mid_run_ends_cleanly(services):
    c = services.client
    run_id = c.prepare("rf", "binary", 42, 10, "steady", "stop-test", 5000)
    c.start(run_id, 10, "steady", 5000)
    wait_sent(c, 50)
    assert c.stop()["stopping"] is True
    eng = c.wait_done(run_id, timeout=30)
    stm = c.streamer("status")["last_run"]
    assert stm["run_id"] == run_id and stm["stopped"] and 50 <= stm["n_sent"] < 5000
    assert eng["eos"] and eng["end_reason"] == "stopped"
    assert eng["processed"] == stm["n_sent"] and eng["malformed"] == 0
    mr = model_run(services.db, run_id)
    assert mr["end_reason"] == "stopped" and mr["ended_at"]
    assert mr["n_pred"] == stm["n_sent"] and mr["null_t4"] == 0 and mr["negative"] == 0
    assert_idle(c)
    out = c.run("iforest", dt_ms=10, scenario="after-stop", n=50, timeout=60)
    assert out["engine"]["processed"] == 50                # next run is unaffected
    print(f"\nstop: run {run_id} stopped after {stm['n_sent']} of 5000 records; "
          f"{mr['n_pred']} predictions, all with t4; end_reason={mr['end_reason']}")


@needs_data
def test_stop_cancels_prepared_run(services):
    c = services.client
    run_id = c.prepare("iforest", "binary", 42, 100, "steady", "cancel", 100)
    assert c.engine("status")["state"] == "prepared"
    r = c.stop()                                            # streamer idle -> engine
    assert r["stopping"] and r["end_reason"] == "cancelled"
    eng = c.wait_done(run_id, timeout=10)
    assert eng["end_reason"] == "cancelled" and eng["processed"] == 0
    mr = model_run(services.db, run_id)
    assert mr["end_reason"] == "cancelled" and mr["ended_at"] and mr["n_pred"] == 0
    assert_idle(c)
    assert c.stop() == {"ok": True, "stopping": False, "state": "idle"}  # nothing to stop


# ---------------------------------------------------------------- set_rate (demo only)
@needs_data
def test_set_rate_repaces_and_marks_run(services):
    c = services.client
    with pytest.raises(ControlError, match="no run"):
        c.set_rate(5)                                       # idle: refused
    run_id = c.prepare("iforest", "binary", 42, 20, "steady", "demo", 300)
    c.start(run_id, 20, "steady", 300)
    wait_sent(c, 30)
    r = c.set_rate(5)
    k = r["from_seq"]
    assert r["run_id"] == run_id and 30 <= k < 300
    eng = c.wait_done(run_id, timeout=60)
    assert eng["end_reason"] == "complete" and eng["processed"] == 300
    mr = model_run(services.db, run_id)
    changes = json.loads(mr["rate_changes"])
    assert len(changes) == 1 and changes[0]["dt_ms"] == 5 and changes[0]["from_seq"] == k
    assert mr["emission_interval_ms"] == 20.0               # the configured value is kept
    assert c.streamer("status")["last_run"]["rate_changes"] == [{"from_seq": k, "dt_ms": 5}]
    gaps = np.diff(np.array(mr["t0"], dtype=np.int64)) / 1e6
    before, after = np.median(gaps[: k - 1]), np.median(gaps[k + 1:])
    assert abs(before - 20) < 1 and abs(after - 5) < 1
    con = ro(services.db)
    try:   # experiment runs are told apart by rate_changes IS NULL
        demo = [r[0] for r in con.execute(
            "SELECT run_id FROM model_run WHERE rate_changes IS NOT NULL")]
    finally:
        con.close()
    assert demo == [run_id]
    print(f"\nset_rate: run {run_id} dt 20 -> 5 ms from seq {k}; median gap "
          f"{before:.2f} -> {after:.2f} ms; MODEL_RUN.rate_changes={mr['rate_changes']}")


@needs_data
def test_set_rate_refused_in_burst(services):
    c = services.client
    run_id = c.prepare("iforest", "binary", 42, 20, "burst", "burst-demo", 120)
    c.start(run_id, 20, "burst", 120, burst_block=40)
    wait_sent(c, 5)
    with pytest.raises(ControlError, match="steady"):
        c.set_rate(5)
    eng = c.wait_done(run_id, timeout=60)
    assert eng["processed"] == 120 and eng["rate_changes"] is None


# ---------------------------------------------------------------- refusals
@needs_data
def test_busy_and_invalid_commands_refused(services):
    c = services.client
    for kw, word in ((dict(model="rf", seed=7), "seed"),          # not in seeds 42-46
                     (dict(model="svm", seed=42), "unknown detector"),
                     (dict(model="iforest", task="multi", seed=42), "tasks")):
        r = c.engine("prepare", dt_ms=10, **kw)
        assert not r["ok"] and word in r["error"], r
    r = c.engine("prepare", model="rf", seed=42)                   # missing dt_ms
    assert not r["ok"] and "dt_ms" in r["error"]
    assert_idle(c)

    run_id = c.prepare("iforest", "binary", 42, 10, "steady", "busy", 400)
    c.start(run_id, 10, "steady", 400)
    wait_sent(c, 10)
    r = c.engine("prepare", model="rf", seed=42, dt_ms=10)
    assert not r["ok"] and "busy" in r["error"]
    with pytest.raises(ControlError, match="busy"):
        c.start(run_id, 10, "steady", 400)
    eng = c.wait_done(run_id, timeout=60)
    assert eng["processed"] == 400 and eng["end_reason"] == "complete"


# ---------------------------------------------------------------- ST2 (NFR6)
@needs_data
def test_service_ports_bind_loopback_only(services):
    c = services.client
    listening = {}
    for name, pid in (("engine", services.engine.pid), ("streamer", services.streamer.pid)):
        conns = psutil.Process(pid).net_connections(kind="inet")
        listening[name] = {(x.laddr.ip, x.laddr.port) for x in conns
                           if x.status == psutil.CONN_LISTEN}
    assert listening["engine"] == {("127.0.0.1", services.data_port),
                                   ("127.0.0.1", c.engine_port)}
    assert listening["streamer"] == {("127.0.0.1", c.streamer_port)}
    print(f"\nlistening sockets: {listening}")


# ---------------------------------------------------------------- run.py (NFR5)
@needs_data
def test_run_py_starts_services_and_shuts_down_cleanly(tmp_path):
    db = tmp_path / "nids.db"                      # absent: run.py creates it
    err = open(tmp_path / "run.log", "w")
    p = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "--no-dashboard",
                          "--db", str(db), "--data-port", "0", "--engine-ctl-port", "0",
                          "--streamer-ctl-port", "0"],
                         cwd=ROOT, stdout=subprocess.PIPE, stderr=err, text=True)
    lines = []
    ready = None
    try:
        for line in p.stdout:
            lines.append(line)
            if line.startswith('{"event": "ready", "service": "run"'):
                ready = json.loads(line)
                break
        assert ready, "".join(lines) + (tmp_path / "run.log").read_text()
        drain = threading.Thread(target=lambda: lines.extend(p.stdout), daemon=True)
        drain.start()
        c = ControlClient(ready["engine_ctl_port"], ready["streamer_ctl_port"])
        run_id = c.prepare("iforest", "binary", 42, 10, "steady", "ctrl-c", 5000)
        c.start(run_id, 10, "steady", 5000)
        wait_sent(c, 50)
        os.kill(p.pid, signal.SIGINT)                       # Ctrl+C
        assert p.wait(60) == 0
        drain.join(5)
    finally:
        if p.poll() is None:
            p.kill()
    out = "".join(lines)
    assert "stopped cleanly" in out and "engine stopped (exit code 0)" in out \
        and "streamer stopped (exit code 0)" in out, out
    for pid in (ready["engine_pid"], ready["streamer_pid"]):
        assert not psutil.pid_exists(pid) or \
            psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    mr = model_run(db, run_id)
    assert mr["end_reason"] == "stopped" and mr["ended_at"] and mr["null_t4"] == 0
    print(f"\nrun.py: Ctrl+C during run {run_id} -> stopped after {mr['n_pred']} records, "
          f"services exited 0")
