"""Tests for the streamer (M3, plan §6.2) and inference engine (M4, plan §6.3).

Unit tests cover sampling, the schedule, pacing and framing. Integration tests (IT1)
run the engine and streamer as separate processes against a temporary database built
by init_db (db/nids.db is never touched):
  - 200 records end to end for each model: PREDICTION count = sent - malformed, no
    negative interval in v_latency (plan §6.3 "Done when");
  - broken lines increase malformed_count and do not stop the engine (FR5);
  - clock check: 1,000 records at 100 ms; every t_tx > 0 and median t_tx < 1 ms (§3.3.6);
  - every run records socket buffers and wal_autocheckpoint in run_conditions.
The two slow ones are marked `slow` (deselect with -m "not slow").
"""
import json
import socket
import sqlite3
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

from nids.prep.convert_xlsx import ROOT
from nids.storage.init_db import init_db
from nids.stream import netconf
from nids.stream import streamer as st

ART = ROOT / "artefacts"
TEST_RAW = ROOT / "data" / "processed" / "test_raw.parquet"
LABEL_MAP = ART / "label_map.json"
FEATURES = json.loads((ART / "feature_list.json").read_text())
HAVE_DATA = TEST_RAW.exists() and (ART / "rf_binary_seed42.joblib").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="test partition / artefacts missing")
COMPONENTS = ("t_tx_ms", "t_pre_ms", "t_inf_ms", "t_ser_ms", "L_ms")


# ---------------------------------------------------------------- streamer units
@pytest.fixture(scope="module")
def test_raw():
    return pd.read_parquet(TEST_RAW, columns=["row_index", "label_7", *FEATURES])


@needs_data
def test_sample_is_stratified_reproducible_and_shuffled(test_raw):
    a = st.draw_sample(test_raw, 2000, 42)
    b = st.draw_sample(test_raw, 2000, 42)
    assert a["row_index"].tolist() == b["row_index"].tolist()        # NFR4
    assert a["row_index"].is_unique and len(a) == 2000
    assert not a["row_index"].is_monotonic_increasing                  # seeded shuffle
    full = test_raw["label_7"].value_counts(normalize=True)
    samp = a["label_7"].value_counts(normalize=True)
    assert np.allclose(samp.reindex(full.index, fill_value=0), full, atol=0.002)
    c = st.draw_sample(test_raw, 2000, 43)
    assert c["row_index"].tolist() != a["row_index"].tolist()


def test_schedule_steady_and_burst():
    s = st.schedule_offsets_ns(5, 10, "steady")
    assert s.tolist() == [0, 10_000_000, 20_000_000, 30_000_000, 40_000_000]
    b = st.schedule_offsets_ns(2000, 100, "burst", 1000)
    dt = 100_000_000
    assert len(b) == 2000 and np.all(np.diff(b) >= 0)
    assert b[499] == 499 * dt and np.all(b[500:1500] == 500 * dt)      # 1,000 back-to-back
    assert b[1500] == 501 * dt and b[1999] == 1000 * dt                 # 500 @ dt after
    with pytest.raises(ValueError):
        st.schedule_offsets_ns(100, 10, "burst", 1000)


def test_wait_until_is_accurate():
    late = []
    for _ in range(20):
        due = time.perf_counter_ns() + 3_000_000
        st.wait_until(due)
        late.append(time.perf_counter_ns() - due)
    assert min(late) >= 0 and np.median(late) < 200_000                # < 0.2 ms


def test_encode_frames_and_stamps_t0_before_dumps():
    before = time.perf_counter_ns()
    raw = st.encode(3, 77, [1.0, 2.5])
    after = time.perf_counter_ns()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    msg = json.loads(raw)
    assert (msg["seq"], msg["row_index"], msg["features"]) == (3, 77, [1.0, 2.5])
    assert before <= msg["t0"] <= after


def test_effective_socket_buffers_probe():
    b = netconf.effective_socket_buffers()
    assert b["so_sndbuf_requested"] == 4 * 1024 * 1024
    assert b["so_sndbuf_effective"] > 0 and b["so_rcvbuf_effective"] > 0


def test_sndbuf_warning_threshold():
    assert netconf.sndbuf_warning(netconf.SEND_BUFFER) is None
    assert netconf.sndbuf_warning(2 * netconf.SEND_BUFFER) is None
    w = netconf.sndbuf_warning(425_984)
    assert w.startswith("WARNING") and "wmem_max" in w


def test_streamer_refuses_non_loopback():
    with pytest.raises(ValueError):
        st.connect("0.0.0.0", 9100, timeout_s=0.1)


# ---------------------------------------------------------------- integration helpers
@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("pipe") / "nids.db"
    init_db(path, TEST_RAW, LABEL_MAP)
    return path


def start_engine(db, model, task="binary", dt_ms=100, mode="steady", n=200, scenario="IT1"):
    p = subprocess.Popen([sys.executable, "-m", "nids.engine.engine", "--model", model,
                          "--task", task, "--seed", "42", "--dt-ms", str(dt_ms),
                          "--mode", mode, "--scenario", scenario, "--n", str(n),
                          "--port", "0", "--db", str(db)],
                         cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in p.stdout:
        if line.startswith("{"):
            ev = json.loads(line)
            if ev.get("event") == "ready":
                return p, ev
    raise RuntimeError(f"engine did not start:\n{p.stderr.read()[-3000:]}")


def finish_engine(p, timeout=120):
    out, err = p.communicate(timeout=timeout)
    assert p.returncode == 0, err[-3000:]
    return json.loads([ln for ln in out.splitlines() if ln.startswith("{")][-1])


def run_pipeline(db, model, task="binary", n=200, dt_ms=100, mode="steady"):
    eng, ready = start_engine(db, model, task, dt_ms, mode, n)
    s = subprocess.run([sys.executable, "-m", "nids.stream.streamer", "--n", str(n),
                        "--dt-ms", str(dt_ms), "--mode", mode, "--port", str(ready["port"])],
                       cwd=ROOT, capture_output=True, text=True, timeout=n * dt_ms / 1000 + 300)
    assert s.returncode == 0, s.stderr[-3000:]
    sent = json.loads(s.stdout.strip().splitlines()[-1])
    low = sent["sndbuf_granted"] < netconf.SEND_BUFFER
    assert sent["sndbuf_warning"] == low and ("WARNING: send buffer" in s.stderr) == low
    return ready["run_id"], sent, finish_engine(eng)


def ro(db):
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def run_conditions(db, run_id):
    con = ro(db)
    try:
        rc = con.execute("SELECT e.run_conditions FROM model_run m JOIN test_environment e "
                         "ON e.env_id = m.env_id WHERE m.run_id=?", (run_id,)).fetchone()[0]
    finally:
        con.close()
    return json.loads(rc)


def check_run_conditions(db, run_id):
    rc = run_conditions(db, run_id)
    assert rc["sqlite_wal_autocheckpoint"] == 1000                    # default, kept
    assert rc["so_sndbuf_requested"] == netconf.SEND_BUFFER
    assert rc["so_sndbuf_effective"] > 0 and rc["so_rcvbuf_effective"] > 0
    assert "net_core_wmem_max" in rc and "net_core_rmem_max" in rc
    return rc


def run_stats(db, run_id):
    con = ro(db)
    try:
        n_pred = con.execute("SELECT COUNT(*) FROM prediction WHERE run_id=?",
                             (run_id,)).fetchone()[0]
        mr = con.execute("SELECT model_name, task, random_seed, emission_interval_ms, "
                         "load_condition, scenario, n_records, malformed_count, env_id, "
                         "hyperparameters, ended_at FROM model_run WHERE run_id=?",
                         (run_id,)).fetchone()
        lat = pd.read_sql_query("SELECT * FROM v_latency WHERE run_id=? ORDER BY seq_no",
                                con, params=(run_id,))
        null_t4 = con.execute("SELECT COUNT(*) FROM prediction WHERE run_id=? AND "
                              "t4_store_ns IS NULL", (run_id,)).fetchone()[0]
    finally:
        con.close()
    return n_pred, mr, lat, null_t4


# ---------------------------------------------------------------- IT1: 200 records each
@needs_data
@pytest.mark.slow
@pytest.mark.parametrize("model", ["iforest", "rf", "lstm"])
def test_end_to_end_200(db, model):
    run_id, sent, done = run_pipeline(db, model)
    n_pred, mr, lat, null_t4 = run_stats(db, run_id)
    assert sent["n_sent"] == 200 and done["eos"]
    assert n_pred == sent["n_sent"] - mr[7] == done["processed"]
    assert mr[:8] == (model, "binary", 42, 100.0, "steady", "IT1", 200, 0)
    assert mr[8] is not None and json.loads(mr[9])["config"] and mr[10]
    assert null_t4 == 0 and len(lat) == 200                         # flush wrote last t4
    check_run_conditions(db, run_id)
    assert (lat[list(COMPONENTS)] >= 0).all().all()                  # no negative interval
    med = lat[lat.seq_no >= 100][list(COMPONENTS)].median()          # Table 3.13 warm-up
    print(f"\n{model}: sent={sent['n_sent']} predictions={n_pred} malformed={mr[7]} "
          f"pacing_late_median={sent['lateness_ms']['median']:.3f}ms  medians(ms, seq>=100): "
          + " ".join(f"{c}={med[c]:.3f}" for c in COMPONENTS))


# ---------------------------------------------------------------- FR5: malformed lines
@needs_data
def test_malformed_lines_counted_engine_continues(db, test_raw):
    eng, ready = start_engine(db, "iforest", n=43, scenario="FR5")
    rows = test_raw.head(40)
    good = [st.encode(i, int(r), [float(v) for v in vals]) for i, (r, vals) in
            enumerate(zip(rows["row_index"], rows[FEATURES].to_numpy()))]
    bad = [b"this is not json\n",
           json.dumps({"seq": 99, "row_index": int(rows.row_index.iloc[0]),
                       "features": [1.0, 2.0], "t0": 1}).encode() + b"\n",   # wrong length
           json.dumps({"seq": 98, "features": [0.0] * len(FEATURES),
                       "t0": 1}).encode() + b"\n",                           # missing key
           json.dumps({"seq": 97, "row_index": 10**9, "features": [0.0] * len(FEATURES),
                       "t0": 1}).encode() + b"\n",                           # unknown flow
           b"[1, 2, 3]\n"]                                                   # not an object
    with socket.create_connection(("127.0.0.1", ready["port"])) as s:
        s.sendall(b"".join(good[:20]) + b"".join(bad) + b"".join(good[20:]) + st.EOS)
        s.shutdown(socket.SHUT_WR)
        s.recv(1)
    done = finish_engine(eng)
    n_pred, mr, lat, _ = run_stats(db, ready["run_id"])
    assert done == {"event": "done", "run_id": ready["run_id"], "processed": 40,
                    "malformed": 5, "eos": True}
    assert n_pred == 40 and mr[7] == 5 and mr[10]
    rc = check_run_conditions(db, ready["run_id"])
    print(f"\nrun_conditions: " + json.dumps({k: rc[k] for k in sorted(rc)
                                             if k.startswith(("so_", "net_", "sqlite_"))}))
    print(f"\nFR5: sent 45 lines (5 broken) -> predictions={n_pred} malformed_count={mr[7]}")


# ---------------------------------------------------------------- clock check (§3.3.6)
@needs_data
@pytest.mark.slow
def test_clock_check_1000_at_100ms(db):
    run_id, sent, done = run_pipeline(db, "iforest", n=1000, dt_ms=100)
    n_pred, mr, lat, _ = run_stats(db, run_id)
    assert n_pred == 1000 and mr[7] == 0
    tx = lat["t_tx_ms"]
    print(f"\nclock check (1,000 @ 100 ms, iforest): t_tx min={tx.min():.4f} "
          f"median={tx.median():.4f} p95={tx.quantile(.95):.4f} p99={tx.quantile(.99):.4f} "
          f"max={tx.max():.4f} ms; all > 0: {bool((tx > 0).all())}; "
          f"pacing late median={sent['lateness_ms']['median']:.3f} ms")
    assert (tx > 0).all() and tx.median() < 1.0
