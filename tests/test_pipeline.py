"""Tests for the streamer (M3, plan §6.2) and inference engine (M4, plan §6.3).

Unit tests cover sampling, the schedule, pacing and framing. Integration tests drive
the engine and streamer services (tests/conftest.py `services`: separate processes,
temporary database built by init_db; db/nids.db is never touched) through the control
channel (plan §7.1):
  - 200 records end to end for each model (IT1): see tests/test_control.py;
  - broken lines increase malformed_count and do not stop the engine (FR5);
  - clock check: 1,000 records at 100 ms; every t_tx > 0 and median t_tx < 1 ms (§3.3.6);
  - every run records socket buffers and wal_autocheckpoint in run_conditions.
The two slow ones are marked `slow` (deselect with -m "not slow").
"""
import json
import socket
import sqlite3
import time

import numpy as np
import pandas as pd
import pytest

from nids.prep.convert_xlsx import ROOT
from nids.stream import netconf
from nids.stream import streamer as st

from tests.conftest import ART, HAVE_DATA, TEST_RAW

FEATURES = json.loads((ART / "feature_list.json").read_text())
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
def run_pipeline(svc, model, task="binary", n=200, dt_ms=100, mode="steady",
                 scenario="IT1"):
    """prepare -> start -> eos through the control channel; returns (run_id, sent, done)."""
    c = svc.client
    run_id = c.prepare(model, task, 42, dt_ms, mode, scenario, n)
    started = c.start(run_id, dt_ms, mode, n, sample_seed=42)
    low = started["sndbuf_granted"] < netconf.SEND_BUFFER
    assert started["sndbuf_warning"] == low
    done = c.wait_done(run_id, timeout=n * dt_ms / 1000 + 300)
    sent = c.streamer("status")["last_run"]
    assert sent["run_id"] == run_id
    return run_id, sent, done


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
                         "hyperparameters, ended_at, end_reason, rate_changes "
                         "FROM model_run WHERE run_id=?", (run_id,)).fetchone()
        lat = pd.read_sql_query("SELECT * FROM v_latency WHERE run_id=? ORDER BY seq_no",
                                con, params=(run_id,))
        null_t4 = con.execute("SELECT COUNT(*) FROM prediction WHERE run_id=? AND "
                              "t4_store_ns IS NULL", (run_id,)).fetchone()[0]
    finally:
        con.close()
    return n_pred, mr, lat, null_t4


# ---------------------------------------------------------------- FR5: malformed lines
@needs_data
def test_malformed_lines_counted_engine_continues(services, test_raw):
    run_id = services.client.prepare("iforest", "binary", 42, 100, "steady", "FR5", 45)
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
    with socket.create_connection(("127.0.0.1", services.data_port)) as s:
        s.sendall(b"".join(good[:20]) + b"".join(bad) + b"".join(good[20:]) + st.EOS)
        s.shutdown(socket.SHUT_WR)
        s.recv(1)
    done = services.client.wait_done(run_id, timeout=60)
    n_pred, mr, lat, _ = run_stats(services.db, run_id)
    assert done == {"run_id": run_id, "processed": 40, "malformed": 5, "eos": True,
                    "end_reason": "complete", "rate_changes": None}
    assert n_pred == 40 and mr[7] == 5 and mr[10] and mr[11] == "complete"
    assert services.client.engine("status")["state"] == "idle"        # still serving
    rc = check_run_conditions(services.db, run_id)
    print("\nrun_conditions: " + json.dumps({k: rc[k] for k in sorted(rc)
                                             if k.startswith(("so_", "net_", "sqlite_"))}))
    print(f"\nFR5: sent 45 lines (5 broken) -> predictions={n_pred} malformed_count={mr[7]}")


# ---------------------------------------------------------------- clock check (§3.3.6)
@needs_data
@pytest.mark.slow
def test_clock_check_1000_at_100ms(services):
    run_id, sent, done = run_pipeline(services, "iforest", n=1000, dt_ms=100,
                                      scenario="clock")
    n_pred, mr, lat, _ = run_stats(services.db, run_id)
    assert n_pred == 1000 and mr[7] == 0
    tx = lat["t_tx_ms"]
    print(f"\nclock check (1,000 @ 100 ms, iforest): t_tx min={tx.min():.4f} "
          f"median={tx.median():.4f} p95={tx.quantile(.95):.4f} p99={tx.quantile(.99):.4f} "
          f"max={tx.max():.4f} ms; all > 0: {bool((tx > 0).all())}; "
          f"pacing late median={sent['lateness_ms']['median']:.3f} ms")
    assert (tx > 0).all() and tx.median() < 1.0
