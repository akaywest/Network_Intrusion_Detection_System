"""nids.evaluation.system_metrics (plan §8.3-8.4): backlog, Eq. 3.7 throughput, S4
recovery, sustainable-rate rule, run selection and the offline agreement check, on
small synthetic runs (no database needed)."""
import numpy as np
import pandas as pd
import pytest

from nids.evaluation import system_metrics as sm

MS = 1_000_000


def make_run(n=300, dt_ms=10.0, service_ms=2.0, mode="steady", block=0):
    """A FIFO single-server queue: emission schedule, fixed service time."""
    if mode == "steady":
        t0 = (np.arange(n) * dt_ms * MS).astype(np.int64)
    else:
        lead = (n - block) // 2
        t0 = np.concatenate([np.arange(lead) * dt_ms, np.full(block, lead * dt_ms),
                             lead * dt_ms + np.arange(1, n - lead - block + 1) * dt_ms])
        t0 = (t0 * MS).astype(np.int64)
    t4 = np.empty(n, dtype=np.int64)
    free = 0
    for i in range(n):
        start = max(t0[i], free)
        t4[i] = free = start + int(service_ms * MS)
    L = (t4 - t0) / MS
    rec = pd.DataFrame({"seq_no": np.arange(n), "row_index": np.arange(n), "is_warmup": 0,
                        "predicted_label": "Normal", "t0_emit_ns": t0, "t4_store_ns": t4,
                        "t_tx_ms": L - service_ms, "t_pre_ms": 0.0,
                        "t_inf_ms": service_ms, "t_ser_ms": 0.0, "L_ms": L})
    run = pd.Series({"run_id": 1, "dt_ms": dt_ms, "mode": mode, "n_records": n,
                     "refreshes": 10, "dashboard_open": 1})
    return rec, run


def test_backlog_counts_unfinished_earlier_records():
    t0 = np.array([0, 10, 20, 30]) * MS
    t4 = np.array([25, 35, 45, 55]) * MS          # each takes 25 ms, FIFO
    assert sm.backlog(t0, t4).tolist() == [0, 1, 2, 2]


def test_stable_run_is_sustained_and_eq_3_7_throughput():
    rec, run = make_run(n=300, dt_ms=10.0, service_ms=2.0)
    m, bl = sm.run_metrics(rec, run, warmup=100, block=0, offline=None)
    assert m["n_post_warmup"] == 200 and m["backlog_max"] == 0
    assert m["L_p95_ms"] == pytest.approx(2.0) and m["sustained"]
    # 200 records / (last t4 - first t0 after warm-up) = 200 / (199*10 + 2) ms
    assert m["throughput_rps"] == pytest.approx(200 / 1.992)


def test_overloaded_run_grows_backlog_and_fails():
    rec, run = make_run(n=300, dt_ms=10.0, service_ms=15.0)
    m, _ = sm.run_metrics(rec, run, warmup=100, block=0, offline=None)
    assert m["backlog_growing"] and not m["sustained"]
    # Eq. 3.7 runs from the first post-warm-up emission (t0 = 1.0 s) to the last
    # commit (300 x 15 ms = 4.5 s), so it includes the backlog the warm-up records
    # left behind and sits below the 66.7/s service rate: 200 / 3.5 s
    assert m["throughput_rps"] == pytest.approx(200 / 3.5)


def test_backlog_tolerance_ignores_sub_record_noise():
    rec, run = make_run(n=300, dt_ms=10.0, service_ms=2.0)
    late = rec.index[-5:]                           # one stall near the end
    rec.loc[late[0], "t4_store_ns"] += 12 * MS
    rec.loc[late[0], "L_ms"] += 12
    strict, _ = sm.run_metrics(rec, run, 100, 0, None, backlog_tol=0.0)
    tolerant, _ = sm.run_metrics(rec, run, 100, 0, None, backlog_tol=1.0)
    assert strict["backlog_growing"] and not tolerant["backlog_growing"]


def test_burst_recovery_and_not_recovered():
    rec, run = make_run(n=400, dt_ms=100.0, service_ms=20.0, mode="burst", block=200)
    m, _ = sm.run_metrics(rec, run, warmup=10, block=200, offline=None)
    # 200 x 20 ms backlog = 4 s, drains at 80 ms per 100 ms tail record
    assert m["peak_L_ms"] == pytest.approx(200 * 20.0)
    assert m["recovered"] and 3.0 < m["recovery_s"] < 6.0
    rec, run = make_run(n=400, dt_ms=100.0, service_ms=90.0, mode="burst", block=200)
    m, _ = sm.run_metrics(rec, run, warmup=10, block=200, offline=None)
    assert not m["recovered"] and np.isnan(m["recovery_s"])


def test_agreement_with_offline_labels():
    rec, run = make_run(n=50)
    off = pd.Series("Normal", index=np.arange(50))
    off.iloc[[3, 7]] = "Malicious"
    m, _ = sm.run_metrics(rec, run, 10, 0, off)
    assert m["n_disagree"] == 2 and m["agreement"] == pytest.approx(48 / 50)


def test_select_runs_rules():
    base = dict(scenario="S3", model="lstm", task="binary", seed=44, dt_ms=1.0,
                mode="steady", n_records=5000, n_pred=5000, dashboard_open=1, refreshes=9)
    runs = pd.DataFrame([{**base, "run_id": 65, "refreshes": 0},        # no refresh
                         {**base, "run_id": 66},                        # its repeat
                         {**base, "run_id": 70, "seed": 45, "n_pred": 4999},
                         {**base, "run_id": 71, "seed": 46, "dashboard_open": None},
                         {**base, "run_id": 80, "seed": 42},
                         {**base, "run_id": 81, "seed": 42}])           # duplicate
    out = sm.select_runs(runs).set_index("run_id")
    assert out.included.to_dict() == {65: False, 66: True, 70: False, 71: False,
                                      80: False, 81: True}
    assert "no refresh" in out.loc[65, "exclude_reason"]
    assert "duplicate" in out.loc[80, "exclude_reason"]


def test_sustainable_rate_needs_all_seeds_and_every_larger_dt():
    rows = []
    for seed in range(42, 47):
        for dt in sm.RATE_STEPS:
            ok = dt >= 20 or (dt == 5 and seed < 46)      # 10 ms fails: stop there
            sc = "S1" if dt == 100 else "S2" if dt == 10 else "S3"
            rows.append(dict(included=True, dashboard_open=1, mode="steady", scenario=sc,
                             model="rf", seed=seed, dt_ms=dt, sustained=ok,
                             L_p95_ms=1.0, backlog_growing=not ok, throughput_rps=1.0))
    sr = sm.sustainable_rate(pd.DataFrame(rows))
    rf = sr[sr.model == "rf"]
    assert rf.sustainable_dt_ms.iloc[0] == 20.0
    assert rf.max_sustainable_rps.iloc[0] == pytest.approx(50.0)


def test_nfr2_judged_on_sustained_rate_not_capped_throughput():
    """S2 offers exactly 100 records/s, so Eq. 3.7 throughput ~100 must not pass NFR2:
    the verdict follows p95(L) <= dt and a non-growing backlog (plan §8.3), and the
    reported value is the S2 p95."""
    rows = []
    for seed in range(42, 47):
        for dt in sm.RATE_STEPS:
            sc = "S1" if dt == 100 else "S2" if dt == 10 else "S3"
            ok = dt >= 50                                # keeps up only down to 50 ms
            rows.append(dict(included=True, dashboard_open=1, mode="steady", scenario=sc,
                             model="lstm", seed=seed, dt_ms=dt, sustained=ok,
                             L_p95_ms=5.0 if ok else 250.0, backlog_growing=not ok,
                             backlog_growth=0.0 if ok else 30.0,
                             throughput_rps=min(1000 / dt, 99.9)))
    nfr2 = sm.nfr_verdicts(pd.DataFrame(rows), backlog_tol=1.0).query(
        "nfr == 'NFR2' and model == 'lstm'")
    r = nfr2.iloc[0]
    assert r.throughput_mean_rps == pytest.approx(99.9)   # looks like 100/s ...
    assert r.verdict == "FAIL" and r.seeds_pass == 0      # ... but is not sustained
    assert r.value_mean == pytest.approx(250.0) and r.unit.startswith("ms")
    assert r.backlog_growing_runs == 5
    assert r.max_sustainable_rps == pytest.approx(20.0)

