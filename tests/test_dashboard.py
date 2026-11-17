"""Tests for the dashboard (M6; plan §7.2, IT2 of plan §9; FR2, FR8, FR12, FR13).

Unit: KPI arithmetic (warm-up exclusion, Eq. 3.7 throughput), detection KPIs vs the
dataset label, the p95/p99-vs-Δt rule, traffic buckets, protocol mix, Δt steps. Integration (tests/conftest.py `services`; Streamlit AppTest, no browser):
  - the app renders without services and without runs;
  - against a finished run, both KPI rows match the database (warm-up excluded);
  - Start and Stop in the sidebar drive a run through the control channel; while it is
    live the settings lock (Δt stays live, FR2) and the dashboard delay is appended to
    the CSV (a temp file, never results/);
  - the connection is read-only.
"""
import csv
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from nids.dashboard import queries as q
from nids.prep.convert_xlsx import ROOT
from tests.conftest import HAVE_DATA

APP = str(ROOT / "src" / "nids" / "dashboard" / "app.py")
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="test partition / artefacts missing")
NS = 1_000_000_000


# ---------------------------------------------------------------- units
def fake_records(n=300, dt_ms=10.0, L_ms=2.0):
    t0 = (np.arange(n) * dt_ms * 1e6).astype(np.int64) + 5 * NS
    lab = np.where(np.arange(n) % 3 == 0, "Malicious", "Normal")
    df = pd.DataFrame({"seq_no": np.arange(n), "is_warmup": 0, "predicted_label": lab,
                       "t0_emit_ns": t0, "t4_store_ns": t0 + int(L_ms * 1e6),
                       "t_tx_ms": 0.1, "t_pre_ms": 0.2, "t_inf_ms": 1.0, "t_ser_ms": 0.7,
                       "L_ms": L_ms, "protocol": np.where(np.arange(n) % 4, "TCP", "UDP"),
                       # ground truth: every 3rd row malicious, but rows 6k+3 labelled
                       # Normal (false positives) and rows 6k+1 malicious (missed)
                       "true_label": lab.copy()})
    i = np.arange(n)
    df.loc[i % 6 == 3, "true_label"] = "Normal"
    df.loc[i % 6 == 1, "true_label"] = "DoS/DDoS"
    df.loc[:99, "L_ms"] = 500.0                          # slow warm-up must not count
    df["malicious"] = df["predicted_label"] != "Normal"
    return df


def test_kpis_exclude_warmup_and_use_eq_3_7():
    df = fake_records()
    k = q.kpis(df, alerts=7, warmup=100)
    assert k["records"] == 300 and k["alerts"] == 7 and k["n_post_warmup"] == 200
    assert k["p50_ms"] == k["p95_ms"] == k["p99_ms"] == pytest.approx(2.0)
    span = (df.t4_store_ns.iloc[-1] - df.t0_emit_ns.iloc[100]) / NS
    assert k["throughput_rps"] == pytest.approx(200 / span)
    early = q.kpis(df.head(50), 0, 100)
    assert early["records"] == 50 and early["p95_ms"] is None
    assert early["throughput_rps"] is None
    assert q.component_means(df, 100) == pytest.approx(
        {"t_tx_ms": 0.1, "t_pre_ms": 0.2, "t_inf_ms": 1.0, "t_ser_ms": 0.7})


def test_detection_kpis_exclude_warmup():
    df = fake_records(n=300)
    post = df[df.seq_no >= 100]
    d = q.detection_kpis(df, 100)
    i = post.seq_no.to_numpy()
    tp = int(((i % 3 == 0) & (i % 6 != 3)).sum())
    fp = int((i % 6 == 3).sum())
    fn = int((i % 6 == 1).sum())
    assert (d["n"], d["tp"], d["fp"], d["fn"], d["false_alarms"]) == (200, tp, fp, fn, fp)
    assert d["precision"] == pytest.approx(tp / (tp + fp))
    assert d["recall"] == pytest.approx(tp / (tp + fn))
    early = q.detection_kpis(df.head(50), 100)
    assert early["n"] == 0 and early["precision"] is None and early["recall"] is None
    # a seven-class prediction of the wrong attack class is still a hit (binary view)
    multi = df.assign(predicted_label=np.where(df.predicted_label == "Normal", "Normal",
                                               "Botnet"))
    assert q.detection_kpis(multi, 100)["tp"] == tp


def test_within_target():
    assert q.within_target(100.0, 100) is True          # NFR1: p95 <= Δt passes
    assert q.within_target(100.1, 100) is False
    assert q.within_target(None, 100) is None


def test_traffic_last_60_seconds_filled():
    df = fake_records(n=8000, dt_ms=10)                   # 80 s of emission
    tr = q.traffic_per_second(df, 60)
    assert len(tr) == 60 and tr["second"].tolist() == list(range(20, 80))
    assert (tr["benign"] + tr["malicious"] == 100).all()  # 100 flows/s
    gap = df[(df.t0_emit_ns - df.t0_emit_ns.min()) // NS != 70]   # a silent second
    assert q.traffic_per_second(gap, 60).set_index("second").loc[70].sum() == 0
    assert q.traffic_per_second(df.iloc[:0]).empty


def test_protocol_mix_window_and_order():
    pm = q.protocol_mix(fake_records(n=8000), 60)
    assert pm["protocol"].tolist() == ["TCP", "UDP", "Other"]   # no ICMP in the data
    assert pm["flows"].sum() == 6001 and pm.set_index("protocol").loc["Other", "flows"] == 0


def test_dt_steps():
    df = fake_records(n=300, dt_ms=10)
    run = {"emission_interval_ms": 10.0, "rate_changes": None}
    assert q.dt_steps(run, df, None) == [(0.0, 10.0)]
    steps = q.dt_steps(run, df, [{"from_seq": 150, "dt_ms": 5}])
    assert steps == [(0.0, 10.0), (pytest.approx(1.5), 5.0)]


# ---------------------------------------------------------------- integration
@pytest.fixture
def app_env(monkeypatch, tmp_path):
    def set_env(svc=None):
        delay = tmp_path / "dashboard_delay.csv"
        monkeypatch.setenv("NIDS_DASHBOARD_DELAY_CSV", str(delay))
        if svc is None:
            monkeypatch.setenv("NIDS_DB", str(tmp_path / "missing.db"))
            monkeypatch.setenv("NIDS_ENGINE_CTL_PORT", "1")
            monkeypatch.setenv("NIDS_STREAMER_CTL_PORT", "2")
        else:
            monkeypatch.setenv("NIDS_DB", str(svc.db))
            monkeypatch.setenv("NIDS_ENGINE_CTL_PORT", str(svc.client.engine_port))
            monkeypatch.setenv("NIDS_STREAMER_CTL_PORT", str(svc.client.streamer_port))
        return delay
    return set_env


def metrics(at) -> dict:
    return {m.label: m.value for m in at.metric}


def test_renders_without_services_or_database(app_env):
    app_env(None)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("not reachable" in e.value for e in at.sidebar.error)
    assert any("read-only" in e.value for e in at.error)


@needs_data
def test_kpis_match_database_for_finished_run(services, app_env):
    app_env(services)
    out = services.client.run("iforest", dt_ms=5, scenario="dash", n=300, timeout=120)
    run_id = out["run_id"]
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    m = metrics(at)
    con = q.open_ro(services.db)
    try:
        df = q.run_records(con, run_id)
        k = q.kpis(df, q.alert_count(con, run_id), 100)
        det = q.detection_kpis(df, 100)
        L = pd.read_sql_query("SELECT L_ms FROM v_latency WHERE run_id=? AND seq_no>=100",
                              con, params=(run_id,))["L_ms"]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            con.execute("DELETE FROM alert")                 # read-only connection
    finally:
        con.close()
    assert m["Records processed"] == "300" and m["Alerts"] == f"{k['alerts']:,}"
    p95 = float(np.percentile(L, 95))
    assert k["p95_ms"] == pytest.approx(p95)
    assert m["L p95"] == (f"{p95:.1f} ms" if p95 >= 10 else f"{p95:.2f} ms")
    assert m["Precision"] == ("—" if det["precision"] is None else f"{det['precision']:.3f}")
    assert m["Recall"] == ("—" if det["recall"] is None else f"{det['recall']:.3f}")
    assert m["False alarms"] == f"{det['false_alarms']:,}"
    assert f"Run {run_id}" in " ".join(c.value for c in at.caption)
    assert {s.value for s in at.subheader} >= {"Live traffic (last 60 s)",
                                               "Protocol mix (last 60 s)", "Latest alerts",
                                               "Latency components",
                                               "Latency L per record"}
    at.toggle(key="latency_log").set_value(True).run()      # log-scale latency chart
    assert not at.exception
    print(f"\ndashboard KPIs for run {run_id}: {m}")


@needs_data
def test_start_stop_from_sidebar_and_delay_log(services, app_env):
    delay_csv = app_env(services)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox(key="model").set_value("iforest")
    at.sidebar.slider(key="dt_ms").set_value(10)
    at.run()                                   # slider moved while idle: no set_rate
    at.button(key="start").click().run()
    assert not at.exception
    assert any("started" in s.value for s in at.sidebar.success), \
        [e.value for e in at.sidebar.error]
    c = services.client
    run_id = c.engine("status")["run"]["run_id"]
    deadline = time.monotonic() + 60
    while c.engine("status")["run"]["processed"] < 150 and time.monotonic() < deadline:
        time.sleep(0.1)
    at.run()                                   # live refresh -> delay logged
    assert metrics(at)["Records processed"] != "0"
    # settings lock while the run is active; Δt stays live in a steady run (FR2)
    assert at.sidebar.selectbox(key="model").disabled and at.button(key="start").disabled
    assert not at.sidebar.slider(key="dt_ms").disabled and not at.button(key="stop").disabled
    at.button(key="stop").click().run()
    eng = c.wait_done(run_id, timeout=30)
    at.run()                                   # run ended -> settings unlock
    assert not at.sidebar.selectbox(key="model").disabled
    assert not at.button(key="start").disabled and at.button(key="stop").disabled
    assert eng["end_reason"] == "stopped"
    con = q.open_ro(services.db)
    try:
        info = q.run_info(con, run_id)
    finally:
        con.close()
    assert (info["model_name"], info["emission_interval_ms"], info["scenario"],
            info["n_records"], info["rate_changes"]) == ("iforest", 10.0, "dashboard",
                                                        5000, None)
    rows = list(csv.DictReader(Path(delay_csv).open()))
    assert rows and rows[-1]["run_id"] == str(run_id) and float(rows[-1]["delay_ms"]) >= 0
    print(f"\ndashboard Start/Stop: run {run_id} stopped after {eng['processed']} records; "
          f"dashboard delay {float(rows[-1]['delay_ms']):.1f} ms")


@needs_data
def test_sidebar_mirrors_run_started_by_experiment_script(services, app_env):
    """A run started outside the dashboard (run_matrix's prepare/start, scenario != the
    dashboard's) is shown in the locked sidebar with its real settings and a note."""
    app_env(services)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox(key="model").set_value("lstm")     # the user's own choices
    at.sidebar.selectbox(key="seed").set_value(46)
    at.sidebar.slider(key="dt_ms").set_value(250)
    at.run()
    c = services.client
    # own label: test_run_matrix counts every rm-test-S1 run in the shared session DB
    run_id = c.prepare("rf", "multi", 44, 20.0, "steady", "dash-test-S1", 300,
                       dashboard_open=True)
    c.start(run_id, 20.0, "steady", 300)
    try:
        at.run()
        assert not at.exception
        sb = at.sidebar
        assert (sb.selectbox(key="model").value, sb.radio(key="task").value,
                sb.selectbox(key="seed").value, sb.slider(key="dt_ms").value,
                sb.radio(key="mode").value) == ("rf", "multi", 44, 20, "steady")
        assert all(w.disabled for w in (sb.selectbox(key="model"), sb.radio(key="task"),
                                        sb.selectbox(key="seed"), sb.slider(key="dt_ms"),
                                        sb.radio(key="mode"), at.button(key="start")))
        notes = [i.value for i in sb.info]
        assert any("started by experiment script" in n and f"Run {run_id}" in n
                   and "dash-test-S1" in n for n in notes), notes
        assert not [w.value for w in sb.warning]           # no Session State warning
    finally:
        c.stop()
        eng = c.wait_done(run_id, timeout=60)
    at.run()                                               # run ended: choices restored
    sb = at.sidebar
    assert (sb.selectbox(key="model").value, sb.selectbox(key="seed").value,
            sb.slider(key="dt_ms").value) == ("lstm", 46, 250)
    assert not sb.selectbox(key="model").disabled and not sb.info
    assert eng["rate_changes"] is None                     # the run was never re-paced
