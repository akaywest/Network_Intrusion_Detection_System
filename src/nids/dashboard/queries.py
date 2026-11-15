"""Read-only queries and KPI arithmetic for the dashboard (M6; plan §7.2, Chapter 3
§3.3.7, Fig. 3.12, Table 3.8; FR12, FR13).

Rules (plan §7.2):
  - the database is opened READ-ONLY (sqlite URI mode=ro), so the dashboard can never
    write to the log or hold a write lock against the engine;
  - every query keeps only rows with t4_store_ns IS NOT NULL (the newest row gets its
    t4 one transaction later, plan §5.3);
  - latency KPIs and throughput exclude the warm-up: the first `warmup` records of the
    run (Table 3.13, config stream.warmup_latency = 100) and any is_warmup row;
  - "live" windows (last 60 s) are measured on the run's own emission clock
    (t0 relative to the newest t0), so a finished run still shows its last minute.
All SQL is literal with ? parameters (NFR6).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

NS = 1_000_000_000
COMPONENTS = ("t_tx_ms", "t_pre_ms", "t_inf_ms", "t_ser_ms")
PROTOCOL_ORDER = ("TCP", "UDP", "ICMP", "Other")


def open_ro(db_path: str | Path) -> sqlite3.Connection:
    """Read-only connection (plan §7.2). Raises sqlite3.OperationalError if absent."""
    return sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True, timeout=2.0)


def latest_run_id(con: sqlite3.Connection) -> int | None:
    row = con.execute("SELECT MAX(run_id) FROM model_run").fetchone()
    return row[0] if row else None


def run_info(con: sqlite3.Connection, run_id: int) -> dict | None:
    cur = con.execute(
        "SELECT run_id, model_name, task, random_seed, emission_interval_ms, "
        "load_condition, scenario, n_records, malformed_count, started_at, ended_at, "
        "rate_changes, end_reason FROM model_run WHERE run_id=?", (run_id,))
    row = cur.fetchone()
    if row is None:
        return None
    d = dict(zip([c[0] for c in cur.description], row))
    d["rate_changes"] = json.loads(d["rate_changes"]) if d["rate_changes"] else None
    return d


def run_records(con: sqlite3.Connection, run_id: int) -> pd.DataFrame:
    """One row per stored prediction of the run (t4 not null), with latency components,
    the binary decision and the flow's protocol."""
    df = pd.read_sql_query(
        "SELECT p.seq_no, p.is_warmup, p.predicted_label, p.t0_emit_ns, p.t4_store_ns, "
        "v.t_tx_ms, v.t_pre_ms, v.t_inf_ms, v.t_ser_ms, v.L_ms, f.protocol "
        "FROM prediction p JOIN v_latency v ON v.prediction_id = p.prediction_id "
        "JOIN flow_record f ON f.flow_id = p.flow_id "
        "WHERE p.run_id=? AND p.t4_store_ns IS NOT NULL ORDER BY p.seq_no",
        con, params=(run_id,))
    # binary and seven-class predictions alike: anything but Normal is malicious (FR10)
    df["malicious"] = (df["predicted_label"] != "Normal") & (df["is_warmup"] == 0)
    return df


def alert_count(con: sqlite3.Connection, run_id: int) -> int:
    return con.execute(
        "SELECT COUNT(*) FROM alert a JOIN prediction p ON p.prediction_id = a.prediction_id "
        "WHERE p.run_id=? AND p.t4_store_ns IS NOT NULL", (run_id,)).fetchone()[0]


def recent_alerts(con: sqlite3.Connection, run_id: int, limit: int = 20) -> pd.DataFrame:
    """Last `limit` alerts: ALERT ⋈ PREDICTION ⋈ FLOW_RECORD ⋈ MODEL_RUN (plan §7.2)."""
    return pd.read_sql_query(
        "SELECT a.raised_at, p.seq_no, p.predicted_label, f.true_label, p.anomaly_score, "
        "p.confidence, a.severity, m.model_name, m.task "
        "FROM alert a JOIN prediction p ON p.prediction_id = a.prediction_id "
        "JOIN flow_record f ON f.flow_id = p.flow_id "
        "JOIN model_run m ON m.run_id = p.run_id "
        "WHERE p.run_id=? AND p.t4_store_ns IS NOT NULL "
        "ORDER BY a.alert_id DESC LIMIT ?", con, params=(run_id, limit))


def max_t4(con: sqlite3.Connection, run_id: int) -> int | None:
    return con.execute("SELECT MAX(t4_store_ns) FROM prediction WHERE run_id=?",
                       (run_id,)).fetchone()[0]


# ---------------------------------------------------------------- arithmetic
def post_warmup(df: pd.DataFrame, warmup: int) -> pd.DataFrame:
    return df[(df["seq_no"] >= warmup) & (df["is_warmup"] == 0)]


def kpis(df: pd.DataFrame, alerts: int, warmup: int) -> dict:
    """KPI row (plan §7.2): records, alerts, p50/p95/p99 of L and throughput (Eq. 3.7),
    the last three over post-warm-up records only."""
    post = post_warmup(df, warmup)
    out = {"records": int(len(df)), "alerts": int(alerts), "n_post_warmup": int(len(post)),
           "p50_ms": None, "p95_ms": None, "p99_ms": None, "throughput_rps": None}
    if len(post):
        p50, p95, p99 = np.percentile(post["L_ms"].to_numpy(), [50, 95, 99])
        out.update(p50_ms=float(p50), p95_ms=float(p95), p99_ms=float(p99))
        span_s = (int(post["t4_store_ns"].max()) - int(post["t0_emit_ns"].min())) / NS
        if span_s > 0:
            out["throughput_rps"] = len(post) / span_s     # Eq. 3.7
    return out


def traffic_per_second(df: pd.DataFrame, window_s: int = 60) -> pd.DataFrame:
    """Benign / malicious flows per second over the last window_s seconds, bucketed by
    (t0 - first t0) in whole seconds (plan §7.2). Empty seconds are filled with 0."""
    if df.empty:
        return pd.DataFrame(columns=["second", "benign", "malicious"])
    sec = (df["t0_emit_ns"] - df["t0_emit_ns"].min()) // NS
    last = int(sec.max())
    lo = max(0, last - window_s + 1)
    keep = sec >= lo
    g = pd.DataFrame({"second": sec[keep], "malicious": df.loc[keep, "malicious"]})
    counts = g.groupby(["second", "malicious"]).size().unstack(fill_value=0)
    counts = counts.reindex(range(lo, last + 1), fill_value=0)
    return pd.DataFrame({"second": counts.index,
                         "benign": counts.get(False, 0),
                         "malicious": counts.get(True, 0)}).reset_index(drop=True)


def protocol_mix(df: pd.DataFrame, window_s: int = 60) -> pd.DataFrame:
    """Flows per protocol over the last window_s seconds of emission (FR13)."""
    if df.empty:
        return pd.DataFrame(columns=["protocol", "flows"])
    recent = df[df["t0_emit_ns"] >= df["t0_emit_ns"].max() - window_s * NS]
    c = recent["protocol"].fillna("Other").value_counts()
    rows = [(p, int(c.get(p, 0))) for p in PROTOCOL_ORDER if p != "ICMP" or c.get(p, 0)]
    rows += [(p, int(n)) for p, n in c.items() if p not in PROTOCOL_ORDER]
    return pd.DataFrame(rows, columns=["protocol", "flows"])


def component_means(df: pd.DataFrame, warmup: int) -> dict[str, float] | None:
    """Mean of each latency component over post-warm-up records (v_latency averages)."""
    post = post_warmup(df, warmup)
    if post.empty:
        return None
    return {c: float(post[c].mean()) for c in COMPONENTS}


def dt_steps(run: dict, df: pd.DataFrame, live_changes: list[dict] | None) -> list[tuple]:
    """Δt reference as (start_s, dt_ms) steps on the L-over-time axis. Normally one step
    at the run's emission interval; demo set_rate changes add steps at their from_seq."""
    changes = live_changes or run.get("rate_changes") or []
    steps = [(0.0, float(run["emission_interval_ms"]))]
    if df.empty:
        return steps
    t0_first = int(df["t0_emit_ns"].min())
    by_seq = df.set_index("seq_no")["t0_emit_ns"]
    for ch in changes:
        k = ch.get("from_seq")
        if k is None:
            continue
        later = by_seq[by_seq.index >= k]
        if len(later):
            steps.append(((int(later.iloc[0]) - t0_first) / NS, float(ch["dt_ms"])))
    return steps
