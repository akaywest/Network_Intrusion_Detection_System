"""System metrics from the streaming log (M8; implementation plan §8.2-8.5, Chapter 3
§3.3.6, §3.3.11, Eq. 3.7, NFR1, NFR2).

Reads db/nids.db read-only. Only completed experiment runs are used: scenario S1-S4,
end_reason = 'complete', rate_changes IS NULL, as many PREDICTION rows as n_records,
and a declared dashboard state (TEST_ENVIRONMENT.run_conditions.dashboard_open).
Dashboard-open runs must also have at least one row in results/dashboard_delay.csv,
the rule experiments/run_matrix.py used to accept a run, so the zero-refresh runs it
repeated (65, 84) are listed but excluded. If a configuration still has two accepted
runs, the latest is used and the other is excluded as a duplicate.

Per run (first stream.warmup_latency = 100 records excluded, plan §8.2):
  L median / p95 / p99 / mean / max, mean of t_tx, t_pre, t_inf, t_ser (v_latency)
  throughput (Eq. 3.7) = post-warm-up records / (last t4 - first t0 after warm-up)
  backlog of record i = number of earlier records j with t4_j > t0_i (plan §8.3);
      mean and max, and "growing" = mean over the last 10% of post-warm-up records >
      mean over the first 10%
  S4: peak L, and recovery = t4 of the first record of the first run of 20 consecutive
      records with L <= 100 ms after the last burst emission, minus that emission's t0
  dashboard refresh rate = dashboard-delay rows for the run / run duration (first t0 to
      last t4); flagged when below 0.5/s (chapter4_notes §3.9)
  agreement (§8.4) = share of records whose streamed label equals the offline label
      for the same model, task, seed and row_index (results/offline)

Across seeds: mean and SD (ddof = 1) per (dashboard state, scenario, dt, mode, model).
S3 sustainable rate (plan §8.3): a run passes at its dt when p95(L) <= dt and its
backlog is not growing, i.e. last-10% mean - first-10% mean <= --backlog-tolerance
records. Default 1 record (BACKLOG_TOL): the plan's literal rule (0) flags stable runs,
whose mean backlog is ~0.01, as growing after a single storage stall
(chapter4_notes §2.17); 0 is still available for comparison. Steps: 100 (S1), 50, 20 (S3), 10 (S2), 5, 1 (S3), all with
the dashboard open. A model's sustainable dt is the smallest dt at which all five
seeds pass and every larger dt passes too; 1000/dt is the max sustainable throughput.
NFR verdicts: NFR1 p95(L) <= 100 ms in S1; NFR2 the S2 (dt = 10 ms, 100 records/s)
pass rule above, required for IF and RF, reported as a finding for the LSTM. Verdict
PASS when all seeds pass, FAIL when the seed-mean fails, MIXED otherwise.

Outputs (overwritten on every run, so it can be re-run after the closed set):
  results/latency_summary.csv         one row per run (excluded runs flagged)
  results/system_summary.csv          mean +- SD per dashboard state/scenario/dt/model
  results/sustainable_rate.csv        S3 pass table per model and dt, and the verdict
  results/nfr_verdicts.csv            NFR1 / NFR2 per model
  results/dashboard_open_vs_closed.csv  S1 comparison (or a note when not run yet)
  results/per_record/latency_records.parquet  per-record L, components, backlog
  results/plots/*.png                 nids.evaluation.plots

Usage:  python -m nids.evaluation.system_metrics [--db db/nids.db] [--out results]
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nids.prep.convert_xlsx import ROOT

MODELS = ("iforest", "rf", "lstm")
MODEL_NAMES = {"iforest": "Isolation Forest", "rf": "Random Forest", "lstm": "LSTM"}
SCENARIOS = ("S1", "S2", "S3", "S4")
COMPONENTS = ("t_tx_ms", "t_pre_ms", "t_inf_ms", "t_ser_ms")
RATE_STEPS = (100.0, 50.0, 20.0, 10.0, 5.0, 1.0)   # S3 search incl. S1 (100) and S2 (10)
RECOVERY_L_MS, RECOVERY_RUN = 100.0, 20             # plan §8.3, S4
NFR1_MS, NFR2_DT_MS = 100.0, 10.0
LOW_REFRESH = 0.5                                   # refreshes/s (chapter4_notes §3.9)
BACKLOG_TOL = 1.0                                   # records (chapter4_notes §2.17)
STAT_COLS = ["L_p50_ms", "L_p95_ms", "L_p99_ms", "L_mean_ms", "L_max_ms",
             *[f"{c}_mean" for c in COMPONENTS], "throughput_rps", "backlog_mean",
             "backlog_max", "peak_L_ms", "recovery_s", "refresh_per_s", "agreement"]


# ---------------------------------------------------------------- run selection
def _ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def delay_rows(csv_path: Path) -> pd.DataFrame:
    try:
        with open(csv_path, newline="") as f:
            rows = [(int(r["run_id"]), _ts(r["wall_time_utc"])) for r in csv.DictReader(f)
                    if r.get("run_id")]
    except OSError:
        rows = []
    return pd.DataFrame(rows, columns=["run_id", "at"])


def candidate_runs(con: sqlite3.Connection, delays: pd.DataFrame) -> pd.DataFrame:
    """Every complete, unmarked S1-S4 run with its dashboard state and refresh count."""
    runs = pd.read_sql_query(
        "SELECT r.run_id, r.scenario, r.model_name AS model, r.task, r.random_seed AS seed, "
        "r.emission_interval_ms AS dt_ms, r.load_condition AS mode, r.n_records, "
        "r.malformed_count, r.started_at, r.ended_at, "
        "json_extract(e.run_conditions, '$.dashboard_open') AS dashboard_open, "
        "(SELECT count(*) FROM prediction p WHERE p.run_id = r.run_id) AS n_pred "
        "FROM model_run r JOIN test_environment e ON e.env_id = r.env_id "
        "WHERE r.end_reason = 'complete' AND r.rate_changes IS NULL "
        "AND r.scenario IN (?, ?, ?, ?) ORDER BY r.run_id", con, params=SCENARIOS)
    counts = []
    for r in runs.itertuples():
        lo, hi = _ts(r.started_at), _ts(r.ended_at)
        d = delays[delays.run_id == r.run_id]
        if lo is not None and hi is not None:
            d = d[(d["at"] >= lo) & (d["at"] <= hi + timedelta(seconds=5))]
        counts.append(len(d))
    runs["refreshes"] = counts
    return runs


def select_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Mark each run included or excluded (with the reason), run_matrix's rules."""
    runs = runs.copy()
    reason = pd.Series("", index=runs.index, dtype=object)
    reason[runs.n_pred != runs.n_records] = "prediction count != n_records"
    reason[(reason == "") & runs.dashboard_open.isna()] = "dashboard state not declared"
    reason[(reason == "") & (runs.dashboard_open == 1) & (runs.refreshes == 0)] = \
        "dashboard open but no refresh (repeated by run_matrix)"
    key = ["scenario", "model", "task", "seed", "dt_ms", "mode", "n_records",
           "dashboard_open"]
    ok = runs[reason == ""]
    dup = ok.duplicated(subset=key, keep="last")
    reason[dup[dup].index] = "duplicate configuration (later run used)"
    runs["included"] = reason == ""
    runs["exclude_reason"] = reason
    return runs


# ---------------------------------------------------------------- per-run metrics
RECORDS_SQL = ("SELECT p.run_id, p.seq_no, p.flow_id AS row_index, p.is_warmup, "
               "p.predicted_label, p.t0_emit_ns, p.t4_store_ns, v.t_tx_ms, v.t_pre_ms, "
               "v.t_inf_ms, v.t_ser_ms, v.L_ms "
               "FROM prediction p JOIN v_latency v ON v.prediction_id = p.prediction_id "
               "WHERE p.run_id = ? ORDER BY p.seq_no")


def load_records(con: sqlite3.Connection, run_ids: list[int]) -> pd.DataFrame:
    """Per-record timestamps and latency components, one bound query per run (NFR6)."""
    return pd.concat([pd.read_sql_query(RECORDS_SQL, con, params=(int(r),))
                      for r in run_ids], ignore_index=True)


def backlog(t0: np.ndarray, t4: np.ndarray) -> np.ndarray:
    """For each record i (emission order): earlier records j < i with t4_j > t0_i."""
    t0, t4 = np.asarray(t0, dtype=np.int64), np.asarray(t4, dtype=np.int64)
    if np.all(np.diff(t4) >= 0):                    # engine is FIFO: t4 rises with seq
        done = np.searchsorted(t4, t0, side="right")  # j with t4_j <= t0_i
        idx = np.arange(len(t0))
        return idx - np.minimum(done, idx)
    out = np.empty(len(t0), dtype=np.int64)         # general case (never seen)
    for i in range(len(t0)):
        out[i] = int((t4[:i] > t0[i]).sum())
    return out


def burst_recovery(rec: pd.DataFrame, n: int, block: int) -> tuple[float, float]:
    """(peak L ms, recovery s) for a burst run; recovery NaN if never recovered."""
    lead = (n - block) // 2
    last_burst = lead + block - 1                   # seq of the last back-to-back record
    after = rec[rec.seq_no > last_burst].reset_index(drop=True)
    ok = (after.L_ms <= RECOVERY_L_MS).to_numpy()
    streak = np.convolve(ok.astype(int), np.ones(RECOVERY_RUN, dtype=int), "valid")
    hits = np.flatnonzero(streak == RECOVERY_RUN)
    t0_last = int(rec.loc[rec.seq_no == last_burst, "t0_emit_ns"].iloc[0])
    recovery = (np.nan if not len(hits)
                else (int(after.t4_store_ns.iloc[hits[0]]) - t0_last) / 1e9)
    return float(rec.L_ms.max()), recovery


def offline_labels(offline_dir: Path, model: str, task: str, seed: int) -> pd.Series:
    df = pd.read_parquet(offline_dir / f"{model}_{task}_seed{seed}.parquet",
                         columns=["row_index", "predicted_label"])
    return df.set_index("row_index")["predicted_label"]


def run_metrics(rec: pd.DataFrame, run: pd.Series, warmup: int, block: int,
                offline: pd.Series | None, backlog_tol: float = 0.0
                ) -> tuple[dict, np.ndarray]:
    rec = rec.sort_values("seq_no").reset_index(drop=True)
    bl = backlog(rec.t0_emit_ns.to_numpy(), rec.t4_store_ns.to_numpy())
    post = (rec.seq_no >= warmup) & (rec.is_warmup == 0)
    p, b = rec[post], bl[post.to_numpy()]
    L = p.L_ms.to_numpy()
    k = max(1, len(b) // 10)
    out = {"n_post_warmup": int(len(p)),
           "L_p50_ms": float(np.median(L)), "L_p95_ms": float(np.percentile(L, 95)),
           "L_p99_ms": float(np.percentile(L, 99)), "L_mean_ms": float(L.mean()),
           "L_max_ms": float(L.max()),
           **{f"{c}_mean": float(p[c].mean()) for c in COMPONENTS},
           "throughput_rps": len(p) / ((p.t4_store_ns.max() - p.t0_emit_ns.min()) / 1e9),
           "backlog_mean": float(b.mean()), "backlog_max": int(b.max()),
           "backlog_first10_mean": float(b[:k].mean()),
           "backlog_last10_mean": float(b[-k:].mean())}
    out["backlog_growth"] = out["backlog_last10_mean"] - out["backlog_first10_mean"]
    out["backlog_growing"] = bool(out["backlog_growth"] > backlog_tol)
    out["p95_within_dt"] = out["L_p95_ms"] <= run.dt_ms
    out["sustained"] = bool(out["p95_within_dt"] and not out["backlog_growing"]
                            and run["mode"] == "steady")
    if run["mode"] == "burst":
        out["peak_L_ms"], out["recovery_s"] = burst_recovery(rec, int(run.n_records), block)
        out["recovered"] = bool(np.isfinite(out["recovery_s"]))
    else:
        out["peak_L_ms"], out["recovery_s"], out["recovered"] = np.nan, np.nan, None
    dur = (rec.t4_store_ns.max() - rec.t0_emit_ns.min()) / 1e9
    out["duration_s"] = dur
    out["refresh_per_s"] = (run.refreshes / dur) if run.dashboard_open == 1 else np.nan
    out["low_refresh"] = bool(run.dashboard_open == 1 and out["refresh_per_s"] < LOW_REFRESH)
    if offline is not None:
        ref = offline.reindex(rec.row_index.to_numpy()).to_numpy()
        same = rec.predicted_label.to_numpy() == ref
        out["agreement"], out["n_disagree"] = float(same.mean()), int((~same).sum())
    else:
        out["agreement"], out["n_disagree"] = np.nan, -1
    return out, bl


# ---------------------------------------------------------------- aggregates
def aggregate(ls: pd.DataFrame) -> pd.DataFrame:
    inc = ls[ls.included]
    g = inc.groupby(["dashboard_open", "scenario", "dt_ms", "mode", "model"], sort=False)
    rows = []
    for key, d in g:
        row = dict(zip(["dashboard_open", "scenario", "dt_ms", "mode", "model"], key))
        row["n_seeds"] = len(d)
        row["seeds"] = " ".join(str(s) for s in sorted(d.seed))
        row["low_refresh_runs"] = int(d.low_refresh.sum())
        row["sustained_runs"] = int(d.sustained.sum()) if key[3] == "steady" else np.nan
        row["recovered_runs"] = (int(d.recovered.fillna(False).astype(bool).sum())
                                 if key[3] == "burst" else np.nan)
        for c in STAT_COLS:
            row[f"{c}_mean"] = d[c].mean()
            row[f"{c}_sd"] = d[c].std(ddof=1) if len(d) > 1 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(MODELS)}
    return out.sort_values(["dashboard_open", "scenario", "dt_ms", "model"],
                           ascending=[False, True, False, True],
                           key=lambda s: s.map(order) if s.name == "model" else s
                           ).reset_index(drop=True)


def rate_search_runs(ls: pd.DataFrame) -> pd.DataFrame:
    """S1, S2 and S3 runs with the dashboard open: the six-step rate search."""
    return ls[ls.included & (ls.dashboard_open == 1) & (ls["mode"] == "steady")
              & ls.scenario.isin(["S1", "S2", "S3"]) & ls.dt_ms.isin(RATE_STEPS)]


def sustainable_rate(ls: pd.DataFrame, n_seeds: int = 5) -> pd.DataFrame:
    rs = rate_search_runs(ls)
    rows = []
    for m in MODELS:
        d = rs[rs.model == m]
        steps = []
        for dt in RATE_STEPS:
            x = d[d.dt_ms == dt]
            steps.append({"model": m, "dt_ms": dt, "offered_rps": 1000 / dt,
                          "n_seeds": len(x), "seeds_pass": int(x.sustained.sum()),
                          "p95_mean_ms": x.L_p95_ms.mean(),
                          "backlog_growing_runs": int(x.backlog_growing.sum()),
                          "throughput_mean_rps": x.throughput_rps.mean()})
        best = None
        for s in steps:                     # largest dt first: stop at the first failure
            if s["n_seeds"] == n_seeds and s["seeds_pass"] == n_seeds:
                best = s["dt_ms"]
            else:
                break
        for s in steps:
            s["sustainable_dt_ms"] = best
            s["max_sustainable_rps"] = 1000 / best if best else np.nan
        rows += steps
        # per-seed sustainable dt (smallest dt passing with every larger dt passing)
    return pd.DataFrame(rows)


def per_seed_sustainable(ls: pd.DataFrame) -> pd.DataFrame:
    rs = rate_search_runs(ls)
    rows = []
    for (m, seed), d in rs.groupby(["model", "seed"]):
        best = None
        for dt in RATE_STEPS:
            x = d[d.dt_ms == dt]
            if len(x) == 1 and bool(x.sustained.iloc[0]):
                best = dt
            else:
                break
        rows.append({"model": m, "seed": seed, "sustainable_dt_ms": best})
    return pd.DataFrame(rows)


def verdict(passes: pd.Series, mean_ok: bool) -> str:
    if len(passes) and passes.all():
        return "PASS"
    return "FAIL" if not mean_ok else "MIXED"


def nfr_verdicts(ls: pd.DataFrame, backlog_tol: float = 0.0) -> pd.DataFrame:
    inc = ls[ls.included & (ls.dashboard_open == 1)]
    rows = []
    for m in MODELS:
        s1 = inc[(inc.scenario == "S1") & (inc.model == m)]
        ok1 = s1.L_p95_ms <= NFR1_MS
        rows.append({"nfr": "NFR1", "model": m,
                     "rule": "p95(L) <= 100 ms in S1 (dt 100 ms, dashboard open)",
                     "n_seeds": len(s1), "seeds_pass": int(ok1.sum()),
                     "value_mean": s1.L_p95_ms.mean(), "value_sd": s1.L_p95_ms.std(ddof=1),
                     "unit": "ms (p95 L)",
                     "verdict": verdict(ok1, s1.L_p95_ms.mean() <= NFR1_MS),
                     "required": "yes"})
        s2 = inc[(inc.scenario == "S2") & (inc.model == m)]
        mean_ok = (s2.L_p95_ms.mean() <= NFR2_DT_MS
                   and s2.backlog_growth.mean() <= backlog_tol)
        rows.append({"nfr": "NFR2", "model": m,
                     "rule": "sustains 100 records/s: S2 (dt 10 ms) p95(L) <= 10 ms and "
                             "backlog not growing",
                     "n_seeds": len(s2), "seeds_pass": int(s2.sustained.sum()),
                     "value_mean": s2.throughput_rps.mean(),
                     "value_sd": s2.throughput_rps.std(ddof=1),
                     "unit": "records/s (Eq. 3.7)",
                     "verdict": verdict(s2.sustained, mean_ok),
                     "required": "no (finding)" if m == "lstm" else "yes"})
    return pd.DataFrame(rows)


def open_vs_closed(ls: pd.DataFrame) -> pd.DataFrame:
    s1 = ls[ls.included & (ls.scenario == "S1")]
    closed = s1[s1.dashboard_open == 0]
    if closed.empty:
        return pd.DataFrame([{"note": "dashboard-closed S1 set not run yet "
                                      "(run_matrix.py --set closed); comparison skipped"}])
    metrics = ["L_p50_ms", "L_p95_ms", "L_p99_ms", "L_mean_ms", "t_inf_ms_mean",
               "t_tx_ms_mean", "throughput_rps"]
    rows = []
    groups = {"open (all)": s1[s1.dashboard_open == 1],
              f"open (refresh >= {LOW_REFRESH}/s)": s1[(s1.dashboard_open == 1)
                                                       & ~s1.low_refresh],
              "closed": closed}
    for m in MODELS:
        for name, d in groups.items():
            x = d[d.model == m]
            row = {"model": m, "group": name, "n_runs": len(x),
                   "refresh_per_s_mean": x.refresh_per_s.mean()}
            for c in metrics:
                row[f"{c}_mean"] = x[c].mean()
                row[f"{c}_sd"] = x[c].std(ddof=1) if len(x) > 1 else np.nan
            rows.append(row)
        base = groups["closed"][groups["closed"].model == m]
        op = groups["open (all)"][groups["open (all)"].model == m]
        rows.append({"model": m, "group": "difference open (all) - closed",
                     "n_runs": np.nan, "refresh_per_s_mean": np.nan,
                     **{f"{c}_mean": op[c].mean() - base[c].mean() for c in metrics}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- driver
def compute(db: Path, offline_dir: Path, delay_csv: Path, cfg: dict,
            backlog_tol: float = 0.0) -> dict:
    warmup = cfg["stream"]["warmup_latency"]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        runs = select_runs(candidate_runs(con, delay_rows(delay_csv)))
        recs = load_records(con, runs.run_id.astype(int).tolist()) if len(runs) else None
    finally:
        con.close()
    if recs is None or recs.empty:
        raise SystemExit(f"no completed S1-S4 experiment runs in {db}")
    rows, parts = [], []
    for run in runs.itertuples(index=False):
        run = pd.Series(run._asdict())
        rec = recs[recs.run_id == run.run_id]
        block = cfg["stream"]["burst_block"]
        try:
            off = offline_labels(offline_dir, run.model, run.task, int(run.seed))
        except FileNotFoundError:
            off = None
        m, bl = run_metrics(rec, run, warmup, block, off, backlog_tol)
        rows.append({**run.drop(["started_at", "ended_at"]).to_dict(), **m})
        r = rec.sort_values("seq_no")
        parts.append(pd.DataFrame({
            "run_id": run.run_id, "seq_no": r.seq_no.to_numpy(),
            "row_index": r.row_index.to_numpy(),
            "t_rel_s": (r.t0_emit_ns.to_numpy() - r.t0_emit_ns.min()) / 1e9,
            **{c: r[c].to_numpy() for c in (*COMPONENTS, "L_ms")}, "backlog": bl}))
    ls = pd.DataFrame(rows)
    ls["dashboard_open"] = ls.dashboard_open.astype("Int64")
    ls["backlog_tolerance"] = backlog_tol
    return {"latency_summary": ls, "records": pd.concat(parts, ignore_index=True),
            "backlog_tol": backlog_tol}


def write_outputs(res: dict, out: Path) -> dict:
    from nids.evaluation import plots
    ls, recs = res["latency_summary"], res["records"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_record").mkdir(exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    tables = {"system_summary": aggregate(ls), "sustainable_rate": sustainable_rate(ls),
              "nfr_verdicts": nfr_verdicts(ls, res["backlog_tol"]),
              "dashboard_open_vs_closed": open_vs_closed(ls)}
    ls.to_csv(out / "latency_summary.csv", index=False)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    recs.to_parquet(out / "per_record" / "latency_records.parquet", index=False)

    inc = ls[ls.included & (ls.dashboard_open == 1)]
    s1_ids = inc[inc.scenario == "S1"].set_index("run_id").model
    s1 = recs[recs.run_id.isin(s1_ids.index)].merge(
        s1_ids.rename("model"), left_on="run_id", right_index=True)
    s1 = s1[s1.seq_no >= 100]
    plots.plot_latency_cdf({m: s1[s1.model == m].L_ms.to_numpy() for m in MODELS},
                           out / "plots" / "latency_cdf_S1.png", NFR1_MS)
    agg = tables["system_summary"]
    a1 = agg[(agg.dashboard_open == 1) & (agg.scenario == "S1")].set_index("model")
    plots.plot_components({m: {c: a1.loc[m, f"{c}_mean_mean"] for c in COMPONENTS}
                           for m in MODELS if m in a1.index},
                          out / "plots" / "components_S1.png")
    rs = rate_search_runs(ls)
    plots.plot_rate_search(rs, out / "plots" / "rate_search_S3.png")
    s4 = inc[inc.scenario == "S4"]
    burst = recs[recs.run_id.isin(s4.run_id)].merge(
        s4[["run_id", "model", "seed"]], on="run_id")
    n, block = 2000, 1000
    lead = (n - block) // 2
    notes = {}
    for m, d in s4.groupby("model"):
        rec_ok = d.recovery_s.dropna()
        notes[m] = (f"recovered {len(rec_ok)}/{len(d)}, {rec_ok.mean():.1f} s mean"
                    if len(rec_ok) else f"not recovered by run end (0/{len(d)})")
    plots.plot_burst(burst, lead, block, out / "plots" / "burst_S4.png", RECOVERY_L_MS,
                     notes)
    return tables


def print_tables(ls: pd.DataFrame, tables: dict) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    inc = ls[ls.included]
    print(f"runs: {len(ls)} candidates, {len(inc)} included "
          f"(open {int((inc.dashboard_open == 1).sum())}, "
          f"closed {int((inc.dashboard_open == 0).sum())})")
    exc = ls[~ls.included]
    if len(exc):
        print("excluded:\n" + exc[["run_id", "scenario", "model", "seed", "dt_ms",
                                   "exclude_reason"]].to_string(index=False))
    agg = tables["system_summary"]
    show = agg[["dashboard_open", "scenario", "dt_ms", "model", "n_seeds"]].copy()
    for c, lab in (("L_p50_ms", "p50"), ("L_p95_ms", "p95"), ("L_p99_ms", "p99"),
                   ("L_max_ms", "max"), ("t_inf_ms_mean", "t_inf"),
                   ("throughput_rps", "thr/s"), ("backlog_max", "bl_max"),
                   ("refresh_per_s", "refr/s")):
        show[lab] = [f"{m:.1f}±{s:.1f}" if pd.notna(s) else f"{m:.1f}"
                     for m, s in zip(agg[f"{c}_mean"], agg[f"{c}_sd"])]
    show["sust"] = agg.sustained_runs
    show["lowref"] = agg.low_refresh_runs
    print("\nper dashboard state / scenario / dt / model (mean ± SD over seeds; ms)")
    print(show.to_string(index=False))
    s4 = agg[agg.scenario == "S4"]
    print("\nS4 burst: peak L (ms) and recovery (s)")
    print(pd.DataFrame({"model": s4.model,
                        "peak_L_ms": [f"{m:.0f}±{s:.0f}" for m, s in
                                      zip(s4.peak_L_ms_mean, s4.peak_L_ms_sd)],
                        "recovered": [f"{int(r)}/{n}" for r, n in
                                      zip(s4.recovered_runs, s4.n_seeds)],
                        "recovery_s": [f"{m:.2f}±{s:.2f}" if pd.notna(m)
                                       else "not recovered within the run"
                                       for m, s in zip(s4.recovery_s_mean,
                                                       s4.recovery_s_sd)]}
                       ).to_string(index=False))
    sr = tables["sustainable_rate"]
    tol = float(ls.backlog_tolerance.iloc[0])
    print(f"\nS3 rate search: seeds passing (p95 <= dt and backlog growth <= {tol:g} "
          "records)")
    piv = sr.pivot(index="model", columns="dt_ms", values="seeds_pass")
    piv = piv[sorted(piv.columns, reverse=True)].reindex(MODELS)
    best = sr.groupby("model").sustainable_dt_ms.first().reindex(MODELS)
    piv["sustainable_dt_ms"] = best
    piv["max_rps"] = 1000 / best
    print(piv.to_string())
    print("\nNFR verdicts")
    print(tables["nfr_verdicts"][["nfr", "model", "n_seeds", "seeds_pass", "value_mean",
                                  "value_sd", "unit", "verdict", "required"]]
          .to_string(index=False))
    print("\n§8.4 agreement (streamed vs offline labels): "
          f"{(inc.agreement == 1).sum()}/{len(inc)} runs at 100%, "
          f"{int(inc.n_disagree.clip(lower=0).sum())} disagreeing records in total, "
          f"min {inc.agreement.min():.4%}")
    print(f"low dashboard refresh (< {LOW_REFRESH}/s): "
          f"{int(inc.low_refresh.sum())}/{int((inc.dashboard_open == 1).sum())} open runs")
    print("\ndashboard open vs closed (S1):")
    print(tables["dashboard_open_vs_closed"].to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(ROOT / cfg["paths"]["db"]))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--offline", default=str(ROOT / "results" / "offline"))
    ap.add_argument("--delay-csv", default=str(ROOT / "results" / "dashboard_delay.csv"))
    ap.add_argument("--backlog-tolerance", type=float, default=BACKLOG_TOL,
                    help="records the backlog may grow (last 10%% vs first 10%%) and still "
                         "count as not growing (default 1; 0 = plan §8.3 literal)")
    a = ap.parse_args(argv)
    res = compute(Path(a.db).resolve(), Path(a.offline), Path(a.delay_csv), cfg,
                  a.backlog_tolerance)
    tables = write_outputs(res, Path(a.out))
    print_tables(res["latency_summary"], tables)
    print(f"\nwrote {a.out}/latency_summary.csv, system_summary.csv, sustainable_rate.csv, "
          "nfr_verdicts.csv, dashboard_open_vs_closed.csv, per_record/, plots/")


if __name__ == "__main__":
    main()
