"""Streamlit dashboard (M6; plan §7.2, Chapter 3 §3.3.7, Fig. 3.12, Table 3.8;
FR2, FR8, FR12, FR13, UC6, NFR5).

Layout (Fig. 3.12):
  sidebar       title; "Run settings": model, task (RF/LSTM only), seed, Δt slider
                1-1,000 ms (with its records/s equivalent), steady/burst; "Actions":
                Start/Stop and a status line with record count  -> control channel (M7)
  KPI row       records, alerts, p50/p95/p99 of L, throughput (warm-up excluded);
                p95/p99 marked against Δt, throughput against 1000/Δt
  detection row running precision, recall, false alarms vs the dataset label
                (warm-up excluded; display aid, not the §8.1 metrics)
  top           live traffic (benign vs malicious flows/s, stacked, last 60 s) |
                protocol mix (neutral grey)
  centre        last 20 alerts (true class marked as ground truth, severity chips,
                false positives highlighted) | latency components (stacked bar)
  bottom        L per record over time with Δt and p95 reference lines, log toggle
The display area is one @st.fragment(run_every="1s") (config dashboard.refresh_s), so
only it re-runs each second; the sidebar does not (its status line is a second 1 s
fragment, which re-runs the whole app once when a run starts or ends so the settings
lock and unlock).

Data: a READ-ONLY SQLite connection, rows with t4 IS NOT NULL only (nids.dashboard.
queries); the same five queries per refresh as before the UI changes (chapter4_notes
§2.15). Colour: benign is always blue and malicious always orange (§3.3.7;
categorical slots 1-2 of the dataviz reference palette, a CVD-safe pair), in light and
dark themes; nothing else uses those two hues. Target checks (p95/p99 vs Δt) use the
reference status colours and always carry an icon and words, never colour alone.

Dashboard delay (§3.3.6, reported separately from L): on each refresh while a run is
active, perf_counter_ns() - max(t4) is appended to results/dashboard_delay.csv
(perf_counter_ns is the same monotonic clock the engine stamps t4 with; see the clock
check in plan §6.3).

Δt during a run (FR2: "before and during a run"): the other run settings lock while a
run is active, but the Δt slider stays live in steady runs; moving it sends set_rate,
which is demo-only; the engine marks that run (MODEL_RUN.rate_changes) so it is never
counted as an experiment run. In burst runs it locks too (the streamer rejects
set_rate there, the burst shape is the experiment).

Environment (set by run.py): NIDS_DB, NIDS_ENGINE_CTL_PORT, NIDS_STREAMER_CTL_PORT;
NIDS_DASHBOARD_DELAY_CSV overrides the delay log path (tests).
Launch: python run.py   (or: streamlit run src/nids/dashboard/app.py with the services up)
"""
from __future__ import annotations

import csv
import math
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from nids.control.client import ControlClient, ControlError
from nids.dashboard import queries as q
from nids.prep.convert_xlsx import ROOT

CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
DB_PATH = Path(os.environ.get("NIDS_DB", ROOT / CFG["paths"]["db"]))
DELAY_CSV = Path(os.environ.get("NIDS_DASHBOARD_DELAY_CSV",
                                ROOT / "results" / "dashboard_delay.csv"))
WARMUP = int(CFG["stream"]["warmup_latency"])
REFRESH = f"{CFG['dashboard']['refresh_s']}s"

TITLE = "Real-Time Network Intrusion Detection System"
SUBTITLE = "Machine learning · IDS2025 replay"

# Model selector (NFR7: a new detector needs one Detector subclass + one entry here).
MODELS = {"iforest": "Isolation Forest", "rf": "Random Forest", "lstm": "LSTM"}
TASKS = {"iforest": ("binary",), "rf": ("binary", "multi"), "lstm": ("binary", "multi")}
TASK_NAMES = {"binary": "Binary", "multi": "Seven-class"}

# Colour roles (dataviz reference palette; light / dark steps of the same hues).
# benign/malicious: categorical slots 1-2; components: slots 3-6 (no blue/orange);
# neutral: protocol mix (greys only, so blue and orange keep their meaning);
# good/critical: reference status colours (fixed across themes).
PALETTE = {
    "light": {"benign": "#2a78d6", "malicious": "#eb6834", "single": "#6250d6",
              "components": ["#1baf7a", "#eda100", "#e87ba4", "#008300"],
              "neutral": "#898781", "ref": "#52514e", "muted": "#898781",
              "good": "#0ca30c", "critical": "#d03b3b", "surface": "#ffffff"},
    "dark": {"benign": "#3987e5", "malicious": "#d95926", "single": "#9085e9",
             "components": ["#199e70", "#c98500", "#d55181", "#008300"],
             "neutral": "#898781", "ref": "#c3c2b7", "muted": "#898781",
             "good": "#0ca30c", "critical": "#d03b3b", "surface": "#0e1117"},
}
PLOTLY_CONFIG = {"displayModeBar": False}  # the toolbar covered the legends
# Pipeline order, left to right (Eq. 3.1: L = t_tx + t_pre + t_inf + t_ser).
COMPONENT_NAMES = {"t_tx_ms": "t_tx transmission", "t_pre_ms": "t_pre preprocessing",
                   "t_inf_ms": "t_inf inference", "t_ser_ms": "t_ser storage"}
SEVERITY_CHIPS = {"High": "▲ High", "Medium": "● Medium"}
FP_TINT = "background-color: rgba(208, 59, 59, 0.14)"   # status critical, translucent

st.set_page_config(page_title=TITLE, page_icon="🛡️", layout="wide")
# Hide Streamlit's Deploy button and main menu (a measurement tool, not a demo app).
# Metric values one step smaller so six bordered cards fit a 1,440 px window untruncated.
st.markdown("<style>[data-testid='stAppDeployButton'],[data-testid='stMainMenu']"
            "{display:none !important}"
            "[data-testid='stMetricValue']{font-size:1.75rem}</style>",
            unsafe_allow_html=True)


def colours() -> dict:
    try:
        mode = st.context.theme.type or "light"
    except AttributeError:
        mode = "light"
    return PALETTE.get(mode, PALETTE["light"])


def rgba(hex_: str, alpha: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"


def client() -> ControlClient:
    return ControlClient.from_config(CFG)  # cheap: ports only, no open connection


def services_status() -> dict | None:
    try:
        return client().status()
    except OSError:
        return None


def active_run(sv: dict | None) -> dict | None:
    return (sv or {}).get("engine", {}).get("run")


def flash(kind: str, text: str) -> None:
    st.session_state["flash"] = (kind, text)


# ---------------------------------------------------------------- sidebar callbacks
def on_start() -> None:
    s = st.session_state
    mode = s["mode"]
    n = CFG["stream"]["steady_n"] if mode == "steady" else CFG["stream"]["burst_n"]
    task = s["task"] if s["model"] != "iforest" else "binary"
    try:
        run_id = client().prepare(s["model"], task, int(s["seed"]), float(s["dt_ms"]),
                                  mode, "dashboard", n, dashboard_open=True)
        try:
            client().start(run_id, float(s["dt_ms"]), mode, n,
                           CFG["stream"]["sample_seed"])
        except Exception:
            client().engine("stop")               # cancel the prepared run
            raise
    except (ControlError, OSError) as e:
        flash("error", f"Could not start: {e}")
        return
    s["view_run"] = run_id
    flash("success", f"Run {run_id} started: {MODELS[s['model']]} "
                     f"({TASK_NAMES[task]}), Δt = {s['dt_ms']} ms, {mode}, {n:,} records")


def on_stop() -> None:
    try:
        r = client().stop()
    except OSError as e:
        flash("error", f"Could not stop: {e}")
        return
    flash("info", f"Stopping run {r['run_id']}" if r.get("stopping") else "No run to stop")


def on_dt_change() -> None:
    """FR2: changing Δt mid-run is a demonstration feature (run gets marked)."""
    st_ = services_status()
    if not st_ or st_["streamer"].get("state") != "streaming":
        return                                         # applies to the next run only
    try:
        r = client().set_rate(float(st.session_state["dt_ms"]))
    except (ControlError, OSError) as e:
        flash("error", f"Δt not changed: {e}")
        return
    flash("warning", f"Δt changed to {st.session_state['dt_ms']} ms from record "
                     f"{r['from_seq']:,} (demo only: run {r['run_id']} is marked and "
                     "excluded from experiments)")


# ---------------------------------------------------------------- sidebar
def rate_text(dt_ms: float) -> str:
    return f"≈ {1000 / dt_ms:,.4g} records/s"


def sidebar() -> None:
    run = active_run(services_status())
    st.session_state["sidebar_locked_for"] = run["run_id"] if run else None
    locked = run is not None
    burst_run = locked and run.get("mode") == "burst"

    st.sidebar.header(TITLE)
    st.sidebar.caption(SUBTITLE)

    st.sidebar.subheader("Run settings")
    st.sidebar.selectbox("Model", list(MODELS), format_func=MODELS.get, key="model",
                         disabled=locked)
    model = st.session_state["model"]
    st.sidebar.radio("Task", TASKS["rf"], format_func=TASK_NAMES.get, key="task",
                     horizontal=True, disabled=locked or len(TASKS[model]) == 1,
                     help="Isolation Forest is binary only" if model == "iforest" else None)
    st.sidebar.selectbox("Seed", CFG["seeds"], key="seed", disabled=locked)
    st.sidebar.slider("Emission interval Δt (ms)", 1, 1000, 100, key="dt_ms",
                      on_change=on_dt_change, disabled=burst_run,
                      help="Moving this during a steady run changes the rate for that "
                           "run (FR2; demo only, the run is marked).")
    dt = float(st.session_state["dt_ms"])
    st.sidebar.caption(rate_text(dt) + (" · changes now apply to the live run (demo)"
                                        if locked and not burst_run else ""))
    st.sidebar.radio("Load", ("steady", "burst"), format_func=str.capitalize, key="mode",
                     horizontal=True, disabled=locked)
    if locked:
        st.sidebar.caption(f"Settings are locked while run {run['run_id']} is active.")

    st.sidebar.subheader("Actions")
    c1, c2 = st.sidebar.columns(2)
    c1.button("Start", type="primary", on_click=on_start, key="start",
              use_container_width=True, disabled=locked)
    c2.button("Stop", on_click=on_stop, key="stop", use_container_width=True,
              disabled=not locked)
    if "flash" in st.session_state:
        kind, text = st.session_state.pop("flash")
        getattr(st.sidebar, kind)(text)
    with st.sidebar:
        status_line()


@st.fragment(run_every=REFRESH)
def status_line() -> None:
    sv = services_status()
    run = active_run(sv)
    # A run started or ended since the sidebar was drawn: redraw the app once so the
    # settings lock / unlock (the sidebar itself is outside the 1 s fragments).
    if (run["run_id"] if run else None) != st.session_state.get("sidebar_locked_for"):
        st.rerun(scope="app")
    if sv is None:
        st.error("Engine/streamer not reachable. Start them with `python run.py`.")
        return
    eng, stm = sv["engine"], sv["streamer"]
    if run:
        sent = (stm.get("run") or {}).get("sent")
        total = run.get("n") or "?"
        st.markdown(f"**● {eng['state'].capitalize()}** · run {run['run_id']} "
                    f"({MODELS.get(run['model'], run['model'])}, "
                    f"{TASK_NAMES.get(run['task'], run['task'])}, seed {run['seed']})  \n"
                    f"{run['processed']:,} processed"
                    + (f" · {sent:,} / {total:,} sent" if sent is not None else "")
                    + (f" · {run['malformed']} malformed" if run["malformed"] else ""))
    else:
        last = eng.get("last_run")
        tail = (f" · last run {last['run_id']}: {last['processed']:,} records "
                f"({last['end_reason']})") if last else ""
        st.markdown(f"**○ Idle**{tail}")


# ---------------------------------------------------------------- charts
def base_layout(fig: go.Figure, height: int, ytitle: str, xtitle: str) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      hovermode="x unified")
    fig.update_xaxes(title_text=xtitle, showgrid=False)
    fig.update_yaxes(title_text=ytitle, rangemode="tozero")
    return fig


def traffic_chart(tr: pd.DataFrame, c: dict) -> go.Figure:
    """Stacked area: benign at the bottom, malicious on top; the top edge is the total."""
    fig = go.Figure()
    for col, name in (("benign", "Benign"), ("malicious", "Malicious")):
        fig.add_trace(go.Scatter(x=tr["second"], y=tr[col], name=name, mode="lines",
                                 stackgroup="traffic", line=dict(color=c[col], width=1.5),
                                 fillcolor=rgba(c[col], 0.55),
                                 hovertemplate=f"{name}: %{{y}} flows/s<extra></extra>"))
    return base_layout(fig, 260, "flows / s", "seconds since run start")


def protocol_chart(pm: pd.DataFrame, c: dict) -> go.Figure:
    fig = go.Figure(go.Bar(x=pm["flows"], y=pm["protocol"], orientation="h",
                           marker=dict(color=c["neutral"], cornerradius=4),
                           text=pm["flows"], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x} flows<extra></extra>"))
    fig = base_layout(fig, 260, "", "flows in last 60 s")
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed")
    return fig


def components_chart(means: dict, c: dict) -> go.Figure:
    fig = go.Figure()
    for (col, name), colour in zip(COMPONENT_NAMES.items(), c["components"]):
        fig.add_trace(go.Bar(x=[means[col]], y=["mean L"], name=name, orientation="h",
                             marker=dict(color=colour,
                                         line=dict(color=c["surface"], width=2)),
                             hovertemplate=f"{name}: %{{x:.3f}} ms<extra></extra>"))
    total = sum(means.values())
    fig.add_annotation(x=total, y="mean L", text=f"{total:.1f} ms", showarrow=False,
                       xanchor="left", xshift=6, font=dict(color=c["muted"]))
    fig = base_layout(fig, 150, "", "ms (mean per record, warm-up excluded)")
    fig.update_layout(barmode="stack", hovermode="closest",
                      legend=dict(traceorder="normal"))   # pipeline order, left to right
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(range=[0, total * 1.15 if total > 0 else 1])
    return fig


def latency_chart(df: pd.DataFrame, steps: list[tuple], p95: float | None, log: bool,
                  c: dict) -> go.Figure:
    x = (df["t0_emit_ns"] - df["t0_emit_ns"].min()) / q.NS
    fig = go.Figure(go.Scattergl(x=x, y=df["L_ms"], name="L per record", mode="lines",
                                 line=dict(color=c["single"], width=1.5),
                                 customdata=df["seq_no"],
                                 hovertemplate="record %{customdata}: L = %{y:.2f} ms"
                                               "<extra></extra>"))
    xmax = float(x.max()) if len(x) else 1.0
    sx = [s for s, _ in steps] + [xmax]
    sy = [d for _, d in steps] + [steps[-1][1]]
    fig.add_trace(go.Scatter(x=sx, y=sy, name="Δt (target)", mode="lines",
                             line_shape="hv", line=dict(color=c["ref"], width=3, dash="dash"),
                             hovertemplate="Δt = %{y} ms<extra></extra>"))
    fig.add_annotation(x=0, y=sy[0], text=f"<b>Δt = {sy[0]:g} ms</b>", showarrow=False,
                       xanchor="left", yanchor="bottom", yshift=4,
                       font=dict(color=c["ref"]), yref="y")
    if p95 is not None:
        ok = q.within_target(p95, sy[-1])
        colour = c["good"] if ok else c["critical"]
        fig.add_trace(go.Scatter(x=[0, xmax], y=[p95, p95], name="p95 of L", mode="lines",
                                 line=dict(color=colour, width=2, dash="dot"),
                                 hovertemplate="p95 = %{y:.1f} ms<extra></extra>"))
        fig.add_annotation(x=xmax, y=p95, showarrow=False, xanchor="right",
                           yanchor="bottom", yshift=4, font=dict(color=colour),
                           text=f"p95 = {p95:.1f} ms {'✓ within' if ok else '✗ over'} Δt")
    fig = base_layout(fig, 280, "L (ms, log scale)" if log else "L (ms)",
                      "seconds since run start")
    fig.update_layout(hovermode="closest")
    if log:
        fig.update_yaxes(type="log", rangemode="normal")
        for a in fig.layout.annotations:    # log axes place annotations by log10(y)
            a.y = _log10(a.y)
    return fig


def _log10(v: float) -> float:
    return math.log10(v) if v and v > 0 else 0.0


def fmt_ms(v: float | None) -> str:
    return "—" if v is None else (f"{v:.1f} ms" if v >= 10 else f"{v:.2f} ms")


def fmt_frac(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def target_delta(v: float | None, dt: float) -> tuple[str | None, str]:
    """p95/p99 against Δt (NFR1 uses p95 ≤ Δt = 100 ms): icon + words + status colour."""
    ok = q.within_target(v, dt)
    if ok is None:
        return None, "off"
    return ("✓ ≤ target Δt", "green") if ok else ("✗ > target Δt", "red")


def log_delay(run_id: int, t4: int | None) -> float | None:
    """§3.3.6 dashboard delay: now - newest t4, appended to results/dashboard_delay.csv."""
    if t4 is None:
        return None
    delay_ms = (time.perf_counter_ns() - int(t4)) / 1e6
    new = not DELAY_CSV.exists()
    DELAY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DELAY_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["wall_time_utc", "session", "run_id", "delay_ms"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    st.session_state.setdefault("session", uuid.uuid4().hex[:8]),
                    run_id, f"{delay_ms:.3f}"])
    return delay_ms


# ---------------------------------------------------------------- display regions
def kpi_rows(k: dict, det: dict, run: dict, dt: float) -> None:
    cols = st.columns(6)
    cols[0].metric("Records processed", f"{k['records']:,}", border=True)
    cols[1].metric("Alerts", f"{k['alerts']:,}", border=True)
    cols[2].metric("L p50", fmt_ms(k["p50_ms"]), border=True)
    for col, key, label in ((cols[3], "p95_ms", "L p95"), (cols[4], "p99_ms", "L p99")):
        delta, colour = target_delta(k[key], dt)
        col.metric(label, fmt_ms(k[key]), delta, delta_color=colour, delta_arrow="off",
                   border=True, help=f"Target: {label[2:]} of L ≤ Δt = {dt:g} ms (NFR1 "
                                     "uses p95 at Δt = 100 ms).")
    tput = k["throughput_rps"]
    if run["load_condition"] == "steady" and tput is not None:
        offered = 1000 / dt
        tdelta = f"{tput / offered:.0%} of 1000/Δt"
        thelp = (f"Eq. 3.7 throughput compared with the offered rate 1000/Δt = "
                 f"{offered:,.4g} records/s.")
    else:
        tdelta = "burst load" if tput is not None else None
        thelp = "Burst load has no single offered rate to compare with."
    cols[5].metric("Throughput", "—" if tput is None else f"{tput:.1f} rec/s", tdelta,
                   delta_color="gray", delta_arrow="off", border=True, help=thelp)

    cols = st.columns([1, 1, 1, 3])
    cols[0].metric("Precision", fmt_frac(det["precision"]), border=True,
                   help="TP / (TP + FP): of the flows flagged malicious, the share that "
                        "are malicious in the dataset labels.")
    cols[1].metric("Recall", fmt_frac(det["recall"]), border=True,
                   help="TP / (TP + FN): of the malicious flows, the share flagged.")
    cols[2].metric("False alarms", f"{det['false_alarms']:,}", border=True,
                   help="Flows flagged malicious whose dataset label is Normal (FP).")
    if k["n_post_warmup"] == 0 and k["records"]:
        note = (f"Latency, throughput and detection figures appear after the first "
                f"{WARMUP} records (warm-up, Table 3.13).")
    else:
        note = (f"Latency, throughput and detection figures exclude the first {WARMUP} "
                "records (warm-up). Detection is running, against the dataset labels "
                f"(any non-Normal class = malicious), over {det['n']:,} records; the "
                "reported metrics use the full test partition offline (§8.1).")
    cols[3].caption(note)


def alert_table(alerts: pd.DataFrame) -> None:
    fp = (alerts["true_label"] == "Normal").to_numpy()
    view = pd.DataFrame({
        "Time (UTC)": pd.to_datetime(alerts["raised_at"]).dt.strftime("%H:%M:%S.%f")
                  .str[:-3],
        "Severity": [[SEVERITY_CHIPS.get(s, s)] for s in alerts["severity"]],
        "Predicted": alerts["predicted_label"],
        "True class (ground truth)": alerts["true_label"],
        "Check": ["✗ False positive" if f else "✓ True alert" for f in fp],
        "Score": alerts["anomaly_score"],
        "Model": alerts["model_name"].map(MODELS).fillna(alerts["model_name"]),
    })
    styled = (view.style
              .apply(lambda r: [FP_TINT if fp[r.name] else ""] * len(r), axis=1)
              .format({"Score": "{:.4f}"}, na_rep="—"))
    st.dataframe(styled, hide_index=True, height=300, key="alerts",
                 column_config={
                     "Severity": st.column_config.MultiselectColumn(
                         "Severity", options=list(SEVERITY_CHIPS.values()),
                         color=["red", "orange"], width="small"),
                     "Predicted": st.column_config.TextColumn(width="small"),
                     "Score": st.column_config.TextColumn(width="small"),
                     "Model": st.column_config.TextColumn(width="small")})
    st.caption(f"True class comes from the dataset labels (ground truth); the engine "
               f"never sees it. Shaded rows: predicted malicious, true class Normal "
               f"({int(fp.sum())} of the last {len(fp)}).")


@st.fragment(run_every=REFRESH)
def display() -> None:
    c = colours()
    sv = services_status()
    active = active_run(sv)
    try:
        con = q.open_ro(DB_PATH)
    except sqlite3.OperationalError as e:
        st.error(f"Cannot open {DB_PATH} read-only: {e}. Run `python run.py` once to "
                 "create it.")
        return
    try:
        run_id = active["run_id"] if active else (st.session_state.get("view_run")
                                                  or q.latest_run_id(con))
        if run_id is None:
            st.info("No runs yet. Choose a model in the sidebar and press **Start**.")
            return
        run = q.run_info(con, run_id)
        df = q.run_records(con, run_id)
        n_alerts = q.alert_count(con, run_id)
        alerts = q.recent_alerts(con, run_id)
        t4 = q.max_t4(con, run_id)
    finally:
        con.close()

    live_changes = None
    if active and sv:
        live_changes = (sv["streamer"].get("run") or {}).get("rate_changes")
    state = "live" if active else (run["end_reason"] or "not finished")
    st.caption(f"Run {run_id} · {MODELS.get(run['model_name'], run['model_name'])} "
               f"({TASK_NAMES.get(run['task'], run['task'])}) · seed {run['random_seed']} "
               f"· Δt {run['emission_interval_ms']:g} ms · {run['load_condition']} · "
               f"scenario {run['scenario']} · {state}"
               + (" · **demo: Δt changed during run**" if (live_changes or
                                                          run["rate_changes"]) else ""))

    steps = q.dt_steps(run, df, live_changes)
    dt = steps[-1][1]                       # the Δt in force now (= the run's, unless demo)
    k = q.kpis(df, n_alerts, WARMUP)
    kpi_rows(k, q.detection_kpis(df, WARMUP), run, dt)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Live traffic (last 60 s)")
        tr = q.traffic_per_second(df)
        if tr.empty:
            st.info("Waiting for records…")
        else:
            st.plotly_chart(traffic_chart(tr, c), key="traffic", config=PLOTLY_CONFIG)
    with right:
        st.subheader("Protocol mix (last 60 s)")
        pm = q.protocol_mix(df)
        if pm.empty:
            st.info("Waiting for records…")
        else:
            st.plotly_chart(protocol_chart(pm, c), key="protocol", config=PLOTLY_CONFIG)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Latest alerts")
        if alerts.empty:
            st.info("No alerts in this run yet.")
        else:
            alert_table(alerts)
    with right:
        st.subheader("Latency components")
        means = q.component_means(df, WARMUP)
        if means is None:
            st.info(f"Shown after the first {WARMUP} records.")
        else:
            st.plotly_chart(components_chart(means, c), key="components", config=PLOTLY_CONFIG)
            st.dataframe(pd.DataFrame({"Component": list(COMPONENT_NAMES.values()),
                                       "Mean (ms)": [means[k] for k in COMPONENT_NAMES]}),
                         hide_index=True, key="components_table",
                         column_config={"Mean (ms)": st.column_config.NumberColumn(
                             format="%.3f")})

    head, toggle = st.columns([4, 1], vertical_alignment="bottom")
    head.subheader("Latency L per record")
    log = toggle.toggle("Log scale", key="latency_log",
                        help="Log y-axis: spikes and the Δt line stay readable together.")
    if df.empty:
        st.info("Waiting for records…")
    else:
        st.plotly_chart(latency_chart(df, steps, k["p95_ms"], log, c),
                        key="latency", config=PLOTLY_CONFIG)

    if active and active["run_id"] == run_id:
        d = log_delay(run_id, t4)
        if d is not None:
            st.caption(f"Dashboard delay (now − newest t4): {d:.0f} ms, logged to "
                       f"{DELAY_CSV.name}")


sidebar()
display()
