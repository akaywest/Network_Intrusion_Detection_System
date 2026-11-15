"""Headless test-matrix driver (M8; plan §8.2, Table 3.13, Chapter 3 §3.3.10-3.3.11).

Drives the engine and streamer through the loopback control channel with exactly the
calls the dashboard makes (ControlClient.prepare -> start -> wait for eos), so a matrix
run and a dashboard run take the same code path (plan §7.1).

Scenarios (binary models only, plan §8.2; n from config.yaml stream.*):
  S1  steady, dt 100 ms, 5,000 records
  S2  steady, dt 10 ms, 5,000 records
  S3  steady rate search, dt 50, 20, 5, 1 ms, 5,000 records (100 and 10 reuse S1/S2)
  S4  burst, 2,000 records: 500 @ 100 ms, 1,000 back-to-back, 500 @ 100 ms

Sets (the dashboard's state cannot change mid-matrix, so each set is its own invocation):
  main    S1-S4 with the dashboard OPEN at a 1 s refresh, as the design specifies
          (§3.3.7, plan §8.2). Requires --dashboard-open.
  closed  S1 with the dashboard CLOSED, to separate the dashboard's load from the
          pipeline's own latency (chapter4_notes §3.6). Refuses --dashboard-open.

Order: repetition r uses model seed 42+r (NFR3; each repeat a different model). Within a
repetition every scenario step runs the three models interleaved IF, RF, LSTM, so heat
or background load does not favour one model (plan §8.2). The record sample is the same
in every run (stream.sample_seed). A pause separates runs (--pause-s).

Checks before the first run (the matrix refuses to start otherwise):
  * socket buffers: net.core.wmem_max and rmem_max >= 4 MB and the probed SO_SNDBUF
    >= 4 MB (plan §6.2; the sysctl is not persistent, chapter4_notes §2.10); every
    streamer start reply is checked again
  * dashboard: --dashboard-open must match whether the dashboard answers on
    127.0.0.1:--dashboard-port; with it open, each run must also have rows in
    results/dashboard_delay.csv (a browser really refreshed during the run)
  * services idle, and the engine writes the same database this script reads
--dashboard-open is sent with prepare and recorded in TEST_ENVIRONMENT.run_conditions.

Resumable: a run is skipped when MODEL_RUN already holds it with end_reason='complete',
rate_changes IS NULL, as many PREDICTION rows as n_records, and the same dashboard
state (and, if open, dashboard-delay rows). A power cut leaves the run in progress
with end_reason NULL, so only that run is repeated; Ctrl+C stops the active run
cleanly (end_reason 'stopped') and it is repeated next time.

Usage:
  python experiments/run_matrix.py --dry-run                 # both sets, status, time
  python experiments/run_matrix.py --set main --dashboard-open
  python experiments/run_matrix.py --set closed [--launch]   # --launch: start services
Smoke test (labelled runs that never count as experiment runs):
  python experiments/run_matrix.py --set closed --launch --db /tmp/x.db --limit 2 \\
      --n-override 200 --scenario-prefix smoke- --pause-s 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nids.control.client import ControlClient, ControlError  # noqa: E402
from nids.stream.netconf import SEND_BUFFER, effective_socket_buffers  # noqa: E402

MODELS = ("iforest", "rf", "lstm")            # interleaving order (plan §8.2)
TASK = "binary"                               # all timing runs use binary models
SETS = ("main", "closed")

# Rough per-run overheads for the time estimate (seconds): untimed model load at
# prepare (LSTM: Keras load + tf.function trace), sample draw + 50 ms lead-in, and
# per-record work outside t_inf (t_pre + t_ser + t_tx, ms; chapter4_notes §3.6).
PREPARE_S = {"iforest": 2.0, "rf": 4.0, "lstm": 10.0}
START_S = 1.0
EXTRA_MS = 3.0
DEFAULT_INF_MS = {"iforest": 12.0, "rf": 38.0, "lstm": 7.0}


@dataclass(frozen=True)
class Step:
    scenario: str
    dt_ms: float
    mode: str
    n: int


@dataclass(frozen=True)
class Run:
    set: str
    rep: int
    seed: int
    model: str
    step: Step
    dashboard_open: bool

    @property
    def key(self) -> tuple:
        s = self.step
        return (s.scenario, self.model, TASK, self.seed, float(s.dt_ms), s.mode, s.n,
                self.dashboard_open)


def scenario_steps(cfg: dict, n_override: int | None = None, prefix: str = "") -> dict:
    """Table 3.13 as Steps, keyed by scenario id."""
    st = cfg["stream"]
    ns, nb = (n_override or st["steady_n"]), (n_override or st["burst_n"])
    return {
        "S1": [Step(prefix + "S1", 100.0, "steady", ns)],
        "S2": [Step(prefix + "S2", 10.0, "steady", ns)],
        "S3": [Step(prefix + "S3", dt, "steady", ns) for dt in (50.0, 20.0, 5.0, 1.0)],
        "S4": [Step(prefix + "S4", 100.0, "burst", nb)],
    }


def build_plan(cfg: dict, which: str, models=MODELS, seeds=None, scenarios=None,
               n_override: int | None = None, prefix: str = "") -> list[Run]:
    """Ordered run list: repetition (seed) -> scenario step -> model, interleaved."""
    steps = scenario_steps(cfg, n_override, prefix)
    wanted = {"main": ["S1", "S2", "S3", "S4"], "closed": ["S1"]}[which]
    if scenarios:
        wanted = [s for s in wanted if s in scenarios]
    seeds = list(seeds or cfg["seeds"])
    models = [m for m in MODELS if m in models]
    plan = []
    for rep, seed in enumerate(seeds):
        for sc in wanted:
            for step in steps[sc]:
                for m in models:
                    plan.append(Run(which, rep + 1, seed, m, step, which == "main"))
    return plan


def burst_block(cfg: dict, n: int) -> int:
    """Back-to-back block size; scaled with n when --n-override shrinks a burst run."""
    st = cfg["stream"]
    return st["burst_block"] if n == st["burst_n"] else max(1, n * st["burst_block"]
                                                            // st["burst_n"])


# ---------------------------------------------------------------- time estimate
def inference_ms(path: Path = ROOT / "results" / "detector_timing.json") -> dict:
    try:
        det = json.loads(path.read_text())["detectors"]
        return {m: float(det[m]["mean_ms"]) for m in MODELS}
    except (OSError, KeyError, ValueError):
        return dict(DEFAULT_INF_MS)


def estimate_s(run: Run, cfg: dict, inf_ms: dict) -> float:
    """Wall time of one run: the schedule, or the engine's service time if that is
    longer (a backlog drains after the last emission), plus untimed set-up."""
    s, svc = run.step, (inf_ms[run.model] + EXTRA_MS) / 1000.0
    dt = s.dt_ms / 1000.0
    if s.mode == "steady":
        span = max((s.n - 1) * dt, s.n * svc)
    else:
        block = burst_block(cfg, s.n)
        lead = (s.n - block) // 2
        tail = s.n - lead - block
        span = lead * dt + max(tail * dt, (block + tail) * svc)
    return PREPARE_S[run.model] + START_S + span


def fmt_dur(sec: float) -> str:
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


# ---------------------------------------------------------------- resume state
def delay_run_ids(csv_path: Path) -> set[int]:
    """run_ids with at least one dashboard-delay row (a browser refreshed during them)."""
    try:
        with open(csv_path, newline="") as f:
            return {int(r["run_id"]) for r in csv.DictReader(f) if r.get("run_id")}
    except (OSError, ValueError, KeyError):
        return set()


def completed_runs(db: Path, delay_ids: set[int]) -> dict[tuple, int]:
    """Run.key -> run_id for every finished, unmarked, complete run in the database."""
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT r.run_id, r.scenario, r.model_name, r.task, r.random_seed, "
            "r.emission_interval_ms, r.load_condition, r.n_records, "
            "json_extract(e.run_conditions, '$.dashboard_open') "
            "FROM model_run r JOIN test_environment e ON e.env_id = r.env_id "
            "WHERE r.end_reason = 'complete' AND r.rate_changes IS NULL "
            "AND r.n_records = (SELECT count(*) FROM prediction p "
            "                   WHERE p.run_id = r.run_id) "
            "ORDER BY r.run_id").fetchall()
    except sqlite3.OperationalError:  # no schema yet / pre-Phase 5 database
        return {}
    finally:
        con.close()
    done = {}
    for run_id, sc, model, task, seed, dt, mode, n, dash in rows:
        if dash is None:
            continue  # dashboard state not declared: not a matrix run
        if dash == 1 and run_id not in delay_ids:
            continue  # "open" but no browser refreshed during it
        done[(sc, model, task, seed, float(dt), mode, n, bool(dash))] = run_id
    return done


# ---------------------------------------------------------------- preflight checks
def check_buffers() -> tuple[bool, dict, list[str]]:
    b = effective_socket_buffers()
    problems = []
    for k in ("net_core_wmem_max", "net_core_rmem_max"):
        if b[k] is not None and b[k] < SEND_BUFFER:
            problems.append(f"{k.replace('_', '.', 2)} = {b[k]} < {SEND_BUFFER}")
    if b["so_sndbuf_effective"] < SEND_BUFFER:
        problems.append(f"probed SO_SNDBUF = {b['so_sndbuf_effective']} < {SEND_BUFFER}")
    return not problems, b, problems


def dashboard_up(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health",
                                    timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


# ---------------------------------------------------------------- service launch
class Launched:
    """run.py started by this script (only with --launch); stopped on exit."""

    def __init__(self, db: Path, dashboard: bool, ports: dict):
        args = [sys.executable, str(ROOT / "run.py"), "--db", str(db),
                "--data-port", str(ports["data"]),
                "--engine-ctl-port", str(ports["engine"]),
                "--streamer-ctl-port", str(ports["streamer"])]
        if not dashboard:
            args.append("--no-dashboard")
        self.log = open(db.with_suffix(".services.log"), "a")
        self.proc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, bufsize=1,
                                     start_new_session=True)
        self.ready: dict | None = None
        self._ev = threading.Event()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self.log.write(line)
            self.log.flush()
            if self.ready is None and line.startswith("{"):
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("event") == "ready" and ev.get("service") == "run":
                    self.ready = ev
                    self._ev.set()
        self._ev.set()

    def wait_ready(self, timeout: float = 240.0) -> dict:
        if not self._ev.wait(timeout) or self.ready is None:
            raise RuntimeError(f"run.py did not start (exit {self.proc.poll()}); "
                               f"see {self.log.name}")
        return self.ready

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)  # run.py stops any run cleanly
            try:
                self.proc.wait(90)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


# ---------------------------------------------------------------- output
def print_plan(plans: list[list[Run]], cfg: dict, done: dict, inf_ms: dict,
               pause_s: float) -> None:
    for plan in plans:
        if not plan:
            continue
        which = plan[0].set
        dash = "OPEN" if plan[0].dashboard_open else "CLOSED"
        print(f"\n=== set '{which}' (dashboard {dash}) - {len(plan)} runs ===")
        print(f"{'#':>4} {'rep':>3} {'scen':<10} {'model':<8} {'seed':>4} {'dt_ms':>6} "
              f"{'mode':<6} {'n':>5} {'est':>7}  status")
        totals: dict[str, list[float]] = {}
        pend_s, pend_n = 0.0, 0
        for i, r in enumerate(plan, 1):
            est = estimate_s(r, cfg, inf_ms)
            rid = done.get(r.key)
            status = f"done (run {rid})" if rid else "pending"
            if not rid:
                pend_s, pend_n = pend_s + est, pend_n + 1
            t = totals.setdefault(f"{r.step.scenario} dt={r.step.dt_ms:g}", [0, 0.0])
            t[0] += 1
            t[1] += est
            print(f"{i:>4} {r.rep:>3} {r.step.scenario:<10} {r.model:<8} {r.seed:>4} "
                  f"{r.step.dt_ms:>6g} {r.step.mode:<6} {r.step.n:>5} "
                  f"{fmt_dur(est):>7}  {status}")
        print(f"\n  per scenario step (runs, est. run time):")
        for k, (cnt, sec) in totals.items():
            print(f"    {k:<16} {cnt:>3} runs  {fmt_dur(sec):>8}")
        all_s = sum(estimate_s(r, cfg, inf_ms) for r in plan)
        pauses = max(0, pend_n - 1) * pause_s
        print(f"  set total: {len(plan)} runs, {fmt_dur(all_s)} of runs "
              f"+ {fmt_dur(max(0, len(plan) - 1) * pause_s)} pauses "
              f"= {fmt_dur(all_s + max(0, len(plan) - 1) * pause_s)}")
        print(f"  remaining: {pend_n} runs, {fmt_dur(pend_s)} + {fmt_dur(pauses)} pauses "
              f"= {fmt_dur(pend_s + pauses)}")


def append_log(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    net = cfg["network"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=SETS, help="required unless --dry-run")
    ap.add_argument("--dashboard-open", action="store_true",
                    help="the dashboard is open in a browser (recorded in run_conditions)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the run list, resume status and time estimate, then exit")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default=None, help="comma list (default config seeds)")
    ap.add_argument("--scenarios", default=None, help="e.g. S1,S4 (default: all in set)")
    ap.add_argument("--pause-s", type=float, default=20.0, help="pause between runs")
    ap.add_argument("--limit", type=int, default=None, help="run at most N pending runs")
    ap.add_argument("--n-override", type=int, default=None,
                    help="records per run (smoke tests only; needs --scenario-prefix)")
    ap.add_argument("--scenario-prefix", default="",
                    help="label test runs (e.g. smoke-) so they never count as S1-S4")
    ap.add_argument("--db", default=str(ROOT / cfg["paths"]["db"]))
    ap.add_argument("--delay-csv", default=str(ROOT / "results" / "dashboard_delay.csv"))
    ap.add_argument("--log", default=str(ROOT / "results" / "run_matrix_log.csv"))
    ap.add_argument("--launch", action="store_true",
                    help="start run.py (services, plus dashboard with --dashboard-open) "
                         "if the services are not already running")
    ap.add_argument("--engine-ctl-port", type=int, default=net["engine_ctl_port"])
    ap.add_argument("--streamer-ctl-port", type=int, default=net["streamer_ctl_port"])
    ap.add_argument("--data-port", type=int, default=net["data_port"])
    ap.add_argument("--dashboard-port", type=int, default=8501)
    a = ap.parse_args(argv)

    if a.n_override and not a.scenario_prefix:
        ap.error("--n-override changes the experiment; label such runs with "
                 "--scenario-prefix")
    if not a.dry_run and not a.set:
        ap.error("--set is required (main or closed)")
    test_runs = bool(a.scenario_prefix)
    if not test_runs and not a.dry_run:
        if a.set == "main" and not a.dashboard_open:
            ap.error("set 'main' runs with the dashboard open (§3.3.7): open it and pass "
                     "--dashboard-open")
        if a.set == "closed" and a.dashboard_open:
            ap.error("set 'closed' is the dashboard-closed comparison: close the "
                     "dashboard and drop --dashboard-open")

    db = Path(a.db).resolve()
    delay_csv = Path(a.delay_csv)
    models = a.models.split(",")
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    scen = a.scenarios.split(",") if a.scenarios else None
    sets = [a.set] if a.set else list(SETS)
    plans = [build_plan(cfg, s, models, seeds, scen, a.n_override, a.scenario_prefix)
             for s in sets]
    if a.set and a.scenario_prefix:  # test runs: dashboard state as declared
        plans = [[Run(r.set, r.rep, r.seed, r.model, r.step, a.dashboard_open)
                  for r in p] for p in plans]
    inf_ms = inference_ms()
    done = completed_runs(db, delay_run_ids(delay_csv))
    ok_buf, buf, problems = check_buffers()

    if a.dry_run:
        print(f"database: {db}  ({len(done)} completed matrix runs found)")
        print(f"inference times for the estimate (mean ms): "
              + ", ".join(f"{m} {v:.1f}" for m, v in inf_ms.items())
              + f"; + {EXTRA_MS:g} ms per record outside t_inf; pause {a.pause_s:g} s")
        print(f"socket buffers: wmem_max={buf['net_core_wmem_max']} "
              f"rmem_max={buf['net_core_rmem_max']} "
              f"SO_SNDBUF={buf['so_sndbuf_effective']} -> "
              + ("OK" if ok_buf else "WOULD REFUSE: " + "; ".join(problems)))
        print(f"dashboard on 127.0.0.1:{a.dashboard_port}: "
              + ("up" if dashboard_up(a.dashboard_port) else "not running"))
        print_plan(plans, cfg, done, inf_ms, a.pause_s)
        if len(plans) > 1:
            tot = sum(estimate_s(r, cfg, inf_ms) for p in plans for r in p)
            nruns = sum(len(p) for p in plans)
            print(f"\nALL SETS: {nruns} runs, {fmt_dur(tot)} of runs + "
                  f"{fmt_dur((nruns - len(plans)) * a.pause_s)} pauses = "
                  f"{fmt_dur(tot + (nruns - len(plans)) * a.pause_s)}")
        return 0

    plan = plans[0]
    pending = [r for r in plan if r.key not in done]
    n_done, n_pending = len(plan) - len(pending), len(pending)
    if a.limit is not None:
        pending = pending[:a.limit]
    print(f"[matrix] set '{a.set}': {len(plan)} runs, {n_done} already complete, "
          f"{n_pending} pending, {len(pending)} to do now", flush=True)
    if not pending:
        return 0

    # ---- preflight (refuse rather than produce runs that cannot be used)
    if not ok_buf:
        print("[matrix] REFUSING: socket buffers below 4 MB: " + "; ".join(problems) +
              f"\n  fix: sudo sysctl -w net.core.wmem_max={2 * SEND_BUFFER} "
              f"net.core.rmem_max={2 * SEND_BUFFER}", file=sys.stderr)
        return 2
    launched = None
    client = ControlClient(a.engine_ctl_port, a.streamer_ctl_port)
    try:
        client.status()
    except OSError:
        if not a.launch:
            print("[matrix] REFUSING: engine/streamer not running; start `python run.py`"
                  " or pass --launch", file=sys.stderr)
            return 2
        print(f"[matrix] launching run.py (db {db})", flush=True)
        launched = Launched(db, a.dashboard_open, {"data": a.data_port,
                                                   "engine": a.engine_ctl_port,
                                                   "streamer": a.streamer_ctl_port})
        ev = launched.wait_ready()
        client = ControlClient(ev["engine_ctl_port"], ev["streamer_ctl_port"])
        if a.dashboard_open:  # give the launched dashboard time to come up
            t_end = time.monotonic() + 60
            while not dashboard_up(a.dashboard_port) and time.monotonic() < t_end:
                time.sleep(1)

    interrupted = False

    def _term(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    code = 0
    try:
        up = dashboard_up(a.dashboard_port)
        if up != a.dashboard_open:
            msg = (f"--dashboard-open given but nothing answers on 127.0.0.1:"
                   f"{a.dashboard_port}" if a.dashboard_open else
                   f"a dashboard answers on 127.0.0.1:{a.dashboard_port} but "
                   f"--dashboard-open was not given")
            if not test_runs:
                print(f"[matrix] REFUSING: {msg}", file=sys.stderr)
                return 2
            print(f"[matrix] warning: {msg} (labelled test runs: continuing)", flush=True)
        st = client.status()
        eng_db = st["engine"].get("db")
        if eng_db is None or Path(eng_db).resolve() != db:
            print(f"[matrix] REFUSING: the engine writes {eng_db or '(unknown: restart '
                  'the services so they report their database)'}, but resume state is "
                  f"read from {db}", file=sys.stderr)
            return 2
        if st["engine"]["state"] != "idle" or st["streamer"]["state"] != "idle":
            print(f"[matrix] REFUSING: services busy (engine {st['engine']['state']}, "
                  f"streamer {st['streamer']['state']}); stop the active run first "
                  "(python -m nids.control.client stop)", file=sys.stderr)
            return 2
        print(f"[matrix] buffers OK (wmem_max={buf['net_core_wmem_max']}, "
              f"rmem_max={buf['net_core_rmem_max']}); dashboard "
              f"{'open' if a.dashboard_open else 'closed'}; pause {a.pause_s:g} s; "
              f"est. {fmt_dur(sum(estimate_s(r, cfg, inf_ms) for r in pending) + max(0, len(pending) - 1) * a.pause_s)}",
              flush=True)
        failures = 0
        for i, r in enumerate(pending, 1):
            if i > 1 and a.pause_s > 0:
                time.sleep(a.pause_s)
            s = r.step
            est = estimate_s(r, cfg, inf_ms)
            print(f"[matrix] {i}/{len(pending)} {s.scenario} {r.model} seed {r.seed} "
                  f"dt {s.dt_ms:g} ms {s.mode} n {s.n} (est. {fmt_dur(est)})", flush=True)
            t_start = time.monotonic()
            started_at = datetime.now(timezone.utc)
            run_id = client.prepare(r.model, TASK, r.seed, s.dt_ms, s.mode, s.scenario,
                                    s.n, dashboard_open=r.dashboard_open)
            try:
                rep = client.start(run_id, s.dt_ms, s.mode, s.n,
                                   cfg["stream"]["sample_seed"],
                                   burst_block(cfg, s.n) if s.mode == "burst" else None)
            except Exception:
                client.engine("stop")  # cancel the prepared run, engine back to idle
                raise
            if rep.get("sndbuf_warning"):
                client.stop()
                client.wait_done(run_id, timeout=120)
                print(f"[matrix] REFUSING: streamer was granted only "
                      f"{rep.get('sndbuf_granted')} bytes of send buffer (run {run_id} "
                      "stopped)", file=sys.stderr)
                return 2
            eng = client.wait_done(run_id, timeout=3 * est + 300)
            took = time.monotonic() - t_start
            strm = client.streamer("status").get("last_run") or {}
            refreshes = None
            if r.dashboard_open:
                time.sleep(1.5)  # let the last refresh land
                try:
                    with open(delay_csv, newline="") as f:
                        refreshes = sum(1 for row in csv.DictReader(f)
                                        if row.get("run_id") == str(run_id)
                                        and datetime.fromisoformat(row["wall_time_utc"])
                                        >= started_at)
                except OSError:
                    refreshes = 0
            ok = (eng.get("end_reason") == "complete" and eng.get("processed") == s.n
                  and eng.get("malformed") == 0)
            lat = strm.get("lateness_ms") or {}
            print(f"[matrix]   run {run_id}: {eng.get('end_reason')}, processed "
                  f"{eng.get('processed')}/{s.n}, malformed {eng.get('malformed')}, "
                  f"took {fmt_dur(took)}, pacing lateness p99 "
                  f"{lat.get('p99', float('nan')):.2f} ms"
                  + (f", dashboard refreshes {refreshes}" if refreshes is not None else "")
                  + ("" if ok else "  <-- NOT COMPLETE, will be repeated"), flush=True)
            append_log(Path(a.log), {
                "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "db": str(db), "run_id": run_id, "set": r.set, "rep": r.rep,
                "scenario": s.scenario, "model": r.model, "seed": r.seed,
                "dt_ms": s.dt_ms, "mode": s.mode, "n": s.n,
                "dashboard_open": r.dashboard_open, "end_reason": eng.get("end_reason"),
                "processed": eng.get("processed"), "malformed": eng.get("malformed"),
                "wall_s": round(took, 1), "est_s": round(est, 1),
                "lateness_p99_ms": lat.get("p99"), "dashboard_refreshes": refreshes})
            if not ok:
                failures += 1
            if r.dashboard_open and not test_runs and not refreshes:
                print("[matrix] STOPPING: the dashboard is up but no browser refreshed "
                      f"during run {run_id} (no rows in {delay_csv.name}); open "
                      f"http://127.0.0.1:{a.dashboard_port} and rerun. Run {run_id} will "
                      "be repeated.", file=sys.stderr)
                return 3
        print(f"[matrix] finished: {len(pending)} runs, {failures} not complete",
              flush=True)
        code = 1 if failures else 0
    except KeyboardInterrupt:
        interrupted = True
        print("\n[matrix] interrupted: stopping the active run cleanly", flush=True)
        try:
            active = client.engine("status", timeout=3).get("run")
            if active:
                client.stop()
                client.wait_done(active["run_id"], timeout=60)
                print(f"[matrix] run {active['run_id']} stopped (end_reason 'stopped'; "
                      "it will be repeated on resume)", flush=True)
        except (OSError, ControlError, TimeoutError) as e:
            print(f"[matrix] could not stop cleanly: {e}", flush=True)
        code = 130
    except (OSError, ControlError, TimeoutError) as e:
        print(f"[matrix] ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        code = 1
    finally:
        if launched is not None:
            launched.stop()
    return code if not interrupted else 130


if __name__ == "__main__":
    sys.exit(main())
