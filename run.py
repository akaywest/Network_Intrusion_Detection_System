"""One-command launcher (plan §7.1, NFR5: activate the venv, `python run.py`, press Start).

  1. creates db/nids.db with init_db if it does not exist yet (schema + FLOW_RECORD)
  2. starts the inference engine and the streamer as background services
     (python -m nids.engine.engine / nids.stream.streamer), each in its own session so
     Ctrl+C reaches only this launcher, and waits until both answer "status"
  3. runs `streamlit run src/nids/dashboard/app.py` bound to 127.0.0.1 (NFR6), headless
     (so Streamlit's first-run e-mail prompt never blocks the launch), and opens the
     browser itself once the dashboard answers its health check
  4. on Ctrl+C (or when the dashboard exits): stops any active run through the control
     channel (eos, t4 flushed, ended_at set), then stops the dashboard, the streamer and
     the engine in that order, escalating to kill only if one does not exit in time.

Options: --no-dashboard keeps only the two services running (used by run_matrix and the
tests; also chosen automatically while the dashboard does not exist yet), --db, and
port overrides (0 = pick free ports, tests). The dashboard receives NIDS_DB,
NIDS_ENGINE_CTL_PORT and NIDS_STREAMER_CTL_PORT in its environment.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nids.control.client import ControlClient  # noqa: E402

APP = ROOT / "src" / "nids" / "dashboard" / "app.py"
READY_TIMEOUT_S = 180.0


class Service:
    """A child process whose stdout is forwarded with a prefix; captures its ready line."""

    def __init__(self, name: str, args: list[str], env: dict):
        self.name = name
        self.proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     text=True, bufsize=1, start_new_session=True)
        self.ready: dict | None = None
        self._got_ready = threading.Event()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            if self.ready is None and line.startswith("{"):
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    ev = {}
                if ev.get("event") == "ready":
                    self.ready = ev
                    self._got_ready.set()
            print(f"[{self.name}] {line}", end="", flush=True)
        self._got_ready.set()  # process ended

    def wait_ready(self, timeout: float) -> dict:
        if not self._got_ready.wait(timeout) or self.ready is None:
            raise RuntimeError(f"{self.name} did not start (exit code {self.proc.poll()})")
        return self.ready

    def stop(self, timeout: float = 15.0) -> int | None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                print(f"[run] {self.name} did not exit in {timeout:.0f} s; killing",
                      flush=True)
                self.proc.kill()
                self.proc.wait()
        return self.proc.returncode


def ensure_db(cfg: dict, db: Path) -> None:
    if db.exists():
        return
    from nids.storage.init_db import init_db
    paths = cfg["paths"]
    print(f"[run] creating {db} (schema + FLOW_RECORD)", flush=True)
    init_db(db, ROOT / paths["processed"] / "test_raw.parquet",
            ROOT / paths["artefacts"] / "label_map.json",
            extra_conditions={"dashboard_refresh_s": cfg["dashboard"]["refresh_s"]})


def _sigterm(*_):
    raise KeyboardInterrupt  # `kill <pid>` shuts down exactly like Ctrl+C


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGTERM, _sigterm)
    # Ctrl+C -> KeyboardInterrupt even if SIGINT was inherited as ignored (e.g. when
    # launched as a background job from a script)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    net = cfg["network"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--db", default=str(ROOT / cfg["paths"]["db"]))
    ap.add_argument("--data-port", type=int, default=net["data_port"])
    ap.add_argument("--engine-ctl-port", type=int, default=net["engine_ctl_port"])
    ap.add_argument("--streamer-ctl-port", type=int, default=net["streamer_ctl_port"])
    ap.add_argument("--dashboard-port", type=int, default=8501)
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open a browser tab for the dashboard")
    a = ap.parse_args(argv)

    db = Path(a.db).resolve()
    ensure_db(cfg, db)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    py = sys.executable

    services: list[Service] = []
    dashboard: subprocess.Popen | None = None
    code = 0
    try:
        engine = Service("engine", [py, "-m", "nids.engine.engine", "--db", str(db),
                                    "--data-port", str(a.data_port),
                                    "--ctl-port", str(a.engine_ctl_port)], env)
        services.append(engine)
        e = engine.wait_ready(READY_TIMEOUT_S)
        streamer = Service("streamer", [py, "-m", "nids.stream.streamer",
                                        "--data-port", str(e["data_port"]),
                                        "--ctl-port", str(a.streamer_ctl_port)], env)
        services.append(streamer)
        s = streamer.wait_ready(READY_TIMEOUT_S)
        client = ControlClient(e["ctl_port"], s["ctl_port"])
        client.wait_ready(30)
        ports = {"data_port": e["data_port"], "engine_ctl_port": e["ctl_port"],
                 "streamer_ctl_port": s["ctl_port"]}
        print(json.dumps({"event": "ready", "service": "run", "pid": os.getpid(),
                          "engine_pid": e["pid"], "streamer_pid": s["pid"], **ports}),
              flush=True)

        show = not a.no_dashboard and APP.exists()
        if not a.no_dashboard and not APP.exists():
            print(f"[run] {APP.relative_to(ROOT)} not found: services only", flush=True)
        if show:
            dash_env = {**env, "NIDS_DB": str(db),
                        "NIDS_ENGINE_CTL_PORT": str(e["ctl_port"]),
                        "NIDS_STREAMER_CTL_PORT": str(s["ctl_port"])}
            dashboard = subprocess.Popen(
                [py, "-m", "streamlit", "run", str(APP),
                 "--server.address", "127.0.0.1",          # loopback only (NFR6)
                 "--server.port", str(a.dashboard_port),
                 "--server.headless", "true",
                 "--browser.gatherUsageStats", "false"],
                cwd=ROOT, env=dash_env, start_new_session=True,
                stdin=subprocess.DEVNULL)
            url = f"http://127.0.0.1:{a.dashboard_port}"
            if _wait_http(url + "/_stcore/health", 60, dashboard) and not a.no_browser:
                webbrowser.open(url)
            print(f"[run] dashboard: {url}  (Ctrl+C to quit)", flush=True)
            dashboard.wait()
        else:
            print("[run] services running (Ctrl+C to quit)", flush=True)
            while all(sv.proc.poll() is None for sv in services):
                time.sleep(0.5)
            code = 1  # a service died on its own
    except KeyboardInterrupt:
        print("\n[run] shutting down", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[run] error: {ex}", file=sys.stderr, flush=True)
        code = 1
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl+C cannot interrupt
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        _shutdown(services, dashboard)
    return code


def _wait_http(url: str, timeout: float, proc: subprocess.Popen) -> bool:
    """True once url answers 200 (Streamlit's health endpoint), False on timeout/exit."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.3)
    return False


def _shutdown(services: list[Service], dashboard: subprocess.Popen | None) -> None:
    by_name = {sv.name: sv for sv in services}
    eng, stm = by_name.get("engine"), by_name.get("streamer")
    if eng and stm and eng.ready and stm.ready and eng.proc.poll() is None:
        client = ControlClient(eng.ready["ctl_port"], stm.ready["ctl_port"])
        try:  # end any active run cleanly: eos -> engine flushes t4 and sets ended_at
            st = client.engine("status", timeout=3)
            run = st.get("run")
            if run:
                client.stop()
                client.wait_done(run["run_id"], timeout=60)
                print(f"[run] run {run['run_id']} stopped cleanly", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[run] could not stop the active run cleanly: {ex}", flush=True)
    if dashboard is not None and dashboard.poll() is None:
        dashboard.send_signal(signal.SIGTERM)
        try:
            dashboard.wait(10)
        except subprocess.TimeoutExpired:
            dashboard.kill()
    for name in ("streamer", "engine"):
        if name in by_name:
            rc = by_name[name].stop()
            print(f"[run] {name} stopped (exit code {rc})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
