"""Inference engine / consumer service (M4; plan §6.3 and §7.1, Chapter 3 §3.3.1,
§3.3.6, Fig. 3.7).

A long-running service. It listens on 127.0.0.1:9100 (data) and 127.0.0.1:9101
(control, nids.control.protocol) and serves any number of runs without restarting.

Threads:
  main     the data path and everything that touches the detector or SQLite: on a
           "prepare" handed over by the control thread it loads the detector (untimed),
           calls reset_state(), inserts MODEL_RUN and replies with run_id; it then
           accepts the streamer's connection and runs processes 2.1-2.5 for each record.
           Inference is therefore single-threaded (§3.3.10) and the LatencyLogger's
           connection is used by one thread only.
  control  answers status / stop / set_rate itself and passes prepare to the main
           thread. It never runs inference and never writes the database.

Per-record path (plan §6.3), unchanged from Phase 4:
  2.1 receive          t1 = perf_counter_ns() as soon as the line is read
  2.2 validate         json.loads, required keys, row_index known to FLOW_RECORD
  2.3 preprocess       Preprocessor.transform (artefacts/preprocessor.json, FR6)
                       t2 after validate + scale
  2.4 detect           detector.predict(x)
                       t3 after predict
  2.5 store            LatencyLogger.log: one transaction per record; t4 is stamped after
                       COMMIT and written in the next transaction (plan §5.3)
Malformed records are counted and discarded without stopping the engine (FR5).

A run ends when the streamer sends {"type":"eos","reason":...} (reason "complete" or
"stopped"), when the stream is lost ("disconnected"), when a prepared run is stopped
before it starts ("cancelled"), or when the service shuts down ("shutdown"). At the
end the last t4 is flushed and MODEL_RUN gets ended_at, malformed_count, end_reason and,
if a demo set_rate happened, rate_changes (plan §7.1: experiment runs keep dt fixed,
so they are the runs with rate_changes IS NULL).

Single-thread inference (§3.3.10): nids.engine.detectors is imported FIRST, because it
sets the thread environment variables before NumPy/TensorFlow are imported.

Usage:  python -m nids.engine.engine [--data-port 9100] [--ctl-port 9101] [--db db/nids.db]
Prints {"event":"ready",...} once both sockets listen. SIGINT/SIGTERM end the active run
cleanly (end_reason "shutdown") and exit.
"""
from __future__ import annotations

from nids.engine import detectors  # noqa: I001  (must be first: thread env vars)

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import queue  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

from nids.control import protocol  # noqa: E402
from nids.engine.preprocessor import Preprocessor  # noqa: E402
from nids.prep.convert_xlsx import ROOT  # noqa: E402
from nids.storage import environment  # noqa: E402
from nids.storage.db import connect, ensure_additions  # noqa: E402
from nids.storage.logger import LatencyLogger, utc_now_iso  # noqa: E402
from nids.stream.netconf import effective_socket_buffers  # noqa: E402

HOST = protocol.LOOPBACK  # loopback only (NFR6)
log = logging.getLogger("nids.engine")


def hyperparameters(det: detectors.Detector) -> dict:
    """What MODEL_RUN stores about the loaded model (NFR3/NFR4)."""
    hp = {"config": det.config, "tau": det.tau}
    if hasattr(det, "score_max"):
        hp["val_score_max"] = det.score_max
    if hasattr(det, "F"):
        hp["formulation"] = "B"
    return hp


def ensure_env(db_path: Path, extra: dict) -> int:
    """env_id for this machine (FR15), ensured at every prepare.

    run_conditions also records the effective socket buffer sizes and SQLite's
    wal_autocheckpoint (left at its default: the periodic checkpoint is a real storage
    cost and shows up in t_ser, so it is reported, not tuned away).
    """
    con = connect(db_path)  # same connect() as LatencyLogger, so the same PRAGMAs
    try:
        extra = {**extra, **effective_socket_buffers(),
                 "sqlite_wal_autocheckpoint":
                     con.execute("PRAGMA wal_autocheckpoint").fetchone()[0]}
        with con:
            return environment.get_or_create_env(con, environment.collect(db_path, extra))
    finally:
        con.close()


def load_flows(db_path: Path) -> set[int]:
    """Known flow_ids (validation, process 2.2); migrates older databases first."""
    con = connect(db_path)
    try:
        ensure_additions(con)
        flows = {r[0] for r in con.execute("SELECT flow_id FROM flow_record")}
    finally:
        con.close()
    if not flows:
        raise RuntimeError(f"{db_path} has no FLOW_RECORD rows; run nids.storage.init_db")
    return flows


@dataclass
class RunStats:
    processed: int = 0
    malformed: int = 0
    last_seq: int | None = None


def serve(conn: socket.socket, pre: Preprocessor, det: detectors.Detector,
          logger: LatencyLogger, run_id: int, flows: set[int],
          stats: RunStats | None = None) -> dict:
    """The per-record loop of plan §6.3. Returns counts and how the stream ended."""
    stats = stats if stats is not None else RunStats()
    end_reason = None
    try:
        for line in conn.makefile("rb"):
            t1 = time.perf_counter_ns()                         # process 2.1
            try:                                                # process 2.2 + 2.3
                msg = json.loads(line)
                if msg.get("type") == "eos":
                    end_reason = "stopped" if msg.get("reason") == "stopped" else "complete"
                    break
                x = pre.transform(msg["features"])
                seq, row_index, t0 = int(msg["seq"]), int(msg["row_index"]), int(msg["t0"])
                if row_index not in flows:
                    raise ValueError(f"unknown row_index {row_index}")
            except (ValueError, KeyError, TypeError, AttributeError):
                stats.malformed += 1                            # FR5: count, discard, carry on
                continue
            t2 = time.perf_counter_ns()
            res = det.predict(x)                                # process 2.4
            t3 = time.perf_counter_ns()
            logger.log(run_id, seq, row_index, res, t0, t1, t2, t3)  # process 2.5 -> 3.0
            stats.processed += 1                                # after t4: outside L
            stats.last_seq = seq
    except OSError:
        pass  # connection reset / shut down by stop: treated as a lost stream
    return {"processed": stats.processed, "malformed": stats.malformed,
            "eos": end_reason is not None, "end_reason": end_reason or "disconnected"}


@dataclass
class ActiveRun:
    run_id: int
    model: str
    task: str
    seed: int
    dt_ms: float
    mode: str
    scenario: str | None
    n: int | None
    det: detectors.Detector = field(repr=False)
    state: str = "prepared"                    # prepared -> running
    stats: RunStats = field(default_factory=RunStats)
    rate_changes: list[dict] = field(default_factory=list)
    cancel_reason: str | None = None
    conn: socket.socket | None = field(default=None, repr=False)

    def describe(self) -> dict:
        return {"run_id": self.run_id, "model": self.model, "task": self.task,
                "seed": self.seed, "dt_ms": self.dt_ms, "mode": self.mode,
                "scenario": self.scenario, "n": self.n, "processed": self.stats.processed,
                "malformed": self.stats.malformed, "rate_changed": bool(self.rate_changes)}


class Engine:
    def __init__(self, cfg: dict, db_path: Path, data_port: int, ctl_port: int):
        self.cfg, self.db_path = cfg, Path(db_path)
        self.art = ROOT / cfg["paths"]["artefacts"]
        self.high_confidence = cfg["alerts"]["high_confidence"]
        self.pre = Preprocessor(self.art / "preprocessor.json")
        self.flows = load_flows(self.db_path)
        self.logger = LatencyLogger(self.db_path, high_confidence=self.high_confidence)
        self.data_srv = socket.create_server((HOST, data_port))   # loopback only (NFR6)
        self.data_srv.settimeout(0.2)
        self.data_port = self.data_srv.getsockname()[1]
        self.jobs: queue.Queue = queue.Queue()
        self.lock = threading.RLock()  # RLock: the signal handler may re-enter on main
        self.active: ActiveRun | None = None
        self.preparing = False
        self.last_run: dict | None = None
        self.shutting_down = threading.Event()
        self.ctl = protocol.ControlServer(ctl_port, self.handle, protocol.ENGINE_CMDS,
                                          name="engine-control")

    # ---- control thread ------------------------------------------------------------
    def handle(self, cmd: dict) -> dict:
        c = cmd["cmd"]
        if c == "status":
            return self.status()
        if c == "stop":
            return self.stop()
        if c == "set_rate":
            return self.set_rate(cmd["dt_ms"], cmd.get("from_seq"))
        # prepare: checked here, executed on the main thread
        with self.lock:
            if self.shutting_down.is_set():
                return {"ok": False, "error": "engine is shutting down"}
            if self.preparing or self.active is not None:
                busy = self.active.run_id if self.active else "a new run"
                return {"ok": False, "error": f"busy: {busy} is not finished"}
            self.preparing = True
        reply: queue.Queue = queue.Queue(maxsize=1)
        self.jobs.put((cmd, reply))
        return reply.get()

    def status(self) -> dict:
        with self.lock:
            run = self.active
            state = run.state if run else ("preparing" if self.preparing else "idle")
            return {"ok": True, "service": "engine", "pid": os.getpid(), "state": state,
                    "data_port": self.data_port, "db": str(self.db_path.resolve()), "run": run.describe() if run else None,
                    "last_run": self.last_run}

    def stop(self, reason: str | None = None) -> dict:
        """Cancel a prepared run or end a running one (fallback if the stream is lost)."""
        with self.lock:
            run = self.active
            if run is None:
                return {"ok": True, "stopping": False, "state": "idle"}
            run.cancel_reason = reason or ("cancelled" if run.state == "prepared"
                                           else "stopped")
            conn = run.conn
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_RDWR)  # ends serve()'s read loop
            except OSError:
                pass
        return {"ok": True, "stopping": True, "run_id": run.run_id,
                "end_reason": run.cancel_reason}

    def set_rate(self, dt_ms: float, from_seq: int | None) -> dict:
        """Demo-only (plan §7.1): record the change; the run is marked in MODEL_RUN."""
        with self.lock:
            run = self.active
            if run is None:
                return {"ok": False, "error": "no active run"}
            run.rate_changes.append({"from_seq": from_seq, "dt_ms": dt_ms,
                                     "at": utc_now_iso()})
        log.warning("set_rate dt_ms=%s from_seq=%s during run %d: demo only, run marked "
                    "(MODEL_RUN.rate_changes) and excluded from experiments",
                    dt_ms, from_seq, run.run_id)
        return {"ok": True, "run_id": run.run_id, "marked": True}

    def request_shutdown(self, *_sig) -> None:
        self.shutting_down.set()
        self.stop("shutdown")

    # ---- main thread ---------------------------------------------------------------
    def serve_forever(self) -> None:
        self.ctl.start()
        print(json.dumps({"event": "ready", "service": "engine", "pid": os.getpid(),
                          "data_port": self.data_port, "ctl_port": self.ctl.port}),
              flush=True)
        log.info("engine ready: data 127.0.0.1:%d, control 127.0.0.1:%d",
                 self.data_port, self.ctl.port)
        try:
            while not self.shutting_down.is_set():
                try:
                    cmd, reply = self.jobs.get(timeout=0.5)
                except queue.Empty:
                    continue
                run = None
                try:
                    run = self._prepare(cmd)
                except Exception as e:  # noqa: BLE001  report, stay up
                    log.exception("prepare failed")
                    reply.put({"ok": False, "error": f"{type(e).__name__}: {e}"})
                    continue
                finally:
                    with self.lock:
                        self.preparing = False
                        self.active = run
                reply.put({"ok": True, "run_id": run.run_id})
                self._execute(run)
        finally:
            while not self.jobs.empty():
                self.jobs.get_nowait()[1].put({"ok": False, "error": "engine stopped"})
            self.ctl.close()
            self.data_srv.close()
            self.logger.close()
            log.info("engine stopped")

    def _prepare(self, cmd: dict) -> ActiveRun:
        """Untimed set-up for one run (plan §7.1 step 1)."""
        seed, task = cmd["seed"], cmd.get("task", "binary")
        if seed not in self.cfg["seeds"]:
            raise ValueError(f"seed must be one of {self.cfg['seeds']} (NFR3)")
        det = detectors.make_detector(cmd["model"], self.art,
                                      high_confidence=self.high_confidence)
        det.load(seed, task)
        det.reset_state()
        env_id = ensure_env(self.db_path, {
            "dashboard_refresh_s": self.cfg["dashboard"]["refresh_s"],
            "dashboard_open": cmd.get("dashboard_open")})  # None = not declared
        self._drain_backlog()
        mode, scenario, n = cmd.get("mode", "steady"), cmd.get("scenario"), cmd.get("n")
        run_id = self.logger.start_run(env_id, cmd["model"], task, hyperparameters(det),
                                       seed, cmd["dt_ms"], mode, scenario, n)
        log.info("prepared run %d: %s/%s seed=%d dt=%sms %s scenario=%s n=%s",
                 run_id, cmd["model"], task, seed, cmd["dt_ms"], mode, scenario, n)
        return ActiveRun(run_id, cmd["model"], task, seed, cmd["dt_ms"], mode, scenario,
                         n, det)

    def _drain_backlog(self) -> None:
        """Close any data connection queued while idle, so it cannot join this run."""
        self.data_srv.setblocking(False)
        try:
            while True:
                stale, _ = self.data_srv.accept()
                stale.close()
                log.warning("closed a stale data connection made while idle")
        except (BlockingIOError, OSError):
            pass
        finally:
            self.data_srv.settimeout(0.2)

    def _accept(self, run: ActiveRun) -> socket.socket | None:
        while run.cancel_reason is None and not self.shutting_down.is_set():
            try:
                conn, _ = self.data_srv.accept()
            except TimeoutError:
                continue
            conn.settimeout(None)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return conn
        return None

    def _execute(self, run: ActiveRun) -> None:
        out = {"eos": False, "end_reason": None}
        try:
            conn = self._accept(run)
            if conn is not None:
                with self.lock:
                    run.conn, run.state = conn, "running"
                    cancelled = run.cancel_reason is not None
                try:
                    if not cancelled:
                        out = serve(conn, self.pre, run.det, self.logger, run.run_id,
                                    self.flows, run.stats)
                finally:
                    conn.close()
        finally:
            reason = out["end_reason"] if out["eos"] else (run.cancel_reason or
                                                           "disconnected")
            self.logger.finish_run(run.run_id, run.stats.malformed, reason,
                                   run.rate_changes)  # flush t4, ended_at
            summary = {"run_id": run.run_id, "processed": run.stats.processed,
                       "malformed": run.stats.malformed, "eos": out["eos"],
                       "end_reason": reason, "rate_changes": run.rate_changes or None}
            with self.lock:
                self.active = None
                self.last_run = summary
            log.info("run %d ended (%s): processed=%d malformed=%d", run.run_id, reason,
                     run.stats.processed, run.stats.malformed)


def main(argv: list[str] | None = None) -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    net = cfg["network"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-port", type=int, default=net["data_port"],
                    help="0 picks a free port (tests)")
    ap.add_argument("--ctl-port", type=int, default=net["engine_ctl_port"])
    ap.add_argument("--db", default=str(ROOT / cfg["paths"]["db"]))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s engine %(levelname)s %(message)s")
    eng = Engine(cfg, Path(a.db), a.data_port, a.ctl_port)
    signal.signal(signal.SIGINT, eng.request_shutdown)
    signal.signal(signal.SIGTERM, eng.request_shutdown)
    eng.serve_forever()


if __name__ == "__main__":
    main()
