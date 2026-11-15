"""Streamer / producer (M3; plan §6.2, Chapter 3 §3.3.1, §3.3.6, §3.3.11; FR1-FR3).

Replays cleaned but UNSCALED rows of the held-out test partition to the inference
engine over a loopback TCP socket; the engine applies artefacts/preprocessor.json
(FR6). Each message carries only seq, row_index, the feature values and t0, so the true
label never leaves FLOW_RECORD (leakage impossible by construction).

Sample and order (plan §6.2). A stratified sample of n rows (on label_7, sample_seed)
is drawn, sorted by row_index, then replayed in a random order fixed by sample_seed (a
seeded permutation). Every run of a scenario therefore sends the same records in the
same order (NFR4).
  Justified deviation from §3.3.11 ("replayed in their original order"): the source
  sheet is sorted by class (99.92% of consecutive rows share a label, plan §3.1 step 3),
  so the original order would send long single-class blocks. The seeded shuffle keeps
  the replay reproducible and mixes classes as live traffic would. No model depends on
  order (IF, RF, LSTM formulation B; plan §4.3).

Schedule (absolute; pacing errors never accumulate): record i is due at
start + offset_i.
  steady  offset_i = i * dt
  burst   lead records at dt, then `block` records back-to-back, then the rest at dt
          (Table 3.13 with n = 2000: 500 @ dt, 1,000 back-to-back, 500 @ dt)
wait_until() sleeps coarsely, then spins for the last ~1.5 ms (plan §6.2).

Wire format: newline-delimited JSON (TCP is a byte stream), TCP_NODELAY on (no Nagle
delay), a large send buffer requested so a burst does not block the sender; a warning
is printed to stderr if the kernel grants less than 4 MB (nids.stream.netconf). t0 is
stamped immediately BEFORE json.dumps, so t_tx includes serialisation (§3.3.6).
The stream ends with {"type":"eos","reason":"complete"} (or "stopped" after a stop).

Service (plan §7.1). The streamer is a long-running process: test_raw.parquet is loaded
once, a control thread listens on 127.0.0.1:9102 (nids.control.protocol) and the main
thread emits one run per "start" command, then returns to idle.
  start     draw the sample, compute the schedule, connect to the engine, reply, emit
  stop      checked before each record; sends eos with reason "stopped"
  set_rate  demo only (FR2): re-anchors the steady schedule from the next record that
            is not yet scheduled (Pacer) and reports that seq as from_seq; the client
            passes it to the engine, which marks the run in MODEL_RUN.rate_changes.
            Refused in burst mode (the burst shape is the experiment).
  status    state, records sent, last run summary (pacing lateness, rate changes)

Usage:  python -m nids.stream.streamer [--data-port 9100] [--ctl-port 9102]
Prints {"event":"ready",...} once the control socket listens. SIGINT/SIGTERM stop the
active run (eos is still sent) and exit.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from nids.control import protocol
from nids.prep.convert_xlsx import ROOT
from nids.stream.netconf import SEND_BUFFER, sndbuf_warning

SPIN_NS = 1_500_000            # spin for the last 1.5 ms
SLEEP_ABOVE_NS = 2_000_000
LEAD_IN_NS = 50_000_000        # 50 ms between "start" and the first record
log = logging.getLogger("nids.streamer")


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def draw_sample(test_raw: pd.DataFrame, n: int, sample_seed: int) -> pd.DataFrame:
    """Stratified sample of n rows on label_7, then a seeded shuffle (plan §6.2)."""
    if not 0 < n <= len(test_raw):
        raise ValueError(f"n must be in 1..{len(test_raw)}, got {n}")
    if n == len(test_raw):
        sample = test_raw
    else:
        sample, _ = train_test_split(test_raw, train_size=n, stratify=test_raw["label_7"],
                                     random_state=sample_seed)
    sample = sample.sort_values("row_index")  # fixed base order, then permute
    perm = np.random.default_rng(sample_seed).permutation(len(sample))
    return sample.iloc[perm].reset_index(drop=True)


def schedule_offsets_ns(n: int, dt_ms: float, mode: str, block: int = 1000) -> np.ndarray:
    """Due time of each record relative to the start, in ns (absolute schedule)."""
    dt = int(round(dt_ms * 1_000_000))
    if mode == "steady":
        return np.arange(n, dtype=np.int64) * dt
    if mode != "burst":
        raise ValueError(f"mode must be steady or burst, got {mode!r}")
    if not 0 < block < n:
        raise ValueError(f"burst block ({block}) must be smaller than n ({n})")
    lead = (n - block) // 2
    tail = n - lead - block
    burst_at = lead * dt
    return np.concatenate([
        np.arange(lead, dtype=np.int64) * dt,               # lead @ dt
        np.full(block, burst_at, dtype=np.int64),           # back-to-back
        burst_at + np.arange(1, tail + 1, dtype=np.int64) * dt,  # tail @ dt
    ])


def wait_until(deadline_ns: int) -> None:
    """Sleep coarsely, then busy-wait the last ~1.5 ms (plan §6.2)."""
    while True:
        remaining = deadline_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > SLEEP_ABOVE_NS:
            time.sleep((remaining - SPIN_NS) / 1e9)


def encode(seq: int, row_index: int, features: list[float]) -> bytes:
    """One newline-delimited JSON message; t0 is stamped just before json.dumps."""
    msg = {"seq": seq, "row_index": row_index, "features": features}
    msg["t0"] = time.perf_counter_ns()  # stamp, THEN serialise (t_tx includes it)
    return (json.dumps(msg) + "\n").encode()


def eos_line(reason: str = "complete") -> bytes:
    """End-of-stream marker; reason is "complete" (n sent) or "stopped" (stop command)."""
    return protocol.encode({"type": "eos", "reason": reason})


EOS = eos_line("complete")


class Pacer:
    """Absolute schedule (plan §6.2) with demo-only rate changes (plan §7.1, FR2).

    due(i) is start + offset_i until set_rate(dt) is called; from then on record j is
    due at due(k) + (j - k) * dt, where k is the last record whose due time was already
    fixed. Errors never accumulate, and from_seq = k + 1 is exact. The lock is taken
    once per record BEFORE wait_until, i.e. before t0, so it is outside L.
    """

    def __init__(self, offsets_ns: np.ndarray, mode: str = "steady"):
        self.offsets, self.mode = offsets_ns, mode
        self.lock = threading.Lock()
        self.start_ns: int | None = None
        self.fixed = -1                     # last seq whose due time is fixed
        self.last_due: int | None = None
        self.anchor: tuple[int, int, int] | None = None   # (due_ns, seq, dt_ns)
        self.changes: list[dict] = []

    def begin(self, start_ns: int) -> None:
        with self.lock:
            self.start_ns = start_ns

    def due(self, i: int) -> int:
        with self.lock:
            if self.anchor is None:
                d = self.start_ns + int(self.offsets[i])
            else:
                a_ns, a_i, dt = self.anchor
                d = a_ns + (i - a_i) * dt
            self.fixed, self.last_due = i, d
            return d

    def set_rate(self, dt_ms: float) -> int:
        if self.mode != "steady":
            raise ValueError("set_rate is supported in steady mode only (the burst shape "
                             "is fixed by Table 3.13)")
        dt = int(round(dt_ms * 1_000_000))
        with self.lock:
            if self.start_ns is None:
                raise ValueError("the stream has not started yet")
            base = self.last_due if self.fixed >= 0 else self.start_ns - dt
            self.anchor = (base, self.fixed, dt)
            from_seq = self.fixed + 1
            self.changes.append({"from_seq": from_seq, "dt_ms": dt_ms})
            return from_seq


def connect(host: str, port: int, timeout_s: float = 30.0) -> socket.socket:
    """Connect to the engine (retrying until it listens), TCP_NODELAY + big send buffer."""
    if host != "127.0.0.1":
        raise ValueError("the data path is loopback only (NFR6)")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            break
        except (ConnectionRefusedError, OSError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUFFER)
    return sock


@dataclass
class Progress:
    sent: int = 0


def stream(sock: socket.socket, sample: pd.DataFrame, features: list[str],
           offsets_ns: np.ndarray, stop: threading.Event | None = None,
           pacer: Pacer | None = None, progress: Progress | None = None) -> dict:
    """Emit every row on schedule (until stop is set), then eos. Returns pacing stats."""
    # Everything that is not part of the measured path is prepared up front.
    rows = sample[features].to_numpy(dtype=np.float64).tolist()
    idx = sample["row_index"].astype(np.int64).tolist()
    pacer = pacer or Pacer(offsets_ns)
    progress = progress or Progress()
    late = np.zeros(len(rows), dtype=np.int64)
    start_ns = time.perf_counter_ns() + LEAD_IN_NS
    pacer.begin(start_ns)
    stopped = False
    sent = 0
    for i, (ri, vals) in enumerate(zip(idx, rows)):
        if stop is not None and stop.is_set():
            stopped = True
            break
        due = pacer.due(i)
        wait_until(due)
        late[i] = time.perf_counter_ns() - due
        sock.sendall(encode(i, ri, vals))
        sent = progress.sent = i + 1
    sock.sendall(eos_line("stopped" if stopped else "complete"))
    end_ns = time.perf_counter_ns()
    late = late[:sent] if sent else np.zeros(1, dtype=np.int64)
    return {"n_sent": sent, "stopped": stopped, "duration_s": (end_ns - start_ns) / 1e9,
            "rate_changes": pacer.changes or None,
            "lateness_ms": {"median": float(np.median(late) / 1e6),
                            "p99": float(np.percentile(late, 99) / 1e6),
                            "max": float(late.max() / 1e6)}}


# ---------------------------------------------------------------- service (plan §7.1)
@dataclass
class StreamRun:
    run_id: int
    dt_ms: float
    mode: str
    n: int
    sample_seed: int
    sample: pd.DataFrame = field(repr=False)
    offsets: np.ndarray = field(repr=False)
    pacer: Pacer = field(repr=False)
    sock: socket.socket = field(repr=False)
    sndbuf: int = 0
    state: str = "streaming"
    stop: threading.Event = field(default_factory=threading.Event)
    progress: Progress = field(default_factory=Progress)

    def describe(self) -> dict:
        return {"run_id": self.run_id, "dt_ms": self.dt_ms, "mode": self.mode,
                "n": self.n, "sample_seed": self.sample_seed, "sent": self.progress.sent,
                "rate_changes": self.pacer.changes or None}


class Streamer:
    def __init__(self, cfg: dict, data_port: int, ctl_port: int):
        self.cfg = cfg
        self.host = cfg["network"]["host"]
        self.data_port = data_port
        self.burst_block = cfg["stream"]["burst_block"]
        self.default_sample_seed = cfg["stream"]["sample_seed"]
        art = ROOT / cfg["paths"]["artefacts"]
        self.features = json.loads((art / "feature_list.json").read_text())
        self.test_raw = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "test_raw.parquet",
                                        columns=["row_index", "label_7", *self.features])
        self.jobs: queue.Queue = queue.Queue()
        self.lock = threading.RLock()
        self.active: StreamRun | None = None
        self.starting = False
        self.last_run: dict | None = None
        self.shutting_down = threading.Event()
        self.ctl = protocol.ControlServer(ctl_port, self.handle, protocol.STREAMER_CMDS,
                                          name="streamer-control")

    # ---- control thread ------------------------------------------------------------
    def handle(self, cmd: dict) -> dict:
        c = cmd["cmd"]
        if c == "status":
            with self.lock:
                run = self.active
                state = run.state if run else ("starting" if self.starting else "idle")
                return {"ok": True, "service": "streamer", "pid": os.getpid(),
                        "state": state, "run": run.describe() if run else None,
                        "last_run": self.last_run}
        if c == "stop":
            with self.lock:
                run = self.active
                if run is None:
                    return {"ok": True, "stopping": False, "state": "idle"}
                run.stop.set()
            log.info("stop requested for run %d", run.run_id)
            return {"ok": True, "stopping": True, "run_id": run.run_id}
        if c == "set_rate":
            with self.lock:
                run = self.active
                if run is None:
                    return {"ok": False, "error": "no run is streaming"}
                from_seq = run.pacer.set_rate(cmd["dt_ms"])  # ValueError -> ok:false
            log.warning("set_rate dt_ms=%s from seq %d in run %d: demo only",
                        cmd["dt_ms"], from_seq, run.run_id)
            return {"ok": True, "run_id": run.run_id, "from_seq": from_seq,
                    "dt_ms": cmd["dt_ms"]}
        # start: checked here, executed on the main thread
        with self.lock:
            if self.shutting_down.is_set():
                return {"ok": False, "error": "streamer is shutting down"}
            if self.starting or self.active is not None:
                return {"ok": False, "error": "busy: a run is streaming"}
            self.starting = True
        reply: queue.Queue = queue.Queue(maxsize=1)
        self.jobs.put((cmd, reply))
        return reply.get()

    def request_shutdown(self, *_sig) -> None:
        self.shutting_down.set()
        with self.lock:
            if self.active is not None:
                self.active.stop.set()

    # ---- main thread ---------------------------------------------------------------
    def serve_forever(self) -> None:
        self.ctl.start()
        print(json.dumps({"event": "ready", "service": "streamer", "pid": os.getpid(),
                          "ctl_port": self.ctl.port, "data_port": self.data_port}),
              flush=True)
        log.info("streamer ready: control 127.0.0.1:%d, engine data port %d",
                 self.ctl.port, self.data_port)
        try:
            while not self.shutting_down.is_set():
                try:
                    cmd, reply = self.jobs.get(timeout=0.5)
                except queue.Empty:
                    continue
                run = None
                try:
                    run = self._setup(cmd)
                except Exception as e:  # noqa: BLE001  report, stay up
                    log.exception("start failed")
                    reply.put({"ok": False, "error": f"{type(e).__name__}: {e}"})
                    continue
                finally:
                    with self.lock:
                        self.starting = False
                        self.active = run
                warning = sndbuf_warning(run.sndbuf)
                if warning:
                    log.warning(warning)
                reply.put({"ok": True, "run_id": run.run_id, "n": run.n,
                           "sndbuf_granted": run.sndbuf,
                           "sndbuf_warning": warning is not None})
                self._execute(run)
        finally:
            while not self.jobs.empty():
                self.jobs.get_nowait()[1].put({"ok": False, "error": "streamer stopped"})
            self.ctl.close()
            log.info("streamer stopped")

    def _setup(self, cmd: dict) -> StreamRun:
        """Untimed: sample, schedule, connection (plan §7.1 step 2)."""
        mode, n = cmd.get("mode", "steady"), cmd["n"]
        seed = cmd.get("sample_seed", self.default_sample_seed)
        sample = draw_sample(self.test_raw, n, seed)
        offsets = schedule_offsets_ns(n, cmd["dt_ms"], mode,
                                      cmd.get("burst_block", self.burst_block))
        sock = connect(self.host, self.data_port, timeout_s=10.0)
        log.info("run %d: %d records, %s, dt=%sms, sample_seed=%d", cmd["run_id"], n,
                 mode, cmd["dt_ms"], seed)
        return StreamRun(cmd["run_id"], cmd["dt_ms"], mode, n, seed, sample, offsets,
                         Pacer(offsets, mode), sock,
                         sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))

    def _execute(self, run: StreamRun) -> None:
        summary = {"run_id": run.run_id, "mode": run.mode, "dt_ms": run.dt_ms,
                   "sample_seed": run.sample_seed, "sndbuf_requested": SEND_BUFFER,
                   "sndbuf_granted": run.sndbuf,
                   "sndbuf_warning": sndbuf_warning(run.sndbuf) is not None}
        try:
            summary |= stream(run.sock, run.sample, self.features, run.offsets,
                              run.stop, run.pacer, run.progress)
            run.sock.shutdown(socket.SHUT_WR)
            run.sock.recv(1)  # wait for the engine to close after eos (run finished)
        except OSError as e:
            log.error("run %d: data socket error: %s", run.run_id, e)
            summary |= {"n_sent": run.progress.sent, "error": str(e)}
        finally:
            run.sock.close()
            with self.lock:
                self.active = None
                self.last_run = summary
            log.info("run %d: sent %d records%s", run.run_id, summary.get("n_sent", 0),
                     " (stopped)" if summary.get("stopped") else "")


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    net = cfg["network"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-port", type=int, default=net["data_port"],
                    help="the engine's data port")
    ap.add_argument("--ctl-port", type=int, default=net["streamer_ctl_port"],
                    help="0 picks a free port (tests)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s streamer %(levelname)s %(message)s")
    st = Streamer(cfg, a.data_port, a.ctl_port)
    signal.signal(signal.SIGINT, st.request_shutdown)
    signal.signal(signal.SIGTERM, st.request_shutdown)
    st.serve_forever()


if __name__ == "__main__":
    main()
