"""Control-channel client shared by the dashboard (M6) and experiments/run_matrix.py
(M7/M8; plan §7.1, Chapter 3 §3.3.1).

A run is three commands (plan §7.1 table):
  1. engine.prepare  -> detector loaded (untimed), reset_state(), MODEL_RUN row, run_id
  2. streamer.start  -> data socket opened, emission begins
  3. the streamer sends eos when n records are sent (or on stop); the engine flushes t4
     and sets ended_at, then both services return to idle for the next run.
set_rate goes to the streamer (which re-paces) and then to the engine, which marks the
run (MODEL_RUN.rate_changes is not NULL) so it is never mistaken for an experiment run.

Ports come from config.yaml (network.*), overridable with the environment variables
NIDS_ENGINE_CTL_PORT / NIDS_STREAMER_CTL_PORT (run.py sets them for the dashboard).

CLI (handy for demos and smoke tests):
  python -m nids.control.client status
  python -m nids.control.client run --model rf --dt-ms 10 --n 200 [--scenario S2]
  python -m nids.control.client stop | set-rate 20
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

from nids.control import protocol
from nids.control.protocol import LOOPBACK

PREPARE_TIMEOUT_S = 180.0  # model load (LSTM: TF import + trace) is untimed but slow
CMD_TIMEOUT_S = 10.0


class ControlError(RuntimeError):
    """A service answered {"ok": false, ...}."""


def _check(reply: dict, what: str) -> dict:
    if not reply.get("ok"):
        raise ControlError(f"{what}: {reply.get('error', reply)}")
    return reply


class ControlClient:
    def __init__(self, engine_port: int = 9101, streamer_port: int = 9102,
                 host: str = LOOPBACK):
        if host != LOOPBACK:
            raise ValueError("the control channel is loopback only (NFR6)")
        self.host, self.engine_port, self.streamer_port = host, engine_port, streamer_port

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "ControlClient":
        if cfg is None:
            from nids.prep.convert_xlsx import ROOT
            cfg = yaml.safe_load(Path(ROOT, "config.yaml").read_text())
        net = cfg["network"]
        return cls(int(os.environ.get("NIDS_ENGINE_CTL_PORT", net["engine_ctl_port"])),
                   int(os.environ.get("NIDS_STREAMER_CTL_PORT", net["streamer_ctl_port"])),
                   net["host"])

    # ---- raw commands --------------------------------------------------------------
    def engine(self, cmd: str, timeout: float = CMD_TIMEOUT_S, **fields) -> dict:
        return protocol.request(self.engine_port, {"cmd": cmd, **fields}, self.host, timeout)

    def streamer(self, cmd: str, timeout: float = CMD_TIMEOUT_S, **fields) -> dict:
        return protocol.request(self.streamer_port, {"cmd": cmd, **fields}, self.host,
                                timeout)

    def status(self) -> dict:
        return {"engine": self.engine("status"), "streamer": self.streamer("status")}

    def wait_ready(self, timeout: float = 120.0) -> dict:
        """Block until both services answer status (used by run.py and tests)."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.status()
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)

    # ---- run lifecycle (plan §7.1) -------------------------------------------------
    def prepare(self, model: str, task: str = "binary", seed: int = 42,
                dt_ms: float = 100.0, mode: str = "steady", scenario: str | None = None,
                n: int | None = None) -> int:
        r = self.engine("prepare", timeout=PREPARE_TIMEOUT_S, model=model, task=task,
                        seed=seed, dt_ms=dt_ms, mode=mode, scenario=scenario, n=n)
        return int(_check(r, "prepare")["run_id"])

    def start(self, run_id: int, dt_ms: float, mode: str = "steady", n: int = 5000,
              sample_seed: int = 42, burst_block: int | None = None) -> dict:
        fields = dict(run_id=run_id, dt_ms=dt_ms, mode=mode, n=n, sample_seed=sample_seed)
        if burst_block is not None:
            fields["burst_block"] = burst_block
        return _check(self.streamer("start", timeout=60.0, **fields), "start")

    def stop(self) -> dict:
        """End the active run cleanly: the streamer sends eos; if it is not streaming, the
        engine cancels a prepared run (or ends one whose stream was lost)."""
        s = self.streamer("stop")
        if s.get("ok") and s.get("stopping"):
            return s
        return self.engine("stop")

    def set_rate(self, dt_ms: float) -> dict:
        """Demo-only rate change (FR2): re-pace the streamer, then mark the run."""
        s = _check(self.streamer("set_rate", dt_ms=dt_ms), "set_rate (streamer)")
        e = _check(self.engine("set_rate", dt_ms=dt_ms, from_seq=s["from_seq"]),
                   "set_rate (engine)")
        return {"ok": True, "from_seq": s["from_seq"], "dt_ms": dt_ms,
                "run_id": e.get("run_id")}

    def wait_done(self, run_id: int, timeout: float | None = None,
                  poll_s: float = 0.2) -> dict:
        """Wait until the engine has finished run_id; returns its last_run summary."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            st = self.engine("status")
            last = st.get("last_run") or {}
            if last.get("run_id") == run_id:
                return last
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"run {run_id} not finished; engine status: {st}")
            time.sleep(poll_s)

    def run(self, model: str, task: str = "binary", seed: int = 42, dt_ms: float = 100.0,
            mode: str = "steady", scenario: str | None = None, n: int = 5000,
            sample_seed: int = 42, burst_block: int | None = None,
            timeout: float | None = None) -> dict:
        """prepare -> start -> wait for eos. Returns {run_id, engine, streamer}."""
        run_id = self.prepare(model, task, seed, dt_ms, mode, scenario, n)
        try:
            self.start(run_id, dt_ms, mode, n, sample_seed, burst_block)
        except Exception:
            self.engine("stop")  # cancel the prepared run so the engine returns to idle
            raise
        eng = self.wait_done(run_id, timeout)
        st = self.streamer("status")
        return {"run_id": run_id, "engine": eng, "streamer": st.get("last_run")}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("status")
    sub.add_parser("stop")
    sr = sub.add_parser("set-rate")
    sr.add_argument("dt_ms", type=float)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--task", default="binary")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--dt-ms", type=float, default=100.0)
    r.add_argument("--mode", default="steady", choices=protocol.MODES)
    r.add_argument("--scenario", default=None)
    r.add_argument("--n", type=int, default=200)
    r.add_argument("--sample-seed", type=int, default=42)
    a = ap.parse_args(argv)
    c = ControlClient.from_config()
    if a.action == "status":
        out = c.status()
    elif a.action == "stop":
        out = c.stop()
    elif a.action == "set-rate":
        out = c.set_rate(a.dt_ms)
    else:
        out = c.run(a.model, a.task, a.seed, a.dt_ms, a.mode, a.scenario, a.n,
                    a.sample_seed)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
