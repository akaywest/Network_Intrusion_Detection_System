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
The stream ends with {"type":"eos"}.

Usage:
  python -m nids.stream.streamer --n 5000 --dt-ms 100 --mode steady [--port 9100]
Prints one JSON summary line at the end (n_sent, send buffer granted, pacing lateness).
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from nids.prep.convert_xlsx import ROOT
from nids.stream.netconf import SEND_BUFFER, sndbuf_warning

SPIN_NS = 1_500_000            # spin for the last 1.5 ms
SLEEP_ABOVE_NS = 2_000_000


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


EOS = b'{"type":"eos"}\n'


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


def stream(sock: socket.socket, sample: pd.DataFrame, features: list[str],
           offsets_ns: np.ndarray) -> dict:
    """Emit every row on schedule, then eos. Returns pacing statistics."""
    # Everything that is not part of the measured path is prepared up front.
    rows = sample[features].to_numpy(dtype=np.float64).tolist()
    idx = sample["row_index"].astype(np.int64).tolist()
    late = np.empty(len(rows), dtype=np.int64)
    start_ns = time.perf_counter_ns() + 50_000_000  # 50 ms lead-in
    for i, (ri, vals) in enumerate(zip(idx, rows)):
        due = start_ns + int(offsets_ns[i])
        wait_until(due)
        late[i] = time.perf_counter_ns() - due
        sock.sendall(encode(i, ri, vals))
    sock.sendall(EOS)
    end_ns = time.perf_counter_ns()
    return {"n_sent": len(rows), "duration_s": (end_ns - start_ns) / 1e9,
            "lateness_ms": {"median": float(np.median(late) / 1e6),
                            "p99": float(np.percentile(late, 99) / 1e6),
                            "max": float(late.max() / 1e6)}}


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    sc = cfg["stream"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=sc["steady_n"])
    ap.add_argument("--dt-ms", type=float, default=100.0)
    ap.add_argument("--mode", choices=("steady", "burst"), default="steady")
    ap.add_argument("--burst-block", type=int, default=sc["burst_block"])
    ap.add_argument("--sample-seed", type=int, default=sc["sample_seed"])
    ap.add_argument("--port", type=int, default=cfg["network"]["data_port"])
    a = ap.parse_args(argv)

    features = json.loads((ROOT / cfg["paths"]["artefacts"] / "feature_list.json").read_text())
    test_raw = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "test_raw.parquet",
                               columns=["row_index", "label_7", *features])
    sample = draw_sample(test_raw, a.n, a.sample_seed)
    offsets = schedule_offsets_ns(a.n, a.dt_ms, a.mode, a.burst_block)

    sock = connect(cfg["network"]["host"], a.port)
    try:
        sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        warning = sndbuf_warning(sndbuf)
        if warning:
            print(warning, file=sys.stderr, flush=True)
        out = stream(sock, sample, features, offsets)
        sock.shutdown(socket.SHUT_WR)
        sock.recv(1)  # wait for the engine to close after eos (run finished)
    finally:
        sock.close()
    print(json.dumps({"event": "done", "mode": a.mode, "dt_ms": a.dt_ms,
                      "sample_seed": a.sample_seed, "sndbuf_requested": SEND_BUFFER,
                      "sndbuf_granted": sndbuf, "sndbuf_warning": warning is not None,
                      **out}), flush=True)


if __name__ == "__main__":
    main()
