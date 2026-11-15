"""Inference engine / consumer (M4; plan §6.3, Chapter 3 §3.3.1, §3.3.6, Fig. 3.7).

Processes 2.1-2.5 of Fig. 3.7 for each newline-delimited JSON record:
  2.1 receive          t1 = perf_counter_ns() as soon as the line is read
  2.2 validate         json.loads, required keys, row_index known to FLOW_RECORD
  2.3 preprocess       Preprocessor.transform (artefacts/preprocessor.json, FR6)
                       t2 after validate + scale
  2.4 detect           detector.predict(x)
                       t3 after predict
  2.5 store            LatencyLogger.log: one transaction per record; t4 is stamped after
                       COMMIT and written in the next transaction (plan §5.3)
Malformed records are counted and discarded without stopping the engine (FR5); the
count is written to MODEL_RUN.malformed_count when the run ends.

Before the first record (untimed): the detector is loaded and reset, the TEST_ENVIRONMENT
row is ensured, and a MODEL_RUN row records model, task, seed, dt, load condition,
scenario, n, hyperparameters and env_id (FR15, NFR3); run_conditions records the
effective socket buffers and wal_autocheckpoint. The socket binds 127.0.0.1 only
(NFR6). For now the run is configured by CLI arguments; the loopback control channel
(plan §7.1, M7) replaces this in Phase 5.

Single-thread inference (§3.3.10): nids.engine.detectors is imported FIRST, because it
sets the thread environment variables before NumPy/TensorFlow are imported.

Usage:
  python -m nids.engine.engine --model rf --task binary --seed 42 --dt-ms 100 \
      --mode steady --scenario S1 --n 5000 [--port 9100] [--db db/nids.db]
Prints {"event":"ready",...} once listening and {"event":"done",...} at the end.
"""
from __future__ import annotations

from nids.engine import detectors  # noqa: I001  (must be first: thread env vars)

import argparse  # noqa: E402
import json  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

from nids.engine.preprocessor import Preprocessor  # noqa: E402
from nids.prep.convert_xlsx import ROOT  # noqa: E402
from nids.storage import environment  # noqa: E402
from nids.storage.db import connect  # noqa: E402
from nids.storage.logger import LatencyLogger  # noqa: E402
from nids.stream.netconf import effective_socket_buffers  # noqa: E402

HOST = "127.0.0.1"  # loopback only (NFR6)


def hyperparameters(det: detectors.Detector) -> dict:
    """What MODEL_RUN stores about the loaded model (NFR3/NFR4)."""
    hp = {"config": det.config, "tau": det.tau}
    if hasattr(det, "score_max"):
        hp["val_score_max"] = det.score_max
    if hasattr(det, "F"):
        hp["formulation"] = "B"
    return hp


def ensure_env(db_path: Path, extra: dict) -> tuple[int, set[int]]:
    """env_id for this machine (FR15) and the set of known flow_ids (for validation).

    run_conditions also records, for every run, the effective socket buffer sizes and
    SQLite's wal_autocheckpoint (left at its default: the periodic checkpoint is a real
    storage cost and shows up in t_ser, so it is reported, not tuned away).
    """
    con = connect(db_path)  # same connect() as LatencyLogger, so the same PRAGMAs
    try:
        extra = {**extra, **effective_socket_buffers(),
                 "sqlite_wal_autocheckpoint":
                     con.execute("PRAGMA wal_autocheckpoint").fetchone()[0]}
        with con:
            env_id = environment.get_or_create_env(con, environment.collect(db_path, extra))
        flows = {r[0] for r in con.execute("SELECT flow_id FROM flow_record")}
    finally:
        con.close()
    if not flows:
        raise RuntimeError(f"{db_path} has no FLOW_RECORD rows; run nids.storage.init_db")
    return env_id, flows


def serve(conn: socket.socket, pre: Preprocessor, det: detectors.Detector,
          logger: LatencyLogger, run_id: int, flows: set[int]) -> dict:
    """The per-record loop of plan §6.3. Returns counts for the run."""
    processed = malformed = 0
    eos = False
    for line in conn.makefile("rb"):
        t1 = time.perf_counter_ns()                         # process 2.1
        try:                                                # process 2.2 + 2.3
            msg = json.loads(line)
            if msg.get("type") == "eos":
                eos = True
                break
            x = pre.transform(msg["features"])
            seq, row_index, t0 = int(msg["seq"]), int(msg["row_index"]), int(msg["t0"])
            if row_index not in flows:
                raise ValueError(f"unknown row_index {row_index}")
        except (ValueError, KeyError, TypeError, AttributeError):
            malformed += 1                                  # FR5: count, discard, carry on
            continue
        t2 = time.perf_counter_ns()
        res = det.predict(x)                                # process 2.4
        t3 = time.perf_counter_ns()
        logger.log(run_id, seq, row_index, res, t0, t1, t2, t3)  # process 2.5 -> 3.0
        processed += 1
    return {"processed": processed, "malformed": malformed, "eos": eos}


def run(model: str, task: str, seed: int, dt_ms: float, mode: str, scenario: str | None,
        n: int | None, port: int, db_path: Path, cfg: dict) -> dict:
    art = ROOT / cfg["paths"]["artefacts"]
    # ---- untimed set-up ------------------------------------------------------------
    det = detectors.make_detector(model, art, high_confidence=cfg["alerts"]["high_confidence"])
    det.load(seed, task)
    det.reset_state()
    pre = Preprocessor(art / "preprocessor.json")
    env_id, flows = ensure_env(db_path, {"dashboard_refresh_s": cfg["dashboard"]["refresh_s"]})
    logger = LatencyLogger(db_path, high_confidence=cfg["alerts"]["high_confidence"])
    run_id = logger.start_run(env_id, model, task, hyperparameters(det), seed, dt_ms,
                              mode, scenario, n)

    srv = socket.create_server((HOST, port))               # loopback only (NFR6)
    try:
        print(json.dumps({"event": "ready", "port": srv.getsockname()[1], "run_id": run_id,
                          "env_id": env_id}), flush=True)
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            out = serve(conn, pre, det, logger, run_id, flows)
            logger.finish_run(run_id, out["malformed"])     # flush t4, ended_at
        finally:
            conn.close()
    finally:
        srv.close()
        logger.close()
    return {"event": "done", "run_id": run_id, **out}


def main(argv: list[str] | None = None) -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(detectors.DETECTORS))
    ap.add_argument("--task", default="binary", choices=("binary", "multi"))
    ap.add_argument("--seed", type=int, default=42, choices=cfg["seeds"])
    ap.add_argument("--dt-ms", type=float, required=True)
    ap.add_argument("--mode", choices=("steady", "burst"), default="steady")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--n", type=int, default=None, help="records the streamer will send")
    ap.add_argument("--port", type=int, default=cfg["network"]["data_port"],
                    help="0 picks a free port (tests)")
    ap.add_argument("--db", default=str(ROOT / cfg["paths"]["db"]))
    a = ap.parse_args(argv)
    out = run(a.model, a.task, a.seed, a.dt_ms, a.mode, a.scenario, a.n, a.port,
              Path(a.db), cfg)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
