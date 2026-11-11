"""Latency logger (M5; plan §5.2-§5.3, Chapter 3 §3.3.6 and §3.3.9, FR10-FR12, NFR6).

One transaction per record (process 2.5 -> data store 3.0). t4 is the moment the
commit has FINISHED, so it cannot be written in the same transaction: it is stamped
right after COMMIT, kept in pending_t4, and written by an UPDATE at the start of the
NEXT record's transaction. flush() writes the last pending t4 at the end of a run.
Consequence (state in Chapter Four): each record's commit also carries one small
UPDATE for the previous record, and the newest row has t4 = NULL until the next
transaction, which is why v_latency filters on t4_store_ns IS NOT NULL.

Alert rule (§5.2, FR10): an alert is raised when the binary decision is Malicious and
the record is not a warm-up record; severity = High if confidence >= 0.90 (config
alerts.high_confidence), otherwise Medium (NaN confidence -> Medium).

Every statement uses bound ? parameters (NFR6). The `res` argument is duck-typed:
any object with is_malicious, label, score, confidence and warmup (the Result
dataclass of plan §6.1, built in Phase 4).
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from nids.storage.db import connect

HIGH_CONFIDENCE = 0.90  # config.yaml alerts.high_confidence


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def severity_for(confidence, high_confidence: float = HIGH_CONFIDENCE) -> str:
    """High when confidence >= high_confidence, otherwise Medium (§5.2)."""
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return "Medium"
    return "High" if math.isfinite(c) and c >= high_confidence else "Medium"


def _real(v):
    """Float for a REAL column; NaN/inf -> NULL (e.g. warm-up rows)."""
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


class LatencyLogger:
    def __init__(self, db_path: str | Path, high_confidence: float = HIGH_CONFIDENCE):
        self.con = connect(db_path, autocommit=True)  # explicit transactions (§5.3)
        self.high_confidence = high_confidence
        self.pending_t4: list[tuple[int, int]] = []  # [(t4_ns, prediction_id)]

    # ---- MODEL_RUN lifecycle (FR15; plan §7.1 "prepare" / "stop") -----------------
    def start_run(self, env_id: int, model_name: str, task: str, hyperparameters: dict,
                  random_seed: int, emission_interval_ms: float, load_condition: str,
                  scenario: str | None, n_records: int | None) -> int:
        cur = self.con.execute(
            "INSERT INTO model_run (env_id, model_name, task, hyperparameters, random_seed, "
            "emission_interval_ms, load_condition, scenario, n_records, started_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (env_id, model_name, task, json.dumps(hyperparameters, sort_keys=True),
             random_seed, emission_interval_ms, load_condition, scenario, n_records,
             utc_now_iso()))
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, malformed_count: int) -> None:
        self.flush()
        self.con.execute("UPDATE model_run SET malformed_count=?, ended_at=? WHERE run_id=?",
                         (malformed_count, utc_now_iso(), run_id))

    # ---- per-record write (process 2.5) ------------------------------------------
    def log(self, run_id: int, seq_no: int, flow_id: int, res,
            t0: int, t1: int, t2: int, t3: int) -> int:
        cur = self.con.cursor()
        cur.execute("BEGIN")
        try:
            if self.pending_t4:  # previous record's t4
                cur.executemany("UPDATE prediction SET t4_store_ns=? WHERE prediction_id=?",
                                self.pending_t4)
            warmup = bool(res.warmup)
            cur.execute(
                "INSERT INTO prediction (run_id, flow_id, seq_no, is_warmup, predicted_label, "
                "anomaly_score, confidence, t0_emit_ns, t1_recv_ns, t2_prep_ns, t3_infer_ns) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, flow_id, seq_no, int(warmup), res.label, _real(res.score),
                 _real(res.confidence), t0, t1, t2, t3))
            pid = int(cur.lastrowid)
            if res.is_malicious and not warmup:  # FR10
                cur.execute(
                    "INSERT INTO alert (prediction_id, severity, raised_at) VALUES (?,?,?)",
                    (pid, severity_for(res.confidence, self.high_confidence), utc_now_iso()))
            cur.execute("COMMIT")
        except BaseException:
            cur.execute("ROLLBACK")  # pending_t4 kept: retried with the next record
            raise
        self.pending_t4.clear()
        self.pending_t4.append((time.perf_counter_ns(), pid))  # t4 = after commit
        return pid

    def flush(self) -> None:
        """Write the last pending t4 (call at the end of every run)."""
        if not self.pending_t4:
            return
        cur = self.con.cursor()
        cur.execute("BEGIN")
        try:
            cur.executemany("UPDATE prediction SET t4_store_ns=? WHERE prediction_id=?",
                            self.pending_t4)
            cur.execute("COMMIT")
        except BaseException:
            cur.execute("ROLLBACK")
            raise
        self.pending_t4.clear()

    def close(self) -> None:
        self.flush()
        self.con.close()
