"""Create the NIDS log database (M5; plan §5.1 "Done when", Chapter 3 §3.3.9).

  1. Creates db/nids.db from schema.sql (WAL, foreign keys, five tables, index, v_latency).
  2. Loads FLOW_RECORD ONCE from data/processed/test_raw.parquet with flow_id = row_index,
     the display fields (protocol, Flow Duration, Total Fwd Packets, Flow Bytess) and the
     true label. Streamed messages then carry only row_index + features, so the label
     never reaches the engine (leakage impossible by construction).
  3. Inserts (or reuses an identical) TEST_ENVIRONMENT row (FR15).

Re-running on an existing database does not reload FLOW_RECORD; it only checks it and
ensures the environment row. --reset deletes the database (and its -wal/-shm files) first.

Usage:  python -m nids.storage.init_db [--config config.yaml] [--reset]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml

from nids.prep.convert_xlsx import ROOT
from nids.prep.prepare_data import DISPLAY_COLS
from nids.storage import environment
from nids.storage.db import SCHEMA_PATH, connect

# IANA protocol numbers -> FLOW_RECORD.protocol (Table 3.10; dashboard protocol mix, FR13).
# IDS2025's test partition holds 6, 17 and 0 (HOPOPT / unset); anything else is "Other".
PROTOCOL_NAMES = {6: "TCP", 17: "UDP", 1: "ICMP"}


def protocol_name(code) -> str:
    try:
        return PROTOCOL_NAMES.get(int(code), "Other")
    except (TypeError, ValueError):
        return "Other"


def _num(v):
    """float/int value for SQLite, with NaN/inf stored as NULL."""
    if v is None:
        return None
    f = float(v)
    return None if not math.isfinite(f) else f


def create_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = connect(db_path)
    try:
        con.executescript(SCHEMA_PATH.read_text())
    finally:
        con.close()


def flow_rows(test_raw: pd.DataFrame, classes: list[str]) -> list[tuple]:
    """FLOW_RECORD tuples: (flow_id, row_index, protocol, flow_duration, fwd_packets,
    bytes_per_s, true_label), with flow_id = row_index (plan §5.1)."""
    inv = {v: k for k, v in DISPLAY_COLS.items()}  # FLOW_RECORD field -> source column
    rows = []
    for r in test_raw[["row_index", *DISPLAY_COLS, "label_7"]].itertuples(index=False):
        d = dict(zip(["row_index", *DISPLAY_COLS, "label_7"], r))
        fwd = _num(d[inv["fwd_packets"]])
        rows.append((int(d["row_index"]), int(d["row_index"]),
                     protocol_name(d[inv["protocol"]]),
                     _num(d[inv["flow_duration"]]),
                     None if fwd is None else int(fwd),
                     _num(d[inv["bytes_per_s"]]),
                     classes[int(d["label_7"])]))
    return rows


def load_flow_records(con, test_raw: pd.DataFrame, classes: list[str]) -> int:
    rows = flow_rows(test_raw, classes)
    with con:  # one transaction
        con.executemany(
            "INSERT INTO flow_record (flow_id, row_index, protocol, flow_duration, "
            "fwd_packets, bytes_per_s, true_label) VALUES (?,?,?,?,?,?,?)", rows)
    return len(rows)


def init_db(db_path: Path, test_raw_path: Path, label_map_path: Path,
            reset: bool = False, extra_conditions: dict | None = None) -> dict:
    db_path = Path(db_path)
    if reset:
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                p.unlink()
    fresh = not db_path.exists()
    if fresh:
        create_schema(db_path)

    test_raw = pd.read_parquet(test_raw_path)
    classes = json.loads(Path(label_map_path).read_text())["classes"]
    con = connect(db_path)
    try:
        n_existing = con.execute("SELECT COUNT(*) FROM flow_record").fetchone()[0]
        if n_existing == 0:
            loaded = load_flow_records(con, test_raw, classes)
        else:
            loaded = 0
            if n_existing != len(test_raw):
                raise RuntimeError(f"flow_record has {n_existing} rows but test_raw has "
                                   f"{len(test_raw)}; rerun with --reset")
        with con:
            env_id = environment.get_or_create_env(
                con, environment.collect(db_path, extra_conditions))
    finally:
        con.close()
    return {"db": str(db_path), "created": fresh, "flow_records_loaded": loaded,
            "env_id": env_id}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--reset", action="store_true",
                    help="delete the existing database (and -wal/-shm) first")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    out = init_db(ROOT / paths["db"],
                  ROOT / paths["processed"] / "test_raw.parquet",
                  ROOT / paths["artefacts"] / "label_map.json",
                  reset=args.reset,
                  extra_conditions={"dashboard_refresh_s": cfg["dashboard"]["refresh_s"]})
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
