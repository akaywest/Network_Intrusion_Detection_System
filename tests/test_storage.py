"""Phase 3 tests (M5; plan §5 "Done when", §9 UT5, NFR6).

- 1,000 fake predictions through LatencyLogger: no FK errors, v_latency returns 1,000
  rows with no negative interval, t4 >= t3, alerts only for malicious non-warm-up rows,
  severity rule High iff confidence >= 0.90.
- FK enforcement on every connection; WAL + synchronous=NORMAL.
- Static check: no SQL built with f-strings, +, % or .format (NFR6).
- init_db on the real test partition (skipped if data/processed is absent).
"""
from __future__ import annotations

import ast
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from nids.storage import environment
from nids.storage.db import connect
from nids.storage.init_db import create_schema, init_db, protocol_name
from nids.storage.logger import LatencyLogger, severity_for

ROOT = Path(__file__).resolve().parents[1]
TEST_RAW = ROOT / "data" / "processed" / "test_raw.parquet"
LABEL_MAP = ROOT / "artefacts" / "label_map.json"


@dataclass
class FakeResult:  # same fields as the plan §6.1 Result dataclass
    is_malicious: int
    label: str
    score: float
    confidence: float
    severity: str | None
    warmup: bool


@pytest.fixture
def db(tmp_path):
    """Fresh schema with one environment row, one run and 50 flow records."""
    path = tmp_path / "nids.db"
    create_schema(path)
    con = connect(path)
    with con:
        env_id = environment.get_or_create_env(con, environment.collect(path))
        con.executemany(
            "INSERT INTO flow_record (flow_id, row_index, protocol, flow_duration, fwd_packets, "
            "bytes_per_s, true_label) VALUES (?,?,?,?,?,?,?)",
            [(i, i, "TCP", 1.0, 1, 1.0, "Normal") for i in range(50)])
    con.close()
    return path, env_id


def _run_fake(path, env_id, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    log = LatencyLogger(path)
    run_id = log.start_run(env_id, "Random Forest", "binary", {"n_estimators": 10}, 42,
                           10.0, "steady", "TEST", n)
    expected = {}
    for i in range(n):
        mal = int(rng.random() < 0.4)
        warm = bool(rng.random() < 0.05)
        conf = float(rng.choice([rng.uniform(0.5, 0.9), rng.uniform(0.9, 1.0), 0.90]))
        if warm:
            conf = float("nan")
        res = FakeResult(mal, "Malicious" if mal else "Normal", conf, conf, None, warm)
        t0 = time.perf_counter_ns()
        t1, t2, t3 = t0 + 1000, t0 + 2000, t0 + 3000
        pid = log.log(run_id, i, int(rng.integers(0, 50)), res, t0, t1, t2, t3)
        expected[pid] = (mal, warm, conf)
    log.finish_run(run_id, malformed_count=3)
    log.close()
    return run_id, expected


def test_wal_and_pragmas_on_every_connection(db):
    path, _ = db
    con = connect(path)
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert con.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    con.close()
    tables = {r[0] for r in sqlite3.connect(path).execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view','index')")}
    assert {"test_environment", "model_run", "flow_record", "prediction", "alert",
            "v_latency", "idx_pred_run_t0"} <= tables


def test_thousand_fake_predictions(db):
    path, env_id = db
    run_id, expected = _run_fake(path, env_id)
    con = connect(path)
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    assert con.execute("SELECT COUNT(*) FROM prediction").fetchone()[0] == 1000

    # every t4 written (incl. the last one via flush), t4 >= t3
    assert con.execute("SELECT COUNT(*) FROM prediction WHERE t4_store_ns IS NULL").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM prediction WHERE t4_store_ns < t3_infer_ns").fetchone()[0] == 0

    lat = con.execute("SELECT t_tx_ms, t_pre_ms, t_inf_ms, t_ser_ms, L_ms FROM v_latency "
                      "WHERE run_id=?", (run_id,)).fetchall()
    assert len(lat) == 1000
    assert np.all(np.array(lat) >= 0)

    # alerts exactly for malicious, non-warm-up predictions, with the §5.2 severity rule
    alerts = dict(con.execute("SELECT prediction_id, severity FROM alert").fetchall())
    want = {pid for pid, (mal, warm, _) in expected.items() if mal and not warm}
    assert set(alerts) == want and len(want) > 0
    for pid, sev in alerts.items():
        assert sev == ("High" if expected[pid][2] >= 0.90 else "Medium")
    assert "High" in alerts.values() and "Medium" in alerts.values()

    # warm-up rows flagged; NaN confidence stored as NULL
    n_warm = sum(w for _, w, _ in expected.values())
    assert con.execute("SELECT COUNT(*) FROM prediction WHERE is_warmup=1 "
                       "AND confidence IS NULL").fetchone()[0] == n_warm

    mr = con.execute("SELECT malformed_count, ended_at, random_seed, env_id FROM model_run "
                     "WHERE run_id=?", (run_id,)).fetchone()
    assert mr[0] == 3 and mr[1] is not None and mr[2] == 42 and mr[3] == env_id
    con.close()


def test_t4_written_in_next_transaction(db):
    path, env_id = db
    log = LatencyLogger(path)
    run_id = log.start_run(env_id, "IF", "binary", {}, 42, 100.0, "steady", "S1", 2)
    r = FakeResult(0, "Normal", 0.1, 0.9, None, False)
    p1 = log.log(run_id, 0, 1, r, 1, 2, 3, 4)
    ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    q = "SELECT t4_store_ns FROM prediction WHERE prediction_id=?"
    assert ro.execute(q, (p1,)).fetchone()[0] is None          # not yet
    p2 = log.log(run_id, 1, 2, r, 5, 6, 7, 8)
    assert ro.execute(q, (p1,)).fetchone()[0] is not None      # written with record 2
    assert ro.execute(q, (p2,)).fetchone()[0] is None
    log.flush()
    assert ro.execute(q, (p2,)).fetchone()[0] is not None      # flush at end of run
    ro.close()
    log.close()


def test_foreign_keys_enforced(db):
    path, env_id = db
    log = LatencyLogger(path)
    run_id = log.start_run(env_id, "IF", "binary", {}, 42, 100.0, "steady", "S1", 1)
    r = FakeResult(1, "Malicious", 1.0, 0.95, None, False)
    with pytest.raises(sqlite3.IntegrityError):
        log.log(run_id, 0, 99_999, r, 1, 2, 3, 4)               # unknown flow_id
    with pytest.raises(sqlite3.IntegrityError):
        log.log(run_id + 1, 0, 1, r, 1, 2, 3, 4)                # unknown run_id
    pid = log.log(run_id, 0, 1, r, 1, 2, 3, 4)                  # logger still usable
    log.close()
    con = connect(path)
    assert con.execute("SELECT COUNT(*) FROM prediction").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM alert").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO model_run (env_id, model_name) VALUES (?, ?)", (12345, "x"))
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO alert (prediction_id, severity) VALUES (?, ?)", (pid, "High"))
    con.close()


def test_severity_rule():
    assert severity_for(0.90) == "High"
    assert severity_for(0.95) == "High"
    assert severity_for(0.8999) == "Medium"
    assert severity_for(float("nan")) == "Medium"
    assert severity_for(None) == "Medium"


def test_environment_row(db, tmp_path):
    path, env_id = db
    env = environment.collect(path)
    assert env["cpu_model"] and env["cpu_cores"] >= 1 and env["ram_gb"] > 0
    assert env["python_version"].startswith("3.")
    libs = json.loads(env["library_versions"])
    assert libs["scikit-learn"] != "not installed" and "sqlite" in libs
    assert json.loads(env["run_conditions"])["sqlite_synchronous"] == "NORMAL"
    con = connect(path)
    with con:
        assert environment.get_or_create_env(con, env) == env_id  # reused, not duplicated
    assert con.execute("SELECT COUNT(*) FROM test_environment").fetchone()[0] == 1
    con.close()


def test_protocol_names():
    assert [protocol_name(c) for c in (6, 17.0, 1, 0, 47, None)] == \
        ["TCP", "UDP", "ICMP", "Other", "Other", "Other"]


# ---- NFR6: no dynamically built SQL --------------------------------------------------
SQL_RE = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|PRAGMA|ALTER|REPLACE)\b")
EXEC_METHODS = {"execute", "executemany", "executescript", "read_sql", "read_sql_query"}


def _python_sources():
    return [p for d in ("src", "experiments") for p in (ROOT / d).rglob("*.py")]


def _module_constants(tree) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


def test_no_sql_built_from_strings():
    problems = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        consts = _module_constants(tree)
        for node in ast.walk(tree):
            # 1. any f-string / concatenation / %-format / .format() that looks like SQL
            if isinstance(node, ast.JoinedStr):
                text = "".join(v.value for v in node.values if isinstance(v, ast.Constant))
                if SQL_RE.search(text):
                    problems.append(f"{path}:{node.lineno} f-string SQL")
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
                for side in (node.left, node.right):
                    if isinstance(side, ast.Constant) and isinstance(side.value, str) \
                            and SQL_RE.search(side.value):
                        problems.append(f"{path}:{node.lineno} SQL built with + or %")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "format" and isinstance(node.func.value, ast.Constant) \
                    and isinstance(node.func.value.value, str) and SQL_RE.search(node.func.value.value):
                problems.append(f"{path}:{node.lineno} SQL built with .format()")
            # 2. every execute*/read_sql argument must be a literal (or a module constant,
            #    or the schema file read for executescript)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in EXEC_METHODS and node.args:
                a = node.args[0]
                ok = (isinstance(a, ast.Constant) and isinstance(a.value, str)) \
                    or (isinstance(a, ast.Name) and a.id in consts) \
                    or (node.func.attr == "executescript" and isinstance(a, ast.Call)
                        and isinstance(a.func, ast.Attribute) and a.func.attr == "read_text")
                if not ok:
                    problems.append(f"{path}:{node.lineno} non-literal SQL in .{node.func.attr}()")
    assert problems == [], "\n".join(problems)


def test_sql_checker_catches_bad_code(tmp_path):
    """The static check above is not vacuous: it flags the patterns it is meant to."""
    bad = ('t = "x"\ncon.execute(f"SELECT * FROM {t}")\n'
           'con.execute("SELECT * FROM " + t)\nq = "DELETE FROM %s" % t\n')
    tree = ast.parse(bad)
    found = [n for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)
             and SQL_RE.search("".join(v.value for v in n.values if isinstance(v, ast.Constant)))]
    found += [n for n in ast.walk(tree) if isinstance(n, ast.BinOp)
              and any(isinstance(s, ast.Constant) and isinstance(s.value, str)
                      and SQL_RE.search(s.value) for s in (n.left, n.right))]
    assert len(found) == 3


# ---- init_db on the real test partition ----------------------------------------------
@pytest.mark.skipif(not TEST_RAW.exists(), reason="data/processed/test_raw.parquet absent")
def test_init_db_real_partition(tmp_path):
    import pandas as pd
    path = tmp_path / "nids.db"
    out = init_db(path, TEST_RAW, LABEL_MAP)
    test_raw = pd.read_parquet(TEST_RAW)
    assert out["created"] and out["flow_records_loaded"] == len(test_raw)
    con = connect(path)
    assert con.execute("SELECT COUNT(*) FROM flow_record WHERE flow_id != row_index").fetchone()[0] == 0
    assert sorted(r[0] for r in con.execute("SELECT row_index FROM flow_record")) == \
        sorted(test_raw["row_index"].tolist())
    classes = json.loads(LABEL_MAP.read_text())["classes"]
    labels = dict(con.execute("SELECT true_label, COUNT(*) FROM flow_record GROUP BY true_label"))
    assert labels == {classes[k]: v for k, v in test_raw["label_7"].value_counts().items()}
    protos = {r[0] for r in con.execute("SELECT DISTINCT protocol FROM flow_record")}
    assert protos <= {"TCP", "UDP", "ICMP", "Other"}
    con.close()
    # second call: no reload, environment row reused
    out2 = init_db(path, TEST_RAW, LABEL_MAP)
    assert not out2["created"] and out2["flow_records_loaded"] == 0
    assert out2["env_id"] == out["env_id"]
