-- NIDS log database (M5; implementation plan §5.1, Chapter 3 §3.3.9, Table 3.10).
-- Fields marked "added" are the extensions listed in plan §1.1 item 3.
-- journal_mode=WAL persists in the file; foreign_keys and synchronous are per-connection
-- and are re-applied on EVERY connection by nids.storage.db.connect().

PRAGMA journal_mode = WAL;

PRAGMA foreign_keys = ON; -- also run on EVERY new connection

CREATE TABLE test_environment (
    env_id INTEGER PRIMARY KEY, cpu_model TEXT, cpu_cores INTEGER, ram_gb REAL,
    storage_type TEXT, -- added (Table 3.14)
    os_name TEXT, python_version TEXT, library_versions TEXT,
    run_conditions TEXT); -- added: power mode, threads, sqlite synchronous

CREATE TABLE model_run (
    run_id INTEGER PRIMARY KEY,
    env_id INTEGER NOT NULL REFERENCES test_environment(env_id),
    model_name TEXT NOT NULL, task TEXT NOT NULL DEFAULT 'binary', -- task added
    hyperparameters TEXT, random_seed INTEGER, emission_interval_ms REAL,
    load_condition TEXT CHECK (load_condition IN ('steady','burst')),
    scenario TEXT, n_records INTEGER, malformed_count INTEGER DEFAULT 0, -- added
    started_at TEXT, ended_at TEXT);

CREATE TABLE flow_record (
    flow_id INTEGER PRIMARY KEY, row_index INTEGER NOT NULL UNIQUE,
    protocol TEXT, flow_duration REAL, fwd_packets INTEGER, bytes_per_s REAL,
    true_label TEXT NOT NULL);

CREATE TABLE prediction (
    prediction_id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES model_run(run_id),
    flow_id INTEGER NOT NULL REFERENCES flow_record(flow_id),
    seq_no INTEGER NOT NULL, is_warmup INTEGER NOT NULL DEFAULT 0, -- added
    predicted_label TEXT, anomaly_score REAL, confidence REAL,
    t0_emit_ns INTEGER, t1_recv_ns INTEGER, t2_prep_ns INTEGER,
    t3_infer_ns INTEGER, t4_store_ns INTEGER);

CREATE TABLE alert (
    alert_id INTEGER PRIMARY KEY,
    prediction_id INTEGER NOT NULL UNIQUE REFERENCES prediction(prediction_id),
    severity TEXT CHECK (severity IN ('High','Medium')), raised_at TEXT);

-- Index 1 of §3.3.9 (run + emission time). Index 2 (unique row_index) is the UNIQUE
-- constraint on flow_record.row_index, which SQLite backs with an automatic index.
CREATE INDEX idx_pred_run_t0 ON prediction(run_id, t0_emit_ns);

CREATE VIEW v_latency AS -- L is derived, never stored (3NF)
SELECT prediction_id, run_id, seq_no, is_warmup,
    (t1_recv_ns - t0_emit_ns) / 1e6 AS t_tx_ms,
    (t2_prep_ns - t1_recv_ns) / 1e6 AS t_pre_ms,
    (t3_infer_ns - t2_prep_ns) / 1e6 AS t_inf_ms,
    (t4_store_ns - t3_infer_ns) / 1e6 AS t_ser_ms,
    (t4_store_ns - t0_emit_ns) / 1e6 AS L_ms
FROM prediction WHERE t4_store_ns IS NOT NULL;
