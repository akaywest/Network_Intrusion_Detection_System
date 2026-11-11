**IMPLEMENTATION PLAN**

**Development of a Real-Time Network Intrusion Detection System Using Machine Learning**

A phase-by-phase build guide aligned with Chapters One to Three and the methodology presentation, with a review of the previous implementation guide

**Student:** Kayode Akinola Samuel (236336)

**Supervisor:** Mr. I. O. Adeyemi

**Department:** Computer Science, Faculty of Computing, University of Ibadan

**Source documents:** Final Project Chapters 1–3; Methodology Presentation (Oct 2026); Implementation Guide and Project Context

**Prepared:** November 2026

**How to use this document.** Section 1 explains where the previous guide falls short. Sections 2 to 10 are the plan to follow: build one phase at a time and check each phase's “Done when” list before moving on. Every rule in the plan points back to the requirement or section of Chapter Three it comes from, so your Chapter Four can cite your own design.

# Contents

# 1. Review of the Previous Implementation Guide

The previous guide (“Implementation Guide and Project Context”) gets the architecture right: DSRM with CRISP-DM nested inside it; Isolation Forest, Random Forest and LSTM behind one Detector interface; a TCP producer and consumer; SQLite in WAL mode with five tables; a Streamlit dashboard; and the t₀ to t₄ timestamps. It is a sound outline.

As a guide to follow, though, it is **not aligned well enough**. It leaves out a large part of what Chapter Three commits you to, and in a few places it contradicts Chapter Three. Your examiners will read Chapter Four against Chapter Three, so every difference will either show up as an unexplained deviation or as a requirement you never met. It also leaves out several engineering details that do not appear in any chapter but that would make the latency figures wrong if they were missed. Table 1 lists the differences. Severity means: **High**, the results or a stated requirement would be wrong or missing; **Medium**, a stated method is not followed; **Low**, completeness or good practice.

| **\#** | **Area**                | **What the old guide says**                                                                                                    | **What Chapter 3 / practice requires**                                                                                                                                                                                                                                                                                                         | **Severity** |
|--------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| 1      | What gets streamed      | Saves the transformed partitions and streams “the 15% test partition CSV”. It does not say whether the rows are raw or scaled. | The engine must apply the saved scaler to every record (FR6, Fig. 3.7, process 2.3). Stream **cleaned but unscaled** rows. Streaming scaled rows would scale them twice and make t_pre meaningless.                                                                                                                                            | High         |
| 2      | t₄ definition           | “Executes the INSERT, grabs the current time as t₄, and commits.”                                                              | t₄ is stamped **when the insert is committed** (§3.3.6, §3.3.9). Stamping it before the commit leaves the commit cost out of L. See §5.3 for a correct write pattern.                                                                                                                                                                          | High         |
| 3      | Five seeds              | Not mentioned.                                                                                                                 | Final training and evaluation are repeated with seeds 42–46 and reported as mean ± SD (§3.3.10, NFR3). Each scenario is run five times.                                                                                                                                                                                                        | High         |
| 4      | Test scenarios          | Steady at 100 ms and 10 ms, plus a burst.                                                                                      | Four scenarios S1–S4, including **S3, a rate search** (100, 50, 20, 10, 5, 1 ms). Samples of 5,000 records for steady runs and 2,000 for burst runs, the first 100 records excluded as warm-up, and five repeats (Table 3.13).                                                                                                                 | High         |
| 5      | Detection metrics       | Queried from the streaming database.                                                                                           | Precision, recall, F1 and ROC-AUC on the **full held-out test partition**, plus macro-F1, per-class recall and a confusion matrix for the seven-class task (§3.3.11). The streamed predictions are then checked against the offline ones.                                                                                                      | High         |
| 6      | Protocol mix            | The dashboard has traffic, latency and alerts only.                                                                            | FR13 and UC6 require a protocol-mix panel (TCP/UDP/ICMP) and a latency-trend chart with a Δt reference line (§3.3.7).                                                                                                                                                                                                                          | High         |
| 7      | Preprocessing           | Drop duplicates, handle inf/NaN, drop identifier and constant columns, then scale.                                             | Also drop one column from each pair with \|r\| \> 0.98, apply log(1+x) to strongly skewed non-negative features, and fill missing values with **training** medians (or drop them if they are under 1%). Stratify on the **seven-class** label. Save a label map (§3.2.4).                                                                      | Medium       |
| 8      | Isolation Forest tuning | Tunes only the threshold τ.                                                                                                    | Grid search over n_estimators, max_samples and max_features on validation F1 / ROC-AUC, then τ (Table 3.11).                                                                                                                                                                                                                                   | Medium       |
| 9      | Random Forest tuning    | RandomizedSearchCV “validated against the validation set”.                                                                     | 30 trials with **stratified 5-fold CV inside the training partition**, scored by F1 (binary) and macro-F1 (seven-class), refitted, then checked on validation. Both binary and seven-class models are needed.                                                                                                                                  | Medium       |
| 10     | LSTM                    | Windows W ∈ {5, 10, 20}, 1–2 layers, early stopping; saved as .h5.                                                             | Random search with 15 trials, class-weighted loss, Adam, at most 50 epochs, windows built inside each partition, a threshold set on validation, W−1 warm-up records at run time, a check that the row order is meaningful (with a fallback if it is not), and a binary plus a seven-class model. Use the .keras format, because .h5 is legacy. | Medium       |
| 11     | Single-thread inference | Not mentioned.                                                                                                                 | Each model is limited to one thread during inference tests (§3.3.10).                                                                                                                                                                                                                                                                          | Medium       |
| 12     | Malformed records       | Not mentioned.                                                                                                                 | Validate every record, then discard and count bad ones without stopping (FR5, process 2.2).                                                                                                                                                                                                                                                    | Medium       |
| 13     | Database details        | Five tables with keys.                                                                                                         | Also a latency **view** (L is not stored), two indexes, PRAGMA foreign_keys=ON on every connection, bound parameters only (NFR6), and TEST_ENVIRONMENT filled for every run (FR15, Table 3.14).                                                                                                                                                | Medium       |
| 14     | Control channel         | Widgets write to a JSON file or a lightweight queue.                                                                           | This is acceptable, but polling a file on the data path costs time. Use a separate loopback control socket (§3.3.1, module M7).                                                                                                                                                                                                                | Low          |
| 15     | TCP framing and Nagle   | Not mentioned.                                                                                                                 | TCP is a byte stream, so messages need delimiters (newline-delimited JSON). Turn on TCP_NODELAY, or Nagle’s algorithm with delayed ACK can add tens of ms per record.                                                                                                                                                                          | High         |
| 16     | Pacing accuracy         | “Wait for the defined interval.”                                                                                               | A plain time.sleep(Δt) drifts and is coarse at 1–10 ms. Use an absolute schedule with a short busy-wait at the end (§6.2).                                                                                                                                                                                                                     | High         |
| 17     | Keras per-record cost   | Detector.predict() wraps the .h5 model.                                                                                        | Keras model.predict() on one record has an overhead of tens of ms that has nothing to do with the LSTM itself. Use a compiled tf.function call (§6.1).                                                                                                                                                                                         | High         |
| 18     | Testing for Chapter 4   | Only an offline-vs-streaming check.                                                                                            | Chapter 3.4 promises unit, integration, system and user acceptance testing (NFR5 has an acceptance test).                                                                                                                                                                                                                                      | Medium       |
| 19     | Reproducibility         | Not mentioned.                                                                                                                 | Pinned library versions, a configuration file and saved artefacts (NFR4).                                                                                                                                                                                                                                                                      | Low          |

*Table 1: Differences between the previous guide and the approved design*

## 1.1 Small fixes to make in Chapters One to Three

While checking the documents against each other I found a few inconsistencies inside your chapters. Fix them now, before Chapter Four depends on them.

1.  **Latency definition (Chapter 1.4 vs §3.3.6).** Chapter One defines latency as running “from a flow’s emission ... to its prediction reaching the dashboard”. Chapter Three defines L = t₄ − t₀, ending at the database commit, and reports the dashboard refresh delay separately. Change Chapter One to match Chapter Three.

2.  **Order of the correlation filter (Fig. 3.3).** The figure removes correlated columns before the split. Under leakage control 1, statistics should come from the training partition only, so compute the constant-column check, the correlation filter, the skewness test and the medians on the training partition after the split (§3.1 below). Update the figure, or state the change in Chapter Four.

3.  **Data dictionary (Table 3.10).** Table 3.14 says storage type and run conditions are recorded, but TEST_ENVIRONMENT has no fields for them. The implementation also needs a few more fields: seq_no and is_warmup on PREDICTION, and task, scenario, n_records and malformed_count on MODEL_RUN. Add them to Table 3.10, or list them in Chapter Four as additions made during implementation.

4.  **Isolation Forest score sign.** Equation 3.3 gives scores near 1 for anomalies. scikit-learn’s score_samples() returns the **negative** of this score, so use s = −score_samples(x) and say so in Chapter Four.

# 2. Project Setup (Phase 0)

## 2.1 Environment

- **Python 3.11** in a virtual environment. TensorFlow supports 3.11 on Windows, macOS and Linux, and from 3.11 time.sleep on Windows uses a high-resolution timer, which helps pacing.

- Install: pandas numpy scikit-learn tensorflow streamlit plotly pyarrow pyyaml psutil py-cpuinfo joblib matplotlib pytest. After the install works, run pip freeze \> requirements.lock.txt and commit the file. This is the pinning that NFR4 requires.

- Run everything on CPU, with no GPU (NFR8). Do the measured runs plugged into power, with the same power plan each time and other applications closed (§3.3.11).

## 2.2 Repository layout

This layout gives each of the eight modules in Table 3.9 its own folder, so Chapter Four can map code to design directly.

> nids/
>
> config.yaml \# every tunable setting (ports, paths, seeds, dt, sample sizes)
>
> requirements.lock.txt
>
> run.py \# one-command launcher (engine + streamer + dashboard) -\> NFR5
>
> CLAUDE.md \# design rules for the coding assistant (see Section 10)
>
> data/raw/ \# IDS2025 daily CSVs (git-ignored)
>
> data/processed/ \# partitions, arrays (git-ignored)
>
> artefacts/ \# models, preprocessor.json, feature_list.json, label_map.json
>
> db/nids.db \# SQLite log (git-ignored)
>
> results/ \# CSV exports, plots for Chapter 4
>
> notebooks/01_data_understanding.ipynb
>
> src/nids/
>
> prep/prepare_data.py \# M1 Data preparation
>
> train/train_iforest.py \# M2 Model training
>
> train/train_rf.py
>
> train/train_lstm.py
>
> train/offline_predict.py
>
> stream/streamer.py \# M3 Streamer
>
> engine/preprocessor.py \# M4 Inference engine
>
> engine/detectors.py
>
> engine/engine.py
>
> storage/schema.sql \# M5 Latency logger and database
>
> storage/logger.py
>
> storage/environment.py
>
> dashboard/app.py \# M6 Dashboard
>
> control/protocol.py \# M7 Control
>
> evaluation/detection_metrics.py \# M8 Evaluation
>
> evaluation/system_metrics.py
>
> evaluation/plots.py
>
> experiments/run_matrix.py \# drives S1-S4 automatically through the control channel
>
> tests/

## 2.3 Configuration file

> paths: {raw: data/raw, processed: data/processed, artefacts: artefacts, db: db/nids.db}
>
> network: {host: 127.0.0.1, data_port: 9100, engine_ctl_port: 9101, streamer_ctl_port: 9102}
>
> split: {train: 0.70, val: 0.15, test: 0.15, seed: 42, stratify: label_7}
>
> prep: {corr_threshold: 0.98, skew_threshold: 1.0, missing_drop_below: 0.01}
>
> seeds: \[42, 43, 44, 45, 46\]
>
> stream: {steady_n: 5000, burst_n: 2000, burst_block: 1000, warmup_latency: 100, sample_seed: 42}
>
> dashboard: {refresh_s: 1}
>
> alerts: {high_confidence: 0.90}

### Done when

- git init is done and pushed to GitHub. The data/, db/ and large model files are in .gitignore.

- python -c "import tensorflow, sklearn, streamlit" runs without errors and the lock file is committed.

# 3. Phase 1 — Data Preparation (Module M1; CRISP-DM phases 1–3)

Goal: turn the IDS2025 daily files into leakage-free partitions and a saved preprocessing state that the engine can apply to a single record. The source is §3.2.4, Fig. 3.3 and leakage controls 1, 2, 3 and 8.

## 3.1 Steps, in this order

1.  **Load and merge** the Monday–Friday CSVs in day order. Add row_index, the global position after the merge (Monday first, file row order kept), and source_day. Strip whitespace from column names, because CICFlowMeter exports often have leading spaces. Record the row count of each file for Chapter Four.

2.  **Data understanding** (notebook). List every column with its type, missing count and number of unique values. Find any identifier columns: Flow ID, Src/Dst IP, Src/Dst Port, Timestamp. The dataset description does not say which ones exist, so check (§3.2.4).

3.  **Order check for the LSTM.** If a Timestamp exists, check that it is non-decreasing within each day. Also measure the label run lengths: if neighbouring rows usually share a label, the file order carries sequence. Write the finding down, because it decides which LSTM formulation you use (§4.3).

4.  **Infinite values:** replace ±inf with NaN and count the affected rows.

5.  **Separate the columns into roles.** features are the numeric CICFlowMeter fields, with no identifiers. display are protocol, flow duration, forward packets and bytes per second, which FLOW_RECORD needs; protocol and the others can also stay as features. meta is row_index, source_day and timestamp, which are never features.

6.  **Duplicates:** after dropping identifiers, remove exact duplicates on features + label (control 2). Also find **conflicting duplicates** (same features, different label) and drop them. Report both counts.

7.  **Missing values:** if the rows with any NaN are under 1% of the data, drop them now. Otherwise keep them and fill them with training medians in step 10.

8.  **Labels:** create label_7 (the seven classes in Table 3.6) and label_bin (Normal = 0, Malicious = 1). Save label_map.json. Compute the class counts and save them; Chapter Four needs this table because the dataset description does not publish it.

9.  **Stratified split** on label_7 with seed 42, done in two calls to train_test_split: 70/30, then the 30 split 50/50 into validation and test. Do this **before** fitting any statistic.

10. **Fit on the training partition only:** (a) drop constant columns; (b) compute \|Pearson r\| and drop the later column of each pair with \|r\| \> 0.98, then print the pairs so you can review them; (c) compute the medians used for filling; (d) mark as log columns the features with min ≥ 0 and \|skewness\| \> 1, then apply log(1+x); (e) fit StandardScaler on the result.

11. **Transform all three partitions** with the fitted state and cast them to float32.

12. **Sort each partition back into row_index order.** train_test_split shuffles the rows. The stream must replay in the original order (§3.2.4), and the LSTM windows need that order too.

## 3.2 Outputs

| **File**                                          | **Contents**                                                                                                                        | **Used by**                               |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| data/processed/{train,val,test}\_raw.parquet      | Cleaned but **unscaled** rows with row_index, display fields and both labels                                                        | Streamer (test only); FLOW_RECORD loading |
| data/processed/X\_{train,val,test}.npy, y\_\*.npy | Scaled float32 matrices and labels in row_index order                                                                               | Model training, offline predictions       |
| artefacts/feature_list.json                       | Ordered list of the F retained features                                                                                             | Engine, training                          |
| artefacts/preprocessor.json                       | Medians, log-column mask, scaler mean\_ and scale\_, all in feature order                                                           | Engine (Preprocessor class)               |
| artefacts/label_map.json                          | Seven-class codes and binary mapping                                                                                                | Training, evaluation, dashboard           |
| results/prep_report.json                          | Row counts, duplicates removed, conflicts, missing handling, dropped columns and correlated pairs, class counts, order-check result | Chapter 4 write-up                        |

> **Why JSON and not scaler.pkl.** Saving the preprocessing state as plain numbers (JSON or .npz) means the engine never has to unpickle a file for it. NFR6 notes that loading a pickle can execute code. The models still have to use joblib or Keras files, and those are loaded only from artefacts/ (§6.1).

## 3.3 The online preprocessor

Fit with scikit-learn offline, then apply the same arithmetic online with NumPy. Transforming one row through a scikit-learn Pipeline costs around a millisecond in overhead; the NumPy version below takes microseconds and gives the same result. A unit test must confirm that the two agree (§9).

> class Preprocessor:
>
> def \_\_init\_\_(self, path):
>
> p = json.load(open(path))
>
> self.features = p\["features"\]
>
> self.median = np.array(p\["median"\])
>
> self.log_mask = np.array(p\["log_mask"\], dtype=bool)
>
> self.mean = np.array(p\["mean"\])
>
> self.scale = np.array(p\["scale"\])
>
> def transform(self, values): \# raises ValueError on bad input (FR5)
>
> x = np.asarray(values, dtype=np.float64)
>
> if x.shape != (len(self.features),):
>
> raise ValueError("wrong number of features")
>
> x\[~np.isfinite(x)\] = np.nan
>
> x = np.where(np.isnan(x), self.median, x)
>
> x\[self.log_mask\] = np.log1p(np.clip(x\[self.log_mask\], 0, None))
>
> return ((x - self.mean) / self.scale).astype(np.float32)

### Done when

- prep_report.json exists and the class-count table looks reasonable.

- No row_index appears in more than one partition, and every class appears in all three.

- Preprocessor.transform(raw_row) equals the matching row of X_test.npy (np.allclose) for 1,000 random test rows.

# 4. Phase 2 — Model Training (Module M2; CRISP-DM phases 4–5)

The pattern is the same for all three models: **search once** with seed 42 using training and validation data only, fix the best configuration, then **retrain that configuration with seeds 42, 43, 44, 45 and 46**. You end up with five saved models per model type, and each repeat of a test scenario uses a different one (§3.3.10, NFR3). Training may use all CPU cores. Record the chosen hyperparameters in artefacts/\<model\>\_config.json, because MODEL_RUN stores them.

## 4.1 Isolation Forest

- Train on the **Normal** rows of the training partition only.

- Grid (Table 3.11): n_estimators {100, 200, 300, 500} × max_samples {256, 512, 1024} × max_features {0.5, 0.75, 1.0}, 36 fits. For each fit, compute s = −score_samples on the validation set, then ROC-AUC, and the best F1 over thresholds (from precision_recall_curve). Choose by F1, breaking ties with ROC-AUC.

- τ is the threshold that maximises validation F1. Save iforest_seed{s}.joblib as a dict of {model, tau, config}.

- Report it as “unsupervised in training, with a supervised threshold”, which is the wording Chapter Three already uses.

## 4.2 Random Forest

- Use RandomizedSearchCV(n_iter=30, cv=StratifiedKFold(5, shuffle=True, random_state=42)) on the **training partition**. Score with f1 for binary and f1_macro for seven-class. The search space is Table 3.11, including class_weight ∈ {balanced, balanced_subsample}.

- Refit the best configuration on the full training partition and check it on validation. Choose the decision threshold on validation, either 0.5 or the F1-maximising value, and record which one you used.

- Train **two** families: rf_binary_seed{s}.joblib (used in all timing scenarios) and rf_multi_seed{s}.joblib (for the seven-class results).

- Also time the chosen configuration on 1,000 single-row predictions with n_jobs=1. Tree count and depth drive per-record latency, so this tells you early whether NFR2 (100 records/s) is realistic.

> If 30 × 5 fits on the full training set takes too long on your laptop, run the search on a stratified subsample of the training partition (for example 20%) and refit the winner on the full partition. This is a legitimate choice, but state it in Chapter Four as a deviation made for compute reasons.

## 4.3 LSTM

- **Choose the formulation using the order check from Phase 1.** (A) **Record windows** (the main design): the input is the last W records, shape (W, F), and the label is that of the last record. (B) **Fallback**: each flow’s F features treated as a sequence of F steps, shape (F, 1), with no buffer and no dependence on order. Use B if the file order carries no sequential meaning, or if A gives no benefit. Either way, say which one you used in Chapter Four (§3.3.10).

- Build the windows **inside each partition** (control 7) with tf.keras.utils.timeseries_dataset_from_array. Do not build a full (N, W, F) array: at W = 20 it can need several gigabytes of RAM.

- Architecture: 1–2 LSTM layers, then dropout, then a Dense output (sigmoid for binary, softmax over 7 for multi). Use Adam and class-weighted cross-entropy, with EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True) and at most 50 epochs.

- Random search, 15 trials: W {5, 10, 20}, layers {1, 2}, units {32, 64, 128}, dropout U(0.2, 0.5), learning rate log-U(10⁻⁴, 10⁻²), batch {128, 256, 512}. Choose by validation F1 (macro-F1 for multi). Set the threshold on validation.

- Seed with tf.keras.utils.set_random_seed(s). Save lstm_binary_seed{s}.keras and lstm_multi_seed{s}.keras, plus a meta JSON with W, τ and the formulation.

## 4.4 Offline predictions

offline_predict.py scores the **full test partition** with every saved model and seed, and saves label, score and confidence per row to results/offline/\<model\>\_\<task\>\_seed\<s\>.parquet. These files are the source of the detection metrics (§8.1) and the reference for the streaming agreement check (§8.4).

### Done when

- There are 25 model files: IF × 5, RF binary × 5, RF multi × 5, LSTM binary × 5, LSTM multi × 5, plus a config or meta JSON for each family.

- Validation metrics for every model are logged. No test-partition data was touched before this step (control 6).

# 5. Phase 3 — Database and Latency Logger (Module M5)

## 5.1 Schema

This schema follows the entities and data dictionary in §3.3.3 and Table 3.10. The fields marked “added” are the extensions listed in §1.1.

> PRAGMA journal_mode = WAL;
>
> PRAGMA foreign_keys = ON; -- also run on EVERY new connection
>
> CREATE TABLE test_environment (
>
> env_id INTEGER PRIMARY KEY, cpu_model TEXT, cpu_cores INTEGER, ram_gb REAL,
>
> storage_type TEXT, -- added (Table 3.14)
>
> os_name TEXT, python_version TEXT, library_versions TEXT,
>
> run_conditions TEXT); -- added: power mode, threads, sqlite synchronous
>
> CREATE TABLE model_run (
>
> run_id INTEGER PRIMARY KEY,
>
> env_id INTEGER NOT NULL REFERENCES test_environment(env_id),
>
> model_name TEXT NOT NULL, task TEXT NOT NULL DEFAULT 'binary', -- task added
>
> hyperparameters TEXT, random_seed INTEGER, emission_interval_ms REAL,
>
> load_condition TEXT CHECK (load_condition IN ('steady','burst')),
>
> scenario TEXT, n_records INTEGER, malformed_count INTEGER DEFAULT 0, -- added
>
> started_at TEXT, ended_at TEXT);
>
> CREATE TABLE flow_record (
>
> flow_id INTEGER PRIMARY KEY, row_index INTEGER NOT NULL UNIQUE,
>
> protocol TEXT, flow_duration REAL, fwd_packets INTEGER, bytes_per_s REAL,
>
> true_label TEXT NOT NULL);
>
> CREATE TABLE prediction (
>
> prediction_id INTEGER PRIMARY KEY,
>
> run_id INTEGER NOT NULL REFERENCES model_run(run_id),
>
> flow_id INTEGER NOT NULL REFERENCES flow_record(flow_id),
>
> seq_no INTEGER NOT NULL, is_warmup INTEGER NOT NULL DEFAULT 0, -- added
>
> predicted_label TEXT, anomaly_score REAL, confidence REAL,
>
> t0_emit_ns INTEGER, t1_recv_ns INTEGER, t2_prep_ns INTEGER,
>
> t3_infer_ns INTEGER, t4_store_ns INTEGER);
>
> CREATE TABLE alert (
>
> alert_id INTEGER PRIMARY KEY,
>
> prediction_id INTEGER NOT NULL UNIQUE REFERENCES prediction(prediction_id),
>
> severity TEXT CHECK (severity IN ('High','Medium')), raised_at TEXT);
>
> CREATE INDEX idx_pred_run_t0 ON prediction(run_id, t0_emit_ns);
>
> CREATE VIEW v_latency AS -- L is derived, never stored (3NF)
>
> SELECT prediction_id, run_id, seq_no, is_warmup,
>
> (t1_recv_ns - t0_emit_ns) / 1e6 AS t_tx_ms,
>
> (t2_prep_ns - t1_recv_ns) / 1e6 AS t_pre_ms,
>
> (t3_infer_ns - t2_prep_ns) / 1e6 AS t_inf_ms,
>
> (t4_store_ns - t3_infer_ns) / 1e6 AS t_ser_ms,
>
> (t4_store_ns - t0_emit_ns) / 1e6 AS L_ms
>
> FROM prediction WHERE t4_store_ns IS NOT NULL;

- **FLOW_RECORD is filled once**, when the database is initialised, from test_raw.parquet with flow_id = row_index. Each streamed message then carries only row_index and the feature values, so the true label never reaches the engine. This makes leakage impossible by construction.

- **TEST_ENVIRONMENT** is filled by environment.py using platform, psutil and cpuinfo, together with library versions read through importlib.metadata. Each MODEL_RUN links to the current env_id (FR15).

- Choose PRAGMA synchronous (NORMAL is a reasonable choice in WAL mode), keep the same setting for every run, and record it in run_conditions. It has a large effect on commit time, and therefore on t_ser.

## 5.2 Alert and severity rule

- An alert is raised when the binary decision is Malicious (FR10). severity = High when confidence ≥ 0.90, otherwise Medium.

- Confidence for RF and LSTM is the probability of the predicted class. For Isolation Forest, define confidence as the position of s between τ and the highest validation score, clipped to \[0, 1\]. State this rule in Chapter Four, because §3.3.9 only says “a fixed rule on the confidence”.

## 5.3 Stamping t₄ correctly

t₄ is the moment the commit has finished, so it cannot be written in the same transaction. The pattern below keeps the definition exact and costs one extra statement in the next transaction. Describe it in Chapter Four.

> class LatencyLogger:
>
> def \_\_init\_\_(self, db_path):
>
> self.con = sqlite3.connect(db_path, isolation_level=None) \# explicit transactions
>
> self.con.execute("PRAGMA foreign_keys = ON")
>
> self.pending_t4 = \[\] \# \[(t4, prediction_id)\]
>
> def log(self, run_id, seq_no, flow_id, res, t0, t1, t2, t3):
>
> cur = self.con.cursor()
>
> cur.execute("BEGIN")
>
> if self.pending_t4: \# previous record's t4
>
> cur.executemany("UPDATE prediction SET t4_store_ns=? WHERE prediction_id=?",
>
> self.pending_t4)
>
> self.pending_t4.clear()
>
> cur.execute("""INSERT INTO prediction (run_id, flow_id, seq_no, is_warmup,
>
> predicted_label, anomaly_score, confidence,
>
> t0_emit_ns, t1_recv_ns, t2_prep_ns, t3_infer_ns)
>
> VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
>
> (run_id, flow_id, seq_no, res.warmup, res.label, res.score,
>
> res.confidence, t0, t1, t2, t3))
>
> pid = cur.lastrowid
>
> if res.is_malicious and not res.warmup:
>
> cur.execute("INSERT INTO alert (prediction_id, severity, raised_at) VALUES (?,?,?)",
>
> (pid, res.severity, utc_now_iso()))
>
> cur.execute("COMMIT")
>
> self.pending_t4.append((time.perf_counter_ns(), pid)) \# t4 = after commit
>
> def flush(self): \# call at end of run
>
> ...

### Done when

- init_db.py creates the schema, loads FLOW_RECORD and inserts a TEST_ENVIRONMENT row.

- Inserting 1,000 fake predictions gives no foreign-key errors, and v_latency returns 1,000 rows with no negative values.

- A search of the code finds no SQL built with f-strings or + (NFR6).

# 6. Phase 4 — Streaming Pipeline (Modules M3, M4)

## 6.1 Detector interface (strategy pattern, NFR7)

> @dataclass
>
> class Result:
>
> is_malicious: int; label: str; score: float; confidence: float
>
> severity: str \| None; warmup: bool
>
> class Detector(ABC):
>
> def \_\_init\_\_(self, artefact_dir: Path): self.dir = artefact_dir.resolve()
>
> def \_path(self, name): \# NFR6: only load from artefacts/
>
> p = (self.dir / name).resolve()
>
> if self.dir not in p.parents: raise PermissionError(p)
>
> return p
>
> @abstractmethod
>
> def load(self, seed: int, task: str = "binary") -\> None: ...
>
> @abstractmethod
>
> def predict(self, x: np.ndarray) -\> Result: ...
>
> def reset_state(self) -\> None: pass
>
> class IsolationForestDetector(Detector):
>
> def load(self, seed, task="binary"):
>
> b = joblib.load(self.\_path(f"iforest_seed{seed}.joblib"))
>
> self.model, self.tau = b\["model"\], b\["tau"\]
>
> self.model.set_params(n_jobs=1)
>
> def predict(self, x):
>
> s = float(-self.model.score_samples(x.reshape(1, -1))\[0\]) \# paper's s(x, psi)
>
> ...
>
> class LSTMDetector(Detector):
>
> def load(self, seed, task="binary"):
>
> self.model = keras.models.load_model(self.\_path(f"lstm\_{task}\_seed{seed}.keras"))
>
> self.W, F = self.model.input_shape\[1\], self.model.input_shape\[2\]
>
> self.buf = collections.deque(maxlen=self.W)
>
> self.\_infer = tf.function(lambda t: self.model(t, training=False),
>
> input_signature=\[tf.TensorSpec((1, self.W, F), tf.float32)\])
>
> self.\_infer(tf.zeros((1, self.W, F))) \# trace once, outside any timed path
>
> def reset_state(self): self.buf.clear() \# called at the start of every run
>
> def predict(self, x):
>
> self.buf.append(x)
>
> if len(self.buf) \< self.W: \# first W-1 records: warm-up
>
> return Result(0, "Warm-up", float("nan"), float("nan"), None, True)
>
> p = self.\_infer(np.stack(self.buf)\[None\]).numpy()\[0\]
>
> ...
>
> **Single-thread inference (§3.3.10).** At the very top of engine.py, **before** importing NumPy or TensorFlow, set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, TF_NUM_INTRAOP_THREADS and TF_NUM_INTEROP_THREADS to "1". Then call tf.config.threading.set_intra_op_parallelism_threads(1) and set_inter_op_parallelism_threads(1), and use n_jobs=1 for the scikit-learn models.

## 6.2 Streamer (producer)

- Load test_raw.parquet. Draw the run sample with sample_seed: 5,000 rows for steady runs or 2,000 for burst runs, stratified. Re-sort the sample by row_index so it replays in the original order (§3.3.11).

- Connect to 127.0.0.1:9100. Call setsockopt(IPPROTO_TCP, TCP_NODELAY, 1) and set a large send buffer (about 4 MB) so that a 1,000-record burst does not block the sender.

- Use an **absolute schedule**: record i is due at start + offset_i, so pacing errors never accumulate. Steady: offset_i = i·Δt. Burst (2,000 records): 500 at Δt = 100 ms, then 1,000 back-to-back, then 500 at 100 ms.

> def wait_until(deadline_ns):
>
> while True:
>
> remaining = deadline_ns - time.perf_counter_ns()
>
> if remaining \<= 0:
>
> return
>
> if remaining \> 2_000_000: \# sleep coarsely ...
>
> time.sleep((remaining - 1_500_000) / 1e9)
>
> \# ... then spin for the last ~1.5 ms for sub-millisecond accuracy
>
> for i, row in enumerate(sample):
>
> wait_until(start_ns + offsets_ns\[i\])
>
> msg = {"seq": i, "row_index": int(row.row_index), "features": row.feature_values}
>
> msg\["t0"\] = time.perf_counter_ns() \# stamp, THEN serialise (t_tx includes it)
>
> sock.sendall((json.dumps(msg) + "\n").encode()) \# newline-delimited JSON framing
>
> sock.sendall(b'{"type":"eos"}\n') \# end of stream

## 6.3 Inference engine (consumer)

> srv = socket.create_server(("127.0.0.1", 9100)) \# loopback only (NFR6)
>
> conn, \_ = srv.accept()
>
> conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
>
> for line in conn.makefile("rb"):
>
> t1 = time.perf_counter_ns() \# process 2.1
>
> try: \# process 2.2 + 2.3
>
> msg = json.loads(line)
>
> if msg.get("type") == "eos": break
>
> x = pre.transform(msg\["features"\])
>
> seq, row_index, t0 = int(msg\["seq"\]), int(msg\["row_index"\]), int(msg\["t0"\])
>
> except (ValueError, KeyError, TypeError):
>
> malformed += 1; continue \# FR5: count, discard, carry on
>
> t2 = time.perf_counter_ns()
>
> res = detector.predict(x) \# process 2.4
>
> t3 = time.perf_counter_ns()
>
> logger.log(run_id, seq, row_index, res, t0, t1, t2, t3) \# process 2.5 -\> 3.0
>
> logger.flush(); finish_run(run_id, malformed)
>
> **Clock check.** t₀ is stamped in the streamer process and t₁ to t₄ in the engine. time.perf_counter_ns() reads the system’s monotonic counter, which all processes on Windows, Linux and macOS share. Python’s documentation only guarantees differences within one process, though, so verify it as §3.3.6 requires: run 1,000 records at 100 ms and confirm that every t_tx is positive and below about 1 ms. If that ever fails, switch every stamp to time.time_ns() and rerun the check.

### Done when

- 200 records stream end to end with each of the three detectors; the PREDICTION count equals the records sent minus the malformed ones.

- Sending a deliberately broken line increases malformed_count and does not stop the engine.

- The clock check passes, and no interval in v_latency is negative.

# 7. Phase 5 — Control Channel and Dashboard (Modules M6, M7)

## 7.1 Control channel

The engine and the streamer each run a small control thread that listens on its own loopback port (9101 and 9102) for one-line JSON commands. Commands therefore never travel on the data path or add to L (§3.3.1).

| **Step** | **Command**                                                                                                                       | **Effect**                                                                                                                                |
|----------|-----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| 1        | Dashboard → engine: {"cmd":"prepare","model":"rf","task":"binary","seed":42,"dt_ms":100,"mode":"steady","scenario":"S1","n":5000} | Engine loads the detector (untimed), calls reset_state(), inserts MODEL_RUN and replies {"ok":true,"run_id":17}                           |
| 2        | Dashboard → streamer: {"cmd":"start","run_id":17,"dt_ms":100,"mode":"steady","n":5000,"sample_seed":42}                           | Streamer opens the data socket and begins emitting                                                                                        |
| 3        | Dashboard → streamer: {"cmd":"stop"}                                                                                              | Streamer sends eos; engine flushes t₄ and sets ended_at                                                                                   |
| —        | {"cmd":"set_rate","dt_ms":20}                                                                                                     | Changes Δt during a run (FR2). Use it for demonstrations only, because experimental runs must keep Δt fixed (MODEL_RUN stores one value). |

run.py starts the engine and the streamer as background processes and then streamlit run. NFR5 is then met in three steps: activate the environment, run python run.py, and press Start. experiments/run_matrix.py sends the same commands without the UI to automate Section 8.

## 7.2 Dashboard layout (Fig. 3.12, Table 3.8)

| **Region**   | **Content**                                                                                                                                                       | **Query / source**                                     |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| Sidebar      | Model selector (IF / RF / LSTM), task (binary / seven-class, RF and LSTM only), Δt slider 1–1,000 ms, Steady / Burst, Start / Stop, status line with record count | Control channel                                        |
| KPI row      | Records processed, alerts, median / p95 / p99 of L, throughput                                                                                                    | v_latency for the active run, excluding warm-up        |
| Top left     | Live traffic: benign vs malicious flows per second, last 60 s                                                                                                     | PREDICTION, bucketed by (t0 − first t0) in seconds     |
| Top right    | Protocol mix: TCP / UDP / ICMP / other (map protocol numbers 6 / 17 / 1)                                                                                          | FLOW_RECORD joined to the run’s predictions, last 60 s |
| Centre left  | Alert table, last 20: time, predicted class, true class (marked as ground truth), score, severity, model                                                          | ALERT ⋈ PREDICTION ⋈ FLOW_RECORD ⋈ MODEL_RUN           |
| Centre right | Latency panel: the four components as a stacked bar (t_tx, t_pre, t_inf, t_ser)                                                                                   | v_latency averages                                     |
| Bottom       | L per record over time with a horizontal reference line at Δt                                                                                                     | v_latency                                              |

- Refresh with @st.fragment(run_every="1s") around the display area. It is built into Streamlit, so no third-party autorefresh package is needed, and the sidebar does not re-run.

- Open the database **read-only**: sqlite3.connect("file:db/nids.db?mode=ro", uri=True). Always filter on t4_store_ns IS NOT NULL, because the newest row gets its t₄ one transaction later.

- Use one colour for benign traffic and one contrasting colour for malicious traffic and alerts, everywhere (§3.3.7).

- Dashboard delay (reported separately, §3.3.6): on each refresh, write perf_counter_ns() − max(t4) to results/dashboard_delay.csv.

# 8. Phase 6 — Empirical Evaluation (Module M8; DSRM activities 4–5)

## 8.1 Detection metrics (objective iii, NFR3)

- Compute from the offline prediction files over the **full test partition**, for each model and seed: precision, recall, F1, ROC-AUC and FPR (binary; positive = Malicious). Report mean ± SD over the five seeds.

- For the seven-class RF and LSTM, also report macro-F1, per-class recall and a confusion matrix (as a plot).

- ROC-AUC uses the continuous score: s for IF, P(malicious) for RF and the binary LSTM, and 1 − P(Normal) for the multi-class models.

## 8.2 Test matrix

| **ID** | **Setting**                                                              | **Runs**                | **Approx. wall time** |
|--------|--------------------------------------------------------------------------|-------------------------|-----------------------|
| S1     | Steady, Δt = 100 ms, 5,000 records                                       | 3 models × 5 seeds = 15 | ~8.3 min each, ~2.1 h |
| S2     | Steady, Δt = 10 ms, 5,000 records                                        | 15                      | ~50 s each            |
| S3     | Rate search: Δt = 50, 20, 5, 1 ms (100 and 10 are reused from S1 and S2) | 15 × 4 = 60             | ~35 min in total      |
| S4     | Burst: 500 @ 100 ms, 1,000 back-to-back, 500 @ 100 ms (2,000 records)    | 15                      | ~100 s each           |

- Use the **binary** models in all timing runs, because that is the primary comparison. Optionally add one S1 run of each seven-class model as a supplementary result.

- Run the matrix with run_matrix.py. **Interleave** the models (IF, RF, LSTM, IF, ...) instead of running all of one model first, so that heat or background load does not favour one model. Keep the dashboard open at a 1 s refresh during every run, as the design specifies, and record that in run_conditions.

- Exclude the first 100 records of each run from the latency statistics. LSTM warm-up rows (is_warmup = 1) are also excluded from the detection checks.

## 8.3 System metrics

| **Metric**            | **Definition used**                                                                                                                                                                                                         |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| L statistics          | Median, p95, p99, mean and max of L_ms per run, plus the mean of each component. Then mean ± SD across the five seeds.                                                                                                      |
| Throughput (Eq. 3.7)  | (records after warm-up) / (last t₄ − first t₀ after warm-up), in records/s                                                                                                                                                  |
| Backlog               | For record i: the number of earlier records j with t₄ⱼ \> t₀ᵢ, which is how many were still unfinished when i was emitted. It is computed from the timestamps after the run, so nothing extra has to be measured during it. |
| Sustainable rate (S3) | The smallest Δt for which p95(L) ≤ Δt **and** the backlog is not growing (mean backlog in the last 10% of the run ≤ that in the first 10% after warm-up). Max sustainable throughput = 1000/Δt records/s.                   |
| Burst recovery (S4)   | Peak L, and the time from the last burst emission until L ≤ 100 ms holds for 20 consecutive records                                                                                                                         |
| NFR checks            | NFR1: p95(L) ≤ 100 ms in S1. NFR2: IF and RF sustain ≥ 100 records/s (S2 passes). For the LSTM, report the figure as a finding.                                                                                             |

## 8.4 Correctness check

For every run, join PREDICTION to the offline prediction file for the same model and seed on row_index; the labels must agree 100% (§3.3.11). For the LSTM, the offline reference must be computed over **the same streamed sequence**, because the windows depend on which rows came before. Any disagreement points to a difference between offline and online preprocessing (control 8).

## 8.5 Outputs for Chapter Four

- results/detection_metrics.csv, per_class_recall.csv, and confusion-matrix PNGs

- results/latency_summary.csv: one row per run with scenario, model, seed, L statistics, components, throughput, max backlog and recovery time

- Plots: latency CDF per model (S1), stacked component bars, throughput and p95 against Δt (S3), L over time during the burst (S4)

- Optional robustness check (§3.2.4): train on Monday–Wednesday and test on Friday, for RF only, and report the drop in F1

# 9. Phase 7 — Testing for Chapter Four

Chapter 3.4 promises unit, integration, system and user acceptance testing. Write the unit and integration tests with pytest while you build each phase rather than at the end.

| **ID** | **Level**   | **Test**                                                                                                  | **Pass criterion**                            | **Req.**    |
|--------|-------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------|-------------|
| UT1    | Unit        | Preprocessor matches the offline scaled arrays                                                            | allclose on 1,000 rows                        | FR6         |
| UT2    | Unit        | Malformed inputs (wrong length, strings, missing key) raise ValueError                                    | All rejected                                  | FR5         |
| UT3    | Unit        | Each Detector returns a valid Result; LSTM returns warm-up for the first W−1 records                      | Types and ranges valid                        | FR7, FR9    |
| UT4    | Unit        | Detector refuses a model path outside artefacts/                                                          | PermissionError                               | NFR6        |
| UT5    | Unit        | Logger writes prediction and alert, t₄ ≥ t₃, FK enforced                                                  | Rows correct                                  | FR10–12     |
| IT1    | Integration | Streamer → engine → DB, 200 records, each model                                                           | Counts match; no negative intervals           | FR1–4, FR11 |
| IT2    | Integration | Dashboard commands start, stop, model switch and rate change                                              | State changes as expected                     | FR2, FR8    |
| IT3    | Integration | Streamed vs offline label agreement                                                                       | 100%                                          | §3.3.11     |
| ST1    | System      | Scenarios S1–S4                                                                                           | Results produced; NFR1/2 verdicts stated      | NFR1, NFR2  |
| ST2    | System      | Sockets bound to loopback only (netstat -an)                                                              | Only 127.0.0.1                                | NFR6        |
| ST3    | System      | Re-running one S1 configuration gives the same detection labels                                           | Identical labels                              | NFR4        |
| UAT    | Acceptance  | 3–5 users, given only a one-page instruction sheet, start a run, switch model, find an alert and read p95 | Done without help; short Likert questionnaire | NFR5        |

NFR7 (modularity) is shown by inspection: point out that adding a model means writing one Detector subclass and one entry in the model selector.

# 10. Build Order and Working in VS Code

| **Phase**    | **Deliverable**                            | **Gate before moving on**       |
|--------------|--------------------------------------------|---------------------------------|
| 0 Setup      | Repo, environment, config, lock file       | Imports work; repo pushed       |
| 1 Data       | Partitions, preprocessor.json, prep_report | Section 3 “Done when”           |
| 2 Models     | 25 model files, offline predictions        | Validation metrics logged       |
| 3 Database   | schema.sql, init_db.py, LatencyLogger      | Fake-insert test passes         |
| 4 Pipeline   | Detectors, streamer, engine                | 200-record run plus clock check |
| 5 Dashboard  | Control channel, run.py, Streamlit app     | Live demo works in three steps  |
| 6 Evaluation | run_matrix.py, metrics scripts, plots      | All CSVs and plots produced     |
| 7 Testing    | pytest suite, UAT sheets                   | All tests pass                  |

The phases are sequential, but the work stays iterative, as DSRM requires (Fig. 3.1). If S1 shows the LSTM missing the target, return to Phase 2 (smaller W or fewer units, for example), retrain, and record that iteration in Chapter Four. That kind of evidence is exactly what Chapter Three says will be gathered.

## 10.1 Keeping the coding assistant on-design

If you use Claude inside VS Code, put a short CLAUDE.md at the root of the repository. The assistant reads it on every task, so it will follow your design instead of falling back on generic defaults. A suggested starting point:

> \# Project rules (real-time NIDS, Chapter 3 design)
>
> \- Split BEFORE any fitting. Scaler, medians, log mask, corr filter: training partition only.
>
> \- Stream cleaned UNSCALED test rows; the engine applies artefacts/preprocessor.json.
>
> \- Timestamps: t0 streamer before json.dumps; t1 on receipt; t2 after validate+scale;
>
> t3 after predict; t4 after COMMIT (written in the next transaction). Never change these.
>
> \- All sockets bind 127.0.0.1. All SQL uses ? parameters. Models load only from artefacts/.
>
> \- Inference is single-threaded. LSTM uses a tf.function call, never model.predict per record.
>
> \- Seeds 42-46; every run stores seed, hyperparameters, env_id in MODEL_RUN.
>
> \- Do not edit data/raw. Do not use the test partition for tuning or thresholds.
