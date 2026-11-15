# Chapter Four notes (running list)

A working list for the Chapter Four write-up: what the implementation added to the Chapter Three design, where it departs from it and why, and what has been found along the way. Append new entries as work goes on (see CLAUDE.md). Every figure has a source file in the repository, so the chapter can cite it.

Last updated: 2026-11-15 (end of Phase 5 §7.1).

## 1. Database fields added beyond Table 3.10

Plan §1.1 item 3 allows these as "additions made during implementation". Schema: `src/nids/storage/schema.sql`.

| Table | Field | Why it was added |
|---|---|---|
| PREDICTION | `seq_no` | Position of the record in the run's emission order. Used for the 100-record warm-up exclusion (Table 3.13) and to order records without relying on timestamps. |
| PREDICTION | `is_warmup` | Flags records excluded from statistics (LSTM warm-up under formulation A). Always 0 under formulation B, but kept so the rule is explicit. |
| MODEL_RUN | `task` | `binary` or `multi`. RF and LSTM each have two model families, so model_name alone does not identify the model. |
| MODEL_RUN | `scenario` | S1–S4 (or a test label). Groups runs into the test matrix (§8.2) without re-deriving it from dt and load condition. |
| MODEL_RUN | `n_records` | Planned number of records for the run (5,000 steady, 2,000 burst). Compared against the PREDICTION count. |
| MODEL_RUN | `malformed_count` | Records discarded by validation (FR5, process 2.2). Without it, FR5 can't be shown to work. |
| TEST_ENVIRONMENT | `storage_type` | Table 3.14 says storage type is recorded, but Table 3.10 had no field for it. Matters because t_ser depends on the disk (see §3, WAL stalls). |
| TEST_ENVIRONMENT | `run_conditions` | JSON with power profile, CPU governor, AC power, inference threads, SQLite journal mode and synchronous level, wal_autocheckpoint, effective socket buffers, net.core.{w,r}mem_max and dashboard refresh. Table 3.14 says run conditions are recorded but gave no field. |
| MODEL_RUN | `rate_changes` | NULL when Δt stayed fixed for the whole run. Otherwise a JSON list of demo `set_rate` changes `[{from_seq, dt_ms, at}]`. Experiment runs are exactly those `WHERE rate_changes IS NULL` (plan §7.1: set_rate is demo-only). |
| MODEL_RUN | `end_reason` | How the run ended: `complete` (eos after n records), `stopped` (stop command), `disconnected` (stream lost), `cancelled` (prepared, never started), `shutdown`. Separates complete experiment runs from interrupted ones. |

Older databases get `rate_changes` and `end_reason` through `nids.storage.db.ensure_additions` (ALTER TABLE, run when the engine or init_db opens the database).

## 2. Deviations from the Chapter Three design

### 2.1 Dataset is one Excel sheet, not daily CSV files (§3.2.4, Fig. 3.3, FR1)
- IDS2025 as obtained is `data/raw/IDS2025.xlsx`, one sheet (`sample_91830_without_text_field`), 91,830 rows × 80 columns. It is not a set of Monday–Friday CSVs.
- No Timestamp, day, Flow ID or IP column (`results/prep_report.json` → `source.timestamp_present = false`, `day_column_present = false`). There is no merge and no `source_day`.
- Consequence: the optional day-based robustness check (train Mon–Wed, test Fri; plan §8.5) cannot be done. State it as a limitation.
- The workbook is read once by `nids.prep.convert_xlsx` (SHA-256 checked before and after, values unchanged), and everything after that reads the Parquet copy.
- `Flow Bytess` / `Flow Packetss` hold text cells (`Infinity` ×50/×127, `NaN` ×77), parsed in Phase 1 step 4.

### 2.2 Port columns dropped as identifiers
- The only identifier columns present are `Source Port` and `Destination Port` (prep_report → `cleaning.roles.identifiers_dropped`). They were dropped from the features. Flow ID, IPs and Timestamp were checked and are absent.
- Why: ports identify endpoints rather than describe flow behaviour, and §3.2.4 / plan §3.1 step 2 list Src/Dst Port among the identifier columns that are never features. 77 numeric features remain before fitting.

### 2.3 Constant and correlation filters fitted after the split (Fig. 3.3, leakage control 1)
- Fig. 3.3 removes correlated columns before the split. Under leakage control 1, every statistic must come from the training partition only. So the constant-column check, the |r| > 0.98 filter, the skewness test and the medians are all computed on the training partition after the stratified split (plan §1.1 item 2, §3.1 step 10).
- Result (prep_report → `fit`): 8 constant columns dropped, 20 columns dropped by the correlation filter, **49 features** retained. Fig. 3.3 should be updated, or this change stated.

### 2.4 Protocol excluded from log(1+x)
- `Protocol` passes the log rule (min ≥ 0, |skew| > 1) but is a categorical code (0/6/17). log1p would distort its spacing without making it less skewed in any meaningful sense, so it is standardised only (`config.yaml` → `prep.log_exclude`; prep_report → `fit.log_excluded_categorical`).

### 2.5 Skewed columns with negative values are not log-transformed
- The rule logs only non-negative columns (§3.2.4). Nine strongly skewed columns contain negative values in training and are left unlogged (standardised only): Flow Bytess, Flow Packetss, Flow IAT Mean, Flow IAT Max, Flow IAT Min, Fwd IAT Min, Fwd Header Length, Init_Win_bytes_forward, Init_Win_bytes_backward (prep_report → `fit.skewed_but_negative`).
- Negative values in these CICFlowMeter fields are known artefacts (e.g. −1 window sizes, negative IATs). 34 columns are log-transformed.

### 2.6 LSTM uses formulation B (§3.3.10, plan §4.3)
- The main design (A) feeds windows of the last W records. The order check showed the sheet is **sorted by class, not time**: 79 label runs over 91,830 rows, P(next row has same label) = 0.9992. 99.78%/99.55%/99.13% of windows of W = 5/10/20 contain one label (prep_report → `order_check`).
- Under A, windows would encode position in a class block and inflate scores without measuring sequential behaviour. **Formulation B** treats each flow's 49 features as a sequence of 49 steps, input (49, 1). It needs no buffer, no W−1 warm-up and no dependence on order. The W search dimension does not apply.

### 2.7 Stream replayed in a seeded random order (§3.3.11 "original order")
- Replaying in original order would send long single-class blocks (a 5,000-record run would be ~1,400 Normal, then ~1,400 DoS, …). That is unrealistic, and dashboard and S4 burst behaviour would depend on which block happened to be emitted.
- The stratified sample is replayed in a permutation fixed by `sample_seed`, so it stays reproducible (NFR4) and mixes classes. No model depends on order (IF, RF, LSTM-B). `nids.stream.streamer.draw_sample`.

### 2.8 LSTM search on a 0.3 stratified fraction of training (compute)
- The 15-trial LSTM random search (binary and multi) ran on a stratified 30% of the training partition (12,030 rows; `artefacts/lstm_*_config.json` → `search.search_frac = 0.3`). The chosen configuration was then retrained on the full training partition with seeds 42–46.
- RF used the **full** training partition for its 30 × 5-fold search (`search_frac = 1.0`, 40,102 rows), so this deviation applies to the LSTM only. Made for compute reasons on a CPU-only laptop, as plan §4.2 allows.

### 2.9 Threshold rule: tune only if validation F1 improves by ≥ 0.005
- For RF and LSTM binary, the F1-maximising threshold replaces 0.5 only if it raises validation F1 by at least 0.005 (`MIN_TUNED_GAIN` in `train_rf.py`; same rule in `train_lstm.py`). This avoids tuning to validation noise.
- Outcome: both kept τ = 0.5. RF gain was +0.0007 (tuned τ 0.371), LSTM gain +0.0024 (tuned τ 0.349). Multi-class models use argmax. Isolation Forest always uses its validation-F1-maximising τ per seed (there is no natural default).

### 2.10 Socket send buffer (plan §6.2, S4)
- The streamer requests a 4 MB SO_SNDBUF so a 1,000-record burst does not block the sender. Linux caps it at `net.core.wmem_max` (default 212,992 bytes), which would silently shrink the buffer. The streamer warns when granted < 4 MB, and the effective value is recorded per run in `run_conditions`.
- On the test machine `net.core.wmem_max` and `rmem_max` were raised to 8,388,608 (`sysctl -w`). **This is not persistent**: it is not in /etc/sysctl.d, so it must be re-applied after a reboot before measured runs. Every run records the value, so a missed reapply would show up.

### 2.11 Control channel and MODEL_RUN marking (Phase 5, plan §7.1)
- Engine and streamer are long-running services with loopback control ports 9101/9102. Many runs happen without restarting processes. The model loads on `prepare`, outside the timed path.
- `set_rate` (FR2) is demonstration-only. It is logged, and the run is marked in `rate_changes` (§1 above). The streamer rejects it in burst mode, because the burst shape is the experiment.

### 2.12 Isolation Forest score sign (plan §1.1 item 4)
- s = −score_samples(x), because scikit-learn returns the negative of the score in Eq. 3.3.

## 3. Findings so far

### 3.1 Isolation Forest has a ~40% false-positive rate
- Test partition, five seeds (`results/detection_metrics.csv`): FPR **0.396 ± 0.017**, precision 0.756, recall 0.953, F1 0.843, ROC-AUC 0.857.
- For comparison, RF binary has FPR 0.031 (F1 0.982, AUC 0.998) and LSTM binary has FPR 0.050 (F1 0.964, AUC 0.994).
- Interpretation: IF is trained on Normal rows only, and τ is chosen to maximise F1. With malicious rows the majority of the cleaned data (32,239 vs 25,050), F1 rewards recall, so τ is set low and many Normal flows are flagged. This is a real limitation of the unsupervised detector in this setting. Report it, and do not re-tune on test.

### 3.2 RF single-record inference takes ~35 ms
- `results/detector_timing.json` (1,000 single predictions, one thread, n_jobs=1): RF binary median **35.3 ms**, p95 48.3 ms, p99 61.8 ms. IF median 10.7 ms. LSTM (tf.function) median 5.9 ms.
- Cause: 382 trees with max_depth 30, walked one row at a time (`rf_binary_config.json`).
- Implication for NFR2 (≥ 100 records/s, i.e. ≤ 10 ms per record): RF inference alone exceeds 10 ms, so RF is not expected to sustain Δt = 10 ms (S2). S3 will measure the sustainable rate. If needed, a DSRM iteration back to Phase 2 (fewer or shallower trees) should be recorded.

### 3.3 WAL checkpoint stalls on the spinning disk
- Test machine storage is an **HDD** (TEST_ENVIRONMENT.storage_type = `HDD (sda, ST500VT000-1DK14)`, i7-5500U).
- `wal_autocheckpoint` is left at its default (1,000 pages) on purpose. The periodic checkpoint is a real storage cost and shows up in t_ser and as queueing in later records. It is reported rather than tuned away (run_conditions records it).
- Evidence so far: clock check (1,000 records @ 100 ms, IF) had median t_tx 0.157 ms and p99 0.99 ms, but **max 178 ms**. That is a rare long stall consistent with a checkpoint/fsync on the HDD delaying the engine. To quantify in S1: line up L spikes with checkpoint intervals and report how much of p99/max they explain.

### 3.4 Clock check passes (§3.3.6)
- All t_tx > 0, median 0.157 ms. `time.perf_counter_ns()` is consistent across the streamer and engine processes on this machine, so no switch to `time_ns()` is needed.

### 3.5 Data cleaning
- 34,474 exact duplicates removed (24,836 of them PortScan, which falls from 25,409 to 554 rows), 38 conflicting-duplicate rows removed, and 29 rows with missing values dropped (0.05% < 1% rule). Final count is 57,289 rows, split 40,102 / 8,593 / 8,594 (prep_report → `cleaning`, `split`). The PortScan collapse matters for per-class results.
