# Chapter Four notes (running list)

A working list for the Chapter Four write-up: what the implementation added to the Chapter Three design, where it departs from it and why, and what has been found along the way. Append new entries as work goes on (see CLAUDE.md). Every figure has a source file in the repository, so the chapter can cite it.

Last updated: 2026-11-16 (Phase 6 §8.3–8.5 system metrics; §2.17, §3.10).

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

| TEST_ENVIRONMENT | `run_conditions.dashboard_open` | `true`/`false` as declared by the caller of `prepare` (the dashboard sends `true`; run_matrix sends its `--dashboard-open`); `null` = not declared. Separates the main S1–S4 runs (dashboard open, §3.3.7) from the dashboard-closed S1 comparison set. |

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

### 2.12 Dashboard details (plan §7.2, Fig. 3.12, Table 3.8)
- **Seed selector** added to the sidebar (seeds 42–46). It is not in Table 3.8, but MODEL_RUN must store the seed (NFR3).
- **Protocol mix shows TCP / UDP / Other.** The test partition has protocol codes 6, 17 and 0 only, so ICMP never occurs. It would appear automatically if present.
- **"Live" windows** (traffic and protocol mix, last 60 s) are measured on the run's emission clock (t0 relative to the newest t0), not wall time, so a finished run still shows its last minute.
- **Warm-up:** latency percentiles, throughput (Eq. 3.7) and component means exclude the first 100 records. Records and alerts count everything stored.
- **Runs started from the dashboard** get `scenario = 'dashboard'`, and run 5,000 (steady) or 2,000 (burst) records from config.
- **Δt slider during a run** sends `set_rate` (FR2, demo-only; the run is marked, §2.11).
- **Colour:** benign is always blue and malicious always orange (a CVD-safe categorical pair). The latency components use four further categorical hues, with a table of values beside the chart.
- **Dashboard delay** (§3.3.6) = `perf_counter_ns()` − max(t4) on each 1 s refresh while a run is live, appended to `results/dashboard_delay.csv`.
- Streamlit runs **headless** and run.py opens the browser itself. Otherwise Streamlit's first-launch e-mail prompt blocks startup, which would break the three-step launch (NFR5).

### 2.13 Isolation Forest score sign (plan §1.1 item 4)
- s = −score_samples(x), because scikit-learn returns the negative of the score in Eq. 3.3.

### 2.14 Test-matrix driver (plan §8.2, `experiments/run_matrix.py`)
- Headless, through the same control-channel calls as the dashboard (`ControlClient.prepare → start → wait for eos`). Binary models only.
- **Order:** repetition r uses model seed 42+r. Inside a repetition, the scenario steps run in order (S1; S2; S3 at 50/20/5/1 ms; S4), and each step runs IF, RF, LSTM interleaved. The record sample is the same for every run (`stream.sample_seed` = 42). There is a 20 s pause between runs.
- **Two sets**, because the dashboard state can't change mid-matrix: `main` = S1–S4 with the dashboard open (105 runs), and `closed` = S1 with the dashboard closed (15 runs). Both use scenario label `S1` and are told apart by `run_conditions.dashboard_open`.
- **Refuses to start** if net.core.wmem_max/rmem_max or the probed SO_SNDBUF is below 4 MB (§2.10), if `--dashboard-open` doesn't match whether the dashboard answers on 127.0.0.1:8501, if the services are busy, or if the engine writes a different database. If any streamer start reports a short send buffer, the matrix stops.
- **Dashboard really open:** a dashboard-open run counts only if `results/dashboard_delay.csv` has rows for it, i.e. a browser refreshed during the run. Otherwise the matrix stops.
- **Resumable:** a run is skipped when MODEL_RUN holds it with `end_reason='complete'`, `rate_changes IS NULL`, PREDICTION count = n_records and the same dashboard state. A power cut leaves the run in progress with end_reason NULL, so only that run is repeated. A per-run log goes to `results/run_matrix_log.csv`.
- Smoke runs use `--scenario-prefix` (e.g. `smoke-S1`) and `--n-override`, so they can never be counted as experiment runs.

### 2.15 Dashboard UI pass (2026-11-15; Fig. 3.12 layout kept, nothing measured differently)
- **Title** "Real-Time Network Intrusion Detection System" in the sidebar and the browser tab, subtitle "Machine learning · IDS2025 replay". Streamlit's Deploy button and main menu are hidden with CSS (this also removes the menu's theme switch; the OS theme still applies).
- **KPI row:** p95 and p99 carry a status chip, green "✓ ≤ target Δt" or red "✗ > target Δt", in the NFR1 form p95(L) ≤ Δt (`queries.within_target`). Throughput shows "x% of 1000/Δt" in grey, with no pass/fail colour, because Eq. 3.7 at a steady 100 ms gives about 9.99 rec/s and any threshold would be invented. Burst runs say "burst load" (there is no single offered rate). The tooltips give the numbers.
- **Detection KPI row (added beyond Table 3.8):** running precision, recall and false alarms (FP count), from PREDICTION vs FLOW_RECORD.true_label over post-warm-up records (the same first-100 exclusion as the latency KPIs). This is a binary view for every task: any non-Normal class counts as malicious, so a seven-class prediction of the wrong attack class still counts as a hit. It is a live display aid only. The reported detection metrics remain the offline full-test-partition figures (plan §8.1), and the caption says so.
- **Alert table:** severity is shown as coloured chips (▲ High red, ● Medium orange). Rows with a malicious prediction but true class Normal are shaded and labelled "✗ False positive" in a Check column, so the cue is never colour alone.
- **Charts:** latency components are a stacked horizontal bar in pipeline order t_tx, t_pre, t_inf, t_ser with the total at its end. The latency chart has a log-scale toggle, a bold dashed Δt line and a dotted p95 line, coloured green or red against Δt and labelled. Live traffic is a stacked area (benign bottom, malicious top). The protocol mix uses one neutral grey, so blue and orange only ever mean benign and malicious.
- **Sidebar:** grouped into "Run settings" and "Actions". The records/s equivalent (1000/Δt) is shown under the Δt slider. While a run is active, model, task, seed, load and Start are locked. **The Δt slider stays live in steady runs**, because FR2 requires Δt to be configurable "before and during a run" (the change is demo-only and marks the run, §2.11). It locks in burst runs, where the streamer rejects set_rate. The sidebar status fragment triggers one full-app rerun when a run starts or ends, so the lock follows runs started elsewhere and runs that end on their own.
- **Cost:** refresh is still 1 s, and the queries are the same five per refresh. `true_label` rides on the FLOW_RECORD join that `protocol` already used. The new figures are pandas arithmetic on the frame already loaded.

### 2.16 Sidebar shows runs started by the experiment script (2026-11-16)
- When the active run was started by `experiments/run_matrix.py` rather than the sidebar (MODEL_RUN.scenario ≠ `dashboard`), the sidebar shows **that run's** model, task, seed, Δt and load, with the note "started by experiment script (scenario …)". Before this, it kept whatever the viewer had last selected, which could misdescribe the run on screen. The viewer's own selections are restored when the run ends.
- For such runs the **Δt slider is locked as well**. For dashboard runs it stays live, as the FR2 demo feature. A `set_rate` on an experiment run would mark it (`rate_changes`), and run_matrix would then have to repeat it.
- Display only: nothing is sent to the engine, and the refresh queries are unchanged. The engine's existing `status` reply already carries the run's settings and scenario.

### 2.17 System-metric definitions used (plan §8.3–8.4; `nids.evaluation.system_metrics`)
- **Runs used:** scenario S1–S4, end_reason `complete`, rate_changes NULL, PREDICTION count = n_records, dashboard state declared. Dashboard-open runs also need ≥ 1 dashboard-delay row (run_matrix's acceptance rule), so zero-refresh runs **65 and 84** are excluded; their repeats are used. If a configuration had two accepted runs, the later would be used (none at present). Excluded runs stay in `latency_summary.csv` with `included = False` and the reason.
- **Warm-up:** the first 100 records (seq_no < 100) are excluded from L, the components, throughput and backlog statistics. Backlog still counts warm-up records as earlier records j, because they occupy the queue.
- **Backlog** of record i = earlier records j with t4_j > t0_i, computed after the run from the timestamps. The engine is FIFO, so t4 rises with seq_no and a sorted search gives it exactly.
- **Throughput (Eq. 3.7)** runs from the first post-warm-up t0 to the last t4. Under overload it includes the backlog inherited from the warm-up records, so it **sits below the engine's service rate** (synthetic check: 15 ms service, 10 ms Δt → 57.1/s, not 66.7/s; `tests/test_system_metrics.py`). Report it as the Eq. 3.7 figure, not as capacity.
- **S4 recovery** = t4 of the first record of the first run of 20 consecutive records with L ≤ 100 ms after the last back-to-back emission, minus that emission's t0. "Not recovered" when no such run occurs before the run ends.
- **Verdicts:** NFR1 (p95 ≤ 100 ms in S1) and NFR2 (S2: p95 ≤ 10 ms and backlog not growing) are **PASS** when all 5 seeds pass, **FAIL** when the seed mean fails, and MIXED otherwise. NFR2 is required for IF and RF and is a finding for the LSTM. Only dashboard-open runs feed the verdicts (the design condition).
- **Sustainable rate (S3):** the smallest Δt in {100, 50, 20, 10, 5, 1} at which all 5 seeds pass and every larger Δt passes too.
- **Backlog "not growing" uses a 1-record tolerance (clarification of plan §8.3; decided 2026-11-16).** A run counts as not growing when its mean backlog in the last 10% of post-warm-up records exceeds the first 10% by **at most 1 record** (`BACKLOG_TOL`, `--backlog-tolerance`). Why: the literal rule (last ≤ first) misfires on stable runs. Their mean backlog is about 0.01 records, so one HDD stall near the end flags them as "growing". Runs within their p95 target show growth of −0.06 to +0.04 records; runs that can't keep up show growth of −6 to 4,337 (median ≈ 2,300), so a 1-record tolerance separates them with a wide margin. Under the literal rule no model would have a sustainable rate even at 100 ms (3–4 of 5 seeds pass). With the tolerance, the backlog check changes no verdict that p95 ≤ Δt alone would not give. The NFR verdicts are the same under both rules.

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

### 3.6 Dashboard delay and load with the dashboard open (preliminary, 2026-11-15)
- One demonstration run (RF binary, seed 42, Δt = 100 ms, steady, stopped after ~2,700 records). The dashboard was open in **headless Chrome on the same 4-thread laptop**, so this is not an S1 measurement.
- Dashboard delay over 26 refreshes: median **289 ms**, min 119 ms, max 617 ms.
- In the same run (records 100–1,870): L p50 37.6 ms, **p95 102.7 ms, p99 484 ms**, throughput 10.0 rec/s. Mean components: t_tx 10.8, t_pre 0.18, t_inf 42.3, t_ser 2.2 ms. L spiked to ~1,000 ms several times.
- The large mean t_tx is queueing: a record waits in the socket while the engine is still busy with an earlier one (RF ≈ 40 ms per record), so spikes propagate.
- To separate the causes in S1: compare runs with the dashboard closed vs open (the design requires it open, §3.3.7), and line up spikes with WAL checkpoints (§3.3). If p95 > 100 ms holds in S1, NFR1 fails for RF under dashboard load. Report that as a finding.

### 3.7 Test-matrix wall time is longer than plan §8.2 says
- Plan §8.2 gives S3 as "~35 min in total". The schedule alone is 15 × (250 + 100 + 25 + 5) s ≈ 95 min, and at Δt = 5 and 1 ms the run is limited by the engine's service time rather than the schedule (RF ≈ 40 ms/record → ~3.4 min per 5,000 records). Estimated S3 total: **~2h36m** (`run_matrix.py --dry-run`, using the `results/detector_timing.json` means + 3 ms per record).
- Estimated totals: main set 105 runs, 5h38m of runs + 35 min of pauses ≈ **6h13m**. Closed set 15 runs ≈ **2h11m**. Overall ≈ **8h24m**.
- 2-run smoke test (200 records, Δt = 100 ms, IF and RF): 23.5 s and 21.5 s against estimates of 22.9 s and 24.9 s, both complete and nothing malformed, with pacing lateness p99 of 1.8 and 0.6 ms.

### 3.8 Dashboard UI pass: before/after p95 is inconclusive (2026-11-15)
- Method: a separate stack (ports 9200–9202, dashboard 8502, its own `db/ui_ab.db`, so `db/nids.db` was untouched), dashboard open in headless Chrome at 1 s, RF binary, seed 42, Δt = 100 ms, steady, 5,000 records, scenario `ui_before` / `ui_after`. Driver scripts were kept outside the repository.
- **Before** (old UI, run 1 in ui_ab.db): L p50 63.2 ms, **p95 184.5 ms**, p99 1,916 ms, max 2,221 ms; 11.9% of records over 100 ms. Mean components: t_tx 45.4, t_pre 0.20, t_inf 63.2, t_ser 1.2 ms. Dashboard delay median 272 ms (475 refreshes). The 5-minute load average was 3.5–4.6 on 4 threads, so the machine was already busy. Compare the RF run 1 in `db/nids.db` (old UI, Brave): p95 116.9 ms, t_inf ≈ 42 ms.
- **After** (new UI, run 2 in ui_ab.db): **invalid**. A second NIDS stack (`run.py` on 9100/8501, dashboard in Brave) was started during it and ran an Isolation Forest run (db/nids.db run 2, 05:02–05:10 UTC), and VS Code restarted. Load average reached 9.9–11. Mean t_inf doubled to 126 ms, the backlog grew, and L reached ~160 s (p95 157,730 ms). This measures the contention, not the UI. The overlap works both ways: **db/nids.db run 2 (IF) ran while this RF run loaded the CPU**, so it should not be used as an experiment run.
- Conclusion: **no valid evidence either way** about whether p95 changed. The query load is unchanged (above), so a difference would have to come from rendering (Styler, two more metric rows, chip column). To settle it, repeat the pair back to back on a quiet machine (one stack, no other browser on the dashboard), ideally two or three pairs given the run-to-run spread seen here (p95 116.9 vs 184.5 ms on the same old UI).

### 3.9 Main matrix set: dashboard conditions and interruptions (2026-11-16)
- **Completed:** all 105 runs of the `main` set (S1–S4, dashboard open) are complete. Each has end_reason `complete`, 5,000 (S4: 2,000) PREDICTION rows, 0 malformed and `rate_changes` NULL (MODEL_RUN, run_ids up to 112). Sources: `results/matrix_main.log` (console) and `results/run_matrix_log.csv` (one row per run).
- **Intended procedure:** during dashboard-open runs the browser tab was kept in the foreground with the screen on, so the dashboard would refresh every 1 s as §3.3.7 specifies.
- **Browser throttling stopped the matrix twice.** Runs **65** (S3, LSTM, seed 44, Δt 1 ms) and **84** (S3, LSTM, seed 45, Δt 5 ms) completed with **zero** dashboard refreshes: the browser had throttled the tab. run_matrix stopped, as designed, and on resume the configuration was repeated (84 → run 85, 37 refreshes). The zero-refresh runs stay in MODEL_RUN but are not counted as matrix runs.
- **Measured refresh rate falls short of the design for most accepted runs.** Refreshes per second of run time (rows in `results/dashboard_delay.csv` ÷ (last t4 − first t0)): median **0.14/s** against an expected ~1/s. **60 of 105** accepted runs are below 0.5/s and 41 are below 0.1/s. Runs 47, 48, 70, 71 and 72 (S1, 500 s) received only 14–15 refreshes, roughly one every 35 s, which is consistent with a backgrounded tab or the screen turning off. The acceptance check required only ≥ 1 refresh per run, so these runs passed it.
  - Consequence: in those runs the dashboard put **less read load on SQLite and the CPU** than the design's 1 s refresh, so their L may be optimistic compared with a fully refreshing dashboard. When comparing with the dashboard-closed S1 set, use the per-run refresh rate as a covariate, or keep only runs near 1/s. Dashboard-delay statistics (§3.3.6) come from far fewer samples in those runs.
  - Possible fix (not yet applied): require a minimum refresh rate (e.g. ≥ 0.8/s) in run_matrix's acceptance check and rerun the throttled configurations.
- **Interruptions and resume.** The main set was interrupted by power cuts and resumed with run_matrix's resume logic. The console log shows two interrupted sessions:
  - During **run 29** (S2, RF, seed 43): it was left with end_reason NULL after 652 records (a hard stop), and run_matrix itself was ended by a signal while waiting.
  - During **run 49** (S1, LSTM, seed 44): stopped cleanly by a signal (end_reason `stopped`, 1,860 records).
  - Both were repeated on resume, after the services had been restarted (the first resume attempt was refused because they were not running). Including the two throttling stops, the set was run in **five sessions** (resumes at 25, 44, 59 and 77 runs complete). No completed run was lost or repeated.

### 3.10 System metrics, main set (dashboard open; 2026-11-16, `results/latency_summary.csv`, `system_summary.csv`, `nfr_verdicts.csv`)
- **NFR1 PASS for all three models** (S1, p95 of L, mean ± SD over 5 seeds, 5/5 seeds each): IF **36.2 ± 9.3 ms**, RF **67.4 ± 16.3 ms**, LSTM **18.5 ± 1.7 ms**. p99: 74.8 / 161.6 / 37.7 ms, so RF's tail exceeds 100 ms. Max L 295–373 ms for all models: isolated spikes (§3.3 WAL/HDD), visible in `plots/burst_S4.png` before the burst as well.
- **NFR2 FAIL for IF and RF** (S2, Δt 10 ms, 0/5 seeds sustain): Eq. 3.7 throughput IF **70.8 ± 11.9/s**, RF **24.8 ± 3.6/s**, and the backlog grows to ~1,400 and ~3,800 records. **LSTM (finding):** throughput 99.9/s and p50 9.6 ms, but p95 267 ms > 10 ms with a backlog of up to ~40 records, so it also fails the p95 ≤ Δt rule at exactly 100 records/s.
- **Inference under dashboard load is slower than the standalone benchmark:** mean t_inf in S1 IF 20.6, RF 44.8, LSTM 10.9 ms vs 11.8 / 37.2 / 6.5 ms in `detector_timing.json` (§3.2). This is consistent with a shared 4-thread CPU. The open-vs-closed comparison will quantify it.
- **S3 sustainable rate** (all 5 seeds, 1-record backlog tolerance, §2.17): **IF 50 ms (20 records/s), LSTM 50 ms (20 records/s), RF 100 ms (10 records/s).** At Δt ≤ 20 ms (IF, LSTM) and ≤ 50 ms (RF) the queue grows without bound, and p95 reaches tens to hundreds of seconds (`sustainable_rate.csv`, `plots/rate_search_S3.png`).
- **S4 burst:** peak L IF 14.1 ± 1.6 s, RF 43.7 ± 6.9 s, LSTM 7.5 ± 1.1 s. Recovery IF **16.3 ± 2.1 s**, LSTM **8.7 ± 1.2 s** (5/5 each). **RF did not recover within the run (0/5):** the 1,000-record backlog at ~44 ms per record (~44 s of work) drains at only ~56 ms per 100 ms tail record, which needs ~77 s against the 50 s tail.
- **§8.4 agreement: 105/105 runs at 100%**, 0 disagreeing records. Streamed labels equal the offline labels for the same model, seed and row_index, so online and offline preprocessing match (control 8).
- **Dashboard refresh:** 60 of 105 runs are below 0.5 refreshes/s (`low_refresh` column; §3.9).
- **Open vs closed (S1):** see §3.13 (closed set completed 2026-11-17).

### 3.11 Testing (plan §9; 2026-11-16, `docs/testing_report.md`)
- Full suite: **124 passed, 2 failed** of 126 (9 min 11 s). Both failures are test faults, not pipeline faults, and no code was changed: (a) the NFR6 static SQL check flags a literal, parameterised query held in a local variable in `system_metrics.load_records` (false positive); (b) the run_matrix headless test counts an `rm-test-S1` run created earlier in the session by a dashboard test that uses the same label. It passes on its own.
- **ST2 PASS:** with `run.py` and the dashboard running, `netstat -an` shows only 127.0.0.1 listeners (9100, 9101, 9102, 8501).
- **ST3 PASS:** S1 run 6 (LSTM binary, seed 42) replayed offline twice through Preprocessor + Detector: 0 of 5,000 labels differ, and score/confidence differences are 0.0.
- UT3's "warm-up for the first W−1 records" clause does not apply under formulation B (§2.6). UAT materials are in `docs/uat/`, and UAT has not been run yet.

### 3.12 Closed set, first attempt: runs 113 and 114 invalidated (2026-11-16/17)
- The first closed-set attempt (22:37–22:57 +0100, 16 Nov) stopped at run 2/15 with `TimeoutError: timed out` (`results/matrix_closed.log`, `db/nids.services.log`). Run 114 (S1, RF, seed 42) had completed, 5,000/5,000. The error was a 10 s control-channel request after the run: the streamer could not answer. The engine had already closed the data socket at 22:55:04.7, but the streamer took 21.5 s to run the next three statements, and both services then needed more than 15 s to exit (killed). Cause: host memory pressure with swapping to the HDD. sar 22:50–23:00: swap-out 510 pages/s (≈5 before), swap +940 MB (43.8 → 66.3%), iowait 37%, sda 73% util, commit ≈190%. No OOM kill. The other heavy processes running were Brave, Spotify, VS Code and snap-store.
- Run 113 (S1, IF, seed 42) also ran under background load: one storage stall of 11 s at 22:43:08–22:43:19 (t_ser up to 961 ms) right after a snap-store launch. p95 105.4 / p99 1,757 / max 2,645 ms, against 51.2 / 96 / 333 ms for the dashboard-open run 4. Run 114: p99 584 ms vs 179 ms for run 5, with storage stalls every 30–60 s.
- Neither run overlapped the testing-report session (05:52–06:24 the same day).
- **Both runs relabelled `scenario = 'invalid-S1'`** (end_reason stays `complete`; PREDICTION rows kept). run_matrix therefore repeats them and system_metrics ignores them. Backup before the change: `db/nids.before_invalidate_113_114.db`. Run 114 never reached `results/run_matrix_log.csv`; run 113's row there is left as an append-only record.
- Robustness gap (not fixed): one 10 s status timeout aborts the whole matrix even when the run finished. Before the rerun, close other heavy applications and check swap (`free -h`) so the closed set runs on a quiet machine.

### 3.13 Dashboard open vs closed, S1 (2026-11-17, `results/dashboard_open_vs_closed.csv`)
- **Closed set completed:** runs **115–129** (S1, Δt 100 ms, steady, 5,000 records, seeds 42–46 × IF/RF/LSTM). All 15 are complete with 0 malformed, finished in one session (≈8m22s each, ending 02:29 +0100, 17 Nov; `results/matrix_closed.log`). **0 dashboard refreshes** were logged in every closed run, which confirms the dashboard really was closed. Invalidated runs 113/114 (§3.12) are excluded. §8.4 agreement is now **120/120 runs at 100%**.
- **p95(L), mean ± SD over 5 seeds (open = main set, §3.10):**

  | Model | Open (all) | Open, refresh ≥ 0.5/s | Closed | Open − closed | Mean t_inf open → closed |
  |---|---|---|---|---|---|
  | IF | 36.2 ± 9.3 ms | 36.1 ms (n = 3) | 39.6 ± 0.8 ms | −3.4 ms | 20.6 → 21.7 ms |
  | RF | 67.4 ± 16.3 ms | 83.9 ms (n = 2) | 51.2 ± 0.9 ms | **+16.2 ms** | 44.8 → 39.7 ms |
  | LSTM | 18.5 ± 1.7 ms | 19.1 ms (n = 2) | 24.6 ± 0.9 ms | −6.0 ms | 10.9 → 12.7 ms |

- **Finding:** only **RF** shows a dashboard cost that holds across runs. With the dashboard open its p95 is 16 ms higher (24 ms higher in the runs near the designed 1/s refresh) and t_inf is 5 ms higher. This fits RF being the most CPU-bound detector on a 4-thread CPU that it shares with Streamlit and the browser. For **IF** the difference is within the open-set spread. For **LSTM** the closed runs are *slower* (+6 ms p95, +1.8 ms t_inf), so that difference cannot be blamed on the dashboard.
- **Confounding (state in Ch. 4):**
  - Open and closed runs were not interleaved. The open set ran over five sessions with varying refresh rates (§3.9). The closed set ran in one overnight session on a quiet machine.
  - Session effects such as background load, CPU frequency and page-cache state therefore mix with the dashboard effect. The closed SDs (≈0.9 ms) are 2–18× smaller than the open SDs, which is consistent with this.
  - The open subset near 1/s has only 2–3 runs per model.
  - Conclusion: dashboard overhead at a 1 s refresh is **small next to the 100 ms budget**. It is measurable for RF and not separable from session noise for IF or LSTM.
- **NFR1 unchanged:** it remains PASS for all three models. NFR1 is judged on dashboard-open runs, as specified. The closed runs pass as well (worst seed p95 52.4 ms, RF seed 43). NFR2 verdicts are unchanged because they come only from S2.
- **The latency spikes are not caused by the dashboard:** closed-run max L is still 244–453 ms (RF p99 121–170 ms). The isolated spikes seen in §3.10 therefore come from the storage path (WAL/HDD), not from dashboard reads.
- **Run 114 vs its repeat, run 116 (RF, seed 42):** p99 584 → 142 ms. This supports the §3.12 diagnosis that the first attempt was distorted by host load.

