# Testing report (plan §9, Chapter 3 §3.4)

Date: 2026-11-17 (final run, commit `c7594ca`). First run 2026-11-16 on commit `50f7b98`; fixes in §4.
Environment: Ubuntu, Python 3.12.3, pytest 9.2.0, CPU only (the test machine recorded in
TEST_ENVIRONMENT: i7-5500U, 4 logical CPUs, HDD).

## 1. Summary

| Item | Outcome |
|---|---|
| Full pytest suite (`python -m pytest`, 127 tests, 7 min 21 s, 2026-11-17) | **127 passed, 0 failed, 0 skipped.** History: 2026-11-16, 124 passed / 2 failed of 126; 2026-11-17 first re-run after fixes, 125 / 1 (a third fault, §4.3); then 126 / 0. Test 127 (NFR2 sustained-rate check) was added with commit `e70d005`. |
| Plan §9 test IDs (UT1–UT5, IT1–IT3, ST1–ST3) | **10 of 11 met**, with every supporting test passing. UAT is pending. |
| ST2: sockets bound to loopback only (`netstat -an`) | **PASS.** All four listening sockets are on 127.0.0.1 (§5). |
| ST3: S1 configuration re-run gives the same labels | **PASS.** Run 6 replayed twice: 5,000 of 5,000 labels identical, score and confidence differences 0.0 (§6). |
| UAT (NFR5) | Not yet run. Materials ready in `docs/uat/` (instruction sheet and questionnaire, Markdown and A4 PDF). |

All three failures seen (§4) were faults in the tests or test isolation, not in the
pipeline. None affected a measured result.

## 2. Results by plan §9 test ID

"Tests" counts the pytest tests whose main purpose is that ID (column 3 of the table in
§3). Some tests also support a second ID, and §3 shows both.

| ID | Level | Req. | Pass criterion (plan §9) | Evidence | Tests | Result |
|---|---|---|---|---|---|---|
| UT1 | Unit | FR6 | allclose on 1,000 rows | `test_done_preprocessor_matches_X_test`: online `Preprocessor.transform` on 1,000 seeded test rows is allclose to offline `X_test.npy`, and the feature list is identical. Five more tests check the fit rules and that fitting uses the training partition only. | 6 | **PASS** |
| UT2 | Unit | FR5 | All rejected | `test_preprocessor_rejects_malformed`: short or long vector, a string value and a nested list all raise ValueError. "Missing key" applies at message level and is covered by `test_malformed_lines_counted_engine_continues`: not-JSON, wrong length, missing `row_index`, unknown flow and non-object lines were all rejected and counted (malformed_count = 5), the 40 good rows were predicted, and the engine kept serving. | 2 | **PASS** |
| UT3 | Unit | FR7, FR9 | Types and ranges valid | `test_detector_matches_offline` ×5 (IF; RF binary and multi; LSTM binary and multi): 500 rows each return a valid Result. Types, label set, finite score, confidence in [0, 1] and severity-iff-malicious all hold. Decision rules and the registry are tested separately. **The warm-up clause does not apply:** the LSTM uses formulation B (chapter4_notes §2.6), so it has no W−1 warm-up. The test asserts `warmup is False` for every record. | 11 | **PASS** (warm-up part N/A) |
| UT4 | Unit | NFR6 | PermissionError | Detector `_path` refuses `../outside.joblib`, `../../etc/passwd` and an absolute path outside `artefacts/`. The offline scorer refuses `../` too. A control case inside `artefacts/` resolves. | 4 | **PASS** |
| UT5 | Unit | FR10–12 | Rows correct | `test_thousand_fake_predictions`: 1,000 rows, no FK violations, every t4 ≥ t3, v_latency non-negative, alerts exactly for malicious non-warm-up rows, High iff confidence ≥ 0.90. `test_t4_written_in_next_transaction` and `test_foreign_keys_enforced` (IntegrityError on unknown flow, run or env) also pass. | 10 | **PASS** |
| IT1 | Integration | FR1–4, FR11 | Counts match; no negative intervals | `test_200_record_run_via_control` ×3 (IF, RF, LSTM): 200 sent = 200 processed = 200 PREDICTION rows, 0 malformed, 0 NULL t4, 0 negative intervals. Clock check (1,000 @ 100 ms, IF): every t_tx > 0, median 0.188 ms. | 11 | **PASS** |
| IT2 | Integration | FR2, FR8 | State changes as expected | Start/Stop from the dashboard sidebar locks and unlocks the settings, with Δt still live. Model switch: dashboard model selector, and IF → RF on the same processes. Stop mid-run gives end_reason `stopped` with all t4 written. Rate change 20 → 5 ms changes the median gap from 20.00 to 5.00 ms and is recorded in MODEL_RUN.rate_changes. A rate change is refused in burst mode, and busy or invalid commands are refused. | 21 | **PASS** |
| IT3 | Integration | §3.3.11 | 100% | Engine path, 500 rows × 5 model/task pairs: 0 label mismatches against the offline files. Streamed: **105/105** main-set runs at 100% agreement, 0 disagreeing records over 480,000 predictions (`results/latency_summary.csv`, `agreement`, `n_disagree`). | 2 (+5 shared with UT3) | **PASS** |
| ST1 | System | NFR1, NFR2 | Results produced; NFR1/2 verdicts stated | Main set complete: S1 15, S2 15, S3 60, S4 15 included runs. `results/nfr_verdicts.csv`: **NFR1 PASS** for IF (p95 36.2 ± 9.3 ms), RF (67.4 ± 16.3 ms) and LSTM (18.5 ± 1.7 ms). **NFR2 FAIL** on the sustained-rate rule (S2 p95(L) ≤ 10 ms and backlog not growing, 0/5 seeds each): IF p95 20.9 s, RF 145 s, both with growing backlogs; S3 max sustainable 20 and 10 records/s. LSTM: p95 267 ms > 10 ms although Eq. 3.7 throughput is 99.9/s, which S2's 100/s offered rate caps (reported as a finding; chapter4_notes §3.15). Dashboard-closed S1 set complete (runs 115–129); `dashboard_open_vs_closed.csv` filled in (chapter4_notes §3.13). Streamed agreement 120/120 runs. | 17 | **PASS** (results and verdicts stated) |
| ST2 | System | NFR6 | Only 127.0.0.1 | `netstat -an` with the full system running: 9100, 9101, 9102 and 8501 all on 127.0.0.1 (§5). `test_service_ports_bind_loopback_only` agrees. | 2 | **PASS** |
| ST3 | System | NFR4 | Identical labels | S1 run 6 (LSTM binary, seed 42) replayed offline twice in fresh processes: stream order identical, 0 of 5,000 labels differ, max score and confidence difference 0.0 (§6). | manual | **PASS** |
| UAT | Acceptance | NFR5 | Done without help; Likert questionnaire | `docs/uat/instructions.md` / `.pdf` and `docs/uat/questionnaire.md` / `.pdf` | — | Pending |

The other **41** tests are phase "Done when" checks with no §9 ID: data conversion and
cleaning, the split, training for each model, offline scoring and the detection-metric
arithmetic (NFR3, NFR4, leakage controls 1 and 5/6). All 41 pass.

NFR7 (modularity) is shown by inspection, as plan §9 says. Adding a model means writing
one `Detector` subclass in `src/nids/engine/detectors.py` and adding one entry to
`DETECTORS` there and to `MODELS` in `src/nids/dashboard/app.py`. `test_registry` checks
the registry.

## 3. Every pytest test, mapped

"supp." means the test supports the ID, but the ID's pass criterion is judged on the
main test. "(+IT)" means a unit-level ID that is also checked through the live services.

| # | pytest test | §9 ID | Requirement | What it checks (pass criterion) | Result |
|---|---|---|---|---|---|
| 1 | `test_control.py::test_encode_decode_round_trip` | IT2 (supp.) | FR2, FR8 | Control-channel framing; malformed frames -> ProtocolError | PASSED |
| 2 | `test_control.py::test_validate_commands` | IT2 (supp.) | FR2, FR8 | Command schema validation (missing/bool/negative/unknown fields, wrong service) | PASSED |
| 3 | `test_control.py::test_control_server_loopback_errors_and_recovery` | IT2 (supp.) / ST2 | NFR6 | Control server refuses 0.0.0.0, binds 127.0.0.1, survives garbage and handler errors; client refuses non-loopback hosts | PASSED |
| 4 | `test_control.py::test_pacer_matches_absolute_schedule_without_changes` | IT2 (supp.) | FR2 | Pacer follows the absolute schedule | PASSED |
| 5 | `test_control.py::test_pacer_set_rate_reanchors_exactly` | IT2 (supp.) | FR2 | Rate change re-anchors from the next record; refused in burst mode | PASSED |
| 6 | `test_control.py::test_200_record_run_via_control[iforest]` | IT1 | FR1-FR4, FR11 | Streamer -> engine -> DB, 200 records, each model: 200 sent = 200 processed = 200 PREDICTION rows, 0 malformed, no NULL t4, no negative interval, MODEL_RUN fields correct | PASSED |
| 7 | `test_control.py::test_200_record_run_via_control[rf]` | IT1 | FR1-FR4, FR11 | Streamer -> engine -> DB, 200 records, each model: 200 sent = 200 processed = 200 PREDICTION rows, 0 malformed, no NULL t4, no negative interval, MODEL_RUN fields correct | PASSED |
| 8 | `test_control.py::test_200_record_run_via_control[lstm]` | IT1 | FR1-FR4, FR11 | Streamer -> engine -> DB, 200 records, each model: 200 sent = 200 processed = 200 PREDICTION rows, 0 malformed, no NULL t4, no negative interval, MODEL_RUN fields correct | PASSED |
| 9 | `test_control.py::test_second_run_without_restart` | IT1 (supp.) | FR1, FR8 | Two runs (IF then RF) on the same engine/streamer processes, same PIDs | PASSED |
| 10 | `test_control.py::test_stop_mid_run_ends_cleanly` | IT2 | FR2 | Stop mid-run: end_reason stopped, sent = processed = PREDICTION rows, all t4 written; next run unaffected | PASSED |
| 11 | `test_control.py::test_stop_cancels_prepared_run` | IT2 | FR2 | Stop on a prepared run -> cancelled, 0 predictions; stop when idle is a no-op | PASSED |
| 12 | `test_control.py::test_set_rate_repaces_and_marks_run` | IT2 | FR2 | Rate change 20 -> 5 ms mid-run: median gap 20 -> 5 ms, MODEL_RUN.rate_changes recorded | PASSED |
| 13 | `test_control.py::test_set_rate_refused_in_burst` | IT2 | FR2, FR3 | Rate change refused during a burst run | PASSED |
| 14 | `test_control.py::test_busy_and_invalid_commands_refused` | IT2 | FR8 | Invalid seed/model/task and busy prepare/start refused without stopping a service | PASSED |
| 15 | `test_control.py::test_service_ports_bind_loopback_only` | ST2 | NFR6 | psutil on the live engine and streamer: listening sockets are exactly 127.0.0.1:{data, engine ctl} and 127.0.0.1:{streamer ctl} | PASSED |
| 16 | `test_control.py::test_run_py_starts_services_and_shuts_down_cleanly` | IT2 (supp.) | NFR5 | run.py starts both services; Ctrl+C stops the run cleanly, both exit 0 | PASSED |
| 17 | `test_convert_xlsx.py::test_convert_preserves_values_and_counts` | - (Phase 1) | FR1 | xlsx -> Parquet changes no values or counts | PASSED |
| 18 | `test_convert_xlsx.py::test_report_merge_keeps_other_keys` | - (Phase 1) | - | prep_report merge keeps other keys | PASSED |
| 19 | `test_convert_xlsx.py::test_label_run_stats` | - (Phase 1) | §3.2.4 | Label-run (order check) statistics | PASSED |
| 20 | `test_dashboard.py::test_kpis_exclude_warmup_and_use_eq_3_7` | IT2 (supp.) | FR13, Eq. 3.7 | KPI arithmetic excludes warm-up; throughput by Eq. 3.7 | PASSED |
| 21 | `test_dashboard.py::test_detection_kpis_exclude_warmup` | IT2 (supp.) | FR13 | Live precision/recall/false alarms vs dataset label, warm-up excluded | PASSED |
| 22 | `test_dashboard.py::test_within_target` | IT2 (supp.) | FR13, NFR1 | p95/p99-vs-dt indicator | PASSED |
| 23 | `test_dashboard.py::test_traffic_last_60_seconds_filled` | IT2 (supp.) | FR13 | Traffic chart buckets for the last 60 s | PASSED |
| 24 | `test_dashboard.py::test_protocol_mix_window_and_order` | IT2 (supp.) | FR13 | Protocol mix window and ordering | PASSED |
| 25 | `test_dashboard.py::test_dt_steps` | IT2 (supp.) | FR2 | dt slider steps cover 1-1,000 ms | PASSED |
| 26 | `test_dashboard.py::test_renders_without_services_or_database` | IT2 (supp.) | FR13 | Dashboard renders and explains when services/DB are absent | PASSED |
| 27 | `test_dashboard.py::test_kpis_match_database_for_finished_run` | IT2 (supp.) | FR13 | KPI tiles (records, alerts, p95, precision, recall, false alarms) equal values computed from the DB; read-only connection | PASSED |
| 28 | `test_dashboard.py::test_start_stop_from_sidebar_and_delay_log` | IT2 | FR2, FR8 | Dashboard: choose model + dt, Start, settings lock (dt stays live), Stop, unlock; MODEL_RUN matches; dashboard delay logged | PASSED |
| 29 | `test_dashboard.py::test_sidebar_mirrors_run_started_by_experiment_script` | IT2 | FR8 | Sidebar shows and locks to a run started by run_matrix, restores user choices after | PASSED |
| 30 | `test_detectors.py::test_path_refuses_parent_escape` | UT4 | NFR6 | ../ and ../../etc/passwd -> PermissionError | PASSED |
| 31 | `test_detectors.py::test_path_refuses_absolute_outside` | UT4 | NFR6 | Absolute path outside artefacts/ -> PermissionError | PASSED |
| 32 | `test_detectors.py::test_path_accepts_inside` | UT4 | NFR6 | Control case: a file inside artefacts/ resolves | PASSED |
| 33 | `test_detectors.py::test_registry` | UT3 (supp.) | FR7, NFR7 | All three detectors registered behind make_detector; unknown name -> ValueError | PASSED |
| 34 | `test_detectors.py::test_iforest_refuses_multi` | UT3 (supp.) | FR7 | IF has no seven-class task -> ValueError | PASSED |
| 35 | `test_detectors.py::test_binary_rule_and_severity` | UT3 (supp.) | FR9, FR10 | Binary decision rule and High/Medium/None severity | PASSED |
| 36 | `test_detectors.py::test_multi_rule` | UT3 (supp.) | FR9 | Seven-class argmax, score = 1 - P(Normal), confidence = max P | PASSED |
| 37 | `test_detectors.py::test_iforest_confidence_rule` | UT3 (supp.) | FR9 | IF confidence clip((s - tau)/(max - tau), 0, 1) | PASSED |
| 38 | `test_detectors.py::test_detector_matches_offline[iforest-binary]` | UT3 + IT3 | FR7, FR9, §3.3.11 | 500 test rows per model/task (IF, RF bin/multi, LSTM bin/multi): valid Result (types, label set, finite score, confidence in [0,1], severity iff malicious, warmup False); 0 label mismatches vs offline; score/confidence within 1e-5 | PASSED |
| 39 | `test_detectors.py::test_detector_matches_offline[rf-binary]` | UT3 + IT3 | FR7, FR9, §3.3.11 | 500 test rows per model/task (IF, RF bin/multi, LSTM bin/multi): valid Result (types, label set, finite score, confidence in [0,1], severity iff malicious, warmup False); 0 label mismatches vs offline; score/confidence within 1e-5 | PASSED |
| 40 | `test_detectors.py::test_detector_matches_offline[rf-multi]` | UT3 + IT3 | FR7, FR9, §3.3.11 | 500 test rows per model/task (IF, RF bin/multi, LSTM bin/multi): valid Result (types, label set, finite score, confidence in [0,1], severity iff malicious, warmup False); 0 label mismatches vs offline; score/confidence within 1e-5 | PASSED |
| 41 | `test_detectors.py::test_detector_matches_offline[lstm-binary]` | UT3 + IT3 | FR7, FR9, §3.3.11 | 500 test rows per model/task (IF, RF bin/multi, LSTM bin/multi): valid Result (types, label set, finite score, confidence in [0,1], severity iff malicious, warmup False); 0 label mismatches vs offline; score/confidence within 1e-5 | PASSED |
| 42 | `test_detectors.py::test_detector_matches_offline[lstm-multi]` | UT3 + IT3 | FR7, FR9, §3.3.11 | 500 test rows per model/task (IF, RF bin/multi, LSTM bin/multi): valid Result (types, label set, finite score, confidence in [0,1], severity iff malicious, warmup False); 0 label mismatches vs offline; score/confidence within 1e-5 | PASSED |
| 43 | `test_detectors.py::test_single_record_timing` | UT3 (supp.) | NFR2 (early check) | Fresh process: 1,000 single predict() calls per detector, one thread, LSTM tf.function threads (1,1); median <= p95 | PASSED |
| 44 | `test_offline_predict.py::test_iforest_confidence_rule` | - (Phase 2) | FR9 | Offline IF confidence rule | PASSED |
| 45 | `test_offline_predict.py::test_artefact_path_refuses_escape` | UT4 | NFR6 | Offline scorer also refuses ../ -> PermissionError | PASSED |
| 46 | `test_offline_predict.py::test_binary_frame_applies_saved_threshold` | - (Phase 2) | controls 5/6 | Saved threshold applied | PASSED |
| 47 | `test_offline_predict.py::test_multi_frame_argmax_score_confidence` | - (Phase 2) | FR9 | Seven-class argmax/score/confidence | PASSED |
| 48 | `test_offline_predict.py::test_binary_metrics_known_values` | - (Phase 6) | NFR3 | Precision/recall/FPR/ROC-AUC on a known example | PASSED |
| 49 | `test_offline_predict.py::test_mean_sd_uses_sample_sd` | - (Phase 6) | NFR3 | Mean +- sample SD over seeds | PASSED |
| 50 | `test_offline_predict.py::test_offline_files_complete_and_consistent` | IT3 (supp.) | §3.3.11, NFR3 | 25 offline files cover every test row; saved threshold applied, never re-tuned | PASSED |
| 51 | `test_phase0.py::test_config_loads` | - (Phase 0) | NFR4 | config.yaml loads with all settings | PASSED |
| 52 | `test_phase0.py::test_module_packages_import` | - (Phase 0) | NFR7, Table 3.9 | Module packages M1-M8 import | PASSED |
| 53 | `test_pipeline.py::test_sample_is_stratified_reproducible_and_shuffled` | IT1 (supp.) / ST3 | FR1, NFR4 | Stream sample: same seed -> identical order; stratified within 0.002; seeded shuffle; other seed differs | PASSED |
| 54 | `test_pipeline.py::test_schedule_steady_and_burst` | IT1 (supp.) | FR3 | Steady and burst (500 / 1,000 back-to-back / 500) emission schedules | PASSED |
| 55 | `test_pipeline.py::test_wait_until_is_accurate` | IT1 (supp.) | FR2 | Pacing wait: never early, median lateness < 0.2 ms | PASSED |
| 56 | `test_pipeline.py::test_encode_frames_and_stamps_t0_before_dumps` | IT1 (supp.) | FR4, FR11 | One JSON line per record; t0 taken before json.dumps | PASSED |
| 57 | `test_pipeline.py::test_effective_socket_buffers_probe` | IT1 (supp.) | FR4 (S4 buffers) | 4 MB SO_SNDBUF requested; effective sizes probed | PASSED |
| 58 | `test_pipeline.py::test_sndbuf_warning_threshold` | IT1 (supp.) | FR4 | Warning when granted send buffer < 4 MB | PASSED |
| 59 | `test_pipeline.py::test_streamer_refuses_non_loopback` | ST2 (supp.) | NFR6 | Streamer refuses to connect to 0.0.0.0 -> ValueError | PASSED |
| 60 | `test_pipeline.py::test_malformed_lines_counted_engine_continues` | UT2 (+IT) | FR5 | Live engine: not-JSON, wrong length, missing row_index key, unknown flow, non-object -> 5 counted as malformed, 40 good rows predicted, engine stays idle/serving | PASSED |
| 61 | `test_pipeline.py::test_clock_check_1000_at_100ms` | IT1 (supp.) | FR11, §3.3.6 | 1,000 records @ 100 ms: every t_tx > 0, median t_tx < 1 ms (same clock across processes) | PASSED |
| 62 | `test_prepare_data.py::test_roles_exclude_ports` | - (Phase 1) | §3.2.4 | Port columns treated as identifiers, not features | PASSED |
| 63 | `test_prepare_data.py::test_replace_inf_parses_text` | - (Phase 1) | §3.2.4 | Text Infinity/NaN parsed, inf -> NaN | PASSED |
| 64 | `test_prepare_data.py::test_duplicates_and_conflicts` | - (Phase 1) | §3.2.4 | Exact and conflicting duplicates removed | PASSED |
| 65 | `test_prepare_data.py::test_missing_rule_drop_and_keep` | - (Phase 1) | §3.2.4 | Missing-value rule (< 1% drop, else fill) | PASSED |
| 66 | `test_prepare_data.py::test_labels` | - (Phase 1) | §3.2.4 | Seven-class and binary labels, label map | PASSED |
| 67 | `test_prepare_data.py::test_clean_real_data` | - (Phase 1) | §3.2.4 | Cleaning on the real data gives the reported counts | PASSED |
| 68 | `test_run_matrix.py::test_main_set_matches_table_3_13` | ST1 (supp.) | NFR1, NFR2 | Main set = S1-S4 x 3 models x 5 seeds (105 runs) | PASSED |
| 69 | `test_run_matrix.py::test_interleaved_order_and_one_seed_per_repetition` | ST1 (supp.) | NFR1, NFR2 | Models interleaved IF, RF, LSTM; one seed per repetition | PASSED |
| 70 | `test_run_matrix.py::test_closed_set_is_s1_without_dashboard` | ST1 (supp.) | NFR1 | Closed set = S1 without dashboard | PASSED |
| 71 | `test_run_matrix.py::test_burst_block_and_estimate` | ST1 (supp.) | NFR2 (S4) | Burst block and wall-time estimate | PASSED |
| 72 | `test_run_matrix.py::test_buffer_check_refuses_small_buffers` | ST1 (supp.) | S4 validity | Matrix refuses to start with socket buffers < 4 MB | PASSED (fixed, §4.3) |
| 73 | `test_run_matrix.py::test_set_and_flag_must_agree` | ST1 (supp.) | §3.3.7 | --set and --dashboard-open must agree | PASSED |
| 74 | `test_run_matrix.py::test_prepare_accepts_dashboard_open` | ST1 (supp.) | §3.3.7 | Dashboard state passed with prepare | PASSED |
| 75 | `test_run_matrix.py::test_dry_run_prints_both_sets` | ST1 (supp.) | - | Dry run lists both sets (120 runs) | PASSED |
| 76 | `test_run_matrix.py::test_headless_run_records_dashboard_state_and_resumes` | ST1 (supp.) | NFR1, NFR2 | Headless closed-set run through real services records dashboard_open = false, and a resume skips it | PASSED (fixed, §4.2) |
| 77 | `test_split_fit.py::test_done_report_class_counts_reasonable` | - (Phase 1) | control 1 | Split fractions and stratified class shares | PASSED |
| 78 | `test_split_fit.py::test_done_partitions_disjoint_and_all_classes` | - (Phase 1) | control 1 | Partitions disjoint, all classes present, row_index order | PASSED |
| 79 | `test_split_fit.py::test_done_preprocessor_matches_X_test` | UT1 | FR6 | Online Preprocessor.transform on 1,000 seeded test rows is allclose to offline X_test.npy; feature list identical | PASSED |
| 80 | `test_split_fit.py::test_fit_rules` | UT1 (supp.) | FR6, control 1 | Constant/correlation filters, log mask, scaler: standardised output mean 0, SD 1 | PASSED |
| 81 | `test_split_fit.py::test_log_exclude_keeps_column_standardised_only` | UT1 (supp.) | FR6 | Excluded column is standardised but never log1p'd | PASSED |
| 82 | `test_split_fit.py::test_protocol_not_logged_in_artefact` | UT1 (supp.) | FR6 | Saved preprocessor.json: Protocol has log_mask = false | PASSED |
| 83 | `test_split_fit.py::test_correlation_chain_keeps_column_only_linked_via_dropped_one` | UT1 (supp.) | FR6 | Correlation filter keeps a column linked only through a dropped one | PASSED |
| 84 | `test_split_fit.py::test_state_depends_on_train_only` | UT1 (supp.) | control 1 (split before fit) | Fitted state depends only on the frame passed in (training partition) | PASSED |
| 85 | `test_split_fit.py::test_split_stratified_disjoint` | - (Phase 1) | control 1, NFR4 | Stratified 70/15/15 split, disjoint, deterministic | PASSED |
| 86 | `test_split_fit.py::test_preprocessor_rejects_malformed` | UT2 | FR5 | Wrong length (short/long), string value, nested list -> ValueError; inf/NaN -> median; caller's array untouched | PASSED |
| 87 | `test_storage.py::test_wal_and_pragmas_on_every_connection` | UT5 (supp.) | FR12 | WAL, foreign_keys = 1, synchronous = NORMAL; all tables, view, index present | PASSED |
| 88 | `test_storage.py::test_thousand_fake_predictions` | UT5 | FR10, FR11, FR12 | 1,000 predictions: no FK violations, every t4 written and t4 >= t3, v_latency 1,000 rows none negative, alerts exactly for malicious non-warm-up rows with High iff conf >= 0.90, MODEL_RUN seed/env_id stored | PASSED |
| 89 | `test_storage.py::test_t4_written_in_next_transaction` | UT5 | FR11 | t4 of record i appears only after record i+1's commit; flush writes the last one | PASSED |
| 90 | `test_storage.py::test_foreign_keys_enforced` | UT5 | FR12 | Unknown flow_id / run_id / env_id / duplicate alert -> IntegrityError; logger stays usable | PASSED |
| 91 | `test_storage.py::test_severity_rule` | UT5 | FR10 | Severity High iff confidence >= 0.90 (NaN/None -> Medium) | PASSED |
| 92 | `test_storage.py::test_environment_row` | UT5 (supp.) | FR15 | TEST_ENVIRONMENT row collected and reused, not duplicated | PASSED |
| 93 | `test_storage.py::test_protocol_names` | UT5 (supp.) | FR12 | Protocol code -> TCP/UDP/ICMP/Other | PASSED |
| 94 | `test_storage.py::test_no_sql_built_from_strings` | UT5 (supp.) | NFR6 (parameterised SQL) | Static AST check over src/ and experiments/: no SQL built with f-string/+/%/.format; every execute*/read_sql argument a literal or module constant | PASSED (fixed, §4.1) |
| 95 | `test_storage.py::test_sql_checker_catches_bad_code` | UT5 (supp.) | NFR6 | The static check flags the three bad patterns (not vacuous) | PASSED |
| 96 | `test_storage.py::test_init_db_real_partition` | UT5 (supp.) | FR12 | FLOW_RECORD loaded once from test_raw (flow_id = row_index, labels, protocols); second call reloads nothing | PASSED |
| 97 | `test_system_metrics.py::test_backlog_counts_unfinished_earlier_records` | ST1 (supp.) | NFR2 | Backlog definition (plan §8.3) | PASSED |
| 98 | `test_system_metrics.py::test_stable_run_is_sustained_and_eq_3_7_throughput` | ST1 (supp.) | NFR1, NFR2 | Stable queue: sustained, Eq. 3.7 throughput exact | PASSED |
| 99 | `test_system_metrics.py::test_overloaded_run_grows_backlog_and_fails` | ST1 (supp.) | NFR2 | Overloaded queue: backlog grows, not sustained | PASSED |
| 100 | `test_system_metrics.py::test_backlog_tolerance_ignores_sub_record_noise` | ST1 (supp.) | NFR2 | One-record backlog tolerance | PASSED |
| 101 | `test_system_metrics.py::test_burst_recovery_and_not_recovered` | ST1 (supp.) | NFR2 (S4) | Burst peak L and recovery time; non-recovery detected | PASSED |
| 102 | `test_system_metrics.py::test_agreement_with_offline_labels` | IT3 (supp.) | §3.3.11, §8.4 | Agreement checker counts streamed-vs-offline mismatches correctly | PASSED |
| 103 | `test_system_metrics.py::test_select_runs_rules` | ST1 (supp.) | - | Run selection: zero-refresh, short, undeclared, duplicate runs excluded | PASSED |
| 104 | `test_system_metrics.py::test_sustainable_rate_needs_all_seeds_and_every_larger_dt` | ST1 (supp.) | NFR2 (S3) | Sustainable-rate rule needs all seeds and every larger dt | PASSED |
| 105 | `test_train_iforest.py::test_grid_is_36_combinations_within_table_3_11` | - (Phase 2) | Table 3.11 | IF grid = 36 combinations | PASSED |
| 106 | `test_train_iforest.py::test_best_f1_threshold_known_example` | - (Phase 2) | controls 5/6 | Validation F1-maximising threshold | PASSED |
| 107 | `test_train_iforest.py::test_score_sign_outliers_score_higher` | - (Phase 2) | Eq. 3.3 | IF score sign: outliers score higher | PASSED |
| 108 | `test_train_iforest.py::test_train_seeds_bundle_and_reload` | - (Phase 2) | NFR4 | Per-seed bundles save and reload | PASSED |
| 109 | `test_train_iforest.py::test_load_partitions_never_reads_test` | - (Phase 2) | controls 5/6 | Training never reads the test partition | PASSED |
| 110 | `test_train_iforest.py::test_done_five_bundles_and_config` | - (Phase 2) | NFR4 | Five IF bundles (seeds 42-46) and config exist | PASSED |
| 111 | `test_train_lstm.py::test_sampled_configs_within_table_3_11` | - (Phase 2) | Table 3.11 | LSTM random-search ranges | PASSED |
| 112 | `test_train_lstm.py::test_formulation_b_shapes` | - (Phase 2) | §3.3.10 | Formulation B input shape (F, 1) | PASSED |
| 113 | `test_train_lstm.py::test_class_weights_balanced` | - (Phase 2) | Table 3.11 | Balanced class weights | PASSED |
| 114 | `test_train_lstm.py::test_train_task_end_to_end` | - (Phase 2) | NFR4 | LSTM end to end (small) | PASSED |
| 115 | `test_train_lstm.py::test_resume_mid_search_matches_uninterrupted` | - (Phase 2) / ST3 (supp.) | NFR4 | Interrupted search resumes to the identical result | PASSED |
| 116 | `test_train_lstm.py::test_resume_mid_seeds_skips_finished_seed` | - (Phase 2) | NFR4 | Resume skips finished seeds | PASSED |
| 117 | `test_train_lstm.py::test_resume_drops_torn_line_and_refits_without_checkpoint` | - (Phase 2) | NFR4 | Torn log line dropped on resume | PASSED |
| 118 | `test_train_lstm.py::test_resume_rejects_other_configuration` | - (Phase 2) | NFR4 | Resume refuses a different configuration | PASSED |
| 119 | `test_train_lstm.py::test_done_ten_models_and_meta` | - (Phase 2) | NFR4 | Ten LSTM models and meta files exist | PASSED |
| 120 | `test_train_rf.py::test_search_space_matches_table_3_11` | - (Phase 2) | Table 3.11 | RF search space | PASSED |
| 121 | `test_train_rf.py::test_threshold_rule_prefers_0_5_unless_tuned_is_better` | - (Phase 2) | controls 5/6 | Threshold rule | PASSED |
| 122 | `test_train_rf.py::test_threshold_rule_ignores_gain_below_0_005` | - (Phase 2) | controls 5/6 | Threshold gain < 0.005 ignored | PASSED |
| 123 | `test_train_rf.py::test_stratified_fraction_keeps_every_class` | - (Phase 2) | - | Stratified search fraction keeps every class | PASSED |
| 124 | `test_train_rf.py::test_train_task_both_end_to_end` | - (Phase 2) | NFR4 | RF binary + multi end to end (small) | PASSED |
| 125 | `test_train_rf.py::test_load_partitions_never_reads_test` | - (Phase 2) | controls 5/6 | Training never reads the test partition | PASSED |
| 126 | `test_train_rf.py::test_done_ten_bundles_and_configs` | - (Phase 2) | NFR4 | Ten RF bundles and configs exist | PASSED |
| 127 | `test_system_metrics.py::test_nfr2_judged_on_sustained_rate_not_capped_throughput` | ST1 (supp.) | NFR2 | NFR2 verdict follows p95(L) ≤ Δt and backlog not growing (plan §8.3), not Eq. 3.7 throughput capped at the offered 100/s | PASSED |

## 4. Failures

Which side was wrong, and what was changed:

| § | Test | Wrong side | Change |
|---|---|---|---|
| 4.1 | `test_no_sql_built_from_strings` | **Test** (too strict: false positive on correct, parameterised code) | Code layout only: the query moved to a module-level constant so the unchanged check accepts it. Behaviour and SQL text unchanged. |
| 4.2 | `test_headless_run_records_dashboard_state_and_resumes` | **Test** (isolation: two tests shared a scenario label) | Test: the dashboard test uses `dash-test-S1` |
| 4.3 | `test_buffer_check_refuses_small_buffers` | **Test** (read the real `db/nids.db`) | Test: empty temporary `--db` / `--delay-csv` |

No pipeline code was at fault.

### 4.1 `test_storage.py::test_no_sql_built_from_strings` (NFR6 static check)

```
AssertionError: src/nids/evaluation/system_metrics.py:139 non-literal SQL in .read_sql_query()
```

The query in `load_records` is a fixed string literal with a `?` parameter, bound
through `params=(int(r),)`. No value is put into the SQL text. The checker accepts
only a literal or a **module-level** constant as the first argument to
`execute*`/`read_sql*`. Here the literal sits in a local variable `q`, so the check flags
it. This is a **false positive**: NFR6 (parameterised queries) is still met. It was
introduced with `system_metrics.py` in commit `50f7b98`. To fix it, move the query to a
module-level constant, or let the checker accept a local name bound once to a string
literal.

**Fixed 2026-11-17:** the query is now the module-level constant `RECORDS_SQL` in
`system_metrics.py`. The SQL text is unchanged and the checker was not loosened.

### 4.2 `test_run_matrix.py::test_headless_run_records_dashboard_state_and_resumes`

```
assert len(rows) == 1   ->  2 == 1
rows: run 14 (end_reason 'stopped', dashboard_open true, 17 predictions)
      run 17 (end_reason 'complete', dashboard_open false, 20 predictions)
```

The integration tests share one engine, streamer and temporary database per session
(`tests/conftest.py`). Earlier in the session,
`test_dashboard.py::test_sidebar_mirrors_run_started_by_experiment_script` (added in
`be90299`) prepares a run with scenario `rm-test-S1` and then stops it. That is run 14.
The run_matrix test then selects every row `WHERE scenario = 'rm-test-S1'` and finds
both runs. The driver itself did the right thing: run 17 completed with 20 predictions
and `dashboard_open = false`. Run on its own, the test **passes** (`1 passed in 11.77 s`).
This is a test-isolation fault caused by two tests using the same scenario label. To
fix it, give one of them a different label or filter on run_id.

**Fixed 2026-11-17:** the dashboard test now uses its own label, `dash-test-S1`. The
dashboard treats any scenario other than its own as script-started, so the test still
checks the same behaviour.

### 4.3 `test_run_matrix.py::test_buffer_check_refuses_small_buffers` (appeared 2026-11-17)

```
assert code == 2   ->  0 == 2
```

This test passed on 16 Nov and failed on the first re-run after the other two fixes. It
called `run_matrix.main()` without `--db`, so it read the real `db/nids.db`. Once the
closed set was complete there (runs 115–129), `main()` found nothing pending and
returned 0 before it reached the socket-buffer preflight. The test had only been
passing because closed runs were still outstanding. **Fixed:** the test passes a
non-existent temporary `--db` and `--delay-csv`, so every run is pending and the
preflight is always exercised. It no longer reads `db/nids.db`. No other test calls
`main()` without `--db` past argument parsing.

## 5. ST2: loopback-only sockets (NFR6)

Method: started the full system with `python run.py --db <scratch db> --no-browser`, on
the default ports and with the dashboard. A scratch database was used so `db/nids.db`
was not touched. After the `[run] dashboard:` line appeared, the listening sockets were
recorded, then Ctrl+C was sent.

```
$ netstat -an | grep LISTEN | grep -E ':(9100|9101|9102|8501)\b'
tcp        0      0 127.0.0.1:9102          0.0.0.0:*               LISTEN     # streamer control
tcp        0      0 127.0.0.1:9101          0.0.0.0:*               LISTEN     # engine control
tcp        0      0 127.0.0.1:9100          0.0.0.0:*               LISTEN     # engine data
tcp        0      0 127.0.0.1:8501          0.0.0.0:*               LISTEN     # dashboard (Streamlit)

$ ss -ltnup     (owning processes)
tcp LISTEN 0 128   127.0.0.1:9102 0.0.0.0:* users:(("python",pid=2073445,fd=3))   # streamer
tcp LISTEN 0 128   127.0.0.1:9101 0.0.0.0:* users:(("python",pid=2073436,fd=7))   # engine
tcp LISTEN 0 128   127.0.0.1:9100 0.0.0.0:* users:(("python",pid=2073436,fd=6))   # engine
tcp LISTEN 0 2048  127.0.0.1:8501 0.0.0.0:* users:(("python",pid=2073515,fd=6))   # streamlit
```

No NIDS process listened on any other address: no `0.0.0.0`, no `[::]`/tcp6 and no UDP.
On shutdown, both services stopped with exit code 0. The pytest check
`test_service_ports_bind_loopback_only` (psutil, ephemeral ports) found the same:
engine `{127.0.0.1:35029, 127.0.0.1:33863}`, streamer `{127.0.0.1:38661}`. The code also
refuses non-loopback addresses (`test_streamer_refuses_non_loopback`,
`test_control_server_loopback_errors_and_recovery`). **Verdict: PASS.**

## 6. ST3: S1 re-run gives identical labels (NFR4)

Configuration: **run 6** in `db/nids.db`, S1, LSTM binary, model seed 42, Δt = 100 ms,
steady, 5,000 records, sample_seed 42, end_reason `complete`, rate_changes NULL. The
LSTM was chosen because its TensorFlow numerics are the most likely of the three models
to vary between runs.

Method: a replay script kept outside the repository opened `db/nids.db` read-only. It
rebuilt the S1 stream with the streamer's own `draw_sample(test_raw, 5000, 42)`, loaded
the model fresh from `artefacts/` (`make_detector` → `load(42, "binary")`) and pushed
each raw row through the engine's `Preprocessor.transform` and `Detector.predict`, the
same per-record path the engine uses. The output was compared row by row with the
PREDICTION rows of run 6. This was done twice, in two separate processes.

| Check | Replay 1 | Replay 2 |
|---|---|---|
| Regenerated stream order = stored `seq_no` order | yes | yes |
| Label disagreements with stored run (of 5,000) | **0** | **0** |
| Warm-up flag disagreements | 0 | 0 |
| Max \|score − stored\| | 0.0 | 0.0 |
| Max \|confidence − stored\| | 0.0 | 0.0 |
| Label disagreements with `results/offline/lstm_binary_seed42.parquet` | 0 | 0 |
| Label counts | Malicious 2,822 / Normal 2,178 | Malicious 2,822 / Normal 2,178 |

The two replays were identical to each other, to the original streamed run and to the
offline reference. Labels do not depend on timing, so this is the label part of NFR4.
Latency is expected to vary from run to run, and the 5 seeds × S1 spread in
`results/system_summary.csv` reports that. Related automated evidence: the same seed
gives the same stream order (`test_sample_is_stratified_reproducible_and_shuffled`), the
split is deterministic (`test_split_stratified_disjoint`), and an interrupted LSTM search
resumes to the identical result (`test_resume_mid_search_matches_uninterrupted`). The
IT3 agreement of 105/105 runs also shows that every streamed run reproduces the offline
labels. **Verdict: PASS.**

## 7. How to reproduce

```bash
source .venv/bin/activate
python -m pytest -v                                   # full suite (~8–9 min; includes slow tests)
python -m pytest tests/test_run_matrix.py::test_headless_run_records_dashboard_state_and_resumes   # §4.2 in isolation
python run.py --no-browser   # then, in another terminal:
netstat -an | grep LISTEN    # ST2
```

Slowest tests: clock check 101 s, single-record timing 54 s, LSTM end-to-end training
45 s. Use `-m "not slow"` to skip the clock check.
