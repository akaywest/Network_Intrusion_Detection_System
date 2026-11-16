# Real-Time Network Intrusion Detection System

A final-year project that detects network intrusions in real time. Three detectors share
one interface: an Isolation Forest, a Random Forest and an LSTM. A streamer replays
flows from the IDS2025 test partition over a loopback TCP socket. An inference engine
validates, scales and classifies each flow, then logs the prediction, any alert and five
timestamps (t0–t4) to SQLite in WAL mode. A Streamlit dashboard shows traffic, alerts and
end-to-end latency live. The design is in `docs/chapters_1_to_3.md` and the build plan in
`docs/implementation_plan.md`.

## Setup

Tested on Ubuntu with Python 3.12.3, CPU only.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
pip install -e .            # installs the nids package from src/
```

**Dataset.** Download IDS2025 from Mendeley Data: <https://doi.org/10.17632/pkskt3fv3v.1>.
Save the workbook as **`data/raw/IDS2025.xlsx`** (the path is set in `config.yaml`,
`paths.source_xlsx`). The pipeline only reads this file and never modifies it.

Every tunable setting (split, seeds, ports, scenario sizes, alert threshold) is in
`config.yaml`.

## Build pipeline (run once, in this order)

Run all commands from the project root with the venv active.

```bash
# 1. Convert the workbook to Parquet (adds row_index; order check -> results/prep_report.json)
python -m nids.prep.convert_xlsx

# 2. Clean, split 70/15/15 (stratified), fit preprocessing on the training partition only
#    -> data/processed/{train,val,test}_raw.parquet, X_*.npy, y_*.npy,
#       artefacts/preprocessor.json, feature_list.json, label_map.json
python -m nids.prep.prepare_data

# 3. Train the models (seeds 42-46) -> artefacts/, results/train_log/
python -m nids.train.train_iforest
python -m nids.train.train_rf                     # binary and seven-class
python -m nids.train.train_lstm --search-frac 0.3 # binary and seven-class; search on 30% of training

# 4. Offline predictions on the full test partition (25 files) -> results/offline/
python -m nids.train.offline_predict

# 5. Create the log database and load FLOW_RECORD from the test partition -> db/nids.db
python -m nids.storage.init_db
```

Notes:

- `train_lstm` resumes after an interruption. Add `--fresh` to start again.
- `--smoke` on any training module runs a small, fast version that writes only under
  `data/processed/smoke_*`.
- `python -m nids.storage.init_db --reset` deletes and recreates the database.
- Optional: `python -m nids.engine.detectors --time` times 1,000 single-record
  predictions per detector and writes `results/detector_timing.json`.

## Launch the system and dashboard

```bash
python run.py
```

`run.py` creates `db/nids.db` if it is missing. It then starts the inference engine and
the streamer, launches the dashboard and opens **<http://127.0.0.1:8501>**. Choose a model
in the sidebar and press **Start**. Press **Ctrl+C** in the terminal to stop the active
run cleanly and shut everything down. All sockets bind to 127.0.0.1 only. To run only the
engine and streamer, use `python run.py --no-dashboard`.

## Run the experiment matrix

The streamer requests a 4 MB socket send buffer. Linux caps it at `net.core.wmem_max`,
so raise the limits first. The setting is lost on reboot, so apply it again before
every measuring session. `run_matrix.py` refuses to start if the buffers are too small.

```bash
sudo sysctl -w net.core.wmem_max=8388608 net.core.rmem_max=8388608
```

Check the run list, resume status and time estimate:

```bash
python experiments/run_matrix.py --dry-run
```

**Main set** (S1–S4, 105 runs, dashboard open at a 1 s refresh, as the design requires).
Start `python run.py` and keep the dashboard tab in the foreground with the screen on.
Then, in a second terminal:

```bash
python experiments/run_matrix.py --set main --dashboard-open
```

**Closed set** (S1 with the dashboard closed, 15 runs). Close the dashboard tab, then:

```bash
python experiments/run_matrix.py --set closed --launch   # --launch starts the services if they are not running
```

Both sets can be resumed: run the same command again and finished runs are skipped.
Ctrl+C stops the active run cleanly, and that run is repeated next time.

## Analysis

```bash
python -m nids.evaluation.detection_metrics   # precision, recall, F1, ROC-AUC, FPR, confusion matrices
python -m nids.evaluation.system_metrics      # latency, throughput, backlog, S3/S4, NFR verdicts, plots
```

Re-run `system_metrics` after the closed set to fill in the dashboard open-vs-closed
comparison. It overwrites all of its outputs.

## Tests

```bash
python -m pytest                  # full suite (includes slow end-to-end checks)
python -m pytest -m "not slow"    # skip the long-running checks
```

The integration tests start their own engine and streamer on free loopback ports with
a temporary database, so they never touch `db/nids.db`. The mapping of tests to the
plan's test IDs and the latest results are in `docs/testing_report.md`.

## Where outputs go

| Path | Contents |
|---|---|
| `artefacts/` | preprocessor, feature list, label map, model files (`*.joblib`, `*.keras`) and per-model configs |
| `results/prep_report.json` | cleaning, split and order-check report |
| `results/train_log/` | training and validation logs per model |
| `results/offline/` | offline test-partition predictions (not committed; regenerable) |
| `results/detection_metrics.csv`, `per_class_recall.csv`, `confusion_*_multi.{csv,png}` | detection metrics |
| `results/latency_summary.csv`, `system_summary.csv`, `sustainable_rate.csv`, `nfr_verdicts.csv`, `dashboard_open_vs_closed.csv` | system metrics |
| `results/per_record/latency_records.parquet` | per-record latency and backlog (not committed) |
| `results/plots/` | latency CDF (S1), component bars (S1), rate search (S3), burst (S4) |
| `results/run_matrix_log.csv`, `matrix_main.log`, `dashboard_delay.csv` | experiment run log and dashboard refresh delays |
| `db/nids.db` | SQLite log of every run (not committed) |
| `docs/chapter4_notes.md` | running notes on deviations and findings for Chapter Four |
| `docs/uat/` | user acceptance test instruction sheet and questionnaire (Markdown + PDF) |
