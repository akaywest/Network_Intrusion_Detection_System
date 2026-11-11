# Real-Time NIDS — final-year project (Kayode Akinola Samuel, 236336)

Real-time network intrusion detection system: Isolation Forest, Random Forest and LSTM
behind one Detector interface. Pipeline: TCP streamer -> inference engine -> SQLite (WAL) -> Streamlit dashboard.
Dataset: IDS2025 (Mendeley Data). Python 3.12.3 (Ubuntu) in a project venv at .venv/, CPU only.

## Source of truth
- `docs/implementation_plan.md` — the build plan. Follow its phases in order and check each phase's "Done when" list.
- `docs/chapters_1_to_3.md` — the approved design (Chapter 3 is binding). Cite FR/NFR numbers and section numbers in code comments where relevant.
- If the plan and Chapter 3 disagree, stop and ask rather than choosing silently.

## Non-negotiable rules
- Split BEFORE any fitting. Scaler, medians, log mask and correlation filter use the training partition only.
- Stream cleaned UNSCALED test rows; the engine applies artefacts/preprocessor.json.
- Timestamps: t0 in the streamer before json.dumps; t1 on receipt; t2 after validate+scale;
  t3 after predict; t4 after COMMIT (written in the next transaction). Never change these definitions.
- All sockets bind 127.0.0.1. All SQL uses ? parameters. Models load only from artefacts/.
- Inference is single-threaded. The LSTM uses a tf.function call, never model.predict per record.
- Seeds 42–46; every run stores seed, hyperparameters and env_id in MODEL_RUN.
- Never edit data/raw. Never use the test partition for tuning or thresholds.

## Working style
- Explain what you are about to change before large edits; keep modules matching Table 3.9 (M1–M8).
- Write pytest tests alongside each module (see plan Section 9).
