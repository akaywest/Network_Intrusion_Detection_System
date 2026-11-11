"""LSTM training (M2, implementation plan §4.3; Chapter 3 §3.3.10, Table 3.11).

Formulation B (fallback named in §3.3.10; decision recorded in plan §4.3): each flow's F
scaled features are treated as a sequence of F steps, input shape (F, 1). No record
windows, no buffer, no W-1 warm-up, no dependence on row order (the source sheet is
class-sorted, results/prep_report.json order_check). The W search dimension of
Table 3.11 therefore does not apply.

Two families, both trained on ALL training records:
  binary  label_bin, sigmoid output, validation F1       -> artefacts/lstm_binary_seed{s}.keras
  multi   label_7,   softmax over 7, validation macro-F1 -> artefacts/lstm_multi_seed{s}.keras
each with a per-seed meta JSON (lstm_{task}_seed{s}.meta.json: formulation, input shape,
tau, threshold rule, config) and a family config (lstm_{task}_config.json).

Procedure per task (Table 3.11):
  1. Random search, 15 trials, seeded (42): layers {1,2}, units {32,64,128},
     dropout U(0.2,0.5), learning rate log-U(1e-4,1e-2), batch {128,256,512}.
     Architecture: LSTM layer(s) -> Dropout -> Dense. Adam, class-weighted cross-entropy
     ("balanced" weights from the training labels), EarlyStopping(val_loss, patience=5,
     restore_best_weights=True), at most 50 epochs. Every trial uses seed 42.
     Selection: validation F1 at 0.5 (binary) / validation macro-F1 (multi).
  2. Decision rule, chosen on validation on the seed-42 model with the same rule as the
     Random Forest: P(Malicious) >= 0.5 unless the validation F1-maximising threshold
     improves validation F1 by >= 0.005 ("tuned"; tau then re-set per seed on validation).
     multi: argmax (no threshold).
  3. Retrain the chosen configuration with seeds 42-46 (tf.keras.utils.set_random_seed,
     §3.3.10, NFR3), same early stopping, and save.
  4. Per-epoch wall time is logged for every fit.
  5. Time 1,000 single-record calls of the saved seed-42 model through a tf.function with
     a fixed input signature (1, F, 1) -- never model.predict per record (plan §6.1) --
     in a child process limited to one thread (§3.3.10). The child loads the .keras file,
     so this also checks the saved model and the tf.function path agree with predict().

Leakage control 5/6 (Table 3.12): only training and validation arrays are loaded.
Op determinism is enabled so a rerun reproduces the same weights (NFR4).

Usage:  python -m nids.train.train_lstm [--config config.yaml] [--task binary|multi|both]
                                        [--search-frac F] [--smoke]
        --search-frac F: trials train on a stratified fraction F of the training partition
                 (plan §4.2-style compute fallback; a Chapter Four deviation if < 1). The
                 seed retrains always use the full partition.
        --smoke: small stratified subsample, 3 trials, at most 6 epochs, seeds 42-43,
                 100 timing calls; writes to data/processed/smoke_lstm/ only.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

from nids.prep.convert_xlsx import ROOT  # noqa: E402
from nids.train.train_iforest import best_f1_threshold  # noqa: E402
from nids.train.train_rf import (SEARCH_SEED, binary_val_metrics,  # noqa: E402
                                 choose_threshold_rule, load_partitions, multi_val_metrics,
                                 smoke_subsample, stratified_fraction)

FORMULATION = "B"
TASKS = {"binary": {"y": "bin", "metric": "f1"},
         "multi": {"y": "7", "metric": "macro_f1"}}
N_TRIALS, MAX_EPOCHS, PATIENCE, N_TIMING = 15, 50, 5, 1000
SMOKE_N_TRIALS, SMOKE_MAX_EPOCHS, SMOKE_N_TIMING = 3, 6, 100
SPACE = {"layers": [1, 2], "units": [32, 64, 128], "dropout": {"uniform": [0.2, 0.5]},
         "learning_rate": {"log_uniform": [1e-4, 1e-2]}, "batch_size": [128, 256, 512]}
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")


def sample_configs(n: int, seed: int = SEARCH_SEED) -> list[dict]:
    """n seeded draws from the Table 3.11 LSTM ranges (W omitted under formulation B)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.log10(SPACE["learning_rate"]["log_uniform"])
    return [{"layers": int(rng.choice(SPACE["layers"])),
             "units": int(rng.choice(SPACE["units"])),
             "dropout": round(float(rng.uniform(*SPACE["dropout"]["uniform"])), 4),
             "learning_rate": float(f"{10 ** rng.uniform(lo, hi):.4g}"),
             "batch_size": int(rng.choice(SPACE["batch_size"]))} for _ in range(n)]


def to_seq(X: np.ndarray) -> np.ndarray:
    """(N, F) scaled matrix -> (N, F, 1) float32 sequences (formulation B)."""
    return np.ascontiguousarray(X, dtype=np.float32).reshape(len(X), X.shape[1], 1)


def build_model(cfg: dict, n_features: int, n_out: int) -> keras.Model:
    """1-2 LSTM layers -> Dropout -> Dense (sigmoid if n_out == 1 else softmax)."""
    inp = keras.Input((n_features, 1), name="flow_features")
    x = inp
    for i in range(cfg["layers"]):
        x = keras.layers.LSTM(cfg["units"], return_sequences=i < cfg["layers"] - 1,
                              name=f"lstm_{i + 1}")(x)
    x = keras.layers.Dropout(cfg["dropout"], name="dropout")(x)
    binary = n_out == 1
    out = keras.layers.Dense(n_out, activation="sigmoid" if binary else "softmax",
                             name="output")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(cfg["learning_rate"]),
                  loss="binary_crossentropy" if binary else "sparse_categorical_crossentropy")
    return model


def class_weights(y: np.ndarray) -> dict[int, float]:
    """'balanced' weights n / (k * n_c) from the training labels (Table 3.11)."""
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(v) for c, v in zip(classes, w)}


class EpochTimer(keras.callbacks.Callback):
    """Records and prints wall time per epoch (plan §4.3 logging)."""

    def __init__(self, tag: str = ""):
        super().__init__()
        self.tag, self.times, self._t = tag, [], 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self._t)
        logs = logs or {}
        print(f"    {self.tag} epoch {epoch + 1}: loss={logs.get('loss', float('nan')):.4f} "
              f"val_loss={logs.get('val_loss', float('nan')):.4f} "
              f"{self.times[-1]:.1f}s", flush=True)


def fit_one(cfg: dict, seed: int, X_tr, y_tr, X_va, y_va, n_out: int, max_epochs: int,
            tag: str = "") -> tuple[keras.Model, dict]:
    keras.utils.set_random_seed(seed)  # == tf.keras.utils.set_random_seed (§3.3.10)
    model = build_model(cfg, X_tr.shape[1], n_out)
    timer = EpochTimer(tag)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                       restore_best_weights=True)
    t = time.perf_counter()
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=max_epochs,
                     batch_size=cfg["batch_size"], class_weight=class_weights(y_tr),
                     callbacks=[es, timer], shuffle=True, verbose=0)
    fit_s = time.perf_counter() - t
    vl = [float(v) for v in hist.history["val_loss"]]
    return model, {"epochs_run": len(vl), "best_epoch": int(np.argmin(vl)) + 1,
                   "stopped_early": len(vl) < max_epochs, "fit_s": round(fit_s, 2),
                   "epoch_s": [round(s, 3) for s in timer.times],
                   "mean_epoch_s": round(float(np.mean(timer.times)), 3),
                   "n_train_rows": int(len(y_tr)),
                   "train_loss": [float(v) for v in hist.history["loss"]], "val_loss": vl}


def predict_scores(model: keras.Model, X: np.ndarray, task: str) -> np.ndarray:
    """Offline batch scoring of a partition: P(Malicious) (binary) or (N, 7) probabilities."""
    p = model.predict(X, batch_size=4096, verbose=0)
    return p[:, 0] if task == "binary" else p


def selection_score(task: str, y: np.ndarray, p: np.ndarray) -> float:
    if task == "binary":
        return float(f1_score(y, (p >= 0.5).astype(int), zero_division=0))
    return float(f1_score(y, np.argmax(p, axis=1), average="macro", zero_division=0))


def time_single_record(model_path: Path, rows: np.ndarray) -> tuple[dict, np.ndarray]:
    """Run the timing child (one thread) on rows (n, F); returns stats and its outputs."""
    with tempfile.TemporaryDirectory() as td:
        rp = Path(td) / "rows.npy"
        np.save(rp, rows.astype(np.float32))
        env = {**os.environ, **{k: "1" for k in THREAD_ENV}, "TF_CPP_MIN_LOG_LEVEL": "2"}
        r = subprocess.run([sys.executable, "-m", "nids.train.train_lstm",
                            "--time-model", str(model_path), "--time-rows", str(rp)],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"timing child failed:\n{r.stderr[-3000:]}")
        return json.loads(r.stdout.strip().splitlines()[-1]), np.load(Path(td) / "out.npy")


def _timing_child(model_path: Path, rows_path: Path) -> None:
    """Child process: single-threaded tf.function calls, one record at a time (plan §6.1)."""
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model = keras.models.load_model(model_path)
    F = int(model.input_shape[1])
    infer = tf.function(lambda t: model(t, training=False),
                        input_signature=[tf.TensorSpec((1, F, 1), tf.float32)])
    infer(tf.zeros((1, F, 1)))  # trace once, untimed
    rows = np.load(rows_path).astype(np.float32)
    n = len(rows)
    t, out = np.empty(n), []
    for i in range(n):
        x = rows[i].reshape(1, F, 1)
        s = time.perf_counter_ns()
        p = infer(x).numpy()[0]
        t[i] = (time.perf_counter_ns() - s) / 1e6
        out.append(p)
    np.save(rows_path.with_name("out.npy"), np.asarray(out))
    print(json.dumps({"n_calls": n, "method": "tf.function, input_signature (1, F, 1)",
                      "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
                      "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
                      "source": "validation rows", "mean_ms": float(t.mean()),
                      "median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
                      "p99_ms": float(np.percentile(t, 99)), "max_ms": float(t.max()),
                      "implied_max_records_per_s": float(1000 / t.mean())}))


def save_bundle(model: keras.Model, art: Path, task: str, seed: int, meta: dict) -> Path:
    path = art / f"lstm_{task}_seed{seed}.keras"
    model.save(path)
    (art / f"lstm_{task}_seed{seed}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def train_task(task: str, d: dict, search_idx: np.ndarray, seeds: list[int], art: Path,
               names: list[str], smoke: bool, search_frac: float) -> tuple[dict, dict]:
    spec = TASKS[task]
    X_tr, y_tr = to_seq(d["X_train"]), d[f"y_train_{spec['y']}"].astype(np.int64)
    X_va, y_va = to_seq(d["X_val"]), d[f"y_val_{spec['y']}"].astype(np.int64)
    F = X_tr.shape[1]
    n_out = 1 if task == "binary" else len(names)
    n_trials = SMOKE_N_TRIALS if smoke else N_TRIALS
    max_epochs = SMOKE_MAX_EPOCHS if smoke else MAX_EPOCHS
    n_timing = SMOKE_N_TIMING if smoke else N_TIMING
    art.mkdir(parents=True, exist_ok=True)

    # 1. Random search (seed 42 for every trial; selection on validation).
    configs = sample_configs(n_trials)
    print(f"\n== LSTM {task} (formulation {FORMULATION}, input ({F}, 1)): {n_trials} trials "
          f"on {len(search_idx)} training rows, select by validation {spec['metric']}")
    trials, best_i, best_model = [], -1, None
    t0 = time.perf_counter()
    for i, cfg in enumerate(configs):
        print(f"  trial {i + 1}/{n_trials}: {cfg}", flush=True)
        model, fl = fit_one(cfg, SEARCH_SEED, X_tr[search_idx], y_tr[search_idx], X_va, y_va,
                            n_out, max_epochs, tag=f"t{i + 1}")
        score = selection_score(task, y_va, predict_scores(model, X_va, task))
        trials.append({"trial": i + 1, "config": cfg, "val_score": score, **fl})
        print(f"  -> val {spec['metric']}={score:.4f}, {fl['epochs_run']} epochs "
              f"(best {fl['best_epoch']}), {fl['mean_epoch_s']:.1f} s/epoch", flush=True)
        if best_i < 0 or score > trials[best_i]["val_score"]:
            best_i, best_model = i, model
        else:
            del model
    search_s = time.perf_counter() - t0
    cfg = configs[best_i]
    print(f"Best trial {best_i + 1}: val {spec['metric']}={trials[best_i]['val_score']:.4f} {cfg}")

    # 2-3. Seed retrains; the decision rule is fixed on the seed-42 model.
    # With search_frac == 1 the best trial already IS the config trained on the full
    # partition with seed 42, so it is reused rather than refitted.
    print(f"Retraining with seeds {seeds}:")
    decision, per_seed, timing = None, [], None
    for seed in seeds:
        if seed == SEARCH_SEED and search_frac >= 1.0:
            model, fl = best_model, {k: v for k, v in trials[best_i].items()
                                     if k not in ("trial", "config", "val_score")}
        else:
            model, fl = fit_one(cfg, seed, X_tr, y_tr, X_va, y_va, n_out, max_epochs,
                                tag=f"s{seed}")
        p = predict_scores(model, X_va, task)
        if task == "binary":
            if decision is None:
                decision = choose_threshold_rule(y_va, p)
                print(f"  threshold rule: {decision['rule']} (val F1 @0.5="
                      f"{decision['val_f1_at_0.5']:.4f}, @tuned {decision['tuned_tau']:.4f}="
                      f"{decision['val_f1_at_tuned']:.4f}, gain {decision['f1_gain']:+.4f}, "
                      f"min {decision['min_gain']})")
            tau = 0.5 if decision["rule"] == "0.5" else best_f1_threshold(y_va, p)["tau"]
            m = binary_val_metrics(y_va, p, tau)
            msg = (f"tau={tau:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
                   f"F1={m['f1']:.4f} AUC={m['roc_auc']:.4f}")
        else:
            decision = decision or {"rule": "argmax"}
            tau = None
            m = multi_val_metrics(y_va, p, np.arange(len(names)), names)
            msg = f"macro-F1={m['macro_f1']:.4f} bin-F1={m['binary_collapsed']['f1']:.4f}"
        print(f"  seed {seed}: {msg} ({fl['epochs_run']} epochs, {fl['mean_epoch_s']:.1f} s/epoch)")
        meta = {"model": "LSTM", "task": task, "formulation": FORMULATION,
                "formulation_desc": "each flow's F scaled features as a sequence of F steps",
                "input_shape": [F, 1], "warmup_records": 0, "features": int(F),
                "tau": tau, "threshold_rule": decision["rule"], "config": cfg, "seed": seed,
                "classes": list(range(n_out)) if task == "multi" else [0, 1],
                "class_names": names if task == "multi" else ["Normal", "Malicious"],
                "output": ("P(Malicious), sigmoid" if task == "binary"
                           else "class probabilities, softmax")}
        path = save_bundle(model, art, task, seed, meta)
        if seed == SEARCH_SEED:
            rng = np.random.default_rng(SEARCH_SEED)
            rows = d["X_val"][rng.choice(len(X_va), size=n_timing, replace=len(X_va) < n_timing)]
            timing, out = time_single_record(path, rows)
            ref = model.predict(to_seq(rows), batch_size=4096, verbose=0)
            timing["max_abs_diff_vs_predict"] = float(np.abs(out - ref).max())
            print(f"  single-record tf.function (1 thread, {n_timing} calls): median "
                  f"{timing['median_ms']:.2f} ms, p95 {timing['p95_ms']:.2f} ms; "
                  f"max |diff| vs predict {timing['max_abs_diff_vs_predict']:.2e}")
        per_seed.append({"seed": seed, "tau": tau, **{k: fl[k] for k in (
            "epochs_run", "best_epoch", "fit_s", "mean_epoch_s", "epoch_s")}, **m})
        if model is not best_model:
            del model
        keras.backend.clear_session()

    keys = (("precision", "recall", "f1", "roc_auc", "fpr") if task == "binary"
            else ("macro_f1", "accuracy"))
    mean_sd = {k: {"mean": float(np.mean([r[k] for r in per_seed])),
                   "sd": float(np.std([r[k] for r in per_seed], ddof=1)) if len(per_seed) > 1
                   else 0.0} for k in keys}
    all_epoch_s = [s for r in trials for s in r["epoch_s"]]

    config = {"model": "LSTM", "task": task, "formulation": FORMULATION,
              "input_shape": [F, 1], "hyperparameters": cfg,
              "architecture": "LSTM x layers -> Dropout -> Dense "
                              + ("(1, sigmoid)" if task == "binary" else f"({n_out}, softmax)"),
              "training": {"optimizer": "Adam",
                           "loss": ("binary_crossentropy" if task == "binary"
                                    else "sparse_categorical_crossentropy"),
                           "class_weight": "balanced (training labels)",
                           "early_stopping": {"monitor": "val_loss", "patience": PATIENCE,
                                              "restore_best_weights": True},
                           "max_epochs": max_epochs, "op_determinism": True},
              "search": {"method": "random search", "n_trials": n_trials, "seed": SEARCH_SEED,
                         "space": SPACE, "selection": f"validation {spec['metric']}"
                         + (" at 0.5" if task == "binary" else ""),
                         "search_frac": search_frac, "n_rows": int(len(search_idx)),
                         "best_val_score": trials[best_i]["val_score"]},
              "decision_rule": ("Malicious iff P(Malicious) >= tau" if task == "binary"
                                else "argmax of class probabilities"),
              "threshold_choice": decision, "seeds": seeds, "class_names": names,
              "tensorflow_version": tf.__version__, "keras_version": keras.__version__,
              "smoke": smoke}
    (art / f"lstm_{task}_config.json").write_text(json.dumps(config, indent=2) + "\n")

    log = {"model": "LSTM", "task": task, "formulation": FORMULATION, "smoke": smoke,
           "test_partition_used": False, "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
           "n_val_malicious": int(d["y_val_bin"].sum()), "search_s": round(search_s, 1),
           "class_weights": class_weights(y_tr), "chosen": cfg,
           "best_val_score": trials[best_i]["val_score"],
           "search_epoch_s_summary": {"n_epochs": len(all_epoch_s),
                                      "mean": float(np.mean(all_epoch_s)),
                                      "min": float(np.min(all_epoch_s)),
                                      "max": float(np.max(all_epoch_s))},
           "search_trials": sorted(trials, key=lambda r: -r["val_score"]),
           "threshold_choice": decision, "per_seed_validation": per_seed,
           "validation_mean_sd": mean_sd, "single_record_timing": timing,
           "python": platform.python_version(), "tensorflow_version": tf.__version__,
           "keras_version": keras.__version__}
    return config, log


def run(cfg: dict, tasks: list[str], smoke: bool = False, search_frac: float = 1.0) -> dict:
    tf.config.experimental.enable_op_determinism()  # reproducible reruns (NFR4)
    proc = ROOT / cfg["paths"]["processed"]
    if smoke:
        out = proc / "smoke_lstm"
        art, log_dir, seeds = out / "artefacts", out / "train_log", cfg["seeds"][:2]
    else:
        art = ROOT / cfg["paths"]["artefacts"]
        log_dir, seeds = ROOT / "results" / "train_log", cfg["seeds"]
    names = json.loads((ROOT / cfg["paths"]["artefacts"] / "label_map.json").read_text())["classes"]

    d = load_partitions(proc)  # train + val only (control 6)
    if smoke:
        d = smoke_subsample(d)
    search_idx = stratified_fraction(d["y_train_7"], search_frac)
    print(f"LSTM: {len(d['y_train_7'])} training rows, {len(d['y_val_7'])} validation rows, "
          f"{os.cpu_count()} CPU threads{' [SMOKE]' if smoke else ''}")

    logs = {}
    log_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        _, log = train_task(task, d, search_idx, seeds, art, names, smoke, search_frac)
        (log_dir / f"lstm_{task}.json").write_text(json.dumps(log, indent=2) + "\n")
        logs[task] = log
    print(f"\nWrote models + meta/configs to {art}; logs to {log_dir}")
    return logs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=ROOT / "config.yaml", type=Path)
    ap.add_argument("--task", choices=["binary", "multi", "both"], default="both")
    ap.add_argument("--search-frac", type=float, default=1.0,
                    help="stratified fraction of the training partition used for the trials")
    ap.add_argument("--smoke", action="store_true",
                    help="small subsample, 3 trials, <=6 epochs, 2 seeds; "
                         "output to data/processed/smoke_lstm/")
    ap.add_argument("--time-model", type=Path, help=argparse.SUPPRESS)  # internal: timing child
    ap.add_argument("--time-rows", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.time_model:
        _timing_child(args.time_model, args.time_rows)
        return
    tasks = ["binary", "multi"] if args.task == "both" else [args.task]
    run(yaml.safe_load(Path(args.config).read_text()), tasks, smoke=args.smoke,
        search_frac=args.search_frac)


if __name__ == "__main__":
    main()
