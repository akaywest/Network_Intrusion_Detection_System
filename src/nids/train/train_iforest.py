"""Isolation Forest training (M2, implementation plan §4.1; Chapter 3 §3.3.10, Table 3.11).

Procedure:
  1. Fit on the NORMAL rows of the training partition only (Table 3.11 "Training data").
  2. Grid search, 36 fits (Table 3.11 ranges): n_estimators {100,200,300,500} x
     max_samples {256,512,1024} x max_features {0.5,0.75,1.0}, random_state 42.
     Each fit is scored on the VALIDATION partition with the anomaly score
     s = -score_samples(x)  (plan §1.1 item 4: scikit-learn returns the negative of the
     Eq. 3.3 score, so negating gives "high = anomalous").
     Selection: best F1 over thresholds (precision_recall_curve), ties broken by ROC-AUC.
  3. Retrain the chosen configuration with seeds 42-46 (§3.3.10, NFR3). tau is chosen
     PER SEED as the validation F1-maximising threshold, because each seed gives a
     different score distribution. Decision rule: Malicious iff s >= tau.
  4. Save artefacts/iforest_seed{s}.joblib = {model, tau, config, seed, val_score_max}
     (val_score_max feeds the confidence rule of plan §5.2), artefacts/iforest_config.json
     and results/train_log/iforest.json (validation metrics per seed, mean +- SD).

Leakage control 5/6 (Table 3.12): thresholds and tuning use validation data only; the
test partition is never loaded by this module.

Training may use all cores (§3.3.10); the engine sets n_jobs=1 at load time (§6.1).

Usage:  python -m nids.train.train_iforest [--config config.yaml] [--smoke]
        --smoke: 2,000 Normal train rows, 2,000 stratified val rows, 9 fits, seeds 42-43;
                 writes to data/processed/smoke_iforest/ only (never artefacts/ or results/).
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from nids.prep.convert_xlsx import ROOT

# Table 3.11 ranges, discretised as in plan §4.1 (4 x 3 x 3 = 36 fits).
GRID_SPACE = {"n_estimators": [100, 200, 300, 500],
              "max_samples": [256, 512, 1024],
              "max_features": [0.5, 0.75, 1.0]}
SMOKE_SPACE = {**GRID_SPACE, "n_estimators": [100]}
SEARCH_SEED = 42
SMOKE_N = 2000
# The only files this module may read (control 6: no test partition).
PARTITION_FILES = ("X_train.npy", "y_train_bin.npy", "X_val.npy", "y_val_bin.npy")


def grid(space: dict = GRID_SPACE) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, v)) for v in itertools.product(*(space[k] for k in keys))]


def load_partitions(proc: Path) -> dict[str, np.ndarray]:
    """Load training and validation arrays only (Table 3.12 control 6)."""
    out = {}
    for name in PARTITION_FILES:
        assert "test" not in name, "test partition must not be used for training/tuning"
        out[name.removesuffix(".npy")] = np.load(proc / name)
    return out


def anomaly_score(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """s(x) of Eq. 3.3: higher = more anomalous (negated score_samples, plan §1.1 item 4)."""
    return -model.score_samples(X)


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> dict:
    """F1-maximising threshold tau on s; rule: Malicious iff s >= tau (Table 3.11)."""
    p, r, thr = precision_recall_curve(y, s)
    p, r = p[:-1], r[:-1]  # last PR point has no threshold
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    i = int(np.argmax(f1))
    return {"tau": float(thr[i]), "f1": float(f1[i]),
            "precision": float(p[i]), "recall": float(r[i])}


def val_metrics(y: np.ndarray, s: np.ndarray, tau: float) -> dict:
    pred = (s >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "roc_auc": float(roc_auc_score(y, s)), "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def fit_iforest(X_normal: np.ndarray, params: dict, seed: int) -> IsolationForest:
    return IsolationForest(**params, random_state=seed, n_jobs=-1).fit(X_normal)


def grid_search(X_normal, X_val, y_val, space: dict = GRID_SPACE) -> tuple[dict, list[dict]]:
    """Fit each combination with seed 42; choose by (val F1, val ROC-AUC)."""
    rows = []
    for params in grid(space):
        t = time.perf_counter()
        s = anomaly_score(fit_iforest(X_normal, params, SEARCH_SEED), X_val)
        best = best_f1_threshold(y_val, s)
        rows.append({**params, "val_f1": best["f1"], "val_roc_auc": float(roc_auc_score(y_val, s)),
                     "tau": best["tau"], "fit_score_s": round(time.perf_counter() - t, 3)})
        print(f"  {params}  F1={best['f1']:.4f}  AUC={rows[-1]['val_roc_auc']:.4f}")
    chosen = max(rows, key=lambda r: (r["val_f1"], r["val_roc_auc"]))
    return {k: chosen[k] for k in space}, rows


def train_seeds(X_normal, X_val, y_val, params: dict, seeds: list[int], art: Path) -> list[dict]:
    """Retrain the chosen config per seed, set tau on validation, save the bundle."""
    art.mkdir(parents=True, exist_ok=True)
    log = []
    for seed in seeds:
        t = time.perf_counter()
        model = fit_iforest(X_normal, params, seed)
        fit_s = time.perf_counter() - t
        s = anomaly_score(model, X_val)
        tau = best_f1_threshold(y_val, s)["tau"]
        bundle = {"model": model, "tau": tau, "config": dict(params), "seed": seed,
                  "val_score_max": float(s.max())}
        joblib.dump(bundle, art / f"iforest_seed{seed}.joblib")
        m = val_metrics(y_val, s, tau)
        log.append({"seed": seed, "tau": tau, "val_score_max": float(s.max()),
                    "fit_s": round(fit_s, 3), **m})
        print(f"  seed {seed}: tau={tau:.4f}  P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f} AUC={m['roc_auc']:.4f}")
    return log


def summarise(per_seed: list[dict]) -> dict:
    keys = ("precision", "recall", "f1", "roc_auc", "fpr", "tau")
    return {k: {"mean": float(np.mean([r[k] for r in per_seed])),
                "sd": float(np.std([r[k] for r in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0}
            for k in keys}


def smoke_subsample(d: dict, seed: int = SEARCH_SEED) -> dict:
    """Small seeded subsample: SMOKE_N Normal train rows, SMOKE_N stratified val rows."""
    rng = np.random.default_rng(seed)
    normal = np.flatnonzero(d["y_train_bin"] == 0)
    tr = np.sort(rng.choice(normal, size=min(SMOKE_N, len(normal)), replace=False))
    va, _ = train_test_split(np.arange(len(d["y_val_bin"])), train_size=SMOKE_N,
                             stratify=d["y_val_bin"], random_state=seed)
    va = np.sort(va)
    return {"X_train": d["X_train"][tr], "y_train_bin": d["y_train_bin"][tr],
            "X_val": d["X_val"][va], "y_val_bin": d["y_val_bin"][va]}


def run(cfg: dict, smoke: bool = False) -> dict:
    proc = ROOT / cfg["paths"]["processed"]
    if smoke:
        out = proc / "smoke_iforest"
        art, log_path = out / "artefacts", out / "train_log" / "iforest.json"
        space, seeds = SMOKE_SPACE, cfg["seeds"][:2]
    else:
        art = ROOT / cfg["paths"]["artefacts"]
        log_path = ROOT / "results" / "train_log" / "iforest.json"
        space, seeds = GRID_SPACE, cfg["seeds"]

    d = load_partitions(proc)
    if smoke:
        d = smoke_subsample(d)
    X_normal = d["X_train"][d["y_train_bin"] == 0]  # Table 3.11: Normal training rows only
    X_val, y_val = d["X_val"], d["y_val_bin"]
    print(f"IF: {len(X_normal)} Normal training rows, {len(y_val)} validation rows "
          f"({int(y_val.sum())} Malicious){' [SMOKE]' if smoke else ''}")

    t0 = time.perf_counter()
    print(f"Grid search ({len(grid(space))} fits, seed {SEARCH_SEED}):")
    params, rows = grid_search(X_normal, X_val, y_val, space)
    print(f"Chosen: {params}")
    print(f"Retraining with seeds {seeds}:")
    per_seed = train_seeds(X_normal, X_val, y_val, params, seeds, art)

    config = {"model": "IsolationForest", "hyperparameters": params,
              "search": {"method": "grid", "space": space, "n_fits": len(rows),
                         "seed": SEARCH_SEED,
                         "selection": "max validation best-threshold F1, tie-break ROC-AUC"},
              "score": "s = -score_samples(x) (higher = more anomalous; plan §1.1 item 4)",
              "decision_rule": "Malicious iff s >= tau; tau = validation F1-maximising "
                               "threshold, set per seed",
              "training_data": "Normal rows of the training partition only",
              "n_train_normal": int(len(X_normal)), "seeds": seeds,
              "sklearn_version": sklearn.__version__, "smoke": smoke}
    art.mkdir(parents=True, exist_ok=True)
    (art / "iforest_config.json").write_text(json.dumps(config, indent=2) + "\n")

    log = {"model": "IsolationForest", "smoke": smoke, "test_partition_used": False,
           "n_train_normal": int(len(X_normal)), "n_val": int(len(y_val)),
           "n_val_malicious": int(y_val.sum()), "chosen": params, "grid": rows,
           "per_seed_validation": per_seed, "validation_mean_sd": summarise(per_seed),
           "total_s": round(time.perf_counter() - t0, 1),
           "python": platform.python_version(), "sklearn_version": sklearn.__version__}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2) + "\n")
    print(f"Wrote {len(seeds)} bundles + iforest_config.json to {art}; log to {log_path}")
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=ROOT / "config.yaml", type=Path)
    ap.add_argument("--smoke", action="store_true",
                    help="small subsample, 9 fits, 2 seeds; output to data/processed/smoke_iforest/")
    args = ap.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()), smoke=args.smoke)


if __name__ == "__main__":
    main()
