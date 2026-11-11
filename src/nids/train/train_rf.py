"""Random Forest training (M2, implementation plan §4.2; Chapter 3 §3.3.10, Table 3.11).

Two families, both trained on ALL training records (Table 3.11 "Training data"):
  binary  label_bin, scoring "f1"        -> artefacts/rf_binary_seed{s}.joblib
  multi   label_7,   scoring "f1_macro"  -> artefacts/rf_multi_seed{s}.joblib

Procedure per task:
  1. RandomizedSearchCV, 30 trials, StratifiedKFold(5, shuffle=True, random_state=42),
     on the TRAINING partition only (Table 3.11 tuning method). Search space = Table 3.11,
     including class_weight in {balanced, balanced_subsample}. n_jobs=-1 for the search
     (§3.3.10: training may use all cores); each forest inside it uses n_jobs=1.
  2. Refit the best configuration on the full training partition (refit=True, seed 42)
     and check it on VALIDATION.
  3. Decision rule, chosen on validation (Table 3.11):
       binary: P(Malicious) >= 0.5, or the validation F1-maximising threshold if that
               improves validation F1 by at least 0.005. The choice and the gain ("0.5" or "tuned") is made once on
               the seed-42 refit and recorded; with "tuned", tau is re-set per seed on
               validation (as for the Isolation Forest).
       multi:  argmax over the seven class probabilities (no threshold).
  4. Retrain the chosen configuration with seeds 42-46 (§3.3.10, NFR3) and save
     {model, task, tau, threshold_rule, config, seed, classes}.
  5. Time 1,000 single-row predict_proba calls with n_jobs=1 on validation rows (seed-42
     model) as an early check of NFR2 (plan §4.2).

Leakage control 5/6 (Table 3.12): tuning, refit checks and thresholds use training and
validation data only; the test partition is never loaded.

Usage:  python -m nids.train.train_rf [--config config.yaml] [--task binary|multi|both]
                                      [--search-frac F] [--smoke]
        --search-frac F: run the search on a stratified fraction F of the training
                 partition (plan §4.2 compute fallback; a Chapter Four deviation if < 1).
                 The refit and the seed retrains always use the full partition.
        --smoke: small stratified subsample, 3 trials, n_estimators 100-150, seeds 42-43,
                 100 timing calls; writes to data/processed/smoke_rf/ only.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
import yaml
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from nids.prep.convert_xlsx import ROOT
from nids.train.train_iforest import best_f1_threshold

TASKS = {"binary": {"y": "bin", "scoring": "f1"},
         "multi": {"y": "7", "scoring": "f1_macro"}}
# The only files this module may read (control 6: no test partition).
PARTITION_FILES = ("X_train.npy", "y_train_bin.npy", "y_train_7.npy",
                   "X_val.npy", "y_val_bin.npy", "y_val_7.npy")
SEARCH_SEED = 42
N_ITER, N_SPLITS, N_TIMING = 30, 5, 1000
SMOKE_N_ITER, SMOKE_N_TIMING, SMOKE_PER_CLASS = 3, 100, 400
MIN_TUNED_GAIN = 0.005  # tuned threshold used only if validation F1 gain >= this


def search_space(smoke: bool = False) -> dict:
    """Table 3.11 Random Forest ranges (integer ranges are inclusive)."""
    return {"n_estimators": randint(100, 151 if smoke else 501),
            "max_depth": [10, 20, 30, None],
            "min_samples_split": randint(2, 11),
            "min_samples_leaf": randint(1, 5),
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced", "balanced_subsample"]}


def describe_space(space: dict) -> dict:
    return {k: ({"randint": [int(v.a), int(v.b)]} if hasattr(v, "rvs") else v)
            for k, v in space.items()}


def load_partitions(proc: Path) -> dict[str, np.ndarray]:
    """Load training and validation arrays only (Table 3.12 control 6)."""
    out = {}
    for name in PARTITION_FILES:
        assert "test" not in name, "test partition must not be used for training/tuning"
        out[name.removesuffix(".npy")] = np.load(proc / name)
    return out


def stratified_fraction(y7: np.ndarray, frac: float, seed: int = SEARCH_SEED) -> np.ndarray:
    """Indices of a seeded, label_7-stratified fraction of the rows (sorted)."""
    if frac >= 1.0:
        return np.arange(len(y7))
    idx, _ = train_test_split(np.arange(len(y7)), train_size=frac, stratify=y7,
                              random_state=seed)
    return np.sort(idx)


def smoke_subsample(d: dict, seed: int = SEARCH_SEED) -> dict:
    """Up to SMOKE_PER_CLASS rows per seven-class label from train and from val."""
    rng = np.random.default_rng(seed)
    out = {}
    for part in ("train", "val"):
        y7 = d[f"y_{part}_7"]
        idx = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(y7 == c), size=min(SMOKE_PER_CLASS, int((y7 == c).sum())),
                       replace=False) for c in np.unique(y7)]))
        out[f"X_{part}"] = d[f"X_{part}"][idx]
        out[f"y_{part}_bin"] = d[f"y_{part}_bin"][idx]
        out[f"y_{part}_7"] = y7[idx]
    return out


def run_search(X, y, scoring: str, space: dict, n_iter: int) -> RandomizedSearchCV:
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEARCH_SEED)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=SEARCH_SEED, n_jobs=1), space, n_iter=n_iter,
        scoring=scoring, cv=cv, n_jobs=-1, refit=False, random_state=SEARCH_SEED,
        return_train_score=False, verbose=1)
    return search.fit(X, y)


def search_table(search: RandomizedSearchCV) -> list[dict]:
    r = search.cv_results_
    return [{"rank": int(r["rank_test_score"][i]),
             "params": {k: (None if v is None else v if isinstance(v, str) else int(v))
                        for k, v in r["params"][i].items()},
             "cv_mean": float(r["mean_test_score"][i]), "cv_sd": float(r["std_test_score"][i]),
             "mean_fit_s": float(r["mean_fit_time"][i])} for i in range(len(r["params"]))]


def clean_params(p: dict) -> dict:
    return {k: (v if v is None or isinstance(v, str) else int(v)) for k, v in p.items()}


def fit_rf(X, y, params: dict, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(**params, random_state=seed, n_jobs=-1).fit(X, y)


def binary_val_metrics(y, p_mal, tau: float) -> dict:
    pred = (p_mal >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p_mal)),
            "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def choose_threshold_rule(y, p_mal, min_gain: float = MIN_TUNED_GAIN) -> dict:
    """'0.5' unless the validation F1-maximising threshold improves validation F1 by at
    least min_gain (Table 3.11 decision rule; avoids tuning for a negligible gain)."""
    f1_05 = binary_val_metrics(y, p_mal, 0.5)["f1"]
    tuned = best_f1_threshold(y, p_mal)
    gain = tuned["f1"] - f1_05
    rule = "tuned" if gain >= min_gain else "0.5"
    return {"rule": rule, "val_f1_at_0.5": f1_05, "tuned_tau": tuned["tau"],
            "val_f1_at_tuned": tuned["f1"], "f1_gain": gain, "min_gain": min_gain}


def multi_val_metrics(y7, proba, classes: np.ndarray, names: list[str]) -> dict:
    pred = classes[np.argmax(proba, axis=1)]
    normal_col = int(np.flatnonzero(classes == 0)[0])
    y_bin, pred_bin = (y7 != 0).astype(int), (pred != 0).astype(int)
    rec = recall_score(y7, pred, labels=classes, average=None, zero_division=0)
    return {"macro_f1": float(f1_score(y7, pred, average="macro", zero_division=0)),
            "accuracy": float((pred == y7).mean()),
            "per_class_recall": {names[int(c)]: float(r) for c, r in zip(classes, rec)},
            "binary_collapsed": {"f1": float(f1_score(y_bin, pred_bin, zero_division=0)),
                                 "roc_auc": float(roc_auc_score(y_bin, 1 - proba[:, normal_col]))},
            "confusion_matrix": confusion_matrix(y7, pred, labels=classes).tolist()}


def time_single_row(model: RandomForestClassifier, X: np.ndarray, n: int,
                    seed: int = SEARCH_SEED) -> dict:
    """n single-row predict_proba calls with n_jobs=1 (plan §4.2; NFR2 early check)."""
    model.set_params(n_jobs=1)
    rows = X[np.random.default_rng(seed).choice(len(X), size=n, replace=len(X) < n)]
    model.predict_proba(rows[:1])  # warm-up, untimed
    t = np.empty(n)
    for i in range(n):
        s = time.perf_counter_ns()
        model.predict_proba(rows[i:i + 1])
        t[i] = (time.perf_counter_ns() - s) / 1e6
    return {"n_calls": n, "n_jobs": 1, "source": "validation rows",
            "mean_ms": float(t.mean()), "median_ms": float(np.median(t)),
            "p95_ms": float(np.percentile(t, 95)), "p99_ms": float(np.percentile(t, 99)),
            "max_ms": float(t.max()), "implied_max_records_per_s": float(1000 / t.mean())}


def train_task(task: str, d: dict, search_idx: np.ndarray, seeds: list[int], art: Path,
               names: list[str], smoke: bool, search_frac: float) -> tuple[dict, dict]:
    spec = TASKS[task]
    X_tr, y_tr = d["X_train"], d[f"y_train_{spec['y']}"]
    X_va, y_va = d["X_val"], d[f"y_val_{spec['y']}"]
    y_va_bin = d["y_val_bin"]
    space = search_space(smoke)
    n_iter = SMOKE_N_ITER if smoke else N_ITER
    n_timing = SMOKE_N_TIMING if smoke else N_TIMING

    print(f"\n== RF {task}: search {n_iter} trials x {N_SPLITS} folds on "
          f"{len(search_idx)} training rows, scoring {spec['scoring']}")
    t0 = time.perf_counter()
    search = run_search(X_tr[search_idx], y_tr[search_idx], spec["scoring"], space, n_iter)
    search_s = time.perf_counter() - t0
    params = clean_params(search.best_params_)
    print(f"Best CV {spec['scoring']}={search.best_score_:.4f}: {params}")

    # Refit on the full training partition (seed 42) and check on validation.
    refit = fit_rf(X_tr, y_tr, params, SEARCH_SEED)
    proba = refit.predict_proba(X_va)
    if task == "binary":
        p_mal = proba[:, list(refit.classes_).index(1)]
        decision = choose_threshold_rule(y_va, p_mal)
        print(f"Threshold rule: {decision['rule']} (val F1 @0.5={decision['val_f1_at_0.5']:.4f}, "
              f"@tuned {decision['tuned_tau']:.4f}={decision['val_f1_at_tuned']:.4f}, "
              f"gain {decision['f1_gain']:+.4f}, min {decision['min_gain']})")
    else:
        decision = {"rule": "argmax"}
    refit_val = (binary_val_metrics(y_va, p_mal, 0.5 if decision["rule"] == "0.5"
                                    else decision["tuned_tau"])
                 if task == "binary" else multi_val_metrics(y_va, proba, refit.classes_, names))

    print(f"Retraining with seeds {seeds}:")
    art.mkdir(parents=True, exist_ok=True)
    per_seed, timing = [], None
    for seed in seeds:
        t = time.perf_counter()
        model = refit if seed == SEARCH_SEED else fit_rf(X_tr, y_tr, params, seed)
        fit_s = time.perf_counter() - t
        proba = model.predict_proba(X_va)
        if task == "binary":
            p_mal = proba[:, list(model.classes_).index(1)]
            tau = 0.5 if decision["rule"] == "0.5" else best_f1_threshold(y_va, p_mal)["tau"]
            m = binary_val_metrics(y_va, p_mal, tau)
            msg = f"tau={tau:.4f} P={m['precision']:.4f} R={m['recall']:.4f} " \
                  f"F1={m['f1']:.4f} AUC={m['roc_auc']:.4f}"
        else:
            tau = None
            m = multi_val_metrics(y_va, proba, model.classes_, names)
            msg = f"macro-F1={m['macro_f1']:.4f} bin-F1={m['binary_collapsed']['f1']:.4f}"
        print(f"  seed {seed}: {msg}")
        if seed == SEARCH_SEED:
            timing = time_single_row(model, X_va, n_timing)
            print(f"  single-row predict_proba (n_jobs=1, {n_timing} calls): "
                  f"median {timing['median_ms']:.2f} ms, p95 {timing['p95_ms']:.2f} ms")
        model.set_params(n_jobs=1)  # inference is single-threaded (§3.3.10)
        joblib.dump({"model": model, "task": task, "tau": tau,
                     "threshold_rule": decision["rule"], "config": params, "seed": seed,
                     "classes": [int(c) for c in model.classes_]},
                    art / f"rf_{task}_seed{seed}.joblib")
        per_seed.append({"seed": seed, "tau": tau, "fit_s": round(fit_s, 2), **m})

    keys = (("precision", "recall", "f1", "roc_auc", "fpr") if task == "binary"
            else ("macro_f1", "accuracy"))
    mean_sd = {k: {"mean": float(np.mean([r[k] for r in per_seed])),
                   "sd": float(np.std([r[k] for r in per_seed], ddof=1)) if len(per_seed) > 1
                   else 0.0} for k in keys}

    config = {"model": "RandomForestClassifier", "task": task, "hyperparameters": params,
              "search": {"method": "RandomizedSearchCV", "n_iter": n_iter,
                         "cv": f"StratifiedKFold({N_SPLITS}, shuffle=True, random_state=42)",
                         "scoring": spec["scoring"], "space": describe_space(space),
                         "data": "training partition", "search_frac": search_frac,
                         "n_rows": int(len(search_idx)), "best_cv_score": float(search.best_score_)},
              "decision_rule": ("Malicious iff P(Malicious) >= tau" if task == "binary"
                                else "argmax of class probabilities"),
              "threshold_choice": decision, "seeds": seeds,
              "class_names": names, "sklearn_version": sklearn.__version__, "smoke": smoke}
    (art / f"rf_{task}_config.json").write_text(json.dumps(config, indent=2) + "\n")

    log = {"model": "RandomForest", "task": task, "smoke": smoke, "test_partition_used": False,
           "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
           "n_val_malicious": int(y_va_bin.sum()), "search_s": round(search_s, 1),
           "best_cv_score": float(search.best_score_), "chosen": params,
           "search_trials": sorted(search_table(search), key=lambda r: r["rank"]),
           "refit_validation": refit_val, "threshold_choice": decision,
           "per_seed_validation": per_seed, "validation_mean_sd": mean_sd,
           "single_row_timing": timing,
           "python": platform.python_version(), "sklearn_version": sklearn.__version__}
    return config, log


def run(cfg: dict, tasks: list[str], smoke: bool = False, search_frac: float = 1.0) -> dict:
    proc = ROOT / cfg["paths"]["processed"]
    if smoke:
        out = proc / "smoke_rf"
        art, log_dir, seeds = out / "artefacts", out / "train_log", cfg["seeds"][:2]
    else:
        art = ROOT / cfg["paths"]["artefacts"]
        log_dir, seeds = ROOT / "results" / "train_log", cfg["seeds"]
    names = json.loads((ROOT / cfg["paths"]["artefacts"] / "label_map.json").read_text())["classes"]

    d = load_partitions(proc)
    if smoke:
        d = smoke_subsample(d)
    search_idx = stratified_fraction(d["y_train_7"], search_frac)
    print(f"RF: {len(d['y_train_7'])} training rows, {len(d['y_val_7'])} validation rows"
          f"{' [SMOKE]' if smoke else ''}")

    logs = {}
    log_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        _, log = train_task(task, d, search_idx, seeds, art, names, smoke, search_frac)
        (log_dir / f"rf_{task}.json").write_text(json.dumps(log, indent=2) + "\n")
        logs[task] = log
    print(f"\nWrote bundles + configs to {art}; logs to {log_dir}")
    return logs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=ROOT / "config.yaml", type=Path)
    ap.add_argument("--task", choices=["binary", "multi", "both"], default="both")
    ap.add_argument("--search-frac", type=float, default=1.0,
                    help="stratified fraction of the training partition used for the search")
    ap.add_argument("--smoke", action="store_true",
                    help="small subsample, 3 trials, 2 seeds; output to data/processed/smoke_rf/")
    args = ap.parse_args()
    tasks = ["binary", "multi"] if args.task == "both" else [args.task]
    run(yaml.safe_load(Path(args.config).read_text()), tasks, smoke=args.smoke,
        search_frac=args.search_frac)


if __name__ == "__main__":
    main()
