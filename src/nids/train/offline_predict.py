"""Offline predictions on the full held-out test partition (M2, implementation plan §4.4).

Scores the TEST partition once with every saved model and seed:
  IF x5 (binary), RF binary x5, RF multi x5, LSTM binary x5, LSTM multi x5  (25 files).

Each model is applied with the decision rule and threshold saved at training time
(validation-chosen; leakage controls 5/6, Table 3.12). Nothing here is tuned on test.

Per row the output keeps row_index, the true labels, the predicted label, a continuous
score and a confidence, in results/offline/<model>_<task>_seed<s>.parquet. These files
are the source of the detection metrics (§8.1, nids.evaluation.detection_metrics) and the
reference for the streamed-vs-offline agreement check (§8.4, §3.3.11).

Score (ROC-AUC input, plan §8.1):
  IF            s = -score_samples(x)           (Eq. 3.3 sign, plan §1.1 item 4)
  RF/LSTM bin   P(Malicious)
  RF/LSTM multi 1 - P(Normal)
Decision:
  IF            Malicious iff s >= tau (tau per seed, validation F1-maximising)
  RF/LSTM bin   Malicious iff P(Malicious) >= tau (tau stored in the bundle / meta JSON)
  RF/LSTM multi argmax of the seven class probabilities
Confidence (plan §5.2):
  RF/LSTM       probability of the predicted class
  IF            clip((s - tau) / (val_score_max - tau), 0, 1): how far s lies between tau
                and the highest validation score. It is 0 for every Normal prediction,
                because it only feeds the alert severity rule (FR10, §3.3.9).

This is the ONLY module that reads the test partition, and it runs after all training.

Usage:  python -m nids.train.offline_predict [--config config.yaml]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from nids.prep.convert_xlsx import ROOT  # noqa: E402

FAMILIES = (("iforest", "binary"), ("rf", "binary"), ("rf", "multi"),
            ("lstm", "binary"), ("lstm", "multi"))
BIN_NAMES = ("Normal", "Malicious")


def iforest_confidence(s: np.ndarray, tau: float, score_max: float) -> np.ndarray:
    """Plan §5.2: position of s between tau and the highest validation score, in [0, 1]."""
    span = score_max - tau
    if span <= 0:
        return (s >= tau).astype(np.float64)
    return np.clip((s - tau) / span, 0.0, 1.0)


def artefact_path(art: Path, name: str) -> Path:
    """Models load only from artefacts/ (NFR6)."""
    art = art.resolve()
    p = (art / name).resolve()
    if art not in p.parents:
        raise PermissionError(p)
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_test(proc: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Scaled test matrix plus row_index and true labels, checked to be aligned."""
    X = np.load(proc / "X_test.npy")
    raw = pd.read_parquet(proc / "test_raw.parquet",
                          columns=["row_index", "class_name", "label_7", "label_bin"])
    y7, yb = np.load(proc / "y_test_7.npy"), np.load(proc / "y_test_bin.npy")
    if not (len(X) == len(raw) == len(y7) and raw["row_index"].is_monotonic_increasing
            and np.array_equal(raw["label_7"].to_numpy(), y7)
            and np.array_equal(raw["label_bin"].to_numpy(), yb)):
        raise ValueError("X_test.npy and test_raw.parquet are not aligned in row_index order")
    return X, raw.reset_index(drop=True)


def base_frame(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"row_index": raw["row_index"].astype(np.int64),
                         "true_label_7": raw["label_7"].astype(np.int8),
                         "true_class": raw["class_name"].astype(str),
                         "true_bin": raw["label_bin"].astype(np.int8)})


def binary_frame(raw, score, tau, confidence) -> pd.DataFrame:
    df = base_frame(raw)
    pred = (score >= tau).astype(np.int8)
    df["pred_code"] = pred
    df["predicted_label"] = np.asarray(BIN_NAMES)[pred]
    df["pred_bin"] = pred
    df["score"] = score.astype(np.float64)
    df["confidence"] = confidence.astype(np.float64)
    return df


def multi_frame(raw, proba: np.ndarray, classes: list[int], names: list[str]) -> pd.DataFrame:
    classes = np.asarray(classes)
    normal_col = int(np.flatnonzero(classes == 0)[0])
    df = base_frame(raw)
    idx = np.argmax(proba, axis=1)
    pred = classes[idx].astype(np.int8)
    df["pred_code"] = pred
    df["predicted_label"] = np.asarray(names)[pred]
    df["pred_bin"] = (pred != 0).astype(np.int8)
    df["score"] = (1.0 - proba[:, normal_col]).astype(np.float64)
    df["confidence"] = proba[np.arange(len(proba)), idx].astype(np.float64)
    for j, c in enumerate(classes):
        df[f"p_{int(c)}"] = proba[:, j].astype(np.float64)
    return df


def score_iforest(art, seed, X, raw):
    path = artefact_path(art, f"iforest_seed{seed}.joblib")
    b = joblib.load(path)
    s = -b["model"].score_samples(X)
    df = binary_frame(raw, s, b["tau"], iforest_confidence(s, b["tau"], b["val_score_max"]))
    return df, path, {"tau": b["tau"], "threshold_rule": "val F1-max", "config": b["config"]}


def score_rf(art, seed, task, X, raw, names):
    path = artefact_path(art, f"rf_{task}_seed{seed}.joblib")
    b = joblib.load(path)
    proba = b["model"].predict_proba(X)
    if task == "binary":
        col = list(b["model"].classes_).index(1)
        p = proba[:, col]
        conf = np.where(p >= b["tau"], p, 1.0 - p)
        df = binary_frame(raw, p, b["tau"], conf)
    else:
        df = multi_frame(raw, proba, b["classes"], names)
    return df, path, {"tau": b["tau"], "threshold_rule": b["threshold_rule"],
                      "config": b["config"]}


def score_lstm(art, seed, task, X, raw, names):
    import keras  # imported lazily: only needed for the two LSTM families
    path = artefact_path(art, f"lstm_{task}_seed{seed}.keras")
    meta = json.loads(artefact_path(art, f"lstm_{task}_seed{seed}.meta.json").read_text())
    if meta["formulation"] != "B":
        raise ValueError("offline reference assumes formulation B (plan §4.3, §8.4)")
    model = keras.models.load_model(path)
    Xs = np.ascontiguousarray(X, dtype=np.float32).reshape(len(X), X.shape[1], 1)
    p = model.predict(Xs, batch_size=4096, verbose=0)  # batch scoring offline only
    keras.backend.clear_session()
    if task == "binary":
        p = p[:, 0].astype(np.float64)
        conf = np.where(p >= meta["tau"], p, 1.0 - p)
        df = binary_frame(raw, p, meta["tau"], conf)
    else:
        df = multi_frame(raw, p.astype(np.float64), meta["classes"], names)
    return df, path, {"tau": meta["tau"], "threshold_rule": meta["threshold_rule"],
                      "config": meta["config"], "formulation": meta["formulation"]}


def run(cfg: dict) -> dict:
    proc = ROOT / cfg["paths"]["processed"]
    art = ROOT / cfg["paths"]["artefacts"]
    out = ROOT / "results" / "offline"
    out.mkdir(parents=True, exist_ok=True)
    names = json.loads((art / "label_map.json").read_text())["classes"]
    X, raw = load_test(proc)
    print(f"Test partition: {len(X)} rows x {X.shape[1]} features")

    manifest = {"n_rows": int(len(X)), "n_features": int(X.shape[1]),
                "test_files": ["X_test.npy", "test_raw.parquet", "y_test_7.npy",
                               "y_test_bin.npy"],
                "files": []}
    for model, task in FAMILIES:
        for seed in cfg["seeds"]:
            t = time.perf_counter()
            if model == "iforest":
                df, path, info = score_iforest(art, seed, X, raw)
            elif model == "rf":
                df, path, info = score_rf(art, seed, task, X, raw, names)
            else:
                df, path, info = score_lstm(art, seed, task, X, raw, names)
            name = f"{model}_{task}_seed{seed}.parquet"
            df.to_parquet(out / name, index=False)
            el = time.perf_counter() - t
            acc = float((df["pred_code"] == (df["true_bin"] if task == "binary"
                                             else df["true_label_7"])).mean())
            print(f"  {name:32s} {el:6.1f}s  accuracy={acc:.4f}")
            manifest["files"].append({"file": name, "model": model, "task": task,
                                      "seed": int(seed), "artefact": path.name,
                                      "artefact_sha256": sha256(path),
                                      "score_s": round(el, 2), **info})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    a = ap.parse_args()
    run(yaml.safe_load(Path(a.config).read_text()))


if __name__ == "__main__":
    main()
