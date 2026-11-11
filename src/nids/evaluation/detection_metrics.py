"""Detection metrics on the full held-out test partition (M8, implementation plan §8.1;
Chapter 3 §3.3.11, objective iii, NFR3).

Reads the offline prediction files written by nids.train.offline_predict
(results/offline/<model>_<task>_seed<s>.parquet) and computes, per model and seed:
  binary task (positive = Malicious): precision, recall, F1, ROC-AUC, FPR
  seven-class task: macro-F1, accuracy, per-class recall, confusion matrix, plus the
      binary metrics above on the collapsed decision (predicted class != Normal),
      with ROC-AUC on 1 - P(Normal) (plan §8.1)
then reports mean +- SD over seeds 42-46 (SD = sample SD, ddof = 1).

Outputs:
  results/detection_metrics.csv  one row per (model, task, metric): mean, sd, seed_42..46
  results/per_class_recall.csv   one row per (model, task, class): mean, sd, seed_42..46
  results/confusion_<model>_multi.png  seven-class confusion matrices (nids.evaluation.plots)
  results/confusion_<model>_multi.csv  the same matrices as counts, summed over seeds

Usage:  python -m nids.evaluation.detection_metrics
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

from nids.prep.convert_xlsx import ROOT

MODEL_NAMES = {"iforest": "Isolation Forest", "rf": "Random Forest", "lstm": "LSTM"}
BINARY_METRICS = ("precision", "recall", "f1", "roc_auc", "fpr")
MULTI_METRICS = ("macro_f1", "accuracy")


def binary_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, score)),
            "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def multi_metrics(y7: np.ndarray, pred7: np.ndarray, n_classes: int) -> dict:
    labels = list(range(n_classes))
    return {"macro_f1": float(f1_score(y7, pred7, labels=labels, average="macro",
                                       zero_division=0)),
            "accuracy": float((y7 == pred7).mean()),
            "per_class_recall": recall_score(y7, pred7, labels=labels, average=None,
                                             zero_division=0).astype(float),
            "confusion": confusion_matrix(y7, pred7, labels=labels)}


def evaluate_file(df: pd.DataFrame, task: str, n_classes: int) -> dict:
    m = binary_metrics(df["true_bin"].to_numpy(), df["pred_bin"].to_numpy(),
                       df["score"].to_numpy())
    if task == "multi":
        m.update(multi_metrics(df["true_label_7"].to_numpy(), df["pred_code"].to_numpy(),
                               n_classes))
    return m


def mean_sd_row(values: dict[int, float]) -> dict:
    v = np.array(list(values.values()), dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "n_seeds": len(v), **{f"seed_{s}": x for s, x in values.items()}}


def run(results: Path = ROOT / "results") -> dict[str, pd.DataFrame]:
    from nids.evaluation.plots import plot_confusion

    offline = results / "offline"
    manifest = json.loads((offline / "manifest.json").read_text())
    names = json.loads((ROOT / "artefacts" / "label_map.json").read_text())["classes"]

    per_seed: dict[tuple[str, str], dict[int, dict]] = {}
    for f in manifest["files"]:
        df = pd.read_parquet(offline / f["file"])
        if len(df) != manifest["n_rows"]:
            raise ValueError(f"{f['file']}: expected {manifest['n_rows']} rows")
        per_seed.setdefault((f["model"], f["task"]), {})[f["seed"]] = \
            evaluate_file(df, f["task"], len(names))

    metric_rows, recall_rows = [], []
    for (model, task), seeds in per_seed.items():
        basis = "binary decision" if task == "binary" else "collapsed (pred != Normal)"
        keys = BINARY_METRICS + (MULTI_METRICS if task == "multi" else ())
        for k in keys:
            metric_rows.append({"model": model, "task": task, "metric": k,
                                "basis": basis if k in BINARY_METRICS else "seven-class",
                                **mean_sd_row({s: m[k] for s, m in seeds.items()})})
        if task == "multi":
            for i, cname in enumerate(names):
                recall_rows.append({"model": model, "task": task, "class": cname,
                                    **mean_sd_row({s: m["per_class_recall"][i]
                                                   for s, m in seeds.items()})})
            cm = sum(m["confusion"] for m in seeds.values())
            pd.DataFrame(cm, index=pd.Index(names, name="true"),
                         columns=pd.Index(names, name="predicted")) \
                .to_csv(results / f"confusion_{model}_multi.csv")
            plot_confusion(cm, names, len(seeds),
                           f"{MODEL_NAMES[model]}, seven-class (test, seeds "
                           f"{min(seeds)}–{max(seeds)})",
                           results / f"confusion_{model}_multi.png")

    metrics = pd.DataFrame(metric_rows)
    recall = pd.DataFrame(recall_rows)
    metrics.to_csv(results / "detection_metrics.csv", index=False, float_format="%.6f")
    recall.to_csv(results / "per_class_recall.csv", index=False, float_format="%.6f")
    return {"metrics": metrics, "per_class_recall": recall}


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Wide 'mean ± SD' table, one row per model/task, for the console and Chapter 4."""
    t = metrics.assign(v=[f"{m:.4f} ± {s:.4f}" for m, s in zip(metrics["mean"], metrics["sd"])])
    w = t.pivot(index=["model", "task"], columns="metric", values="v")
    cols = [c for c in BINARY_METRICS + MULTI_METRICS if c in w.columns]
    order = [(m, t) for m, t in (("iforest", "binary"), ("rf", "binary"), ("lstm", "binary"),
                                 ("rf", "multi"), ("lstm", "multi")) if (m, t) in w.index]
    return w.loc[order, cols].fillna("—")


def main() -> None:
    out = run()
    pd.set_option("display.width", 200)
    print(summary_table(out["metrics"]).to_string())
    r = out["per_class_recall"]
    if len(r):
        print()
        print(r.assign(v=[f"{m:.4f} ± {s:.4f}" for m, s in zip(r["mean"], r["sd"])])
              .pivot(index="class", columns="model", values="v").to_string())


if __name__ == "__main__":
    main()
