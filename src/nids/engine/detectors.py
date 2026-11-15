"""Detector interface (M4 inference engine, plan §6.1; FR7, FR9, NFR6, NFR7, §3.3.10).

Strategy pattern: Isolation Forest, Random Forest and LSTM sit behind one Detector
interface, so the engine (§6.3) calls `detector.predict(x)` for every record without
knowing which model is active. A new model needs one Detector subclass plus one
`@register("name")` line (NFR7).

Decision rules are identical to nids.train.offline_predict, so streamed labels can be
checked against the offline reference row by row (§8.4, §3.3.11, test IT3):
  IF            s = -score_samples(x) (plan §1.1 item 4); Malicious iff s >= tau;
                confidence = clip((s - tau) / (val_score_max - tau), 0, 1) (plan §5.2)
  RF/LSTM bin   p = P(Malicious); Malicious iff p >= tau; confidence = p or 1 - p
  RF/LSTM multi argmax of the seven class probabilities; score = 1 - P(Normal);
                confidence = probability of the predicted class
Severity: High when confidence >= alerts.high_confidence (0.90), otherwise Medium
(FR10, §3.3.9, plan §5.2); None for benign records.

The LSTM uses formulation B (plan §4.3): input (F, 1) per flow, no record buffer, so no
warm-up records and reset_state() is a no-op. Inference goes through a tf.function with
a fixed input signature, traced once in load(); model.predict is never called per record.

Single-thread inference (§3.3.10): the thread environment variables are set at the top
of this module, before NumPy, scikit-learn or TensorFlow are imported. engine.py must
therefore import this module before anything that imports NumPy. TF intra/inter-op
threads are set to 1 when the LSTM loads, and the sklearn models use n_jobs=1.

Models load only from artefacts/ (NFR6): Detector._path refuses any other location.

Timing:  python -m nids.engine.detectors --time   (writes results/detector_timing.json)
"""
from __future__ import annotations

import os

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
for _k in THREAD_ENV:  # must run before numpy / TF are imported (§3.3.10)
    os.environ.setdefault(_k, "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json  # noqa: E402
import warnings  # noqa: E402
from abc import ABC, abstractmethod  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from nids.storage.logger import HIGH_CONFIDENCE, severity_for  # noqa: E402

BIN_NAMES = ("Normal", "Malicious")


@dataclass
class Result:
    """One prediction, as logged by nids.storage.logger.LatencyLogger (FR9, FR11)."""
    is_malicious: int
    label: str
    score: float
    confidence: float
    severity: str | None
    warmup: bool = False


def configure_tf_threads() -> tuple[int, int]:
    """Limit TensorFlow to one intra-op and one inter-op thread (§3.3.10)."""
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:  # TF runtime already initialised in this process
        pass
    got = (tf.config.threading.get_intra_op_parallelism_threads(),
           tf.config.threading.get_inter_op_parallelism_threads())
    if got != (1, 1):
        warnings.warn(f"TensorFlow threads are {got}, not (1, 1): import "
                      "nids.engine.detectors before TensorFlow is initialised", stacklevel=2)
    return got


# ---------------------------------------------------------------- registry (NFR7)
DETECTORS: dict[str, type["Detector"]] = {}


def register(name: str):
    """Class decorator: make a Detector selectable by name (model selector, MODEL_RUN)."""
    def deco(cls):
        if name in DETECTORS:
            raise ValueError(f"detector {name!r} already registered")
        cls.name = name
        DETECTORS[name] = cls
        return cls
    return deco


def make_detector(name: str, artefact_dir: str | Path, **kw) -> "Detector":
    try:
        cls = DETECTORS[name]
    except KeyError:
        raise ValueError(f"unknown detector {name!r}; choose from {sorted(DETECTORS)}") from None
    return cls(artefact_dir, **kw)


# ---------------------------------------------------------------- interface
class Detector(ABC):
    name: str = ""
    tasks: tuple[str, ...] = ("binary",)

    def __init__(self, artefact_dir: str | Path, high_confidence: float = HIGH_CONFIDENCE):
        self.dir = Path(artefact_dir).resolve()
        self.high_confidence = float(high_confidence)
        self.task: str | None = None
        self.seed: int | None = None

    def _path(self, name: str) -> Path:
        """NFR6: models load only from artefacts/; anything resolving elsewhere is refused."""
        p = (self.dir / name).resolve()
        if self.dir not in p.parents:
            raise PermissionError(f"refusing to load outside {self.dir}: {p}")
        return p

    def _check_task(self, seed: int, task: str) -> None:
        if task not in self.tasks:
            raise ValueError(f"{self.name} supports tasks {self.tasks}, not {task!r}")
        self.seed, self.task = int(seed), task

    @abstractmethod
    def load(self, seed: int, task: str = "binary") -> None: ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> Result: ...

    def reset_state(self) -> None:
        """Called at the start of every run; stateless detectors need nothing."""

    # shared decision rules (same as offline_predict.binary_frame / multi_frame)
    def _binary_result(self, score: float, tau: float, confidence: float) -> Result:
        mal = int(score >= tau)
        return Result(mal, BIN_NAMES[mal], float(score), float(confidence),
                      severity_for(confidence, self.high_confidence) if mal else None)

    def _multi_result(self, proba: np.ndarray, classes: np.ndarray,
                      names: list[str]) -> Result:
        idx = int(np.argmax(proba))
        code = int(classes[idx])
        conf = float(proba[idx])
        normal_col = int(np.flatnonzero(classes == 0)[0])
        mal = int(code != 0)
        return Result(mal, names[code], float(1.0 - proba[normal_col]), conf,
                      severity_for(conf, self.high_confidence) if mal else None)

    def _class_names(self) -> list[str]:
        return json.loads(self._path("label_map.json").read_text())["classes"]


def iforest_confidence(s: float, tau: float, score_max: float) -> float:
    """Plan §5.2: position of s between tau and the highest validation score, in [0, 1]."""
    span = score_max - tau
    if span <= 0:
        return float(s >= tau)
    return float(min(max((s - tau) / span, 0.0), 1.0))


# ---------------------------------------------------------------- detectors
@register("iforest")
class IsolationForestDetector(Detector):
    tasks = ("binary",)

    def load(self, seed, task="binary"):
        self._check_task(seed, task)
        b = joblib.load(self._path(f"iforest_seed{seed}.joblib"))
        self.model, self.tau = b["model"], float(b["tau"])
        self.score_max = float(b["val_score_max"])
        self.config = b["config"]
        self.model.set_params(n_jobs=1)  # saved with n_jobs=-1 (§3.3.10)

    def predict(self, x):
        s = float(-self.model.score_samples(x.reshape(1, -1))[0])  # paper's s(x, psi)
        return self._binary_result(s, self.tau, iforest_confidence(s, self.tau, self.score_max))


@register("rf")
class RandomForestDetector(Detector):
    tasks = ("binary", "multi")

    def load(self, seed, task="binary"):
        self._check_task(seed, task)
        b = joblib.load(self._path(f"rf_{task}_seed{seed}.joblib"))
        self.model, self.config = b["model"], b["config"]
        self.tau = None if b["tau"] is None else float(b["tau"])
        self.model.set_params(n_jobs=1)
        if task == "binary":
            self.col = list(self.model.classes_).index(1)  # P(Malicious) column
        else:
            self.classes = np.asarray(b["classes"])
            self.names = self._class_names()

    def predict(self, x):
        proba = self.model.predict_proba(x.reshape(1, -1))[0]
        if self.task == "binary":
            p = float(proba[self.col])
            return self._binary_result(p, self.tau, p if p >= self.tau else 1.0 - p)
        return self._multi_result(proba, self.classes, self.names)


@register("lstm")
class LSTMDetector(Detector):
    tasks = ("binary", "multi")

    def load(self, seed, task="binary"):
        self._check_task(seed, task)
        meta = json.loads(self._path(f"lstm_{task}_seed{seed}.meta.json").read_text())
        if meta["formulation"] != "B":
            raise ValueError("LSTMDetector implements formulation B only (plan §4.3)")
        self.threads = configure_tf_threads()
        import keras
        import tensorflow as tf
        model = keras.models.load_model(self._path(f"lstm_{task}_seed{seed}.keras"))
        F = int(meta["features"])
        if tuple(model.input_shape[1:]) != (F, 1):
            raise ValueError(f"model input {model.input_shape} != (None, {F}, 1)")
        self.model, self.F, self.config = model, F, meta["config"]
        self.tau = meta["tau"]
        self.classes = np.asarray(meta["classes"])
        self.names = self._class_names() if task == "multi" else list(BIN_NAMES)
        self._infer = tf.function(lambda t: model(t, training=False),
                                  input_signature=[tf.TensorSpec((1, F, 1), tf.float32)])
        self._infer(tf.zeros((1, F, 1)))  # trace once, outside any timed path

    def predict(self, x):
        out = self._infer(np.asarray(x, dtype=np.float32).reshape(1, self.F, 1)).numpy()[0]
        if self.task == "binary":
            p = float(out[0])
            return self._binary_result(p, self.tau, p if p >= self.tau else 1.0 - p)
        return self._multi_result(out.astype(np.float64), self.classes, self.names)


# ---------------------------------------------------------------- timing CLI
TIMED = (("iforest", "binary"), ("rf", "binary"), ("lstm", "binary"))


def time_detectors(n: int = 1000, warmup: int = 50, seed: int = 42) -> dict:
    """Time n single predict() calls per detector on preprocessed test rows (one thread)."""
    import time

    import pandas as pd
    from threadpoolctl import threadpool_info

    from nids.engine.preprocessor import Preprocessor
    from nids.prep.convert_xlsx import ROOT

    art = ROOT / "artefacts"
    pre = Preprocessor(art / "preprocessor.json")
    raw = pd.read_parquet(ROOT / "data" / "processed" / "test_raw.parquet")
    rows = raw.sample(n=n + warmup, random_state=seed)[pre.features].to_numpy()
    X = np.stack([pre.transform(r) for r in rows])

    report = {"n_calls": n, "warmup_calls": warmup, "seed": seed,
              "source": "test partition rows (timing only, nothing tuned)",
              "env": {k: os.environ.get(k) for k in THREAD_ENV},
              "blas_threads": sorted({d["num_threads"] for d in threadpool_info()}),
              "detectors": {}}
    for name, task in TIMED:
        d = make_detector(name, art)
        d.load(seed, task)
        for i in range(warmup):
            d.predict(X[i])
        t = np.empty(n)
        for i in range(n):
            s = time.perf_counter_ns()
            d.predict(X[warmup + i])
            t[i] = (time.perf_counter_ns() - s) / 1e6
        r = {"task": task, "median_ms": float(np.median(t)),
             "p95_ms": float(np.percentile(t, 95)), "p99_ms": float(np.percentile(t, 99)),
             "mean_ms": float(t.mean()), "max_ms": float(t.max())}
        if name == "lstm":
            r["tf_threads"] = list(d.threads)
        report["detectors"][name] = r
    out = ROOT / "results" / "detector_timing.json"
    out.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Detector single-record timing (plan §6.1)")
    ap.add_argument("--time", action="store_true", help="time 1,000 predict() calls each")
    ap.add_argument("-n", type=int, default=1000)
    a = ap.parse_args()
    if not a.time:
        ap.print_help()
        return
    rep = time_detectors(n=a.n)
    print(f"{'detector':10s} {'task':7s} {'median ms':>10s} {'p95 ms':>8s} {'p99 ms':>8s}")
    for name, r in rep["detectors"].items():
        print(f"{name:10s} {r['task']:7s} {r['median_ms']:10.3f} {r['p95_ms']:8.3f} "
              f"{r['p99_ms']:8.3f}")
    print(json.dumps({"env": rep["env"], "blas_threads": rep["blas_threads"]}))


if __name__ == "__main__":
    main()
