"""Tests for the Detector interface (M4, plan §6.1; tests UT3, UT4, part of IT3).

Artefact tests skip until the models and results/offline/ exist. The agreement test
drives the real engine path: raw test row -> Preprocessor -> Detector.predict, and
requires exact label agreement with the offline reference (§8.4, §3.3.11).
"""
import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from nids.engine import detectors as det
from nids.engine.preprocessor import Preprocessor
from nids.prep.convert_xlsx import ROOT

ART = ROOT / "artefacts"
OFF = ROOT / "results" / "offline"
TEST_RAW = ROOT / "data" / "processed" / "test_raw.parquet"
SEED = 42
N_AGREE = 500
CASES = [("iforest", "binary"), ("rf", "binary"), ("rf", "multi"),
         ("lstm", "binary"), ("lstm", "multi")]


def _have(model, task):
    art = (f"iforest_seed{SEED}.joblib" if model == "iforest"
           else f"{model}_{task}_seed{SEED}.{'keras' if model == 'lstm' else 'joblib'}")
    return ((ART / art).exists() and (OFF / f"{model}_{task}_seed{SEED}.parquet").exists()
            and TEST_RAW.exists())


# ---------------------------------------------------------------- UT4 (NFR6)
def test_path_refuses_parent_escape(tmp_path):
    d = det.IsolationForestDetector(tmp_path / "artefacts")
    with pytest.raises(PermissionError):
        d._path("../outside.joblib")
    with pytest.raises(PermissionError):
        d._path("../../etc/passwd")


def test_path_refuses_absolute_outside(tmp_path):
    d = det.RandomForestDetector(tmp_path / "artefacts")
    with pytest.raises(PermissionError):
        d._path(str(tmp_path / "evil.joblib"))


def test_path_accepts_inside(tmp_path):
    d = det.LSTMDetector(tmp_path)
    assert d._path("lstm_binary_seed42.keras") == (tmp_path / "lstm_binary_seed42.keras").resolve()


# ---------------------------------------------------------------- registry (NFR7)
def test_registry():
    assert {"iforest", "rf", "lstm"} <= set(det.DETECTORS)
    assert isinstance(det.make_detector("rf", ART), det.RandomForestDetector)
    with pytest.raises(ValueError):
        det.make_detector("bogus", ART)


def test_iforest_refuses_multi():
    with pytest.raises(ValueError):
        det.make_detector("iforest", ART).load(SEED, "multi")


# ---------------------------------------------------------------- decision rules
def test_binary_rule_and_severity():
    d = det.RandomForestDetector(ART)
    hi, med, ben = (d._binary_result(.95, .5, .95), d._binary_result(.6, .5, .6),
                    d._binary_result(.2, .5, .8))
    assert (hi.is_malicious, hi.label, hi.severity) == (1, "Malicious", "High")
    assert (med.is_malicious, med.severity) == (1, "Medium")
    assert (ben.is_malicious, ben.label, ben.severity) == (0, "Normal", None)


def test_multi_rule():
    d = det.RandomForestDetector(ART)
    names = ["Normal", "Botnet", "Brute Force"]
    r = d._multi_result(np.array([.1, .3, .6]), np.array([0, 1, 2]), names)
    assert (r.label, r.is_malicious, r.severity) == ("Brute Force", 1, "Medium")
    assert r.score == pytest.approx(.9) and r.confidence == pytest.approx(.6)
    n = d._multi_result(np.array([.7, .2, .1]), np.array([0, 1, 2]), names)
    assert (n.label, n.is_malicious, n.severity) == ("Normal", 0, None)


def test_iforest_confidence_rule():
    c = [det.iforest_confidence(s, .5, .7) for s in (.3, .5, .6, .7, .9)]
    assert np.allclose(c, [0, 0, .5, 1, 1])


# ---------------------------------------------------------------- UT3 + agreement
@pytest.fixture(scope="module")
def sample():
    pre = Preprocessor(ART / "preprocessor.json")
    raw = pd.read_parquet(TEST_RAW).sample(n=N_AGREE, random_state=SEED)
    X = np.stack([pre.transform(r) for r in raw[pre.features].to_numpy()])
    return raw["row_index"].to_numpy(), X


@pytest.mark.parametrize("model,task", CASES)
def test_detector_matches_offline(model, task, request):
    if not _have(model, task):
        pytest.skip("artefacts or offline predictions missing")
    rows, X = request.getfixturevalue("sample")
    d = det.make_detector(model, ART)
    d.load(SEED, task)
    d.reset_state()
    res = [d.predict(x) for x in X]

    valid = set(det.BIN_NAMES) if task == "binary" else set(
        json.loads((ART / "label_map.json").read_text())["classes"])
    for r in res:  # UT3: valid Result
        assert isinstance(r, det.Result) and r.warmup is False
        assert r.is_malicious in (0, 1) and r.label in valid
        assert math.isfinite(r.score) and 0.0 <= r.confidence <= 1.0
        assert (r.severity in ("High", "Medium")) == bool(r.is_malicious)
        assert r.is_malicious == int(r.label != "Normal")

    off = pd.read_parquet(OFF / f"{model}_{task}_seed{SEED}.parquet").set_index("row_index")
    ref = off.loc[rows]
    got = np.array([r.label for r in res])
    mismatch = int((got != ref["predicted_label"].to_numpy()).sum())
    assert mismatch == 0, f"{mismatch}/{N_AGREE} labels differ from offline reference"
    assert np.allclose([r.score for r in res], ref["score"].to_numpy(), atol=1e-5)
    assert np.allclose([r.confidence for r in res], ref["confidence"].to_numpy(), atol=1e-5)


# ---------------------------------------------------------------- timing (fresh process)
@pytest.mark.skipif(not all(_have(m, t) for m, t in det.TIMED), reason="artefacts missing")
def test_single_record_timing(tmp_path):
    out = tmp_path / "detector_timing.json"  # never overwrite results/ from a test
    r = subprocess.run([sys.executable, "-m", "nids.engine.detectors", "--time",
                        "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    print(r.stdout)
    rep = json.loads(out.read_text())
    assert rep["n_calls"] == 1000 and set(rep["detectors"]) == {"iforest", "rf", "lstm"}
    assert all(v == "1" for v in rep["env"].values())
    assert rep["detectors"]["lstm"]["tf_threads"] == [1, 1]
    for v in rep["detectors"].values():
        assert 0 < v["median_ms"] <= v["p95_ms"]
