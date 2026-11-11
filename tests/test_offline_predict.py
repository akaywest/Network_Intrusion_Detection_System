"""Tests for offline predictions (M2, plan §4.4) and detection metrics (M8, plan §8.1).

Synthetic tests run always; the artefact checks skip until
`python -m nids.train.offline_predict` has been run.
"""
import json

import numpy as np
import pandas as pd
import pytest

from nids.evaluation import detection_metrics as dm
from nids.prep.convert_xlsx import ROOT
from nids.train import offline_predict as op

OFF = ROOT / "results" / "offline"
SEEDS = [42, 43, 44, 45, 46]


def test_iforest_confidence_rule():
    c = op.iforest_confidence(np.array([0.3, 0.5, 0.6, 0.7, 0.9]), tau=0.5, score_max=0.7)
    assert np.allclose(c, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_artefact_path_refuses_escape(tmp_path):
    with pytest.raises(PermissionError):
        op.artefact_path(tmp_path, "../outside.joblib")


def test_binary_frame_applies_saved_threshold():
    raw = pd.DataFrame({"row_index": [1, 2, 3], "class_name": ["Normal", "PortScan", "Normal"],
                        "label_7": [0, 5, 0], "label_bin": [0, 1, 0]})
    df = op.binary_frame(raw, np.array([0.2, 0.6, 0.5]), 0.5, np.array([.8, .6, .5]))
    assert df["pred_bin"].tolist() == [0, 1, 1]
    assert df["predicted_label"].tolist() == ["Normal", "Malicious", "Malicious"]


def test_multi_frame_argmax_score_confidence():
    raw = pd.DataFrame({"row_index": [1, 2], "class_name": ["Normal", "Botnet"],
                        "label_7": [0, 1], "label_bin": [0, 1]})
    p = np.array([[.7, .2, .1], [.1, .3, .6]])
    df = op.multi_frame(raw, p, [0, 1, 2], ["Normal", "Botnet", "Brute Force"])
    assert df["pred_code"].tolist() == [0, 2] and df["pred_bin"].tolist() == [0, 1]
    assert np.allclose(df["score"], [.3, .9]) and np.allclose(df["confidence"], [.7, .6])


def test_binary_metrics_known_values():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    m = dm.binary_metrics(y, pred, np.array([.1, .2, .3, .6, .7, .8, .9, .4]))
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75)
    assert m["fpr"] == pytest.approx(0.25) and m["roc_auc"] == pytest.approx(0.9375)


def test_mean_sd_uses_sample_sd():
    r = dm.mean_sd_row({42: 1.0, 43: 3.0})
    assert r["mean"] == 2.0 and r["sd"] == pytest.approx(np.sqrt(2))


@pytest.mark.skipif(not (OFF / "manifest.json").exists(), reason="offline_predict not run")
def test_offline_files_complete_and_consistent():
    man = json.loads((OFF / "manifest.json").read_text())
    assert len(man["files"]) == 25
    test = pd.read_parquet(ROOT / "data/processed/test_raw.parquet", columns=["row_index"])
    for f in man["files"]:
        df = pd.read_parquet(OFF / f["file"])
        assert df["row_index"].tolist() == test["row_index"].tolist()
        assert df["confidence"].between(0, 1).all() and df["score"].notna().all()
        if f["task"] == "binary":  # saved threshold applied, never re-tuned
            assert ((df["score"] >= f["tau"]).astype(int) == df["pred_bin"]).all()
        else:
            assert ((df["pred_code"] != 0).astype(int) == df["pred_bin"]).all()
