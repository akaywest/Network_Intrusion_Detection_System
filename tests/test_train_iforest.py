"""Tests for Isolation Forest training (M2, plan §4.1).

Synthetic tests run always; the last one checks the real artefacts once
`python -m nids.train.train_iforest` has been run, and skips otherwise.
"""
import json

import joblib
import numpy as np
import pytest

from nids.prep.convert_xlsx import ROOT
from nids.train import train_iforest as ti

ART = ROOT / "artefacts"
SEEDS = [42, 43, 44, 45, 46]


def test_grid_is_36_combinations_within_table_3_11():
    g = ti.grid()
    assert len(g) == 36 and len({tuple(sorted(p.items())) for p in g}) == 36
    for p in g:
        assert 100 <= p["n_estimators"] <= 500
        assert 256 <= p["max_samples"] <= 1024
        assert 0.5 <= p["max_features"] <= 1.0


def test_best_f1_threshold_known_example():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.6, 0.5, 0.7, 0.9])
    best = ti.best_f1_threshold(y, s)
    # s >= 0.5 -> TP=3, FP=1 -> P=0.75, R=1, F1=6/7 (best over all thresholds)
    assert best["tau"] == pytest.approx(0.5)
    assert best["f1"] == pytest.approx(6 / 7)


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, (600, 5)).astype(np.float32)
    val = np.vstack([rng.normal(0, 1, (100, 5)), rng.normal(6, 1, (100, 5))]).astype(np.float32)
    return normal, val, np.r_[np.zeros(100, int), np.ones(100, int)]


def test_score_sign_outliers_score_higher():
    normal, X_val, y_val = _toy()
    m = ti.fit_iforest(normal, {"n_estimators": 50, "max_samples": 256, "max_features": 1.0}, 42)
    s = ti.anomaly_score(m, X_val)
    assert s[y_val == 1].mean() > s[y_val == 0].mean()


def test_train_seeds_bundle_and_reload(tmp_path):
    normal, X_val, y_val = _toy()
    params = {"n_estimators": 50, "max_samples": 256, "max_features": 1.0}
    log = ti.train_seeds(normal, X_val, y_val, params, [42, 43], tmp_path)
    assert [r["seed"] for r in log] == [42, 43]
    for seed in (42, 43):
        b = joblib.load(tmp_path / f"iforest_seed{seed}.joblib")
        assert {"model", "tau", "config", "seed", "val_score_max"} <= set(b)
        assert b["config"] == params and b["seed"] == seed
        s = ti.anomaly_score(b["model"], X_val)
        assert s.max() == pytest.approx(b["val_score_max"])
        assert ((s >= b["tau"]) == y_val).mean() > 0.95


def test_load_partitions_never_reads_test(tmp_path, monkeypatch):
    for name in ti.PARTITION_FILES:
        np.save(tmp_path / name, np.zeros(3))
    opened = []
    real_load = np.load
    monkeypatch.setattr(ti.np, "load", lambda p, *a, **k: opened.append(str(p)) or real_load(p))
    ti.load_partitions(tmp_path)
    assert opened and not any("test" in p.rsplit("/", 1)[-1] for p in opened)


@pytest.mark.skipif(not all((ART / f"iforest_seed{s}.joblib").exists() for s in SEEDS),
                    reason="run python -m nids.train.train_iforest first")
def test_done_five_bundles_and_config():
    cfg = json.loads((ART / "iforest_config.json").read_text())
    assert cfg["smoke"] is False and cfg["seeds"] == SEEDS
    log = json.loads((ROOT / "results" / "train_log" / "iforest.json").read_text())
    assert log["test_partition_used"] is False and len(log["grid"]) == 36
    for s in SEEDS:
        b = joblib.load(ART / f"iforest_seed{s}.joblib")
        assert b["config"] == cfg["hyperparameters"] and b["seed"] == s
