"""Tests for Random Forest training (M2, plan §4.2).

Synthetic tests run always; the last one checks the real artefacts once
`python -m nids.train.train_rf` has been run, and skips otherwise.
"""
import json

import joblib
import numpy as np
import pytest

from nids.prep.convert_xlsx import ROOT
from nids.train import train_rf as tr

ART = ROOT / "artefacts"
SEEDS = [42, 43, 44, 45, 46]
NAMES = ["Normal", "Botnet", "Brute Force", "DoS/DDoS", "Infiltration", "PortScan", "Web Attack"]


def test_search_space_matches_table_3_11():
    sp = tr.search_space()
    ne = sp["n_estimators"].rvs(2000, random_state=1)
    assert ne.min() >= 100 and ne.max() <= 500
    mss = sp["min_samples_split"].rvs(2000, random_state=1)
    assert mss.min() == 2 and mss.max() == 10
    msl = sp["min_samples_leaf"].rvs(2000, random_state=1)
    assert msl.min() == 1 and msl.max() == 4
    assert sp["max_depth"] == [10, 20, 30, None]
    assert sp["max_features"] == ["sqrt", "log2"]
    assert sp["class_weight"] == ["balanced", "balanced_subsample"]


def test_threshold_rule_prefers_0_5_unless_tuned_is_better():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert tr.choose_threshold_rule(y, np.array([.1, .2, .3, .7, .8, .9]))["rule"] == "0.5"
    d = tr.choose_threshold_rule(y, np.array([.1, .2, .3, .35, .4, .45]))
    assert d["rule"] == "tuned" and d["tuned_tau"] == pytest.approx(0.35)
    assert d["f1_gain"] == pytest.approx(1.0) and d["min_gain"] == 0.005


def test_threshold_rule_ignores_gain_below_0_005():
    # 200 Malicious, one scored 0.45; 200 Normal, one scored 0.42.
    # @0.5: TP=199, FN=1, FP=0 -> F1=398/399; tuned @0.45 -> F1=1; gain ~0.0025 < 0.005
    y = np.r_[np.ones(200, int), np.zeros(200, int)]
    p = np.r_[np.full(199, 0.9), 0.45, np.full(199, 0.1), 0.42]
    d = tr.choose_threshold_rule(y, p)
    assert d["f1_gain"] == pytest.approx(1 - 398 / 399)
    assert d["rule"] == "0.5"
    assert tr.choose_threshold_rule(y, p, min_gain=0.002)["rule"] == "tuned"


def test_stratified_fraction_keeps_every_class():
    y7 = np.repeat(np.arange(7), [500, 50, 100, 400, 10, 20, 40])
    idx = tr.stratified_fraction(y7, 0.5)
    assert set(np.unique(y7[idx])) == set(range(7)) and np.all(np.diff(idx) > 0)
    assert len(tr.stratified_fraction(y7, 1.0)) == len(y7)


def _toy(n=700, seed=0):
    rng = np.random.default_rng(seed)
    y7 = np.repeat(np.arange(7), n // 7)
    X = (rng.normal(0, 1, (len(y7), 6)) + y7[:, None] * 2.0).astype(np.float32)
    return X, y7


def test_train_task_both_end_to_end(tmp_path):
    X, y7 = _toy()
    Xv, yv7 = _toy(seed=1)
    d = {"X_train": X, "y_train_7": y7, "y_train_bin": (y7 != 0).astype(int),
         "X_val": Xv, "y_val_7": yv7, "y_val_bin": (yv7 != 0).astype(int)}
    idx = np.arange(len(y7))
    for task in ("binary", "multi"):
        config, log = tr.train_task(task, d, idx, [42, 43], tmp_path, NAMES, smoke=True,
                                    search_frac=1.0)
        assert log["test_partition_used"] is False and len(log["per_seed_validation"]) == 2
        assert log["single_row_timing"]["n_calls"] == tr.SMOKE_N_TIMING
        for seed in (42, 43):
            b = joblib.load(tmp_path / f"rf_{task}_seed{seed}.joblib")
            assert b["task"] == task and b["seed"] == seed and b["config"] == config["hyperparameters"]
            assert b["model"].n_jobs == 1
            assert b["model"].get_params()["random_state"] == seed
        if task == "binary":
            assert log["threshold_choice"]["rule"] in ("0.5", "tuned")
            assert log["validation_mean_sd"]["f1"]["mean"] > 0.9
        else:
            assert b["tau"] is None and b["classes"] == list(range(7))
            assert log["validation_mean_sd"]["macro_f1"]["mean"] > 0.5
        assert (tmp_path / f"rf_{task}_config.json").exists()


def test_load_partitions_never_reads_test(tmp_path, monkeypatch):
    for name in tr.PARTITION_FILES:
        np.save(tmp_path / name, np.zeros(3))
    opened = []
    real_load = np.load
    monkeypatch.setattr(tr.np, "load", lambda p, *a, **k: opened.append(str(p)) or real_load(p))
    tr.load_partitions(tmp_path)
    assert len(opened) == 6 and not any("test" in p.rsplit("/", 1)[-1] for p in opened)


@pytest.mark.skipif(not all((ART / f"rf_{t}_seed{s}.joblib").exists()
                            for t in ("binary", "multi") for s in SEEDS),
                    reason="run python -m nids.train.train_rf first")
def test_done_ten_bundles_and_configs():
    for task in ("binary", "multi"):
        cfg = json.loads((ART / f"rf_{task}_config.json").read_text())
        assert cfg["smoke"] is False and cfg["seeds"] == SEEDS and cfg["search"]["n_iter"] == 30
        log = json.loads((ROOT / "results" / "train_log" / f"rf_{task}.json").read_text())
        assert log["test_partition_used"] is False and len(log["search_trials"]) == 30
        for s in SEEDS:
            b = joblib.load(ART / f"rf_{task}_seed{s}.joblib")
            assert b["config"] == cfg["hyperparameters"] and b["seed"] == s
