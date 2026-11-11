"""Tests for Phase 1 steps 9-12 (M1) and the online Preprocessor (M4).

The first three real-data tests are the "Done when" checks of plan §3.2/§3.3; they
run on the files written by `python -m nids.prep.prepare_data` and skip otherwise.
"""
import json

import numpy as np
import pandas as pd
import pytest

from nids.engine.preprocessor import Preprocessor
from nids.prep.convert_xlsx import ROOT
from nids.prep.prepare_data import (CLASSES, PARTITIONS, fit_preprocessing, split,
                                    transform_offline)

PROC = ROOT / "data" / "processed"
ART = ROOT / "artefacts"
REPORT = ROOT / "results" / "prep_report.json"
built = pytest.mark.skipif(
    not all((PROC / f"{k}_raw.parquet").exists() for k in PARTITIONS)
    or not (ART / "preprocessor.json").exists(),
    reason="run python -m nids.prep.prepare_data first")


# ------------------------------------------------------------- Done when (plan §3.2/§3.3)
@built
def test_done_report_class_counts_reasonable():
    rep = json.loads(REPORT.read_text())
    cc, rows = rep["split"]["class_counts"], rep["split"]["rows"]
    total = rep["cleaning"]["row_flow"]["final"]
    assert sum(rows.values()) == total
    for k, frac in rep["split"]["fractions"].items():
        assert abs(rows[k] / total - frac) < 0.001
        assert sum(cc[k]["seven_class"].values()) == rows[k]
    for c in CLASSES:  # stratified: each class keeps its share in every partition
        share = rep["cleaning"]["labels"]["class_counts_clean"]["seven_class"][c] / total
        for k in PARTITIONS:
            assert abs(cc[k]["seven_class"][c] / rows[k] - share) < 0.01


@built
def test_done_partitions_disjoint_and_all_classes():
    parts = {k: pd.read_parquet(PROC / f"{k}_raw.parquet") for k in PARTITIONS}
    ids = [set(p["row_index"]) for p in parts.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    for k, p in parts.items():
        assert set(p["class_name"]) == set(CLASSES), k
        assert p["row_index"].is_monotonic_increasing, k  # step 12
        assert len(np.load(PROC / f"X_{k}.npy")) == len(p)
        assert (np.load(PROC / f"y_{k}_7.npy") == p["label_7"].to_numpy()).all()


@built
def test_done_preprocessor_matches_X_test():
    pre = Preprocessor(ART / "preprocessor.json")
    assert pre.features == json.loads((ART / "feature_list.json").read_text())
    raw = pd.read_parquet(PROC / "test_raw.parquet")
    X = np.load(PROC / "X_test.npy")
    assert X.dtype == np.float32 and X.shape == (len(raw), len(pre.features))
    idx = np.random.default_rng(42).choice(len(raw), size=min(1000, len(raw)), replace=False)
    vals = raw[pre.features].to_numpy(np.float64)
    for i in idx:
        assert np.allclose(pre.transform(vals[i]), X[i]), int(raw["row_index"].iat[i])


# ------------------------------------------------------------- synthetic unit tests
def _train_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return pd.DataFrame({
        "const": 5.0,
        "a": a,
        "a_copy": 2 * a + 1,                      # |r| = 1 with a -> dropped
        "a_copy2": -a,                            # |r| = 1 with a -> dropped
        "skewed": rng.exponential(size=n) ** 3,   # min >= 0, heavy right tail -> log
        "neg_skewed": rng.exponential(size=n) ** 3 - 1,  # skewed but negative -> no log
        "plain": rng.normal(size=n),
    })


def test_fit_rules():
    tr = _train_frame()
    feats = list(tr.columns)
    state, rep = fit_preprocessing(tr, feats, 0.98, 1.0)
    assert rep["constant_columns_dropped"] == ["const"]
    assert rep["correlation_dropped"] == {"a_copy": "a", "a_copy2": "a"}
    assert state["features"] == ["a", "skewed", "neg_skewed", "plain"]
    assert rep["log_columns"] == ["skewed"]
    assert rep["skewed_but_negative"] == ["neg_skewed"]
    X = transform_offline(tr, state)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-5) and np.allclose(X.std(axis=0), 1, atol=1e-4)


def test_log_exclude_keeps_column_standardised_only():
    tr = _train_frame()
    state, rep = fit_preprocessing(tr, list(tr.columns), 0.98, 1.0, log_exclude=["skewed"])
    assert rep["log_columns"] == [] and rep["log_excluded_categorical"] == ["skewed"]
    assert "skewed" in state["features"]


@built
def test_protocol_not_logged_in_artefact():
    st = json.loads((ART / "preprocessor.json").read_text())
    assert "Protocol" in st["features"]
    assert st["log_mask"][st["features"].index("Protocol")] is False


def test_correlation_chain_keeps_column_only_linked_via_dropped_one():
    rng = np.random.default_rng(1)
    n = 20_000
    a = rng.normal(size=n)
    b = a + 0.16 * rng.normal(size=n)             # r(a,b) ~ 0.988
    c = b + 0.16 * rng.normal(size=n)             # r(b,c) ~ 0.988, r(a,c) ~ 0.975
    tr = pd.DataFrame({"a": a, "b": b, "c": c})
    state, rep = fit_preprocessing(tr, ["a", "b", "c"], 0.98, 1.0)
    assert rep["correlation_dropped"] == {"b": "a"}
    assert state["features"] == ["a", "c"]


def test_state_depends_on_train_only():
    tr = _train_frame()
    s1, _ = fit_preprocessing(tr, list(tr.columns), 0.98, 1.0)
    s2, _ = fit_preprocessing(tr.copy(), list(tr.columns), 0.98, 1.0)
    assert s1 == s2  # only the frame passed in is used: the function never sees val/test


def test_split_stratified_disjoint():
    df = pd.DataFrame({"row_index": range(1000),
                       "label_7": [0] * 600 + [1] * 300 + [2] * 100})
    p = split(df, 0.70, 0.15, 0.15, 42)
    assert [len(p[k]) for k in PARTITIONS] == [700, 150, 150]
    assert sum(len(set(p[k]["row_index"])) for k in PARTITIONS) == 1000
    assert (p["test"]["label_7"].value_counts().sort_index() == [90, 45, 15]).all()
    q = split(df, 0.70, 0.15, 0.15, 42)
    assert p["test"]["row_index"].tolist() == q["test"]["row_index"].tolist()


def test_preprocessor_rejects_malformed(tmp_path):
    p = tmp_path / "pre.json"
    p.write_text(json.dumps({"features": ["x", "y"], "median": [1.0, 2.0],
                             "log_mask": [True, False], "mean": [0.0, 0.0],
                             "scale": [1.0, 2.0]}))
    pre = Preprocessor(p)
    assert np.allclose(pre.transform([np.e - 1, 4.0]), [1.0, 2.0])
    assert np.allclose(pre.transform([float("inf"), float("nan")]), [np.log1p(1.0), 1.0])
    for bad in ([1.0], [1.0, 2.0, 3.0], ["a", 1.0], [[1.0, 2.0]]):
        with pytest.raises(ValueError):
            pre.transform(bad)
    v = np.array([0.5, 1.0])
    pre.transform(v)
    assert v.tolist() == [0.5, 1.0]  # caller's array untouched
