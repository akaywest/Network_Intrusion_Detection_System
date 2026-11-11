"""Tests for Phase 1 steps 2-8 (M1, nids.prep.prepare_data)."""
import numpy as np
import pandas as pd
import pytest

from nids.prep.convert_xlsx import LABEL_COL, ROOT
from nids.prep.prepare_data import (CLASSES, assign_roles, clean, drop_duplicates,
                                    handle_missing, label_map, make_labels, replace_inf)


def frame(rows):
    """rows: list of (src_port, flow_bytes_text, protocol, label)."""
    df = pd.DataFrame(rows, columns=["Source Port", "Flow Bytess", "Protocol", LABEL_COL])
    df.insert(0, "row_index", range(len(df)))
    df.insert(2, "Destination Port", 80)
    df["Flow Duration"] = 10
    df["Total Fwd Packets"] = 2
    df["Flow Bytess"] = df["Flow Bytess"].astype("string")
    df[LABEL_COL] = df[LABEL_COL].astype("string")
    return df


@pytest.fixture
def raw():
    return frame([
        (1, "1.5", 6, "Normal"),
        (2, "1.5", 6, "Normal"),          # exact duplicate once ports are excluded
        (3, "Infinity", 6, "Normal"),
        (4, "NaN", 17, "PortScan"),
        (5, "2.0", 17, "PortScan"),
        (6, "2.0", 17, "Dos/DDos"),       # conflicts with the row above
        (7, "3.0", 6, "Botnet ARES"),
    ])


def test_roles_exclude_ports(raw):
    r = assign_roles(raw)
    assert "Source Port" not in r["features"] and "Destination Port" not in r["features"]
    assert r["identifiers_dropped"] == ["Source Port", "Destination Port"]
    assert "row_index" not in r["features"] and LABEL_COL not in r["features"]
    assert set(r["display"]) <= set(r["features"])


def test_replace_inf_parses_text(raw):
    feats = assign_roles(raw)["features"]
    df, rep = replace_inf(raw, feats)
    assert df["Flow Bytess"].dtype == np.float64
    assert rep["inf_cells_total"] == 1 and rep["rows_with_inf"] == 1
    assert rep["per_column"]["Flow Bytess"] == {"inf_cells": 1, "nan_cells_before": 1}
    assert not np.isinf(df[feats].to_numpy()).any()
    assert rep["rows_with_nan_after"] == 2


def test_duplicates_and_conflicts(raw):
    feats = assign_roles(raw)["features"]
    df, _ = replace_inf(raw, feats)
    out, rep = drop_duplicates(df, feats)
    assert rep["exact_duplicates_removed"] == 1
    assert 0 in out["row_index"].tolist() and 1 not in out["row_index"].tolist()  # first kept
    assert rep["conflicting_groups"] == 1 and rep["conflicting_rows_removed"] == 2
    assert not {4, 5} & set(out["row_index"])
    assert rep["rows_after"] == len(raw) - 3


def test_missing_rule_drop_and_keep():
    df = pd.DataFrame({"a": [np.nan] + [1.0] * 199, LABEL_COL: ["Normal"] * 200})
    out, rep = handle_missing(df, ["a"], 0.01)          # 0.5 % < 1 % -> drop
    assert rep["decision"] == "drop" and len(out) == 199
    df2 = pd.DataFrame({"a": [np.nan] * 3 + [1.0] * 97, LABEL_COL: ["Normal"] * 100})
    out2, rep2 = handle_missing(df2, ["a"], 0.01)       # 3 % -> keep for train medians
    assert rep2["decision"] == "fill_with_training_medians" and len(out2) == 100


def test_labels():
    df = pd.DataFrame({LABEL_COL: ["Normal", "Dos/DDos", "Botnet ARES", "Web Attack"]})
    out = make_labels(df)
    assert out["label_7"].tolist() == [0, 3, 1, 6]
    assert out["label_bin"].tolist() == [0, 1, 1, 1]
    lm = label_map()
    assert lm["classes"][0] == "Normal" and len(lm["classes"]) == 7
    with pytest.raises(ValueError):
        make_labels(pd.DataFrame({LABEL_COL: ["Normal", "Heartbleed"]}))


REAL = ROOT / "data" / "processed" / "ids2025_raw.parquet"


@pytest.mark.skipif(not REAL.exists(), reason="real parquet not present")
def test_clean_real_data():
    raw = pd.read_parquet(REAL)
    df, roles, sec = clean(raw, 0.01)
    feats = roles["features"]
    assert len(feats) == 77
    if sec["missing"]["decision"] == "drop":
        assert not df[feats].isna().any().any()
    assert not df.duplicated(subset=feats).any()
    assert set(df["class_name"]) == set(CLASSES)
    assert df["row_index"].is_unique
