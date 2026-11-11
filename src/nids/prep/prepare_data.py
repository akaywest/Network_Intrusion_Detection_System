"""Data preparation (M1, Phase 1 steps 2-8 of the implementation plan, §3.2.4).

Reads data/processed/ids2025_raw.parquet (written once by nids.prep.convert_xlsx) and
performs, in this order:

  step 2  column profile (type, missing, unique, range)           -> report "columns"
  step 5  column roles: features / display / meta / identifiers   -> report "roles"
          (done before step 4 so that step 4 knows which columns are features)
  step 4  text "Infinity"/"NaN" parsed, +-inf -> NaN, counted      -> report "infinite"
  step 6  exact duplicates and conflicting duplicates removed     -> report "duplicates"
  step 7  missing-value rule (< 1 % of rows -> drop, else fill)   -> report "missing"
  step 8  label_7, label_bin, artefacts/label_map.json, counts    -> report "labels"

Nothing is split and nothing is fitted here. Constant columns, the correlation filter,
medians, the log mask and the scaler are fitted on the training partition only
(step 10, CLAUDE.md, Table 3.12 control 1). Steps 9-12 are not implemented yet; they
will call clean() in the same process and then split.

Usage:  python -m nids.prep.prepare_data [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nids.prep.convert_xlsx import LABEL_COL, ROOT, update_report

# Leakage control 3 (Table 3.12): identifiers are never features. Only the two ports
# exist in IDS2025 as obtained; the other names are checked so the report says so.
IDENTIFIER_CANDIDATES = ("Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
                         "Source Port", "Src Port", "Destination Port", "Dst Port",
                         "Timestamp")
META_COLS = ("row_index",)  # no source_day / timestamp in this dataset (plan §3.1 step 1)

# Display fields for FLOW_RECORD (plan §5.1): column -> FLOW_RECORD field.
# They stay in the feature set as well (plan §3.1 step 5).
DISPLAY_COLS = {"Protocol": "protocol", "Flow Duration": "flow_duration",
                "Total Fwd Packets": "fwd_packets", "Flow Bytess": "bytes_per_s"}

# Table 3.6 class names. Normal = 0 (negative class); attacks in Table 3.6 order.
CLASSES = ("Normal", "Botnet", "Brute Force", "DoS/DDoS", "Infiltration", "PortScan",
           "Web Attack")
# Raw spellings in the sheet -> Table 3.6 names.
RAW_TO_CLASS = {"Normal": "Normal", "Botnet ARES": "Botnet", "Brute Force": "Brute Force",
                "Dos/DDos": "DoS/DDoS", "Infiltration": "Infiltration",
                "PortScan": "PortScan", "Web Attack": "Web Attack"}

TEXT_VALUES = {"Infinity": np.inf, "+Infinity": np.inf, "-Infinity": -np.inf,
               "inf": np.inf, "-inf": -np.inf, "NaN": np.nan, "nan": np.nan}


# --------------------------------------------------------------------------- step 2
def profile_columns(df: pd.DataFrame) -> dict:
    """Type, missing count, unique count and numeric range of every column (§3.2.4)."""
    out = {}
    for c in df.columns:
        s = df[c]
        info = {"dtype": str(s.dtype), "missing": int(s.isna().sum()),
                "unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            info["min"], info["max"] = s.min().item(), s.max().item()
        info["constant"] = info["unique"] <= 1  # reported only; dropped in step 10 (train)
        out[c] = info
    return out


# --------------------------------------------------------------------------- step 5
def assign_roles(df: pd.DataFrame) -> dict:
    """Split columns into features / display / meta / identifiers / label."""
    if LABEL_COL not in df.columns:
        raise KeyError(f"label column {LABEL_COL!r} missing")
    identifiers = [c for c in df.columns if c in IDENTIFIER_CANDIDATES]
    meta = [c for c in META_COLS if c in df.columns]
    excluded = set(identifiers) | set(meta) | {LABEL_COL}
    features = [c for c in df.columns if c not in excluded]  # sheet order
    missing_display = [c for c in DISPLAY_COLS if c not in features]
    if missing_display:
        raise KeyError(f"display columns missing: {missing_display}")
    return {
        "features": features,
        "display": dict(DISPLAY_COLS),
        "meta": meta,
        "label": LABEL_COL,
        "identifiers_dropped": identifiers,
        "identifiers_checked_absent": [c for c in IDENTIFIER_CANDIDATES
                                       if c not in df.columns],
    }


# --------------------------------------------------------------------------- step 4
def _to_float(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")
    obj = s.astype(object).where(s.notna(), np.nan)
    mapped = obj.map(lambda v: TEXT_VALUES.get(v.strip(), v) if isinstance(v, str) else v)
    return pd.to_numeric(mapped, errors="raise").astype("float64")


def replace_inf(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    """Parse text cells, cast features to float64, replace +-inf by NaN, and count."""
    df = df.copy()
    per_col = {}
    inf_rows = pd.Series(False, index=df.index)
    for c in features:
        x = _to_float(df[c])
        is_inf = np.isinf(x)
        n_inf, n_nan = int(is_inf.sum()), int(x.isna().sum())
        if n_inf or n_nan:
            per_col[c] = {"inf_cells": n_inf, "nan_cells_before": n_nan}
        inf_rows |= is_inf
        df[c] = x.mask(is_inf)
    report = {
        "per_column": per_col,
        "inf_cells_total": sum(v["inf_cells"] for v in per_col.values()),
        "rows_with_inf": int(inf_rows.sum()),
        "rows_with_nan_after": int(df[features].isna().any(axis=1).sum()),
    }
    return df, report


# --------------------------------------------------------------------------- step 6
def drop_duplicates(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    """Leakage control 2: exact duplicates (features + label) then conflicting duplicates.

    Identifiers are already excluded from `features`, so two rows differing only in
    port count as duplicates (plan §3.1 step 6). NaN compares equal to NaN here.
    The first copy (lowest row_index) of an exact duplicate is kept. A conflicting
    group (same features, more than one label) is removed entirely, because no copy
    can be trusted.
    """
    df = df.sort_values("row_index", kind="stable")
    n0 = len(df)
    exact = df.duplicated(subset=features + [LABEL_COL], keep="first")
    removed_exact = df.loc[exact, LABEL_COL].value_counts()
    df = df.loc[~exact]

    # After exact dedup, any rows that still share a feature vector differ in label.
    conflict = df.duplicated(subset=features, keep=False)
    n_conflict_rows = int(conflict.sum())
    n_groups = n_conflict_rows - int(df.duplicated(subset=features, keep="first").sum())
    removed_conflict = df.loc[conflict, LABEL_COL].value_counts()
    df = df.loc[~conflict]

    report = {
        "rows_before": n0,
        "exact_duplicates_removed": int(exact.sum()),
        "exact_duplicates_removed_by_class": {k: int(v) for k, v in removed_exact.items()},
        "conflicting_groups": n_groups,
        "conflicting_rows_removed": n_conflict_rows,
        "conflicting_rows_removed_by_class": {k: int(v)
                                              for k, v in removed_conflict.items()},
        "rows_after": len(df),
    }
    return df.reset_index(drop=True), report


# --------------------------------------------------------------------------- step 7
def handle_missing(df: pd.DataFrame, features: list[str],
                   drop_below: float) -> tuple[pd.DataFrame, dict]:
    """Drop rows with any NaN if they are under `drop_below` of the data (§3.2.4).

    Otherwise keep them; they are filled with TRAINING medians in step 10.
    """
    nan_rows = df[features].isna().any(axis=1)
    share = float(nan_rows.mean()) if len(df) else 0.0
    per_col = {c: int(n) for c, n in df[features].isna().sum().items() if n}
    decision = "drop" if share < drop_below else "fill_with_training_medians"
    report = {
        "rows_with_nan": int(nan_rows.sum()),
        "share": round(share, 6),
        "threshold": drop_below,
        "decision": decision,
        "nan_cells_per_column": per_col,
        "rows_with_nan_by_class": {k: int(v) for k, v in
                                   df.loc[nan_rows, LABEL_COL].value_counts().items()},
    }
    if decision == "drop":
        df = df.loc[~nan_rows].reset_index(drop=True)
    report["rows_after"] = len(df)
    return df, report


# --------------------------------------------------------------------------- step 8
def label_map() -> dict:
    return {
        "classes": list(CLASSES),
        "code_of": {c: i for i, c in enumerate(CLASSES)},
        "raw_to_class": dict(RAW_TO_CLASS),
        "binary": {"Normal": 0, "Malicious": 1},
        "binary_rule": "label_bin = 0 if label_7 == code_of['Normal'] else 1",
    }


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add label_7 (Table 3.6 codes) and label_bin (Normal 0, Malicious 1)."""
    unknown = set(df[LABEL_COL].dropna().unique()) - set(RAW_TO_CLASS)
    if unknown or df[LABEL_COL].isna().any():
        raise ValueError(f"unknown or missing labels: {sorted(unknown)}")
    df = df.copy()
    df["class_name"] = df[LABEL_COL].map(RAW_TO_CLASS)
    df["label_7"] = df["class_name"].map(label_map()["code_of"]).astype("int64")
    df["label_bin"] = (df["label_7"] != 0).astype("int64")
    return df


def class_counts(df: pd.DataFrame) -> dict:
    names = df[LABEL_COL].map(RAW_TO_CLASS)
    seven = {c: int((names == c).sum()) for c in CLASSES}
    n_norm = seven["Normal"]
    return {"seven_class": seven,
            "binary": {"Normal": n_norm, "Malicious": int(len(df) - n_norm)},
            "total": int(len(df))}


# --------------------------------------------------------------------------- driver
def clean(raw: pd.DataFrame, drop_below: float) -> tuple[pd.DataFrame, dict, dict]:
    """Run steps 2-8. Returns (cleaned frame, roles, report section)."""
    columns = profile_columns(raw)
    roles = assign_roles(raw)
    feats = roles["features"]
    counts_before = class_counts(raw)

    df, inf_rep = replace_inf(raw, feats)
    df, dup_rep = drop_duplicates(df, feats)
    df, miss_rep = handle_missing(df, feats, drop_below)
    df = make_labels(df)
    # keep only the columns later steps need (identifiers are gone for good)
    df = df[roles["meta"] + feats + [LABEL_COL, "class_name", "label_7", "label_bin"]]

    section = {
        "columns": columns,
        "roles": {**roles, "n_features": len(feats)},
        "infinite": inf_rep,
        "duplicates": dup_rep,
        "missing": miss_rep,
        "labels": {
            "label_map": "artefacts/label_map.json",
            "class_counts_raw": counts_before,
            "class_counts_clean": class_counts(df),
        },
        "row_flow": {
            "loaded": len(raw),
            "after_exact_duplicates": dup_rep["rows_before"] - dup_rep["exact_duplicates_removed"],
            "after_conflicting_duplicates": dup_rep["rows_after"],
            "after_missing_rule": miss_rep["rows_after"],
            "final": len(df),
        },
    }
    return df, roles, section


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=ROOT / "config.yaml", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    raw = pd.read_parquet(ROOT / cfg["paths"]["raw_parquet"])
    df, roles, section = clean(raw, cfg["prep"]["missing_drop_below"])

    lm_path = ROOT / cfg["paths"]["artefacts"] / "label_map.json"
    lm_path.parent.mkdir(parents=True, exist_ok=True)
    lm_path.write_text(json.dumps(label_map(), indent=2) + "\n")
    update_report(ROOT / "results" / "prep_report.json", {"cleaning": section})

    rf, inf, dup, miss = section["row_flow"], section["infinite"], section["duplicates"], section["missing"]
    print(f"Identifiers dropped: {roles['identifiers_dropped']}; meta: {roles['meta']}")
    print(f"Infinite cells -> NaN: {inf['inf_cells_total']} {inf['per_column']}")
    print(f"Exact duplicates removed: {dup['exact_duplicates_removed']} "
          f"{dup['exact_duplicates_removed_by_class']}")
    print(f"Conflicting duplicates: {dup['conflicting_groups']} groups, "
          f"{dup['conflicting_rows_removed']} rows removed "
          f"{dup['conflicting_rows_removed_by_class']}")
    print(f"Missing: {miss['rows_with_nan']} rows ({miss['share']:.4%}) -> {miss['decision']}")
    print(f"Row flow: {rf}")
    print("Class counts (raw -> clean):")
    raw_c, cl_c = section["labels"]["class_counts_raw"], section["labels"]["class_counts_clean"]
    for i, c in enumerate(CLASSES):
        print(f"  {i} {c:<13} {raw_c['seven_class'][c]:>7} -> {cl_c['seven_class'][c]:>7}")
    print(f"  binary: {cl_c['binary']}")
    print(f"Features ({len(roles['features'])}):")
    for i, f in enumerate(roles["features"], 1):
        print(f"  {i:2d} {f}")


if __name__ == "__main__":
    main()
