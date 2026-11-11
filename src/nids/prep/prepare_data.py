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

  step 9  stratified 70/15/15 split on label_7, seed 42            -> report "split"
  step 10 fit on the TRAINING partition only: constant columns,    -> report "fit"
          |r| > 0.98 filter, medians, log1p mask, StandardScaler
  step 11 transform all three partitions, cast to float32
  step 12 sort each partition back into row_index order

Steps 2-8 fit nothing (they are row-wise or exact-match rules). Every statistic in
step 10 sees the training partition only (CLAUDE.md; Table 3.12 control 1). This moves
the constant-column and correlation filters after the split, unlike Fig. 3.3
(plan §1.1 item 2) -- a deviation to state in Chapter Four.

Outputs (plan §3.2): data/processed/{train,val,test}_raw.parquet (cleaned, UNSCALED),
X_*.npy (scaled float32), y_*_7.npy / y_*_bin.npy, artefacts/feature_list.json,
artefacts/preprocessor.json, artefacts/label_map.json, results/prep_report.json.

Usage:  python -m nids.prep.prepare_data [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nids.prep.convert_xlsx import LABEL_COL, ROOT, update_report

PARTITIONS = ("train", "val", "test")
LABEL_COLS = ["class_name", "label_7", "label_bin"]

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


# --------------------------------------------------------------------------- step 9
def split(df: pd.DataFrame, train: float, val: float, test: float,
          seed: int) -> dict[str, pd.DataFrame]:
    """Stratified split on label_7 in two calls: train vs rest, then rest -> val/test."""
    if not np.isclose(train + val + test, 1.0):
        raise ValueError("split fractions must sum to 1")
    tr, rest = train_test_split(df, train_size=train, stratify=df["label_7"],
                                random_state=seed)
    va, te = train_test_split(rest, test_size=test / (val + test),
                              stratify=rest["label_7"], random_state=seed)
    return {"train": tr, "val": va, "test": te}


# --------------------------------------------------------------------------- step 10
def fit_preprocessing(train: pd.DataFrame, features: list[str], corr_threshold: float,
                      skew_threshold: float,
                      log_exclude: tuple[str, ...] | list[str] = ()) -> tuple[dict, dict]:
    """Fit every preprocessing statistic on the TRAINING partition only.

    (a) drop constant columns; (b) |Pearson r| > corr_threshold: walk the columns in
    order and drop a column if it is too correlated with a column already KEPT (so the
    later column of each pair goes, and a column is never dropped because of a column
    that was itself dropped); (c) medians; (d) log mask: min >= 0 and |skew| >
    skew_threshold, except columns in log_exclude (categorical codes such as Protocol,
    which are not measurements and are only standardised); (e) StandardScaler on the
    log-transformed values.
    Returns (state for preprocessor.json, report).
    """
    X = train[features].astype("float64")

    constant = [c for c in features if X[c].nunique(dropna=True) <= 1]
    cand = [c for c in features if c not in constant]

    r = X[cand].corr(method="pearson")
    a = r.abs().to_numpy()
    pairs = [(cand[i], cand[j], float(r.iat[i, j]))
             for i in range(len(cand)) for j in range(i + 1, len(cand))
             if a[i, j] > corr_threshold]
    kept, corr_dropped = [], {}
    for j, c in enumerate(cand):
        partner = next((k for k in kept if a[cand.index(k), j] > corr_threshold), None)
        if partner is None:
            kept.append(c)
        else:
            corr_dropped[c] = partner

    median = X[kept].median()
    Xf = X[kept].fillna(median)
    skew = Xf.skew()
    log_mask = ((Xf.min() >= 0) & (skew.abs() > skew_threshold)
                & ~Xf.columns.isin(list(log_exclude)))
    Xl = apply_log(Xf.to_numpy(), log_mask.to_numpy())
    scaler = StandardScaler().fit(Xl)

    state = {
        "features": kept,
        "median": [float(v) for v in median],
        "log_mask": [bool(v) for v in log_mask],
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "fitted_on": "train",
        "n_train_rows": int(len(train)),
    }
    report = {
        "fitted_on": "train",
        "n_train_rows": int(len(train)),
        "n_features_in": len(features),
        "constant_columns_dropped": constant,
        "correlated_pairs": [{"a": x, "b": y, "r": round(v, 6),
                              "dropped": y if corr_dropped.get(y) == x else None}
                             for x, y, v in pairs],
        "correlation_dropped": corr_dropped,  # dropped column -> kept partner
        "log_columns": [c for c, m in zip(kept, log_mask) if m],
        "log_excluded_categorical": [c for c in log_exclude if c in kept],
        "skewed_but_negative": [c for c in kept
                                if abs(skew[c]) > skew_threshold and Xf[c].min() < 0],
        "n_features_final": len(kept),
    }
    return state, report


def apply_log(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """log(1+x) on the masked columns; values below 0 are clipped to 0 (same as online)."""
    x = np.array(x, dtype=np.float64)
    x[:, mask] = np.log1p(np.clip(x[:, mask], 0, None))
    return x


# --------------------------------------------------------------------------- step 11
def transform_offline(df: pd.DataFrame, state: dict) -> np.ndarray:
    """Apply the fitted state to a whole partition (same arithmetic as the engine's
    Preprocessor, plan §3.3)."""
    x = df[state["features"]].to_numpy(dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
    x = np.where(np.isnan(x), np.array(state["median"]), x)
    x = apply_log(x, np.array(state["log_mask"], dtype=bool))
    return ((x - np.array(state["mean"])) / np.array(state["scale"])).astype(np.float32)


def class_table(parts: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for name, p in parts.items():
        seven = {c: int((p["class_name"] == c).sum()) for c in CLASSES}
        out[name] = {"seven_class": seven,
                     "binary": {"Normal": seven["Normal"],
                                "Malicious": int(len(p) - seven["Normal"])},
                     "total": int(len(p))}
    return out


def split_and_fit(df: pd.DataFrame, features: list[str], cfg: dict):
    """Steps 9-12. Returns (sorted partitions, scaled arrays, state, report section)."""
    s, prep = cfg["split"], cfg["prep"]
    if s["stratify"] != "label_7":
        raise ValueError("Chapter 3 §3.2.4 stratifies on the seven-class label")
    parts = split(df, s["train"], s["val"], s["test"], s["seed"])
    state, fit_rep = fit_preprocessing(parts["train"], features, prep["corr_threshold"],
                                       prep["skew_threshold"],
                                       prep.get("log_exclude", []))
    # step 12: fixed, reproducible order in every partition file
    parts = {k: p.sort_values("row_index", kind="stable").reset_index(drop=True)
             for k, p in parts.items()}
    arrays = {k: transform_offline(p, state) for k, p in parts.items()}

    ids = [set(p["row_index"]) for p in parts.values()]
    section = {
        "split": {
            "seed": s["seed"], "stratify": "label_7",
            "fractions": {"train": s["train"], "val": s["val"], "test": s["test"]},
            "rows": {k: int(len(p)) for k, p in parts.items()},
            "row_index_overlap": int(len(ids[0] & ids[1]) + len(ids[0] & ids[2])
                                     + len(ids[1] & ids[2])),
            "class_counts": class_table(parts),
        },
        "fit": fit_rep,
    }
    return parts, arrays, state, section


def save_outputs(parts, arrays, state, features: list[str], cfg: dict) -> None:
    processed = ROOT / cfg["paths"]["processed"]
    art = ROOT / cfg["paths"]["artefacts"]
    processed.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)
    for k, p in parts.items():
        # cleaned, UNSCALED rows (streamer uses test only; FLOW_RECORD loading)
        p[["row_index"] + features + LABEL_COLS].to_parquet(processed / f"{k}_raw.parquet",
                                                            index=False)
        np.save(processed / f"X_{k}.npy", arrays[k])
        np.save(processed / f"y_{k}_7.npy", p["label_7"].to_numpy(np.int64))
        np.save(processed / f"y_{k}_bin.npy", p["label_bin"].to_numpy(np.int64))
    (art / "feature_list.json").write_text(json.dumps(state["features"], indent=2) + "\n")
    (art / "preprocessor.json").write_text(json.dumps(state, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=ROOT / "config.yaml", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    raw = pd.read_parquet(ROOT / cfg["paths"]["raw_parquet"])
    df, roles, section = clean(raw, cfg["prep"]["missing_drop_below"])
    parts, arrays, state, sf = split_and_fit(df, roles["features"], cfg)
    save_outputs(parts, arrays, state, roles["features"], cfg)

    lm_path = ROOT / cfg["paths"]["artefacts"] / "label_map.json"
    lm_path.parent.mkdir(parents=True, exist_ok=True)
    lm_path.write_text(json.dumps(label_map(), indent=2) + "\n")
    update_report(ROOT / "results" / "prep_report.json", {"cleaning": section, **sf})

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

    fit, sp = sf["fit"], sf["split"]
    print(f"\nSplit (seed {sp['seed']}, stratify label_7): {sp['rows']}, "
          f"row_index overlap = {sp['row_index_overlap']}")
    cc = sp["class_counts"]
    print(f"  {'class':<14}" + "".join(f"{k:>8}" for k in PARTITIONS) + f"{'total':>8}")
    for c in CLASSES:
        n = [cc[k]["seven_class"][c] for k in PARTITIONS]
        print(f"  {c:<14}" + "".join(f"{v:>8}" for v in n) + f"{sum(n):>8}")
    for b in ("Normal", "Malicious"):
        n = [cc[k]["binary"][b] for k in PARTITIONS]
        print(f"  [bin] {b:<8}" + "".join(f"{v:>8}" for v in n) + f"{sum(n):>8}")

    print(f"\nFitted on train ({fit['n_train_rows']} rows): {fit['n_features_in']} -> "
          f"{fit['n_features_final']} features")
    print(f"Constant columns dropped ({len(fit['constant_columns_dropped'])}): "
          f"{fit['constant_columns_dropped']}")
    print(f"Correlated pairs |r| > {cfg['prep']['corr_threshold']} "
          f"({len(fit['correlated_pairs'])}):")
    for p in fit["correlated_pairs"]:
        tag = f"drop {p['dropped']}" if p["dropped"] else "(no drop: other column already dropped)"
        print(f"  {p['r']:+.4f}  {p['a']}  ~  {p['b']}   {tag}")
    print(f"Correlation-dropped ({len(fit['correlation_dropped'])}): "
          f"{list(fit['correlation_dropped'])}")
    print(f"log1p columns ({len(fit['log_columns'])}): {fit['log_columns']}")
    print(f"Excluded from log1p (categorical, standardised only): "
          f"{fit['log_excluded_categorical']}")
    print(f"Skewed but min < 0, not logged: {fit['skewed_but_negative']}")
    print(f"Final features ({len(state['features'])}): {state['features']}")


if __name__ == "__main__":
    main()
