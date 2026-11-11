"""Tests for the one-off xlsx -> Parquet conversion (M1, Phase 1 step 1)."""
import json
import zipfile

import openpyxl
import pandas as pd
import pytest

from nids.prep.convert_xlsx import convert, label_run_stats, sha256

LONG = "74.755601799999994"  # literal taken from the real sheet


@pytest.fixture
def tiny_xlsx(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "sheet"
    ws.append([" Source Port", "Flow Bytess ", "Protocol", "newLabel"])
    ws.append([80, 1.5, 6, "Normal"])
    ws.append([443, "Infinity", 6, "Normal"])
    ws.append([22, "NaN", 17, "PortScan"])
    ws.append([53, 0.25, 17, "Normal"])
    ws.append([53, 7.0, 17, "Normal"])  # float column cell
    tmp = tmp_path / "w.xlsx"
    wb.save(tmp)
    # Inject a 17-significant-digit literal, as the real IDS2025 sheet stores
    # (openpyxl's writer would shorten it).
    p = tmp_path / "tiny.xlsx"
    with zipfile.ZipFile(tmp) as zin, zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "xl/worksheets/sheet1.xml":
                data = data.replace(b"<v>0.25</v>", f"<v>{LONG}</v>".encode())
            zout.writestr(item, data)
    return p


def test_convert_preserves_values_and_counts(tiny_xlsx, tmp_path):
    before = sha256(tiny_xlsx)
    out, rep = tmp_path / "out.parquet", tmp_path / "report.json"
    s = convert(tiny_xlsx, out, rep)
    df = pd.read_parquet(out)

    assert sha256(tiny_xlsx) == before                    # source never modified
    assert list(df.columns) == ["row_index", "Source Port", "Flow Bytess", "Protocol", "newLabel"]
    assert df["row_index"].tolist() == [0, 1, 2, 3, 4]
    assert df["Source Port"].dtype == "int64"             # all-numeric stays numeric
    assert df["Flow Bytess"].tolist()[:3] == ["1.5", "Infinity", "NaN"]
    assert float(df["Flow Bytess"][3]) == float(LONG)     # exact double round-trip
    assert df["Flow Bytess"][4] == "7"                    # stored as <v>7</v>, kept as such
    assert s["source"]["parquet_rows"] == s["source"]["excel_data_rows"] == 5
    assert s["source"]["parquet_columns"] == s["source"]["excel_columns"] + 1
    assert s["source"]["identifier_columns_found"] == ["Source Port"]
    assert s["source"]["timestamp_present"] is False
    assert "order_check" in json.loads(rep.read_text())


def test_report_merge_keeps_other_keys(tiny_xlsx, tmp_path):
    rep = tmp_path / "report.json"
    rep.write_text(json.dumps({"later_step": 1}))
    convert(tiny_xlsx, tmp_path / "out.parquet", rep)
    assert json.loads(rep.read_text())["later_step"] == 1


def test_label_run_stats():
    s = label_run_stats(pd.Series(["A", "A", "B", "A", "A", "A"]))
    assert s["n_runs"] == 3
    assert s["p_next_same_label"] == round(3 / 5, 6)
    assert s["per_class"]["A"] == {"rows": 5, "runs": 2, "mean": 2.5, "median": 2.5, "max": 3}
    assert s["single_label_window_share"]["W=5"] == 0.0
