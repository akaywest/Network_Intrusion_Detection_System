"""Phase 0 smoke tests: config loads and the module packages (Table 3.9) import."""
import importlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_config_loads():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert cfg["network"]["host"] == "127.0.0.1"  # NFR6
    assert cfg["seeds"] == [42, 43, 44, 45, 46]  # NFR3
    s = cfg["split"]
    assert abs(s["train"] + s["val"] + s["test"] - 1.0) < 1e-9


def test_module_packages_import():
    for m in ["prep", "train", "stream", "engine", "storage", "dashboard", "control", "evaluation"]:
        importlib.import_module(f"nids.{m}")
