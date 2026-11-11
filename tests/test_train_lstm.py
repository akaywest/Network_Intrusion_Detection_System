"""Tests for LSTM training (M2, plan §4.3, formulation B).

Synthetic tests run always; the last one checks the real artefacts once
`python -m nids.train.train_lstm` has been run, and skips otherwise.
"""
import json

import numpy as np
import pytest
import tensorflow as tf

from nids.prep.convert_xlsx import ROOT
from nids.train import train_lstm as tl

ART = ROOT / "artefacts"
SEEDS = [42, 43, 44, 45, 46]
NAMES = ["Normal", "Botnet", "Brute Force", "DoS/DDoS", "Infiltration", "PortScan", "Web Attack"]


def test_sampled_configs_within_table_3_11():
    cs = tl.sample_configs(500)
    assert {c["layers"] for c in cs} == {1, 2}
    assert {c["units"] for c in cs} == {32, 64, 128}
    assert {c["batch_size"] for c in cs} == {128, 256, 512}
    assert all(0.2 <= c["dropout"] <= 0.5 for c in cs)
    assert all(1e-4 <= c["learning_rate"] <= 1e-2 for c in cs)
    assert tl.sample_configs(15) == tl.sample_configs(15)  # seeded, reproducible
    assert "W" not in cs[0]  # formulation B: no window search


def test_formulation_b_shapes():
    X = np.arange(12, dtype=np.float64).reshape(3, 4)
    s = tl.to_seq(X)
    assert s.shape == (3, 4, 1) and s.dtype == np.float32 and np.array_equal(s[:, :, 0], X)
    cfg = {"layers": 2, "units": 8, "dropout": 0.3, "learning_rate": 1e-3, "batch_size": 128}
    assert tl.build_model(cfg, 49, 1).output_shape == (None, 1)
    m = tl.build_model(cfg, 49, 7)
    assert m.input_shape == (None, 49, 1) and m.output_shape == (None, 7)
    assert [l.name for l in m.layers][1:] == ["lstm_1", "lstm_2", "dropout", "output"]


def test_class_weights_balanced():
    w = tl.class_weights(np.array([0] * 90 + [1] * 10))
    assert w[0] == pytest.approx(100 / (2 * 90)) and w[1] == pytest.approx(100 / (2 * 10))


def _toy(n_per=40, F=6, seed=0):
    rng = np.random.default_rng(seed)
    y7 = np.repeat(np.arange(7), n_per)
    X = (rng.normal(0, 1, (len(y7), F)) + y7[:, None] * 1.5).astype(np.float32)
    return X, y7


def test_train_task_end_to_end(tmp_path):
    X, y7 = _toy()
    Xv, yv7 = _toy(seed=1)
    d = {"X_train": X, "y_train_7": y7, "y_train_bin": (y7 != 0).astype(int),
         "X_val": Xv, "y_val_7": yv7, "y_val_bin": (yv7 != 0).astype(int)}
    idx = np.arange(len(y7))
    for task in ("binary", "multi"):
        config, log = tl.train_task(task, d, idx, [42, 43], tmp_path, NAMES, smoke=True,
                                    search_frac=1.0, log_dir=tmp_path / "log",
                                    ckpt_dir=tmp_path / "ckpt")
        assert log["test_partition_used"] is False and len(log["search_trials"]) == tl.SMOKE_N_TRIALS
        assert all(len(t["epoch_s"]) == t["epochs_run"] for t in log["search_trials"])
        t = log["single_record_timing"]
        assert t["n_calls"] == tl.SMOKE_N_TIMING and t["intra_op_threads"] == 1
        assert t["max_abs_diff_vs_predict"] < 1e-4
        for seed in (42, 43):
            assert (tmp_path / f"lstm_{task}_seed{seed}.keras").exists()
            meta = json.loads((tmp_path / f"lstm_{task}_seed{seed}.meta.json").read_text())
            assert meta["formulation"] == "B" and meta["input_shape"] == [6, 1]
            assert meta["seed"] == seed and meta["config"] == config["hyperparameters"]
            assert meta["warmup_records"] == 0
            assert (meta["tau"] is None) == (task == "multi")
        assert (tmp_path / f"lstm_{task}_config.json").exists()


# ---- resume ---------------------------------------------------------------------------

class Interrupted(Exception):
    pass


def _toy_data():
    X, y7 = _toy()
    Xv, yv7 = _toy(seed=1)
    return {"X_train": X, "y_train_7": y7, "y_train_bin": (y7 != 0).astype(int),
            "X_val": Xv, "y_val_7": yv7, "y_val_bin": (yv7 != 0).astype(int)}


def _run(root, monkeypatch, crash_at=None, fresh=False):
    """train_task('binary') in root; if crash_at=k, the k-th fit_one call raises.
    Returns (log or None, number of fit_one calls made)."""
    calls = []
    real = tl.fit_one

    def counting(*a, **k):
        calls.append(k.get("tag"))
        if crash_at is not None and len(calls) == crash_at:
            raise Interrupted
        return real(*a, **k)

    monkeypatch.setattr(tl, "fit_one", counting)
    d = _toy_data()
    try:
        _, log = tl.train_task("binary", d, np.arange(len(d["y_train_7"])), [42, 43],
                               root / "art", NAMES, smoke=True, search_frac=1.0,
                               log_dir=root / "log", ckpt_dir=root / "ckpt", fresh=fresh)
    except Interrupted:
        log = None
    finally:
        monkeypatch.setattr(tl, "fit_one", real)
    return log, calls


def _same_results(a, b):
    ta = sorted(a["search_trials"], key=lambda r: r["trial"])
    tb = sorted(b["search_trials"], key=lambda r: r["trial"])
    assert [t["config"] for t in ta] == [t["config"] for t in tb]
    for x, y in zip(ta, tb):
        assert x["val_score"] == pytest.approx(y["val_score"], rel=1e-6)
        assert x["val_loss"] == pytest.approx(y["val_loss"], rel=1e-5)
        assert x["epochs_run"] == y["epochs_run"]
    assert a["chosen"] == b["chosen"] and a["threshold_choice"] == pytest.approx(b["threshold_choice"])
    for x, y in zip(a["per_seed_validation"], b["per_seed_validation"]):
        for k in ("seed", "epochs_run", "best_epoch", "tp", "fp", "fn", "tn"):
            assert x[k] == y[k]
        for k in ("tau", "f1", "roc_auc"):
            assert x[k] == pytest.approx(y[k], rel=1e-5)


@pytest.fixture(scope="module")
def deterministic():
    tf.config.experimental.enable_op_determinism()  # as run() does


@pytest.fixture(scope="module")
def reference(tmp_path_factory, deterministic):
    mp = pytest.MonkeyPatch()
    log, calls = _run(tmp_path_factory.mktemp("ref"), mp)
    # 3 trials + seed 43 (seed 42 reuses the best trial)
    assert len(calls) == tl.SMOKE_N_TRIALS + 1 and log["resume"]["trials_loaded"] == 0
    return log


def test_resume_mid_search_matches_uninterrupted(tmp_path, monkeypatch, reference):
    log, calls = _run(tmp_path, monkeypatch, crash_at=3)  # dies in trial 3
    assert log is None and calls == ["t1", "t2", "t3"]
    lines = (tmp_path / "log" / "lstm_binary_trials.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["type"] == "header" and len(lines) == 1 + 2
    assert not (tmp_path / "art" / "lstm_binary_seed42.meta.json").exists()

    log, calls = _run(tmp_path, monkeypatch)
    assert calls == ["t3", "s43"]  # trials 1-2 skipped; seed 42 from memory or checkpoint
    assert log["resume"] == {"trials_loaded": 2, "seeds_loaded": [],
                             "trials_file": "lstm_binary_trials.jsonl"}
    _same_results(log, reference)


def test_resume_mid_seeds_skips_finished_seed(tmp_path, monkeypatch, reference):
    log, calls = _run(tmp_path, monkeypatch, crash_at=4)  # dies in the seed-43 retrain
    assert log is None and calls == ["t1", "t2", "t3", "s43"]
    assert (tmp_path / "art" / "lstm_binary_seed42.keras").exists()
    assert (tmp_path / "art" / "lstm_binary_seed42.meta.json").exists()
    assert not (tmp_path / "art" / "lstm_binary_seed43.meta.json").exists()

    log, calls = _run(tmp_path, monkeypatch)
    assert calls == ["s43"]
    assert log["resume"]["trials_loaded"] == 3 and log["resume"]["seeds_loaded"] == [42]
    assert log["single_record_timing"]["n_calls"] == tl.SMOKE_N_TIMING  # kept from seed 42
    _same_results(log, reference)

    log, calls = _run(tmp_path, monkeypatch)  # everything done: no fitting at all
    assert calls == [] and log["resume"]["seeds_loaded"] == [42, 43]
    _same_results(log, reference)


def test_resume_drops_torn_line_and_refits_without_checkpoint(tmp_path, monkeypatch,
                                                              reference):
    _run(tmp_path, monkeypatch, crash_at=3)
    tp = tmp_path / "log" / "lstm_binary_trials.jsonl"
    with open(tp, "a") as f:
        f.write('{"type": "trial", "trial": 3, "conf')  # crash mid-write
    for p in (tmp_path / "ckpt").iterdir():
        p.unlink()  # lost checkpoint: best trial must be refitted if it is 1 or 2
    log, calls = _run(tmp_path, monkeypatch)
    assert calls[0] == "t3" and calls[-1] == "s43"
    assert len(tp.read_text().splitlines()) == 1 + tl.SMOKE_N_TRIALS
    _same_results(log, reference)


def test_resume_rejects_other_configuration(tmp_path, monkeypatch, deterministic):
    _run(tmp_path, monkeypatch, crash_at=2)
    tp = tmp_path / "log" / "lstm_binary_trials.jsonl"
    lines = tp.read_text().splitlines()
    head = json.loads(lines[0])
    head["run_fingerprint"]["max_epochs"] = 99
    tp.write_text("\n".join([json.dumps(head)] + lines[1:]) + "\n")
    with pytest.raises(tl.ResumeMismatch):
        _run(tmp_path, monkeypatch)
    _, calls = _run(tmp_path, monkeypatch, crash_at=1, fresh=True)  # --fresh starts over
    assert calls == ["t1"]


@pytest.mark.skipif(not all((ART / f"lstm_{t}_seed{s}.keras").exists()
                            for t in ("binary", "multi") for s in SEEDS),
                    reason="run python -m nids.train.train_lstm first")
def test_done_ten_models_and_meta():
    for task in ("binary", "multi"):
        cfg = json.loads((ART / f"lstm_{task}_config.json").read_text())
        assert cfg["smoke"] is False and cfg["seeds"] == SEEDS
        assert cfg["search"]["n_trials"] == 15 and cfg["formulation"] == "B"
        log = json.loads((ROOT / "results" / "train_log" / f"lstm_{task}.json").read_text())
        assert log["test_partition_used"] is False and len(log["search_trials"]) == 15
        for s in SEEDS:
            meta = json.loads((ART / f"lstm_{task}_seed{s}.meta.json").read_text())
            assert meta["config"] == cfg["hyperparameters"] and meta["seed"] == s
