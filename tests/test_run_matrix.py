"""experiments/run_matrix.py (plan §8.2): run list, interleaving, estimate, preflight
refusals, and a resumable headless run through the real services (IT2-style)."""
import json
import sqlite3
import sys

import pytest
import yaml

from nids.control import protocol
from nids.prep.convert_xlsx import ROOT

sys.path.insert(0, str(ROOT))
from experiments import run_matrix as rm  # noqa: E402

from tests.conftest import HAVE_DATA  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def test_main_set_matches_table_3_13():
    plan = rm.build_plan(CFG, "main")
    assert len(plan) == 3 * 5 * 7                           # 3 models x 5 seeds x 7 steps
    steps = {(r.step.scenario, r.step.dt_ms, r.step.mode, r.step.n) for r in plan}
    assert steps == {("S1", 100, "steady", 5000), ("S2", 10, "steady", 5000),
                     ("S3", 50, "steady", 5000), ("S3", 20, "steady", 5000),
                     ("S3", 5, "steady", 5000), ("S3", 1, "steady", 5000),
                     ("S4", 100, "burst", 2000)}
    assert all(r.dashboard_open for r in plan)
    assert {r.model for r in plan} == {"iforest", "rf", "lstm"}
    assert len({r.key for r in plan}) == len(plan)          # no duplicate run


def test_interleaved_order_and_one_seed_per_repetition():
    plan = rm.build_plan(CFG, "main")
    assert [r.model for r in plan[:6]] == ["iforest", "rf", "lstm"] * 2
    for rep in range(1, 6):
        assert {r.seed for r in plan if r.rep == rep} == {41 + rep}
    reps = [r.rep for r in plan]
    assert reps == sorted(reps)                             # repetition is the outer loop


def test_closed_set_is_s1_without_dashboard():
    plan = rm.build_plan(CFG, "closed")
    assert len(plan) == 15
    assert all(r.step.scenario == "S1" and not r.dashboard_open for r in plan)
    # same scenario label as the open S1 runs, told apart by the dashboard state
    assert not {r.key for r in plan} & {r.key for r in rm.build_plan(CFG, "main")}


def test_burst_block_and_estimate():
    assert rm.burst_block(CFG, 2000) == 1000
    assert rm.burst_block(CFG, 200) == 100
    inf = {"iforest": 10.0, "rf": 40.0, "lstm": 5.0}
    s1 = rm.Run("main", 1, 42, "rf", rm.Step("S1", 100.0, "steady", 5000), True)
    s3 = rm.Run("main", 1, 42, "rf", rm.Step("S3", 1.0, "steady", 5000), True)
    # schedule-bound at 100 ms, service-bound at 1 ms (RF cannot keep up)
    assert rm.estimate_s(s1, CFG, inf) == pytest.approx(4999 * 0.1 + 5, abs=0.01)
    assert rm.estimate_s(s3, CFG, inf) == pytest.approx(5000 * 0.043 + 5, abs=0.01)


def test_buffer_check_refuses_small_buffers(monkeypatch):
    monkeypatch.setattr(rm, "effective_socket_buffers", lambda: {
        "so_sndbuf_effective": 425984, "net_core_wmem_max": 212992,
        "net_core_rmem_max": 212992})
    ok, _, problems = rm.check_buffers()
    assert not ok and len(problems) == 3
    code = rm.main(["--set", "closed", "--pause-s", "0", "--engine-ctl-port", "1",
                    "--streamer-ctl-port", "1", "--dashboard-port", "1"])
    assert code == 2


def test_set_and_flag_must_agree():
    with pytest.raises(SystemExit):
        rm.main(["--set", "main"])                          # dashboard flag missing
    with pytest.raises(SystemExit):
        rm.main(["--set", "closed", "--dashboard-open"])
    with pytest.raises(SystemExit):
        rm.main(["--set", "closed", "--n-override", "10"])  # unlabelled short runs


def test_prepare_accepts_dashboard_open():
    ok = protocol.validate({"cmd": "prepare", "model": "rf", "seed": 42, "dt_ms": 1,
                            "dashboard_open": False}, protocol.ENGINE_CMDS)
    assert ok["dashboard_open"] is False
    with pytest.raises(protocol.ProtocolError, match="dashboard_open"):
        protocol.validate({"cmd": "prepare", "model": "rf", "seed": 42, "dt_ms": 1,
                           "dashboard_open": "yes"}, protocol.ENGINE_CMDS)


def test_dry_run_prints_both_sets(capsys, tmp_path):
    assert rm.main(["--dry-run", "--db", str(tmp_path / "none.db")]) == 0
    out = capsys.readouterr().out
    assert "set 'main'" in out and "set 'closed'" in out and "ALL SETS: 120 runs" in out


@pytest.mark.skipif(not HAVE_DATA, reason="needs data and artefacts")
def test_headless_run_records_dashboard_state_and_resumes(services, tmp_path, capsys):
    args = ["--set", "closed", "--scenario-prefix", "rm-test-", "--n-override", "20",
            "--models", "iforest", "--seeds", "42", "--pause-s", "0",
            "--db", str(services.db), "--log", str(tmp_path / "log.csv"),
            "--engine-ctl-port", str(services.client.engine_port),
            "--streamer-ctl-port", str(services.client.streamer_port),
            "--dashboard-port", "1"]
    assert rm.main(args) == 0
    con = sqlite3.connect(services.db)
    rows = con.execute(
        "SELECT r.run_id, r.end_reason, e.run_conditions, "
        "(SELECT count(*) FROM prediction p WHERE p.run_id = r.run_id) "
        "FROM model_run r JOIN test_environment e USING (env_id) "
        "WHERE r.scenario = 'rm-test-S1'").fetchall()
    con.close()
    assert len(rows) == 1
    run_id, end_reason, cond, n_pred = rows[0]
    assert end_reason == "complete" and n_pred == 20
    assert json.loads(cond)["dashboard_open"] is False
    capsys.readouterr()
    assert rm.main(args) == 0                               # resume: nothing left to do
    assert "1 already complete, 0 pending" in capsys.readouterr().out
