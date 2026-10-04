"""一次网流量基线的反演、代际迁移与审计一致性。"""

from __future__ import annotations

import dataclasses

import pytest

from heatnet.app import build_platform
from heatnet.config import ConfigError, config_to_mapping, default_config, default_inversion
from heatnet.console.api import Api
from heatnet.errors import InterlockError
from heatnet.primary.inversion import BaselineRegistry, FlowSample, SampleWindow

STATIONS = ["st01", "st02", "st03", "st04", "st05", "st06"]


def _feed(platform, total_t_h, count=None, missing=(), mismatch=False):
    """按当前配置的窗口长度喂实测样本；``missing`` 里的站记缺项。"""

    window = platform.config.inversion.window_size if count is None else count
    per_station = total_t_h / len(STATIONS)
    for _ in range(window):
        flows = {name: (100.0 if mismatch else per_station) for name in STATIONS}
        for name in missing:
            flows[name] = None
        platform.collect_flow_sample(total_t_h, flows, "test")


def _auto_platform(tmp_path, clock):
    """冲突策略为 auto 的平台。"""

    config = default_config()
    inversion = dataclasses.replace(config.inversion, loop_policies={"primary": "auto"})
    config = dataclasses.replace(config, inversion=inversion)
    return build_platform(config, clock=clock, storage_root=str(tmp_path / "state"))


def test_window_is_sliding_not_full_history():
    window = SampleWindow(3)
    for index in range(5):
        window.append(FlowSample(ts=float(index), total_t_h=100.0 + index, stations={}))
    samples = window.samples()
    assert [sample.ts for sample in samples] == [2.0, 3.0, 4.0]
    assert window.full() is True


def test_inversion_uses_only_the_sliding_window():
    limits = dataclasses.replace(default_inversion(), window_size=4)
    registry = BaselineRegistry("primary", limits, default_config().primary, 2400.0)
    for index, total in enumerate([2000.0] * 4 + [2600.0] * 4):
        registry.add_sample(
            FlowSample(ts=float(index), total_t_h=total, stations={"st01": total})
        )
    round_ = registry.invert(100.0, ["st01"])
    assert round_.ok is True
    assert round_.candidate_baseline_t_h == pytest.approx(2600.0)


def test_complete_window_activates_one_generation_per_round(platform):
    _feed(platform, 2450.0)
    first = platform.invert_flow_baseline("test")
    assert first["ok"] is True
    assert first["decision"] == "activated"
    assert first["active_generation"] == 2
    assert platform.network.baseline_t_h() == pytest.approx(2450.0)
    assert platform.network.threshold_t_h() == pytest.approx(
        2450.0 - platform.config.primary.flow_tolerance_t_h
    )
    assert first["recommended_threshold_t_h"] == pytest.approx(
        platform.network.threshold_t_h()
    )
    assert platform.network.snapshot().generation == 2

    _feed(platform, 2455.0)
    second = platform.invert_flow_baseline("test")
    assert second["active_generation"] == 3
    generations = platform.baseline_report()["generations"]
    assert [item["generation"] for item in generations] == [1, 2, 3]
    assert generations[0]["source"] == "commissioning"
    assert generations[1]["source"] == "inversion"
    assert generations[0]["status"] == "retired"
    assert generations[-1]["status"] == "active"


def test_missing_measurements_are_partial_failure_and_block_migration(platform):
    _feed(platform, 2500.0, missing=("st03",))
    report = platform.invert_flow_baseline("test")
    assert report["ok"] is False
    assert report["partial"] is True
    assert report["decision"] == "none"
    assert "missing_measurements" in report["reasons"]
    assert report["missing"]["st03"] == platform.config.inversion.window_size
    assert platform.network.baseline_t_h() == pytest.approx(2400.0)
    assert platform.network.threshold_t_h() == pytest.approx(2220.0)
    assert len(platform.baseline_report()["generations"]) == 1
    entries = [
        entry for entry in platform.audit.entries(limit=50) if entry["action"] == "inversion_round"
    ]
    assert entries and entries[0]["severity"] == "warning"


def test_incomplete_window_is_partial_failure(platform):
    _feed(platform, 2500.0, count=3)
    report = platform.invert_flow_baseline("test")
    assert report["ok"] is False
    assert "window_incomplete" in report["reasons"]
    assert report["missing"]["window"] == platform.config.inversion.window_size - 3
    assert len(platform.baseline_report()["generations"]) == 1


def test_total_mismatch_is_partial_failure(platform):
    _feed(platform, 2500.0, mismatch=True)
    report = platform.invert_flow_baseline("test")
    assert report["ok"] is False
    assert "total_mismatch" in report["reasons"]
    assert len(platform.baseline_report()["generations"]) == 1


def test_manual_policy_parks_conflicting_generation(platform):
    _feed(platform, 2800.0)
    report = platform.invert_flow_baseline("test")
    assert report["ok"] is True
    assert report["conflict"] is True
    assert report["decision"] == "pending"
    assert report["policy"] == "manual"
    # 迁移挂起期间联锁仍按旧代判。
    assert platform.network.baseline_t_h() == pytest.approx(2400.0)
    assert platform.network.threshold_t_h() == pytest.approx(2220.0)
    status = platform.status()["primary"]
    assert status["generation"] == 1
    assert status["pending_generation"] == 2
    report_view = platform.baseline_report()
    assert report_view["migration_in_progress"] is True
    assert report_view["interlock_generation"] == 1
    assert report_view["pending"]["baseline_t_h"] == pytest.approx(2800.0)


def test_confirm_migration_switches_interlock_generation(platform):
    platform.network.set_flow(2300.0)
    assert platform.network.low_flow() is False
    _feed(platform, 2800.0)
    platform.invert_flow_baseline("test")
    # 挂起代门限 2620 会判 2300 低流量，但联锁只按现役代（2220）判。
    assert platform.network.low_flow() is False

    confirmed = platform.confirm_flow_baseline(True, "tester")
    assert confirmed["accepted"] is True
    assert confirmed["active_generation"] == 2
    assert platform.network.baseline_t_h() == pytest.approx(2800.0)
    assert platform.network.threshold_t_h() == pytest.approx(2620.0)
    assert platform.network.low_flow() is True
    generations = platform.baseline_report()["generations"]
    assert generations[0]["status"] == "retired"
    assert generations[1]["status"] == "active"
    entries = [
        entry
        for entry in platform.audit.entries(limit=50)
        if entry["action"] == "baseline_confirmed"
    ]
    assert entries and entries[0]["payload"]["baseline_t_h"] == pytest.approx(2800.0)
    assert entries[0]["payload"]["threshold_t_h"] == pytest.approx(2620.0)


def test_reject_keeps_active_generation(platform):
    _feed(platform, 2800.0)
    platform.invert_flow_baseline("test")
    rejected = platform.confirm_flow_baseline(False, "tester")
    assert rejected["accepted"] is False
    assert platform.network.baseline_t_h() == pytest.approx(2400.0)
    generations = platform.baseline_report()["generations"]
    assert generations[1]["status"] == "rejected"
    with pytest.raises(InterlockError):
        platform.confirm_flow_baseline(True, "tester")


def test_auto_policy_adopts_conflicting_inversion(tmp_path, clock):
    platform = _auto_platform(tmp_path, clock)
    _feed(platform, 2800.0)
    report = platform.invert_flow_baseline("test")
    assert report["decision"] == "activated"
    assert report["policy"] == "auto"
    assert platform.network.baseline_t_h() == pytest.approx(2800.0)
    assert platform.baseline_report()["migration_in_progress"] is False


def test_newer_round_supersedes_pending_generation(platform):
    _feed(platform, 2800.0)
    platform.invert_flow_baseline("test")
    _feed(platform, 2900.0)
    report = platform.invert_flow_baseline("test")
    assert report["decision"] == "pending"
    generations = platform.baseline_report()["generations"]
    assert generations[1]["status"] == "superseded"
    assert generations[2]["status"] == "pending"
    platform.confirm_flow_baseline(True, "tester")
    assert platform.network.baseline_t_h() == pytest.approx(2900.0)


def test_recalibrate_records_a_manual_generation(platform):
    report = platform.calibrate_flow(2600.0)
    assert report["generation"] == 2
    generations = platform.baseline_report()["generations"]
    assert generations[1]["source"] == "manual"
    assert generations[1]["baseline_t_h"] == pytest.approx(2600.0)
    assert platform.network.threshold_t_h() == pytest.approx(2420.0)


def test_generations_and_window_survive_restart(platform, clock):
    _feed(platform, 2450.0)
    platform.invert_flow_baseline("test")

    restarted = build_platform(default_config(), clock=clock, storage_root=platform.store.root)
    assert restarted.network.baseline_t_h() == pytest.approx(2450.0)
    report = restarted.baseline_report()
    assert report["interlock_generation"] == 2
    assert report["rounds"] == 1
    assert report["window"]["full"] is True
    assert report["window"]["rule"] == "sliding"


def test_audit_threshold_and_console_agree_with_inversion(platform):
    _feed(platform, 2450.0)
    platform.invert_flow_baseline("test")
    entries = [
        entry for entry in platform.audit.entries(limit=50) if entry["action"] == "inversion_round"
    ]
    assert entries
    payload = entries[0]["payload"]
    assert payload["active_baseline_t_h"] == pytest.approx(platform.network.baseline_t_h())
    assert payload["active_threshold_t_h"] == pytest.approx(platform.network.threshold_t_h())

    api = Api(platform)
    status, report = api.handle("GET", "/api/primary/baseline", {}, {})
    assert status == 200
    assert report["active"]["baseline_t_h"] == pytest.approx(platform.network.baseline_t_h())
    assert report["threshold_t_h"] == pytest.approx(platform.network.threshold_t_h())
    assert report["window"]["rule"] == "sliding"
    assert len(report["generations"]) == 2


def test_console_endpoints_cover_sample_invert_and_confirm(platform):
    api = Api(platform)
    per_station = 2800.0 / len(STATIONS)
    for _ in range(platform.config.inversion.window_size):
        status, sample = api.handle(
            "POST",
            "/api/primary/sample",
            {},
            {"total_t_h": 2800.0, "stations": {name: per_station for name in STATIONS}},
        )
        assert status == 200
        assert sample["complete"] is True
    status, inversion = api.handle("POST", "/api/primary/invert", {}, {"origin": "console"})
    assert status == 200
    assert inversion["decision"] == "pending"
    status, confirmed = api.handle(
        "POST", "/api/primary/baseline/confirm", {}, {"accept": True, "operator": "console"}
    )
    assert status == 200
    assert confirmed["baseline_t_h"] == pytest.approx(2800.0)
    status, _ = api.handle("POST", "/api/primary/baseline/confirm", {}, {"accept": True})
    assert status == 409


def test_inversion_config_roundtrip_and_validation():
    config = default_config()
    mapping = config_to_mapping(config)
    assert mapping["limits"]["inversion"]["window_size"] == config.inversion.window_size
    assert type(config).from_mapping(mapping) == config

    broken = config_to_mapping(config)
    broken["limits"]["inversion"]["default_policy"] = "sometimes"
    with pytest.raises(ConfigError):
        type(config).from_mapping(broken)

    broken_window = config_to_mapping(config)
    broken_window["limits"]["inversion"]["window_size"] = 1
    with pytest.raises(ConfigError):
        type(config).from_mapping(broken_window)
