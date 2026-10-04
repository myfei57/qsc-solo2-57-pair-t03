"""一次网基线的滑动窗口反演、代际迁移与人工裁决。"""

from __future__ import annotations

import pytest

from heatnet.app import build_platform
from heatnet.clock import ManualClock
from heatnet.config import PrimaryLimits, default_config
from heatnet.console.api import Api
from heatnet.errors import InterlockError
from heatnet.primary.baseline import BaselineLedger, StationReading
from heatnet.primary.network import PrimaryNetwork


def _readings(flow_t_h: float, demand_pct: float = 60.0, count: int = 3) -> list[StationReading]:
    """构造一组自洽的各站实测：分站合计等于总表。"""

    per_station = flow_t_h * demand_pct / 100.0 / count
    return [
        StationReading(station=f"st{i:02d}", flow_t_h=per_station, demand_pct=demand_pct)
        for i in range(1, count + 1)
    ]


def _total(flow_t_h: float, demand_pct: float = 60.0) -> float:
    """与 _readings 配套的全网实测总流量。"""

    return flow_t_h * demand_pct / 100.0


def _ledger(window: int = 3, adoption: str = "manual") -> BaselineLedger:
    limits = PrimaryLimits(2400.0, 180.0, 4.0, adoption, window)
    return BaselineLedger(limits)


def test_inversion_matches_dispatch_formula():
    """反演公式与调度下发互逆：满负荷基线 = 实测总流量 / 平均负荷率。"""

    ledger = _ledger()
    outcome = ledger.run_round(_total(2400.0), _readings(2400.0), ts=1.0)
    assert outcome.round.ok is True
    assert outcome.round.implied_t_h == pytest.approx(2400.0)
    for value in outcome.round.station_implied.values():
        assert value == pytest.approx(2400.0)


def test_complete_round_mints_a_generation_and_migrates():
    """每一次完整反演记成一代；偏差不越容差时新代直接生效。"""

    ledger = _ledger()
    outcome = ledger.run_round(_total(2450.0), _readings(2450.0), ts=1.0)
    assert outcome.decision == "adopted"
    assert outcome.migration is not None
    assert outcome.migration.reason == "no-conflict"
    assert outcome.migration.previous.index == 0
    assert outcome.migration.previous.source == "config"
    assert ledger.generation_index() == 1
    assert ledger.baseline_t_h() == pytest.approx(2450.0)
    # 门限与基线同源：基线 − 容差。
    assert ledger.threshold_t_h() == pytest.approx(2450.0 - 180.0)


def test_sliding_window_is_fixed_and_drops_oldest_samples():
    """反演按定死的滑动窗口：窗口外的旧样本不再参与候选。"""

    ledger = _ledger(window=3, adoption="auto")
    for step, implied in enumerate((2400.0, 2400.0, 2400.0, 3000.0), start=1):
        ledger.run_round(_total(implied), _readings(implied), ts=float(step))
    # 窗口 [2400, 2400, 3000]，最早的 2400 已被挤出。
    assert ledger.baseline_t_h() == pytest.approx((2400.0 + 2400.0 + 3000.0) / 3.0)
    samples = ledger.describe()["window"]
    assert samples["size"] == 3
    assert len(samples["samples"]) == 3


def test_missing_station_reading_is_partial_and_never_migrates():
    """实测缺失属于部分失败：不进窗口、不产代、不换代。"""

    ledger = _ledger()
    readings = _readings(2400.0)
    readings[1] = StationReading(station="st02", flow_t_h=None, demand_pct=60.0)
    outcome = ledger.run_round(_total(2400.0), readings, ts=1.0)
    assert outcome.decision == "partial"
    assert outcome.round.ok is False
    assert outcome.round.reason == "missing-readings"
    assert "st02.flow_t_h" in outcome.round.missing
    assert outcome.generation is None
    assert ledger.generation_index() == 0
    assert ledger.baseline_t_h() == pytest.approx(2400.0)
    assert ledger.partial_rounds() == 1
    assert ledger.describe()["window"]["samples"] == []


def test_missing_total_flow_is_partial():
    """总表缺失同样是部分失败，这一轮不许换代。"""

    ledger = _ledger()
    outcome = ledger.run_round(None, _readings(2400.0), ts=1.0)
    assert outcome.decision == "partial"
    assert "total_flow_t_h" in outcome.round.missing
    assert ledger.generation_index() == 0


def test_imbalanced_readings_are_rejected_without_migration():
    """分站合计与总表对不上：实测自相矛盾，本轮作废。"""

    ledger = _ledger()
    readings = _readings(2400.0)
    outcome = ledger.run_round(_total(2400.0) + 500.0, readings, ts=1.0)
    assert outcome.decision == "partial"
    assert outcome.round.reason == "imbalanced"
    assert outcome.round.imbalance_t_h == pytest.approx(500.0)
    assert ledger.generation_index() == 0


def test_conflict_pends_under_manual_policy_and_interlock_stays_on_active():
    """manual 回路冲突时挂起：迁移期间联锁仍按现役代判。"""

    ledger = _ledger(adoption="manual")
    outcome = ledger.run_round(_total(3000.0), _readings(3000.0), ts=1.0)
    assert outcome.decision == "pending"
    assert outcome.migration is None
    pending = ledger.pending()
    assert pending is not None and pending.index == 1
    # 联锁口径不变：基线与门限仍是第 0 代。
    assert ledger.generation_index() == 0
    assert ledger.baseline_t_h() == pytest.approx(2400.0)
    assert ledger.threshold_t_h() == pytest.approx(2400.0 - 180.0)


def test_confirm_migrates_pending_generation():
    """人工确认后换代：联锁改按新代判，旧代退场。"""

    ledger = _ledger(adoption="manual")
    ledger.run_round(_total(3000.0), _readings(3000.0), ts=1.0)
    decision = ledger.decide(True)
    assert decision.accepted is True
    assert decision.migration is not None
    assert decision.migration.reason == "confirmed"
    assert decision.migration.previous.index == 0
    assert ledger.generation_index() == 1
    assert ledger.baseline_t_h() == pytest.approx(3000.0)
    assert ledger.threshold_t_h() == pytest.approx(3000.0 - 180.0)
    assert ledger.pending() is None
    statuses = {item["index"]: item["status"] for item in ledger.describe()["generations"]}
    assert statuses[0] == "superseded"
    assert statuses[1] == "active"


def test_reject_keeps_active_generation():
    """人工否决后现役代不变，被否的代留档为 rejected。"""

    ledger = _ledger(adoption="manual")
    ledger.run_round(_total(3000.0), _readings(3000.0), ts=1.0)
    decision = ledger.decide(False)
    assert decision.accepted is False
    assert decision.migration is None
    assert ledger.generation_index() == 0
    assert ledger.baseline_t_h() == pytest.approx(2400.0)
    assert ledger.pending() is None
    statuses = {item["index"]: item["status"] for item in ledger.describe()["generations"]}
    assert statuses[1] == "rejected"


def test_decide_without_pending_is_an_interlock_error():
    ledger = _ledger()
    with pytest.raises(InterlockError):
        ledger.decide(True)


def test_new_round_supersedes_stale_pending():
    """待确认期间来了新一轮反演：旧待确认代被取代，人工只对最新代裁决。"""

    ledger = _ledger(adoption="manual")
    ledger.run_round(_total(3000.0), _readings(3000.0), ts=1.0)
    ledger.run_round(_total(3100.0), _readings(3100.0), ts=2.0)
    pending = ledger.pending()
    assert pending is not None and pending.index == 2
    statuses = {item["index"]: item["status"] for item in ledger.describe()["generations"]}
    assert statuses[1] == "superseded"
    assert statuses[2] == "pending"


def test_auto_policy_adopts_conflicting_candidate_immediately():
    """auto 回路冲突时直接采用反演值换代。"""

    ledger = _ledger(adoption="auto")
    outcome = ledger.run_round(_total(3000.0), _readings(3000.0), ts=1.0)
    assert outcome.decision == "adopted"
    assert outcome.migration is not None
    assert outcome.migration.reason == "auto"
    assert ledger.generation_index() == 1
    assert ledger.baseline_t_h() == pytest.approx(3000.0)


def test_recalibrate_is_a_generation_and_restarts_window():
    """手工标定同样记一代并强制生效，旧工况样本不再参与反演。"""

    network = PrimaryNetwork(default_config().primary)
    network.invert(_total(2400.0), _readings(2400.0), 1.0)
    migration = network.recalibrate(2500.0, 2.0)
    assert migration.reason == "recalibrate"
    assert migration.generation.source == "recalibrate"
    assert network.generation() == 2
    assert network.baseline_t_h() == pytest.approx(2500.0)
    assert network.recalibrations() == 1
    assert network.baseline_report()["window"]["samples"] == [2500.0]


def test_generations_survive_restart(platform, clock):
    """代际账落盘：重启后联锁仍按现役代判，待确认代也在。"""

    platform.start()
    platform.invert_baseline(_total(3000.0), [r.__dict__ for r in _readings(3000.0)])
    assert platform.network.generation() == 0  # manual 回路，冲突挂起

    restarted = build_platform(default_config(), clock=clock, storage_root=platform.store.root)
    report = restarted.baseline_report()
    assert report["pending"]["index"] == 1
    assert report["rounds"] == 1
    decision = restarted.decide_baseline(True, "night-shift")
    assert decision["accepted"] is True
    assert restarted.network.baseline_t_h() == pytest.approx(3000.0)

    again = build_platform(default_config(), clock=clock, storage_root=platform.store.root)
    assert again.network.generation() == 1
    assert again.network.baseline_t_h() == pytest.approx(3000.0)
    assert again.network.threshold_t_h() == pytest.approx(3000.0 - 180.0)


def test_audit_trail_matches_generations(platform):
    """反演、门限与审计一致：事件里的代际字段与代际账同源。"""

    platform.start()
    platform.invert_baseline(_total(3000.0), [r.__dict__ for r in _readings(3000.0)])
    platform.decide_baseline(True, "night-shift")
    platform.invert_baseline(None, [])  # 部分失败轮

    entries = platform.audit.entries(limit=50)
    migrations = [item for item in entries if item["action"] == "baseline_migration"]
    assert len(migrations) == 1
    payload = migrations[0]["payload"]
    assert payload["from_generation"] == 0
    assert payload["to_generation"] == 1
    assert payload["threshold_t_h"] == pytest.approx(payload["baseline_t_h"] - 180.0)

    rounds = [item for item in entries if item["action"] == "baseline_round"]
    assert len(rounds) == 2
    assert rounds[0]["severity"] == "warning"  # 最新的在前：部分失败轮
    assert rounds[0]["payload"]["decision"] == "partial"
    assert rounds[1]["payload"]["decision"] == "pending"

    pending_events = [item for item in entries if item["action"] == "baseline_pending"]
    assert len(pending_events) == 1
    assert pending_events[0]["severity"] == "warning"
    decisions = [item for item in entries if item["action"] == "baseline_decision"]
    assert decisions[0]["payload"]["accepted"] is True

    report = platform.baseline_report()
    assert report["active"]["index"] == 1
    assert report["partial_rounds"] == 1
    assert report["generations"][0]["index"] == 1  # 最新代在前


def test_partial_failure_round_does_not_persist_a_generation(platform):
    """部分失败轮不落代：重启后代际账里没有它。"""

    platform.start()
    report = platform.invert_baseline(None, [])
    assert report["ok"] is False
    assert report["decision"] == "partial"
    assert platform.network.generation() == 0

    restarted = build_platform(
        default_config(), clock=ManualClock(), storage_root=platform.store.root
    )
    assert restarted.network.generation() == 0
    assert restarted.baseline_report()["partial_rounds"] == 1


def test_api_baseline_endpoints(platform):
    """控制台：代际报告、反演触发与人工裁决三个接口。"""

    api = Api(platform)
    status, report = api.handle("GET", "/api/primary/baseline", {}, {})
    assert status == 200
    assert report["active"]["index"] == 0
    assert report["policy"] == "manual"
    assert report["window"]["size"] == 8

    stations = [dict(r.__dict__) for r in _readings(3000.0)]
    status, inverted = api.handle(
        "POST",
        "/api/primary/baseline/invert",
        {},
        {"total_flow_t_h": _total(3000.0), "stations": stations},
    )
    assert status == 200
    assert inverted["ok"] is True
    assert inverted["decision"] == "pending"
    assert inverted["active_generation"] == 0

    status, decided = api.handle(
        "POST", "/api/primary/baseline/decide", {}, {"accept": True, "operator": "console"}
    )
    assert status == 200
    assert decided["accepted"] is True
    assert decided["active_generation"] == 1

    status, report = api.handle("GET", "/api/primary/baseline", {}, {})
    assert status == 200
    assert report["active"]["index"] == 1
    assert report["pending"] is None

    # 没有待确认代时裁决是联锁错误，控制台转成 409。
    status, payload = api.handle("POST", "/api/primary/baseline/decide", {}, {"accept": True})
    assert status == 409
    assert payload["component"] == "primary"


def test_api_invert_with_missing_readings_is_partial(platform):
    """控制台入口同样守部分失败：缺实测不换代。"""

    api = Api(platform)
    status, inverted = api.handle(
        "POST",
        "/api/primary/baseline/invert",
        {},
        {"stations": [{"station": "st01", "demand_pct": 60.0}]},
    )
    assert status == 200
    assert inverted["ok"] is False
    assert inverted["decision"] == "partial"
    assert "total_flow_t_h" in inverted["round"]["missing"]
    assert "st01.flow_t_h" in inverted["round"]["missing"]
    assert platform.network.generation() == 0


def test_status_exposes_generation_and_pending(platform):
    """状态快照带现役代序号，待确认代在状态里可见。"""

    platform.start()
    status = platform.status()
    assert status["primary"]["generation"] == 0
    assert status["primary"]["pending_generation"] is None
    platform.invert_baseline(_total(3000.0), [r.__dict__ for r in _readings(3000.0)])
    status = platform.status()
    assert status["primary"]["generation"] == 0
    assert status["primary"]["pending_generation"] == 1
