"""一次网基线反演编排。

把实测样本接进滑动窗口、按窗口反演、按回路策略处置冲突、把人工确认落到代际
链上。反演结果（候选基线、门限推荐值、代际迁移）全部来自
:class:`BaselineRegistry` 的同一批对象，平台层原样落审计，保证反演、门限与
审计一致。
"""

from __future__ import annotations

from ..primary.inversion import FlowSample
from ..primary.network import PrimaryNetwork


def collect_flow_sample(
    network: PrimaryNetwork,
    stations: list[str],
    total_t_h: float | None,
    station_flows: dict,
    now: float,
) -> dict:
    """把一次实测（总流量 + 各站分配）接进滑动窗口。"""

    registry = network.baseline_registry()
    expected = [str(name) for name in stations]
    flows = {
        name: (None if station_flows.get(name) is None else float(station_flows[name]))
        for name in expected
    }
    sample = FlowSample(
        ts=float(now),
        total_t_h=None if total_t_h is None else float(total_t_h),
        stations=flows,
    )
    registry.add_sample(sample)
    missing = sample.missing(expected)
    return {
        "sample": sample.as_dict(),
        "missing": missing,
        "complete": not missing,
        "window": registry.window_status(),
    }


def invert_flow_baseline(
    network: PrimaryNetwork,
    stations: list[str],
    now: float,
    origin: str = "console",
) -> dict:
    """按滑动窗口反演一轮基线；实测缺失时本轮记部分失败、不换代。"""

    registry = network.baseline_registry()
    round_ = registry.invert(float(now), [str(name) for name in stations], origin=origin)
    payload = round_.as_dict()
    steps = [{"component": "primary", "action": "invert_baseline", "detail": payload}]
    violations = [] if round_.ok else ["partial_inversion"]
    return {
        "ok": round_.ok,
        "stage": "done" if round_.ok else "inversion",
        "steps": steps,
        "violations": violations,
        **payload,
    }


def confirm_flow_baseline(
    network: PrimaryNetwork,
    accept: bool,
    operator: str,
    now: float,
) -> dict:
    """人工确认或拒绝挂起代；确认后联锁即刻按新代判。"""

    registry = network.baseline_registry()
    migration = registry.confirm(operator, now) if accept else registry.reject(operator, now)
    active = registry.active()
    pending = registry.pending()
    return {
        "ok": True,
        "accepted": bool(accept),
        "operator": str(operator),
        "migration": migration.as_dict(),
        "active_generation": active.generation,
        "baseline_t_h": round(active.baseline_t_h, 2),
        "threshold_t_h": round(active.threshold_t_h, 2),
        "pending_generation": None if pending is None else pending.generation,
    }
