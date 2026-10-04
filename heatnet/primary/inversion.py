"""一次网流量基线的反演与代际管理。

调试期拍下来的固定基线会随管道结垢、换热站增容慢慢偏离实际，联锁门限跟着
失真。这里按实测反演基线，并把规则定死：

- **滑动窗口**：反演只取最近 ``window_size`` 次实测（总流量 + 各站分配），
  不用整段历史——整段历史会把调试期的旧值永久锚进反演结果，正是要解决的
  问题。窗口语义在 :class:`SampleWindow` 里写死，配置只能改窗口长度。
- **一次反演一代**：每一轮数据完整的反演都记成一代
  （:class:`BaselineGeneration`），代际链只进不出，控制台与审计读同一条链。
- **冲突按回路定**：反演值与现役基线之差超过冲突带时，``auto`` 回路直接
  采用反演值，``manual`` 回路把新代挂起等人工确认；不冲突时两种回路都直接
  生效。
- **换代是迁移**：任何时刻联锁只按 ``active`` 代判，挂起（``pending``）代
  不参与联锁；人工确认的那一刻新代原子生效、旧代退役，拒绝则挂起代作废。
  每次换代都落成 :class:`Migration`，谁退谁进有据可查。
- **实测缺失是部分失败**：窗口未满、缺总表、缺站、站合计与总表对不上，本轮
  只落审计，不产生新代、不发生迁移。

反演给出的门限推荐值与联锁门限用同一条公式
（``PrimaryLimits.low_flow_threshold``），基线、门限、审计三处数字同源。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from ..config import InversionLimits, PrimaryLimits
from ..errors import InterlockError, StorageError

# 落链的代际状态；candidate 只在反演进行中出现，不会留在链上。
STATUSES = ("candidate", "active", "pending", "retired", "rejected", "superseded")


@dataclass(frozen=True)
class FlowSample:
    """一次实测：总流量 + 各站分配；缺项用 ``None`` 表示。"""

    ts: float
    total_t_h: float | None
    stations: dict[str, float | None]

    def missing(self, expected: list[str]) -> list[str]:
        """列出缺项：``total`` 表示总表缺失，其余为站名。"""

        missing = ["total"] if self.total_t_h is None else []
        missing.extend(name for name in expected if self.stations.get(name) is None)
        return missing

    def as_dict(self) -> dict:
        return {
            "ts": round(self.ts, 3),
            "total_t_h": None if self.total_t_h is None else round(self.total_t_h, 2),
            "stations": {
                name: (None if value is None else round(float(value), 2))
                for name, value in sorted(self.stations.items())
            },
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "FlowSample":
        stations = {
            str(name): (None if value is None else float(value))
            for name, value in dict(raw.get("stations", {})).items()
        }
        total = raw.get("total_t_h")
        return cls(
            ts=float(raw.get("ts", 0.0)),
            total_t_h=None if total is None else float(total),
            stations=stations,
        )


class SampleWindow:
    """定长滑动窗口：只保留最近 ``size`` 条实测样本，更早的一律挤出。"""

    def __init__(self, size: int) -> None:
        self._size = max(2, int(size))
        self._samples: list[FlowSample] = []

    def append(self, sample: FlowSample) -> None:
        self._samples.append(sample)
        if len(self._samples) > self._size:
            del self._samples[: len(self._samples) - self._size]

    def samples(self) -> list[FlowSample]:
        return list(self._samples)

    def full(self) -> bool:
        return len(self._samples) >= self._size


@dataclass(frozen=True)
class BaselineGeneration:
    """一代基线：取值、门限、来源、状态与反演窗口证据。"""

    generation: int
    loop: str
    baseline_t_h: float
    threshold_t_h: float
    source: str
    status: str
    created_at: float
    shares: dict[str, float] = field(default_factory=dict)
    window: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "generation": self.generation,
            "loop": self.loop,
            "baseline_t_h": round(self.baseline_t_h, 2),
            "threshold_t_h": round(self.threshold_t_h, 2),
            "source": self.source,
            "status": self.status,
            "created_at": round(self.created_at, 3),
            "shares": {name: round(value, 4) for name, value in sorted(self.shares.items())},
            "window": dict(self.window),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "BaselineGeneration":
        return cls(
            generation=int(raw["generation"]),
            loop=str(raw["loop"]),
            baseline_t_h=float(raw["baseline_t_h"]),
            threshold_t_h=float(raw["threshold_t_h"]),
            source=str(raw["source"]),
            status=str(raw["status"]),
            created_at=float(raw["created_at"]),
            shares={str(k): float(v) for k, v in dict(raw.get("shares", {})).items()},
            window=dict(raw.get("window", {})),
        )


@dataclass(frozen=True)
class GenerationShift:
    """一次代际状态迁移。"""

    generation: int
    from_status: str
    to_status: str

    def as_dict(self) -> dict:
        return {"generation": self.generation, "from": self.from_status, "to": self.to_status}


@dataclass(frozen=True)
class Migration:
    """一次换代迁移：落链的新代、全部状态迁移记录、发生时间与来源。"""

    generation: BaselineGeneration
    shifts: tuple[GenerationShift, ...]
    at: float
    operator: str

    def as_dict(self) -> dict:
        return {
            "generation": self.generation.as_dict(),
            "shifts": [shift.as_dict() for shift in self.shifts],
            "at": round(self.at, 3),
            "operator": self.operator,
        }


@dataclass(frozen=True)
class InversionRound:
    """一轮反演的完整结果；部分失败时候选字段为 ``None``。"""

    round: int
    loop: str
    ts: float
    origin: str
    ok: bool
    partial: bool
    reasons: tuple[str, ...]
    missing: dict[str, int]
    window: dict
    candidate_baseline_t_h: float | None
    recommended_threshold_t_h: float | None
    conflict: bool
    policy: str
    decision: str
    migration: Migration | None
    active_generation: int
    active_baseline_t_h: float
    active_threshold_t_h: float
    shares: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "round": self.round,
            "loop": self.loop,
            "ts": round(self.ts, 3),
            "origin": self.origin,
            "ok": self.ok,
            "partial": self.partial,
            "reasons": list(self.reasons),
            "missing": dict(self.missing),
            "window": dict(self.window),
            "candidate_baseline_t_h": (
                None if self.candidate_baseline_t_h is None else round(self.candidate_baseline_t_h, 2)
            ),
            "recommended_threshold_t_h": (
                None
                if self.recommended_threshold_t_h is None
                else round(self.recommended_threshold_t_h, 2)
            ),
            "conflict": self.conflict,
            "policy": self.policy,
            "decision": self.decision,
            "migration": None if self.migration is None else self.migration.as_dict(),
            "active_generation": self.active_generation,
            "active_baseline_t_h": round(self.active_baseline_t_h, 2),
            "active_threshold_t_h": round(self.active_threshold_t_h, 2),
            "station_shares": {
                name: round(value, 4) for name, value in sorted(self.shares.items())
            },
        }


class BaselineRegistry:
    """单回路的基线代际登记处：滑动窗口、代际链与迁移。"""

    def __init__(
        self,
        loop: str,
        limits: InversionLimits,
        primary: PrimaryLimits,
        seed_baseline_t_h: float,
    ) -> None:
        self._loop = str(loop)
        self._limits = limits
        self._primary = primary
        self._window = SampleWindow(limits.window_size)
        self._rounds = 0
        seed = BaselineGeneration(
            generation=1,
            loop=self._loop,
            baseline_t_h=float(seed_baseline_t_h),
            threshold_t_h=primary.low_flow_threshold(seed_baseline_t_h),
            source="commissioning",
            status="active",
            created_at=0.0,
        )
        self._generations: list[BaselineGeneration] = [seed]

    def loop(self) -> str:
        """返回本登记处负责的回路名。"""

        return self._loop

    def policy(self) -> str:
        """返回本回路的冲突处置策略（按回路定）。"""

        return self._limits.policy_for(self._loop)

    def rounds(self) -> int:
        """返回累计反演轮数（含部分失败的轮次）。"""

        return self._rounds

    def generations(self) -> list[BaselineGeneration]:
        """返回整条代际链（只进不出）。"""

        return list(self._generations)

    def active(self) -> BaselineGeneration:
        """返回现役代——联锁只按这一代判。"""

        return next(g for g in self._generations if g.status == "active")

    def pending(self) -> BaselineGeneration | None:
        """返回挂起待确认的代（迁移进行中），没有则 ``None``。"""

        return next((g for g in self._generations if g.status == "pending"), None)

    def window_status(self) -> dict:
        """返回滑动窗口现状，供控制台展示。"""

        samples = self._window.samples()
        return {
            "rule": "sliding",
            "size": len(samples),
            "required": self._limits.window_size,
            "full": self._window.full(),
            "oldest_ts": round(samples[0].ts, 3) if samples else None,
            "newest_ts": round(samples[-1].ts, 3) if samples else None,
        }

    def add_sample(self, sample: FlowSample) -> None:
        """把一次实测接进滑动窗口。"""

        self._window.append(sample)

    # ---- 反演 ---------------------------------------------------------

    def invert(self, now: float, expected: list[str], origin: str = "console") -> InversionRound:
        """按滑动窗口反演一轮基线；实测缺失时本轮不换代。"""

        self._rounds += 1
        samples = self._window.samples()
        reasons: list[str] = []
        missing: dict[str, int] = {}
        if not self._window.full():
            reasons.append("window_incomplete")
            missing["window"] = self._limits.window_size - len(samples)
        complete: list[FlowSample] = []
        for sample in samples:
            lack = sample.missing(expected)
            if lack:
                for name in lack:
                    missing[name] = missing.get(name, 0) + 1
            else:
                complete.append(sample)
        if any(name != "window" for name in missing):
            reasons.append("missing_measurements")
        mismatch = 0
        for sample in complete:
            total = float(sample.total_t_h)
            summed = sum(float(sample.stations[name]) for name in expected)
            if abs(summed - total) > self._limits.station_sum_tolerance_t_h:
                mismatch += 1
        if mismatch:
            reasons.append("total_mismatch")
            missing["total_mismatch"] = mismatch
        window_info = self.window_status()
        active = self.active()
        if reasons:
            return InversionRound(
                round=self._rounds,
                loop=self._loop,
                ts=float(now),
                origin=str(origin),
                ok=False,
                partial=True,
                reasons=tuple(reasons),
                missing=missing,
                window=window_info,
                candidate_baseline_t_h=None,
                recommended_threshold_t_h=None,
                conflict=False,
                policy=self.policy(),
                decision="none",
                migration=None,
                active_generation=active.generation,
                active_baseline_t_h=active.baseline_t_h,
                active_threshold_t_h=active.threshold_t_h,
                shares={},
            )

        candidate = max(
            1.0, sum(float(sample.total_t_h) for sample in complete) / len(complete)
        )
        shares = {
            name: sum(float(sample.stations[name]) for sample in complete) / len(complete) / candidate
            for name in expected
        }
        recommended = self._primary.low_flow_threshold(candidate)
        conflict = abs(candidate - active.baseline_t_h) > self._limits.conflict_band_t_h
        generation = self._issue(candidate, recommended, "inversion", now, shares, window_info)
        if conflict and self.policy() == "manual":
            migration = self._park(generation, now, origin)
            decision = "pending"
        else:
            migration = self._activate(generation, now, origin)
            decision = "activated"
        current = self.active()
        return InversionRound(
            round=self._rounds,
            loop=self._loop,
            ts=float(now),
            origin=str(origin),
            ok=True,
            partial=False,
            reasons=(),
            missing={},
            window=window_info,
            candidate_baseline_t_h=candidate,
            recommended_threshold_t_h=recommended,
            conflict=conflict,
            policy=self.policy(),
            decision=decision,
            migration=migration,
            active_generation=current.generation,
            active_baseline_t_h=current.baseline_t_h,
            active_threshold_t_h=current.threshold_t_h,
            shares=shares,
        )

    # ---- 迁移 ---------------------------------------------------------

    def confirm(self, operator: str, now: float) -> Migration:
        """人工确认挂起代：新代原子生效，旧代退役。"""

        pending = self.pending()
        if pending is None:
            raise InterlockError(f"回路 {self._loop} 没有待确认的基线代", component="primary")
        return self._activate(pending, now, operator)

    def reject(self, operator: str, now: float) -> Migration:
        """人工拒绝挂起代：挂起代作废，现役代不变。"""

        pending = self.pending()
        if pending is None:
            raise InterlockError(f"回路 {self._loop} 没有待确认的基线代", component="primary")
        rejected = self._set_status(pending, "rejected")
        return Migration(
            generation=rejected,
            shifts=(GenerationShift(pending.generation, "pending", "rejected"),),
            at=float(now),
            operator=str(operator),
        )

    def manual_baseline(self, measured_t_h: float, now: float) -> Migration:
        """人工标定直接落一代并生效（人工下发本身就是确认）。"""

        baseline = max(1.0, float(measured_t_h))
        generation = self._issue(
            baseline, self._primary.low_flow_threshold(baseline), "manual", now, {}, {}
        )
        return self._activate(generation, now, "manual")

    # ---- 落盘 ---------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "loop": self._loop,
            "rounds": self._rounds,
            "window": {
                "size": self._limits.window_size,
                "samples": [sample.as_dict() for sample in self._window.samples()],
            },
            "generations": [generation.as_dict() for generation in self._generations],
        }

    def restore(self, data: dict) -> None:
        """从落盘状态恢复代际链与滑动窗口；链不完整时拒绝恢复。"""

        generations = [BaselineGeneration.from_dict(item) for item in data.get("generations", [])]
        if not generations:
            raise StorageError("基线代际状态缺少 generations", component="primary")
        for generation in generations:
            if generation.status not in STATUSES:
                raise StorageError(
                    f"基线代 {generation.generation} 状态非法: {generation.status}",
                    component="primary",
                )
        if len([g for g in generations if g.status == "active"]) != 1:
            raise StorageError("基线代际状态必须恰好一个 active 代", component="primary")
        if len([g for g in generations if g.status == "pending"]) > 1:
            raise StorageError("基线代际状态至多一个 pending 代", component="primary")
        self._generations = generations
        self._rounds = int(data.get("rounds", 0))
        self._window = SampleWindow(self._limits.window_size)
        for item in dict(data.get("window", {})).get("samples", []):
            self._window.append(FlowSample.from_dict(item))

    # ---- 内部 ---------------------------------------------------------

    def _issue(
        self,
        baseline_t_h: float,
        threshold_t_h: float,
        source: str,
        now: float,
        shares: dict[str, float],
        window_info: dict,
    ) -> BaselineGeneration:
        generation = BaselineGeneration(
            generation=self._generations[-1].generation + 1,
            loop=self._loop,
            baseline_t_h=float(baseline_t_h),
            threshold_t_h=float(threshold_t_h),
            source=source,
            status="candidate",
            created_at=float(now),
            shares=dict(shares),
            window=dict(window_info),
        )
        self._generations.append(generation)
        return generation

    def _set_status(self, generation: BaselineGeneration, status: str) -> BaselineGeneration:
        updated = dataclasses.replace(generation, status=status)
        self._generations = [
            updated if item.generation == generation.generation else item
            for item in self._generations
        ]
        return updated

    def _activate(
        self, generation: BaselineGeneration, now: float, operator: str
    ) -> Migration:
        shifts: list[GenerationShift] = []
        parked = self.pending()
        if parked is not None and parked.generation != generation.generation:
            self._set_status(parked, "superseded")
            shifts.append(GenerationShift(parked.generation, "pending", "superseded"))
        current = self.active()
        if current.generation != generation.generation:
            self._set_status(current, "retired")
            shifts.append(GenerationShift(current.generation, "active", "retired"))
        before = generation.status
        activated = self._set_status(generation, "active")
        shifts.append(GenerationShift(generation.generation, before, "active"))
        return Migration(
            generation=activated, shifts=tuple(shifts), at=float(now), operator=str(operator)
        )

    def _park(self, generation: BaselineGeneration, now: float, operator: str) -> Migration:
        shifts: list[GenerationShift] = []
        parked = self.pending()
        if parked is not None:
            self._set_status(parked, "superseded")
            shifts.append(GenerationShift(parked.generation, "pending", "superseded"))
        before = generation.status
        held = self._set_status(generation, "pending")
        shifts.append(GenerationShift(generation.generation, before, "pending"))
        return Migration(
            generation=held, shifts=tuple(shifts), at=float(now), operator=str(operator)
        )
