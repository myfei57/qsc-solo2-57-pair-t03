"""一次网流量基线的滑动窗口反演与代际管理。

基线不再只是调试时拍下的固定值：每一轮反演拿实测总流量与各站分配（流量 +
目标开度）反推隐含基线，喂进**定死的滑动窗口**（``baseline_window`` 轮有效
样本，不用整段历史——调试期的旧值会永远稀释结垢、增容后的新工况），窗口
均值就是这一轮的候选基线。反演公式与调度下发互逆：下发时每站目标流量是
``基线 × 开度 / 站数``，反演时由实测流量反除回来；总表与分站合计对不上
（偏差越容差）说明实测自相矛盾，本轮作废。

每一次完整反演记成一代（``BaselineGeneration``），联锁门限与基线同源派生
（基线 − 流量容差），控制台展示与审计流水读的都是这同一份代记录。换代规则
（定死）：

- 候选与现役基线偏差不越容差：新代直接生效；
- 偏差越容差（冲突）：按本回路配置的 ``baseline_adoption`` 策略——``auto``
  直接采用反演值，``manual`` 挂起新代等人工裁决（``decide``）；
- 换代是迁移：联锁永远只按 **active 代** 判，pending 代不参与联锁；切换是
  现役指针的一次原子替换，迁移期间不存在"新旧各判一半"的中间态；
- 实测缺失（总表或任一站）或总分对不上：本轮是部分失败，不进窗口、不产代、
  不换代，只把失败原因记清供审计。
"""

from __future__ import annotations

import dataclasses
from collections import deque
from dataclasses import dataclass

from ..config import PrimaryLimits
from ..errors import InterlockError

# 代际历史只保留最近若干代，更早的完整流水在审计日志里。
HISTORY_LIMIT = 64

# 代状态：现役 / 待确认 / 被取代 / 被否决。
STATUS_ACTIVE = "active"
STATUS_PENDING = "pending"
STATUS_SUPERSEDED = "superseded"
STATUS_REJECTED = "rejected"


@dataclass(frozen=True)
class StationReading:
    """一个站的一次实测：流量与目标开度；``None`` 表示该路实测缺失。"""

    station: str
    flow_t_h: float | None
    demand_pct: float | None


@dataclass(frozen=True)
class InversionRound:
    """一轮反演的结果：无论成败都完整记录，供审计与控制台核对。"""

    ok: bool
    reason: str  # "ok" | "missing-readings" | "imbalanced"
    missing: tuple[str, ...]
    implied_t_h: float | None
    imbalance_t_h: float
    station_implied: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "missing": list(self.missing),
            "implied_t_h": None if self.implied_t_h is None else round(self.implied_t_h, 2),
            "imbalance_t_h": round(self.imbalance_t_h, 2),
            "station_implied": {
                station: round(value, 2)
                for station, value in sorted(self.station_implied.items())
            },
        }


@dataclass(frozen=True)
class BaselineGeneration:
    """一代基线：反演基线与联锁门限的同源记录。"""

    index: int
    baseline_t_h: float
    threshold_t_h: float
    samples: int
    source: str  # "config" | "inversion" | "recalibrate"
    status: str  # active / pending / superseded / rejected
    created_ts: float

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "baseline_t_h": round(self.baseline_t_h, 2),
            "threshold_t_h": round(self.threshold_t_h, 2),
            "samples": self.samples,
            "source": self.source,
            "status": self.status,
            "created_ts": round(self.created_ts, 3),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "BaselineGeneration":
        """从落盘字段还原一代。"""

        return cls(
            index=int(raw.get("index", 0)),
            baseline_t_h=float(raw.get("baseline_t_h", 0.0)),
            threshold_t_h=float(raw.get("threshold_t_h", 0.0)),
            samples=int(raw.get("samples", 0)),
            source=str(raw.get("source", "inversion")),
            status=str(raw.get("status", STATUS_SUPERSEDED)),
            created_ts=float(raw.get("created_ts", 0.0)),
        )


@dataclass(frozen=True)
class Migration:
    """一次换代：哪一代换下哪一代、为什么。"""

    generation: BaselineGeneration
    previous: BaselineGeneration
    reason: str  # "no-conflict" | "auto" | "confirmed" | "recalibrate"

    def as_dict(self) -> dict:
        return {
            "from_generation": self.previous.as_dict(),
            "to_generation": self.generation.as_dict(),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class InversionOutcome:
    """一轮反演落账后的结果：产没产代、换没换代、是不是在等人工。"""

    round: InversionRound
    decision: str  # "partial" | "adopted" | "pending"
    generation: BaselineGeneration | None
    migration: Migration | None

    def as_dict(self) -> dict:
        return {
            "decision": self.decision,
            "round": self.round.as_dict(),
            "generation": None if self.generation is None else self.generation.as_dict(),
            "migration": None if self.migration is None else self.migration.as_dict(),
        }


@dataclass(frozen=True)
class Decision:
    """人工对待确认代的裁决结果。"""

    accepted: bool
    generation: BaselineGeneration
    migration: Migration | None

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "generation": self.generation.as_dict(),
            "migration": None if self.migration is None else self.migration.as_dict(),
        }


class BaselineLedger:
    """一次网基线的代际账：滑动窗口反演、代际迁移与人工裁决。"""

    def __init__(self, limits: PrimaryLimits) -> None:
        self._limits = limits
        self._window: deque[float] = deque(maxlen=limits.baseline_window)
        initial = BaselineGeneration(
            index=0,
            baseline_t_h=limits.flow_baseline_t_h,
            threshold_t_h=limits.low_flow_threshold(limits.flow_baseline_t_h),
            samples=0,
            source="config",
            status=STATUS_ACTIVE,
            created_ts=0.0,
        )
        self._generations: list[BaselineGeneration] = [initial]
        self._active = initial
        self._pending: BaselineGeneration | None = None
        self._next_index = 1
        self._rounds = 0
        self._partial = 0

    # ---- 查询 -------------------------------------------------

    def active(self) -> BaselineGeneration:
        """返回现役代（联锁唯一依据）。"""

        return self._active

    def pending(self) -> BaselineGeneration | None:
        """返回待人工确认的代，没有则返回 ``None``。"""

        return self._pending

    def baseline_t_h(self) -> float:
        """返回现役代基线。"""

        return self._active.baseline_t_h

    def threshold_t_h(self) -> float:
        """返回现役代的低流量联锁门限。"""

        return self._active.threshold_t_h

    def generation_index(self) -> int:
        """返回现役代序号。"""

        return self._active.index

    def rounds(self) -> int:
        """返回累计反演轮数（含部分失败轮）。"""

        return self._rounds

    def partial_rounds(self) -> int:
        """返回累计部分失败轮数。"""

        return self._partial

    # ---- 反演 -------------------------------------------------

    def run_round(
        self,
        total_flow_t_h: float | None,
        stations: list[StationReading],
        ts: float,
    ) -> InversionOutcome:
        """跑一轮反演：完整轮产一代并按策略落状态，部分失败轮只记账不换代。"""

        self._rounds += 1
        round_ = self._invert(total_flow_t_h, stations)
        if not round_.ok:
            self._partial += 1
            return InversionOutcome(round=round_, decision="partial", generation=None, migration=None)

        self._window.append(float(round_.implied_t_h))
        candidate = sum(self._window) / len(self._window)
        generation = self._mint(candidate, samples=len(self._window), source="inversion", ts=ts)
        conflict = (
            abs(candidate - self._active.baseline_t_h) > self._limits.flow_tolerance_t_h
        )
        if not conflict:
            migration = self._migrate(generation, "no-conflict")
            return InversionOutcome(round_, "adopted", migration.generation, migration)
        if self._limits.baseline_adoption == "auto":
            migration = self._migrate(generation, "auto")
            return InversionOutcome(round_, "adopted", migration.generation, migration)
        self._pend(generation)
        return InversionOutcome(round_, "pending", generation, None)

    def decide(self, accept: bool) -> Decision:
        """人工裁决待确认代：确认即迁移生效，否决即作废。"""

        if self._pending is None:
            raise InterlockError("没有待确认的基线代", component="primary")
        pending = self._pending
        if accept:
            migration = self._migrate(pending, "confirmed")
            return Decision(True, migration.generation, migration)
        rejected = dataclasses.replace(pending, status=STATUS_REJECTED)
        self._upsert(rejected)
        self._pending = None
        return Decision(False, rejected, None)

    def recalibrate(self, measured_t_h: float, ts: float) -> Migration:
        """手工标定：强制换代并重启窗口（旧工况样本不再参与反演）。"""

        baseline = max(1.0, float(measured_t_h))
        self._window.clear()
        self._window.append(baseline)
        generation = self._mint(baseline, samples=1, source="recalibrate", ts=ts)
        if self._pending is not None:
            self._upsert(dataclasses.replace(self._pending, status=STATUS_SUPERSEDED))
            self._pending = None
        return self._migrate(generation, "recalibrate")

    # ---- 汇总与持久化 ------------------------------------------

    def describe(self) -> dict:
        """汇总代际账，供控制台展示（与审计同源的字段口径）。"""

        pending = self.pending()
        return {
            "policy": self._limits.baseline_adoption,
            "window": {
                "size": self._limits.baseline_window,
                "samples": [round(value, 2) for value in self._window],
            },
            "active": self.active().as_dict(),
            "pending": None if pending is None else pending.as_dict(),
            "rounds": self.rounds(),
            "partial_rounds": self.partial_rounds(),
            "generations": [item.as_dict() for item in reversed(self._generations)],
        }

    def export(self) -> dict:
        """导出代际账，供落盘；重启后按此恢复，联锁仍按现役代判。"""

        return {
            "next_index": self._next_index,
            "rounds": self._rounds,
            "partial_rounds": self._partial,
            "window": [round(value, 4) for value in self._window],
            "generations": [item.as_dict() for item in self._generations],
        }

    def restore(self, payload: dict) -> None:
        """从落盘状态恢复代际账；没有唯一现役代时放弃恢复，保持出厂代。"""

        generations = [
            BaselineGeneration.from_dict(item)
            for item in payload.get("generations", [])
            if isinstance(item, dict)
        ]
        actives = [item for item in generations if item.status == STATUS_ACTIVE]
        if not generations or len(actives) != 1:
            return
        pendings = [item for item in generations if item.status == STATUS_PENDING]
        self._generations = generations[-HISTORY_LIMIT:]
        self._active = actives[0]
        self._pending = pendings[-1] if pendings else None
        self._next_index = max(
            int(payload.get("next_index", 0)),
            max(item.index for item in generations) + 1,
        )
        self._window = deque(
            (float(value) for value in payload.get("window", [])),
            maxlen=self._limits.baseline_window,
        )
        self._rounds = int(payload.get("rounds", 0))
        self._partial = int(payload.get("partial_rounds", 0))

    # ---- 内部 -------------------------------------------------

    def _invert(
        self, total_flow_t_h: float | None, stations: list[StationReading]
    ) -> InversionRound:
        """纯反演：校验实测完整性与自洽性，反推隐含基线，不碰任何状态。"""

        readings = list(stations)
        missing: list[str] = []
        total = None if total_flow_t_h is None else float(total_flow_t_h)
        if total is None or total <= 0.0:
            missing.append("total_flow_t_h")
        if not readings:
            missing.append("stations")
        for reading in readings:
            if reading.flow_t_h is None or float(reading.flow_t_h) <= 0.0:
                missing.append(f"{reading.station}.flow_t_h")
            if reading.demand_pct is None or float(reading.demand_pct) <= 0.0:
                missing.append(f"{reading.station}.demand_pct")
        if missing:
            return InversionRound(False, "missing-readings", tuple(missing), None, 0.0, {})

        summed = sum(float(reading.flow_t_h) for reading in readings)
        imbalance = abs(summed - total)
        station_implied = {
            reading.station: float(reading.flow_t_h) * 100.0 * len(readings) / float(reading.demand_pct)
            for reading in readings
        }
        if imbalance > self._limits.flow_tolerance_t_h:
            return InversionRound(False, "imbalanced", (), None, imbalance, station_implied)
        mean_demand = sum(float(reading.demand_pct) for reading in readings) / len(readings)
        implied = total * 100.0 / mean_demand
        return InversionRound(True, "ok", (), implied, imbalance, station_implied)

    def _mint(self, baseline_t_h: float, *, samples: int, source: str, ts: float) -> BaselineGeneration:
        """铸一代新基线（初始为待确认状态，由迁移或挂起落最终状态）。"""

        generation = BaselineGeneration(
            index=self._next_index,
            baseline_t_h=float(baseline_t_h),
            threshold_t_h=self._limits.low_flow_threshold(baseline_t_h),
            samples=samples,
            source=source,
            status=STATUS_PENDING,
            created_ts=float(ts),
        )
        self._next_index += 1
        return generation

    def _pend(self, generation: BaselineGeneration) -> None:
        """把新代挂起待确认；上一轮没裁决的待确认代被新反演取代。"""

        if self._pending is not None:
            self._upsert(dataclasses.replace(self._pending, status=STATUS_SUPERSEDED))
        self._upsert(generation)
        self._pending = generation

    def _migrate(self, generation: BaselineGeneration, reason: str) -> Migration:
        """换代：现役指针一次原子替换，旧代退场，待确认态随之结清。"""

        previous = dataclasses.replace(self._active, status=STATUS_SUPERSEDED)
        self._upsert(previous)
        adopted = dataclasses.replace(generation, status=STATUS_ACTIVE)
        self._upsert(adopted)
        self._active = adopted
        if self._pending is not None and self._pending.index == adopted.index:
            self._pending = None
        return Migration(generation=adopted, previous=previous, reason=reason)

    def _upsert(self, generation: BaselineGeneration) -> None:
        """按代序号写入或替换一代，历史超出上限时裁掉最旧的。"""

        for position, existing in enumerate(self._generations):
            if existing.index == generation.index:
                self._generations[position] = generation
                return
        self._generations.append(generation)
        if len(self._generations) > HISTORY_LIMIT:
            del self._generations[: len(self._generations) - HISTORY_LIMIT]
