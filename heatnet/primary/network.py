"""一次网管网状态。

网里只有一个"当前总流量"字段，它是全网的结果而不是中间量：任何按站计算的过程
都必须用本地变量累加，算完再一次性写回。中间步骤里反复改写这个共享字段，后一个
站就会把前一个站的累加值当成自己的起点。

流量基线交给 :class:`BaselineLedger` 按代管理：联锁门限永远只跟现役代走，反演、
手工标定与人工裁决都落在同一本代际账上，不会出现"门限一套、基线一套"。
"""

from __future__ import annotations

from ..config import PrimaryLimits
from .baseline import (
    BaselineLedger,
    Decision,
    InversionOutcome,
    Migration,
    StationReading,
)
from .state import NetworkSnapshot


class PrimaryNetwork:
    """一次网：供回水温度、总流量与按代际管理的流量基线。"""

    def __init__(self, limits: PrimaryLimits) -> None:
        self._limits = limits
        self._supply_c = 0.0
        self._return_c = 0.0
        self._flow_t_h = 0.0
        self._ledger = BaselineLedger(limits)
        self._recalibrations = 0
        self._dispatches = 0

    @property
    def limits(self) -> PrimaryLimits:
        return self._limits

    def supply_c(self) -> float:
        """返回一次网供水温度。"""

        return self._supply_c

    def return_c(self) -> float:
        """返回一次网回水温度。"""

        return self._return_c

    def flow_t_h(self) -> float:
        """返回当前总流量。"""

        return self._flow_t_h

    def baseline_t_h(self) -> float:
        """返回现役代流量基线。"""

        return self._ledger.baseline_t_h()

    def threshold_t_h(self) -> float:
        """返回现役代的低流量联锁门限。"""

        return self._ledger.threshold_t_h()

    def generation(self) -> int:
        """返回现役基线代序号。"""

        return self._ledger.generation_index()

    def low_flow(self) -> bool:
        """总流量是否跌破联锁门限。"""

        return self._flow_t_h < self.threshold_t_h()

    def recalibrations(self) -> int:
        """返回累计重新标定次数。"""

        return self._recalibrations

    def dispatches(self) -> int:
        """返回累计下发次数。"""

        return self._dispatches

    def set_supply(self, temperature_c: float) -> float:
        """更新一次网供水温度。"""

        self._supply_c = float(temperature_c)
        return self._supply_c

    def set_return(self, temperature_c: float) -> float:
        """更新一次网回水温度。"""

        self._return_c = float(temperature_c)
        return self._return_c

    def set_flow(self, flow_t_h: float) -> float:
        """把总流量写为给定值，返回写后的流量。"""

        self._flow_t_h = max(0.0, float(flow_t_h))
        return self._flow_t_h

    def recalibrate(self, measured_t_h: float, ts: float) -> Migration:
        """按实测流量强制换代标定基线，联锁门限随新代变化。"""

        migration = self._ledger.recalibrate(measured_t_h, ts)
        self._recalibrations += 1
        return migration

    def invert(
        self,
        total_flow_t_h: float | None,
        stations: list[StationReading],
        ts: float,
    ) -> InversionOutcome:
        """用实测总流量与各站分配跑一轮基线反演。"""

        return self._ledger.run_round(total_flow_t_h, stations, ts)

    def decide_baseline(self, accept: bool) -> Decision:
        """裁决待确认的基线代：确认即迁移生效，否决即作废。"""

        return self._ledger.decide(accept)

    def baseline_report(self) -> dict:
        """返回基线代际报告，供控制台展示。"""

        return self._ledger.describe()

    def export_baseline(self) -> dict:
        """导出基线代际账，供落盘。"""

        return self._ledger.export()

    def restore_baseline(self, payload: dict) -> None:
        """从落盘状态恢复基线代际账。"""

        self._ledger.restore(payload)

    def count_dispatch(self) -> int:
        """记一次下发。"""

        self._dispatches += 1
        return self._dispatches

    def deviation_t_h(self) -> float:
        """返回当前总流量相对基线的偏差。"""

        return self._flow_t_h - self.baseline_t_h()

    def snapshot(self) -> NetworkSnapshot:
        """返回一次网状态快照。"""

        return NetworkSnapshot(
            supply_c=self._supply_c,
            return_c=self._return_c,
            flow_t_h=self._flow_t_h,
            baseline_t_h=self.baseline_t_h(),
            threshold_t_h=self.threshold_t_h(),
            low_flow=self.low_flow(),
            generation=self._ledger.generation_index(),
            recalibrations=self._recalibrations,
            dispatches=self._dispatches,
        )
