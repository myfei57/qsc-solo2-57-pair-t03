"""一次网。

一次网承接热源，把流量分给各换热站。流量基线按代管理：每次反演记一代，可以
重新标定，也可以由实测总流量与各站分配反演换代；调度下发前先按优先级排好站
的调节顺序；每个站的目标流量必须独立算出来，不允许用上一站的结果往下滚。
"""

from __future__ import annotations

from .baseline import (
    BaselineGeneration,
    BaselineLedger,
    Decision,
    InversionOutcome,
    InversionRound,
    Migration,
    StationReading,
)
from .flow import StationFlow, batch_flows
from .network import PrimaryNetwork
from .order import priority_order
from .state import NetworkSnapshot

__all__ = [
    "BaselineGeneration",
    "BaselineLedger",
    "Decision",
    "InversionOutcome",
    "InversionRound",
    "Migration",
    "NetworkSnapshot",
    "PrimaryNetwork",
    "StationFlow",
    "StationReading",
    "batch_flows",
    "priority_order",
]
