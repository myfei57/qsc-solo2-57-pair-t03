"""一次网。

一次网承接热源，把流量分给各换热站。流量基线按代管理：可以拿实测总流量与各站
分配按滑动窗口反演出新代，也可以人工标定；换代是迁移，联锁始终只按现役代判。
调度下发前先按优先级排好站的调节顺序；每个站的目标流量必须独立算出来，不允许
用上一站的结果往下滚。
"""

from __future__ import annotations

from .flow import StationFlow, batch_flows
from .inversion import (
    BaselineGeneration,
    BaselineRegistry,
    FlowSample,
    GenerationShift,
    InversionRound,
    Migration,
)
from .network import PrimaryNetwork
from .order import priority_order
from .state import NetworkSnapshot

__all__ = [
    "BaselineGeneration",
    "BaselineRegistry",
    "FlowSample",
    "GenerationShift",
    "InversionRound",
    "Migration",
    "NetworkSnapshot",
    "PrimaryNetwork",
    "StationFlow",
    "batch_flows",
    "priority_order",
]
