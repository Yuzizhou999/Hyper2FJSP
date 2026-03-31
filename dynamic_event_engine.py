from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class DynamicEventSnapshot:
    breakdown_mask: np.ndarray
    deadline_delta: np.ndarray
    energy_multiplier: np.ndarray
    job_arrival_signal: np.ndarray
    event_flags: np.ndarray


class DynamicEventEngine:
    """FJSP（柔性车间调度问题）环境的随机动态事件生成器。

    该引擎设计得非常轻量，且对环境内部状态无副作用：
    它采样事件信号并返回张量数组，环境可以据此自行更新自己的状态。
    """

    EVENT_JOB_ARRIVAL = "job_arrival"          # 新工件到达事件
    EVENT_MACHINE_BREAKDOWN = "machine_breakdown"  # 机器故障事件
    EVENT_DEADLINE_SHIFT = "deadline_shift"    # 交货期提前/延后事件
    EVENT_ENERGY_SPIKE = "energy_spike"        # 能耗激增事件

    _EVENT_ORDER = (
        EVENT_JOB_ARRIVAL,
        EVENT_MACHINE_BREAKDOWN,
        EVENT_DEADLINE_SHIFT,
        EVENT_ENERGY_SPIKE,
    )

    def __init__(
        self,
        enabled: bool = False,
        event_prob: float = 0.0,
        event_types: Iterable[str] | None = None,
        breakdown_duration: int = 2,
        deadline_shift_scale: float = 0.1,
        energy_duration: int = 2,
        energy_scale: float = 0.2,
        seed: int | None = None,
    ) -> None:
        """
        初始化动态事件生成器。

        :param enabled: 是否启用事件生成器
        :param event_prob: 每个步长发生随机事件的概率
        :param event_types: 允许发生的事件类型列表，若为 None 则默认开启所有支持的事件
        :param breakdown_duration: 机器发生故障时停机的持续步数（时间）
        :param deadline_shift_scale: 交货期波动的最大比例缩放系数
        :param energy_duration: 机器能耗波动的持续步数（时间）
        :param energy_scale: 机器能耗系数波动的最大幅度
        :param seed: 随机数种子（可选）
        """
        self.enabled = bool(enabled)
        self.event_prob = float(max(0.0, event_prob))
        self.event_types = set(
            event_types if event_types is not None else self._EVENT_ORDER
        )
        self.breakdown_duration = int(max(1, breakdown_duration))
        self.deadline_shift_scale = float(max(0.0, deadline_shift_scale))
        self.energy_duration = int(max(1, energy_duration))
        self.energy_scale = float(max(0.0, energy_scale))
        self.rng = np.random.default_rng(seed)

        self.num_envs = 0
        self.num_jobs = 0
        self.num_machines = 0
        self.breakdown_remaining = np.zeros((0, 0), dtype=np.int64)
        self.energy_remaining = np.zeros((0, 0), dtype=np.int64)
        self.energy_multiplier = np.ones((0, 0), dtype=np.float64)
        self.last_deadline_delta = np.zeros((0, 0), dtype=np.float64)
        self.last_job_arrival_signal = np.zeros((0,), dtype=np.float64)
        self.last_event_flags = np.zeros((0, len(self._EVENT_ORDER)), dtype=np.float64)

    @property
    def context_dim(self) -> int:
        """
        获取全局事件上下文特征的维度大小。
        特征包括：
        [arrival_flag, breakdown_ratio_mean, breakdown_time_mean, 
         deadline_shift_mean, deadline_shift_abs_mean, energy_multiplier_mean, 
         energy_multiplier_std, any_breakdown_flag, any_deadline_shift_flag, 
         any_energy_spike_flag]
        """
        return 10

    @property
    def machine_context_dim(self) -> int:
        """
        获取机器级别事件上下文特征的维度大小。
        特征包括：
        [broken_flag (是否故障), breakdown_remaining_norm (归一化的剩余故障时间), 
         energy_multiplier_minus_one (超出正常基准的能耗系数差值)]
        """
        return 3

    def reset(self, num_envs: int, num_jobs: int, num_machines: int) -> None:
        """
        重置/初始化引擎，为下一次迭代或评测准备各个状态张量。

        :param num_envs: 并行环境（Batch Size）数量
        :param num_jobs: 每个环境中的最大工件数
        :param num_machines: 每个环境中的机器数量
        """
        self.num_envs = int(num_envs)
        self.num_jobs = int(num_jobs)
        self.num_machines = int(num_machines)
        self.breakdown_remaining = np.zeros(
            (self.num_envs, self.num_machines), dtype=np.int64
        )
        self.energy_remaining = np.zeros(
            (self.num_envs, self.num_machines), dtype=np.int64
        )
        self.energy_multiplier = np.ones((self.num_envs, self.num_machines))
        self.last_deadline_delta = np.zeros((self.num_envs, self.num_jobs))
        self.last_job_arrival_signal = np.zeros((self.num_envs,))
        self.last_event_flags = np.zeros((self.num_envs, len(self._EVENT_ORDER)))

    def sample(self, current_deadlines: np.ndarray, done_mask: np.ndarray) -> DynamicEventSnapshot:
        """
        在新一个时间步采样并生成各类随机动态事件，更新内部状态。

        :param current_deadlines: 每个工件当前的交货期数组 [num_envs, num_jobs]
        :param done_mask: 标记该环境是否已完成调度的布尔掩码数组 [num_envs]
        :return: 包含各类事件信息的快照对象 DynamicEventSnapshot
        """
        if self.num_envs == 0:
            raise RuntimeError("在采样动态事件之前请先调用 reset() 方法。")

        done_mask = np.asarray(done_mask, dtype=bool)
        active_mask = ~done_mask  # 仅未完成的环境产生事件

        self.last_deadline_delta.fill(0.0)
        self.last_job_arrival_signal.fill(0.0)
        self.last_event_flags.fill(0.0)

        # 如果未启用或概率为 0，则仅更新时间衰减并直接返回
        if not self.enabled or self.event_prob <= 0.0:
            self._tick_durations(active_mask)
            return self._snapshot()

        self._tick_durations(active_mask)

        # === 机器故障事件生成 ===
        if self.EVENT_MACHINE_BREAKDOWN in self.event_types:
            trigger = active_mask & (self.rng.random(self.num_envs) < self.event_prob)
            trigger_idxs = np.where(trigger)[0]
            for env_idx in trigger_idxs:
                mch_idx = int(self.rng.integers(0, self.num_machines))
                self.breakdown_remaining[env_idx, mch_idx] = max(
                    self.breakdown_remaining[env_idx, mch_idx], self.breakdown_duration
                )
            self.last_event_flags[trigger, self._EVENT_ORDER.index(self.EVENT_MACHINE_BREAKDOWN)] = 1.0

        # === 交货期变动事件生成 ===
        if self.EVENT_DEADLINE_SHIFT in self.event_types and self.deadline_shift_scale > 0:
            trigger = active_mask & (self.rng.random(self.num_envs) < self.event_prob)
            trigger_idxs = np.where(trigger)[0]
            for env_idx in trigger_idxs:
                signs = self.rng.choice([-1.0, 1.0], size=self.num_jobs)
                magnitude = self.rng.uniform(0.0, self.deadline_shift_scale, size=self.num_jobs)
                ratio = signs * magnitude
                delta = ratio * np.maximum(np.asarray(current_deadlines[env_idx]), 1e-6)
                self.last_deadline_delta[env_idx] = delta
            self.last_event_flags[trigger, self._EVENT_ORDER.index(self.EVENT_DEADLINE_SHIFT)] = 1.0

        # === 能耗激增事件生成 ===
        if self.EVENT_ENERGY_SPIKE in self.event_types and self.energy_scale > 0:
            trigger = active_mask & (self.rng.random(self.num_envs) < self.event_prob)
            trigger_idxs = np.where(trigger)[0]
            for env_idx in trigger_idxs:
                mch_idx = int(self.rng.integers(0, self.num_machines))
                spike = 1.0 + self.rng.uniform(0.0, self.energy_scale)
                self.energy_multiplier[env_idx, mch_idx] = spike
                self.energy_remaining[env_idx, mch_idx] = self.energy_duration
            self.last_event_flags[trigger, self._EVENT_ORDER.index(self.EVENT_ENERGY_SPIKE)] = 1.0

        # === 新工件到达事件生成 ===
        if self.EVENT_JOB_ARRIVAL in self.event_types:
            trigger = active_mask & (self.rng.random(self.num_envs) < self.event_prob)
            self.last_job_arrival_signal[trigger] = 1.0
            self.last_event_flags[trigger, self._EVENT_ORDER.index(self.EVENT_JOB_ARRIVAL)] = 1.0

        return self._snapshot()

    def build_global_context(self) -> np.ndarray:
        """
        构建供网络模型（如HyperActor中的事件适配器）使用的全局事件状态表征。

        :return: 全局环境上下文特征 [num_envs, context_dim]
        """
        if self.num_envs == 0:
            return np.zeros((0, self.context_dim), dtype=np.float32)

        # 机器故障相关的统计指标
        broken = (self.breakdown_remaining > 0).astype(np.float64)
        breakdown_ratio = broken.mean(axis=1) # 发生故障的机器比例
        breakdown_time = (
            self.breakdown_remaining.astype(np.float64) / float(self.breakdown_duration)
        ).mean(axis=1) # 归一化的平均维修时间

        # 交货期变动的均值和绝对均值（归一化为比例）
        if self.num_jobs > 0:
            deadline_ratio = self.last_deadline_delta / (
                np.abs(self.last_deadline_delta).mean(axis=1, keepdims=True) + 1e-8
            )
            deadline_shift_mean = np.mean(deadline_ratio, axis=1)
            deadline_shift_abs = np.mean(np.abs(deadline_ratio), axis=1)
        else:
            deadline_shift_mean = np.zeros(self.num_envs)
            deadline_shift_abs = np.zeros(self.num_envs)

        # 机器能耗变动指标
        energy_delta = self.energy_multiplier - 1.0
        energy_mean = energy_delta.mean(axis=1)
        energy_std = energy_delta.std(axis=1)

        # 有无相关事件发生的指示器
        any_breakdown = (breakdown_ratio > 0).astype(np.float64)
        any_deadline = (np.abs(self.last_deadline_delta).sum(axis=1) > 0).astype(np.float64)
        any_energy = (energy_delta.max(axis=1) > 0).astype(np.float64)

        # 聚合成最终的全局上下文向量
        context = np.stack(
            [
                self.last_job_arrival_signal,
                breakdown_ratio,
                breakdown_time,
                deadline_shift_mean,
                deadline_shift_abs,
                energy_mean,
                energy_std,
                any_breakdown,
                any_deadline,
                any_energy,
            ],
            axis=1,
        )
        return context.astype(np.float32)

    def build_machine_context(self) -> np.ndarray:
        """
        构建特定于单台机器的局部事件特征表征。

        :return: 机器级事件上下文特征 [num_envs, num_machines, machine_context_dim]
        """
        if self.num_envs == 0:
            return np.zeros((0, 0, self.machine_context_dim), dtype=np.float32)

        # 机器是否故障、归一化的修复剩余时间和增加的能耗乘子
        broken = (self.breakdown_remaining > 0).astype(np.float64)
        remain_norm = self.breakdown_remaining.astype(np.float64) / float(
            self.breakdown_duration
        )
        energy_delta = self.energy_multiplier - 1.0
        
        context = np.stack([broken, remain_norm, energy_delta], axis=-1)
        return context.astype(np.float32)

    def _snapshot(self) -> DynamicEventSnapshot:
        """打包并返回当前时间步事件的发生快照。"""
        return DynamicEventSnapshot(
            breakdown_mask=(self.breakdown_remaining > 0),
            deadline_delta=self.last_deadline_delta.copy(),
            energy_multiplier=self.energy_multiplier.copy(),
            job_arrival_signal=self.last_job_arrival_signal.copy(),
            event_flags=self.last_event_flags.copy(),
        )

    def _tick_durations(self, active_mask: np.ndarray) -> None:
        """
        随着时间步推进，更新机器状态，持续时间扣减等。

        :param active_mask: 掩码指示哪些环境仍在发生交互没有完成
        """
        active_col = active_mask[:, None]
        # 逐时间步消减故障时间
        self.breakdown_remaining[active_col & (self.breakdown_remaining > 0)] -= 1
        self.breakdown_remaining = np.maximum(self.breakdown_remaining, 0)

        # 逐时间步消减高能耗状态的持续时间，恢复原能耗 (1.0)
        self.energy_remaining[active_col & (self.energy_remaining > 0)] -= 1
        self.energy_remaining = np.maximum(self.energy_remaining, 0)
        self.energy_multiplier[self.energy_remaining == 0] = 1.0
