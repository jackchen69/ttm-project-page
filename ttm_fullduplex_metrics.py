"""
TTM Full-Duplex Evaluation Metrics
====================================
覆盖指标:
  ★ 核心全双工指标
    - FTED  : First Token End-to-End Delay (首token时延)
    - BargeinLatency : 打断响应时延
    - TOR   : Takeover Rate (打断成功率)
    - TurnTakingF1  : 换轮时机 F1
    - OverlapHandling : 重叠/回音处理评估

  ▶ 重要指标
    - BackchannelMetrics : 回应词频率/适时性
    - IPUStats  : IPU/Gap/Overlap 对话节律统计
    - MotionInterruptSync : 动作-打断协同 (TTM独有)
    - MOS : 主观自然度评分 (汇总)

  ▶ TTM 原有指标 (保留兼容)
    - FID / FGD
    - RPrecision / Diversity
    - BeatConsistency

依赖:  numpy, scipy, sklearn, torch (FID/FGD only)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics import f1_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

@dataclass
class DialogueTurn:
    """单次对话轮次的时间戳信息"""
    speaker: str          # "user" or "system"
    start_time: float     # 秒
    end_time: float       # 秒
    text: Optional[str] = None
    is_backchannel: bool = False  # 是否为 backchannel (嗯/好的)

@dataclass
class BargeInEvent:
    """打断事件"""
    user_start: float     # 用户开始说话的时刻
    system_stop: float    # 系统停止输出的时刻
    success: bool         # 系统是否成功停止 (在timeout内)
    timeout: float = 0.5  # 打断成功判定阈值 (秒), 目标 <500ms

@dataclass
class MotionFrame:
    """单帧动作特征"""
    timestamp: float
    pose: np.ndarray      # (J, 3) joint positions

@dataclass
class ChunkLatencyRecord:
    """chunk-level 时延记录"""
    user_input_end: float    # 用户输入chunk结束时刻
    first_token_out: float   # 系统第一个输出token时刻


# ===========================================================================
# ★ 核心全双工指标
# ===========================================================================

class FTEDMetric:
    """
    First Token End-to-End Delay (首token时延, FTED)
    --------------------------------------------------
    FTED = first_token_out - user_input_end

    V1 baseline : ~450 ms  (sentence-level)
    V3 目标      : ~200 ms  (chunk-level full-duplex)
    """

    def __init__(self, target_ms: float = 200.0):
        self.target_ms = target_ms
        self._records: List[float] = []

    def update(self, record: ChunkLatencyRecord) -> float:
        delay_ms = (record.first_token_out - record.user_input_end) * 1000.0
        self._records.append(delay_ms)
        return delay_ms

    def compute(self) -> Dict[str, float]:
        if not self._records:
            return {}
        arr = np.array(self._records)
        return {
            "FTED_mean_ms"   : float(np.mean(arr)),
            "FTED_median_ms" : float(np.median(arr)),
            "FTED_p90_ms"    : float(np.percentile(arr, 90)),
            "FTED_p99_ms"    : float(np.percentile(arr, 99)),
            "FTED_pass_rate" : float(np.mean(arr <= self.target_ms)),  # 达标率
            "target_ms"      : self.target_ms,
        }

    def reset(self):
        self._records.clear()


class BargeInMetric:
    """
    打断响应时延 & 打断成功率 (TOR)
    ----------------------------------
    Barge-in Latency = system_stop - user_start
    TOR (Takeover Rate) = 成功打断事件 / 全部打断事件

    CT-Duplex 参考值: TOR=94.05%, 平均时延=0.54s
    TTM 目标         : TOR>90%,   时延<500ms
    """

    def __init__(self, latency_threshold_ms: float = 500.0):
        self.threshold = latency_threshold_ms / 1000.0
        self._events: List[BargeInEvent] = []

    def update(self, event: BargeInEvent):
        # 自动判定 success (若外部未指定)
        latency = event.system_stop - event.user_start
        event.success = latency <= self.threshold
        self._events.append(event)

    def compute(self) -> Dict[str, float]:
        if not self._events:
            return {}
        latencies = np.array([e.system_stop - e.user_start for e in self._events]) * 1000
        successes = np.array([e.success for e in self._events])
        return {
            "TOR"                       : float(np.mean(successes)),
            "barge_in_latency_mean_ms"  : float(np.mean(latencies)),
            "barge_in_latency_median_ms": float(np.median(latencies)),
            "barge_in_latency_p90_ms"   : float(np.percentile(latencies, 90)),
            "total_barge_in_events"     : len(self._events),
        }

    def reset(self):
        self._events.clear()


class TurnTakingF1Metric:
    """
    Turn-Taking F1 Score
    ----------------------
    将对话中每个时间帧标注为:
        SHIFT  - 发话权转移 (user→system 或 system→user)
        HOLD   - 发话权保持
        BACKCHANNEL - 简短回应 (非换轮)

    y_true / y_pred: List[str], 每个元素为 "SHIFT"/"HOLD"/"BC"
    参考: VAP (Voice Activity Projection) 评估框架
    """

    LABELS = ["SHIFT", "HOLD", "BC"]

    def __init__(self):
        self._y_true: List[str] = []
        self._y_pred: List[str] = []

    def update(self, y_true: List[str], y_pred: List[str]):
        assert len(y_true) == len(y_pred)
        self._y_true.extend(y_true)
        self._y_pred.extend(y_pred)

    def compute(self) -> Dict[str, float]:
        if not self._y_true:
            return {}
        results = {}
        for avg in ("macro", "weighted"):
            results[f"turn_taking_f1_{avg}"] = float(
                f1_score(self._y_true, self._y_pred, labels=self.LABELS,
                         average=avg, zero_division=0)
            )
            results[f"turn_taking_precision_{avg}"] = float(
                precision_score(self._y_true, self._y_pred, labels=self.LABELS,
                                average=avg, zero_division=0)
            )
            results[f"turn_taking_recall_{avg}"] = float(
                recall_score(self._y_true, self._y_pred, labels=self.LABELS,
                             average=avg, zero_division=0)
            )
        # Per-class F1
        per_class = f1_score(self._y_true, self._y_pred,
                             labels=self.LABELS, average=None, zero_division=0)
        for label, score in zip(self.LABELS, per_class):
            results[f"turn_taking_f1_{label}"] = float(score)
        return results

    def reset(self):
        self._y_true.clear()
        self._y_pred.clear()


class OverlapHandlingMetric:
    """
    重叠/回音处理质量
    ------------------
    Full-Duplex-Bench v2 定义 4 类 overlap 场景:
        1. user-backchannel : 用户插入 backchannel, 系统继续说
        2. user-interruption: 用户实质打断, 系统应停止
        3. system-overlap   : 系统抢话, 应避免
        4. crosstalk        : 双方同时长时间说话 (异常)

    本指标统计各场景发生率及系统正确行为率。
    """

    OVERLAP_TYPES = ["user_backchannel", "user_interruption", "system_overlap", "crosstalk"]

    def __init__(self):
        self._total: Dict[str, int] = {t: 0 for t in self.OVERLAP_TYPES}
        self._correct: Dict[str, int] = {t: 0 for t in self.OVERLAP_TYPES}

    def update(self, overlap_type: str, system_correct: bool):
        assert overlap_type in self.OVERLAP_TYPES, f"Unknown type: {overlap_type}"
        self._total[overlap_type] += 1
        if system_correct:
            self._correct[overlap_type] += 1

    def compute(self) -> Dict[str, float]:
        results = {}
        total_all = sum(self._total.values())
        correct_all = sum(self._correct.values())
        results["overlap_handling_overall_acc"] = (
            correct_all / total_all if total_all > 0 else 0.0
        )
        for t in self.OVERLAP_TYPES:
            results[f"overlap_{t}_count"] = self._total[t]
            results[f"overlap_{t}_acc"] = (
                self._correct[t] / self._total[t] if self._total[t] > 0 else 0.0
            )
        return results

    def reset(self):
        for t in self.OVERLAP_TYPES:
            self._total[t] = 0
            self._correct[t] = 0


# ===========================================================================
# ▶ 重要指标
# ===========================================================================

class BackchannelMetric:
    """
    Backchannel 频率 & 适时性
    ---------------------------
    频率 = BC 次数 / 对话总时长 (分钟)
    适时性 = BC 发生在对话节律 "适当位置" 的比例

    适当位置定义 (参考 Switchboard 标注):
        - 对方话语内的短暂停顿 (0.2~1.5s)
        - 对方话语中的语调下降后
    """

    def __init__(self, dialogue_duration_sec: float = 0.0):
        self.duration_sec = dialogue_duration_sec
        self._bc_events: List[Tuple[float, bool]] = []  # (timestamp, is_timely)

    def update(self, timestamp: float, is_timely: bool):
        self._bc_events.append((timestamp, is_timely))

    def set_duration(self, duration_sec: float):
        self.duration_sec = duration_sec

    def compute(self) -> Dict[str, float]:
        if not self._bc_events:
            return {"backchannel_count": 0, "backchannel_rate_per_min": 0.0,
                    "backchannel_timeliness": 0.0}
        n = len(self._bc_events)
        timely = sum(1 for _, t in self._bc_events if t)
        rate = (n / self.duration_sec * 60.0) if self.duration_sec > 0 else 0.0
        return {
            "backchannel_count"       : n,
            "backchannel_rate_per_min": float(rate),
            "backchannel_timeliness"  : float(timely / n),
        }

    def reset(self):
        self._bc_events.clear()


class IPUStatsMetric:
    """
    Inter-Pausal Unit (IPU) / Gap / Overlap 对话节律统计
    ------------------------------------------------------
    IPU  : 无内部停顿 (≥threshold) 的连续语音段
    Gap  : 换轮之间的静默时长
    Overlap: 双方同时发声的时长

    参考: Moshi 使用 Fisher Corpus 与人类对比
    """

    def __init__(self, pause_threshold_sec: float = 0.2):
        self.pause_threshold = pause_threshold_sec
        self._gaps: List[float] = []
        self._overlaps: List[float] = []
        self._ipu_durations: List[float] = []

    def update_gap(self, gap_sec: float):
        self._gaps.append(gap_sec)

    def update_overlap(self, overlap_sec: float):
        self._overlaps.append(overlap_sec)

    def update_ipu(self, ipu_duration_sec: float):
        self._ipu_durations.append(ipu_duration_sec)

    def update_from_turns(self, turns: List[DialogueTurn]):
        """从 DialogueTurn 列表自动计算 gap / overlap / IPU"""
        sorted_turns = sorted(turns, key=lambda t: t.start_time)
        for i in range(1, len(sorted_turns)):
            prev, curr = sorted_turns[i - 1], sorted_turns[i]
            if prev.speaker != curr.speaker:
                gap = curr.start_time - prev.end_time
                if gap > 0:
                    self._gaps.append(gap)
                elif gap < 0:
                    self._overlaps.append(-gap)
            # IPU: same speaker continuous
            ipu_dur = prev.end_time - prev.start_time
            self._ipu_durations.append(ipu_dur)

    def compute(self) -> Dict[str, float]:
        def _stats(arr: List[float], prefix: str) -> Dict[str, float]:
            if not arr:
                return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0,
                        f"{prefix}_count": 0}
            a = np.array(arr)
            return {
                f"{prefix}_mean" : float(np.mean(a)),
                f"{prefix}_std"  : float(np.std(a)),
                f"{prefix}_median": float(np.median(a)),
                f"{prefix}_count": len(arr),
            }
        results = {}
        results.update(_stats(self._gaps,         "gap_sec"))
        results.update(_stats(self._overlaps,     "overlap_sec"))
        results.update(_stats(self._ipu_durations,"ipu_sec"))
        return results

    def reset(self):
        self._gaps.clear()
        self._overlaps.clear()
        self._ipu_durations.clear()


# ===========================================================================
# ★ TTM 独有指标
# ===========================================================================

class MotionInterruptSyncMetric:
    """
    Motion-Interrupt Sync (动作-打断协同)
    ----------------------------------------
    TTM 独有需求: 被打断时, Mover 需在 ~200ms 内平滑停止/过渡动作序列

    衡量维度:
        1. stop_latency_ms : 从打断信号到动作停止的时延
        2. smoothness_score: 停止过渡的平滑度 (用相邻帧速度差衡量)
        3. success_rate    : 在 target_stop_ms 内完成停止的比例
    """

    def __init__(self, target_stop_ms: float = 200.0, fps: float = 30.0):
        self.target_stop_ms = target_stop_ms
        self.fps = fps
        self._stop_latencies: List[float] = []
        self._smoothness_scores: List[float] = []

    def update(
        self,
        interrupt_time: float,
        motion_frames_after: List[MotionFrame],
        stop_threshold_velocity: float = 0.01,
    ):
        """
        interrupt_time          : 打断信号时刻 (秒)
        motion_frames_after     : 打断信号后的动作帧序列
        stop_threshold_velocity : 判定"已停止"的速度阈值
        """
        stop_time = None
        velocities = []

        for i in range(1, len(motion_frames_after)):
            dt = motion_frames_after[i].timestamp - motion_frames_after[i - 1].timestamp
            if dt <= 0:
                continue
            vel = np.linalg.norm(
                motion_frames_after[i].pose - motion_frames_after[i - 1].pose
            ) / dt
            velocities.append(vel)
            if vel < stop_threshold_velocity and stop_time is None:
                stop_time = motion_frames_after[i].timestamp

        # Stop latency
        if stop_time is not None:
            latency_ms = (stop_time - interrupt_time) * 1000.0
        else:
            latency_ms = float("inf")
        self._stop_latencies.append(latency_ms)

        # Smoothness: lower acceleration = smoother
        if len(velocities) >= 2:
            accel = np.diff(velocities)
            smoothness = float(1.0 / (1.0 + np.mean(np.abs(accel))))
        else:
            smoothness = 0.0
        self._smoothness_scores.append(smoothness)

    def compute(self) -> Dict[str, float]:
        if not self._stop_latencies:
            return {}
        finite_latencies = [l for l in self._stop_latencies if np.isfinite(l)]
        arr = np.array(finite_latencies) if finite_latencies else np.array([0.0])
        return {
            "motion_stop_latency_mean_ms"  : float(np.mean(arr)),
            "motion_stop_latency_median_ms": float(np.median(arr)),
            "motion_stop_latency_p90_ms"   : float(np.percentile(arr, 90)),
            "motion_stop_success_rate"     : float(
                np.mean(np.array(self._stop_latencies) <= self.target_stop_ms)
            ),
            "motion_smoothness_mean"       : float(np.mean(self._smoothness_scores)),
            "target_stop_ms"               : self.target_stop_ms,
        }

    def reset(self):
        self._stop_latencies.clear()
        self._smoothness_scores.clear()


# ===========================================================================
# ▶ TTM 原有指标 (保留兼容)
# ===========================================================================

def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray,
                eps: float = 1e-6) -> float:
    """
    Fréchet Inception Distance (FID) / Fréchet Gesture Distance (FGD)
    -------------------------------------------------------------------
    real_feats, fake_feats: (N, D) feature arrays
    """
    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    diff = mu_r - mu_f
    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))
    return fid


def compute_r_precision(motion_feats: np.ndarray,
                         text_feats: np.ndarray,
                         top_k: int = 3) -> Dict[str, float]:
    """
    R-Precision@k : 对每个 motion, 在候选文本中检索正确文本是否在 top-k
    motion_feats / text_feats: (N, D)
    """
    N = len(motion_feats)
    hits = {k: 0 for k in range(1, top_k + 1)}
    for i in range(N):
        sims = motion_feats[i] @ text_feats.T   # (N,)
        ranked = np.argsort(-sims)
        for k in range(1, top_k + 1):
            if i in ranked[:k]:
                hits[k] += 1
    return {f"R_Precision_top{k}": hits[k] / N for k in range(1, top_k + 1)}


def compute_diversity(motion_feats: np.ndarray,
                      num_samples: int = 300,
                      seed: int = 42) -> float:
    """
    Diversity: 随机采样对之间的平均欧氏距离
    """
    rng = np.random.default_rng(seed)
    N = len(motion_feats)
    if N < 2:
        return 0.0
    num_samples = min(num_samples, N // 2)
    idx1 = rng.choice(N, num_samples, replace=False)
    idx2 = rng.choice(N, num_samples, replace=False)
    diffs = motion_feats[idx1] - motion_feats[idx2]
    return float(np.mean(np.linalg.norm(diffs, axis=1)))


def compute_beat_consistency(motion_vels: np.ndarray,
                              audio_beats: np.ndarray,
                              fps: float = 30.0,
                              window_ms: float = 50.0) -> float:
    """
    Beat Consistency (BC)
    ----------------------
    motion_vels : (T,) per-frame velocity magnitude
    audio_beats : (B,) beat timestamps in seconds
    fps         : motion frame rate
    window_ms   : ±window around beat to search for motion peak
    """
    window_frames = int(window_ms / 1000.0 * fps)
    T = len(motion_vels)
    hits = 0
    for beat_sec in audio_beats:
        beat_frame = int(beat_sec * fps)
        lo = max(0, beat_frame - window_frames)
        hi = min(T - 1, beat_frame + window_frames)
        if lo >= hi:
            continue
        local_max = np.argmax(motion_vels[lo:hi + 1]) + lo
        if abs(local_max - beat_frame) <= window_frames:
            hits += 1
    return hits / len(audio_beats) if len(audio_beats) > 0 else 0.0


# ===========================================================================
# MOS Aggregator
# ===========================================================================

class MOSAggregator:
    """
    MOS (Mean Opinion Score) 主观自然度评分汇总
    语音+动作整体 MOS, 1~5 分
    """

    def __init__(self):
        self._scores: List[float] = []

    def update(self, scores: List[float]):
        self._scores.extend(scores)

    def compute(self) -> Dict[str, float]:
        if not self._scores:
            return {}
        arr = np.array(self._scores)
        return {
            "MOS_mean"  : float(np.mean(arr)),
            "MOS_std"   : float(np.std(arr)),
            "MOS_95ci"  : float(1.96 * np.std(arr) / np.sqrt(len(arr))),
            "MOS_n"     : len(arr),
        }

    def reset(self):
        self._scores.clear()


# ===========================================================================
# Unified Evaluator
# ===========================================================================

class TTMFullDuplexEvaluator:
    """
    统一评测入口 —— 汇总所有全双工指标
    用法示例见文件末尾 __main__ 块
    """

    def __init__(self,
                 fted_target_ms: float = 200.0,
                 barge_in_threshold_ms: float = 500.0,
                 motion_stop_target_ms: float = 200.0,
                 fps: float = 30.0):
        self.fted      = FTEDMetric(target_ms=fted_target_ms)
        self.barge_in  = BargeInMetric(latency_threshold_ms=barge_in_threshold_ms)
        self.turn_f1   = TurnTakingF1Metric()
        self.overlap   = OverlapHandlingMetric()
        self.backchan  = BackchannelMetric()
        self.ipu       = IPUStatsMetric()
        self.mot_sync  = MotionInterruptSyncMetric(
            target_stop_ms=motion_stop_target_ms, fps=fps)
        self.mos       = MOSAggregator()

    # ----- convenience update wrappers -----

    def log_chunk(self, user_input_end: float, first_token_out: float):
        self.fted.update(ChunkLatencyRecord(user_input_end, first_token_out))

    def log_barge_in(self, user_start: float, system_stop: float):
        self.barge_in.update(BargeInEvent(user_start, system_stop,
                                          success=False))  # auto-computed

    def log_turn_labels(self, y_true: List[str], y_pred: List[str]):
        self.turn_f1.update(y_true, y_pred)

    def log_overlap(self, overlap_type: str, correct: bool):
        self.overlap.update(overlap_type, correct)

    def log_backchannel(self, timestamp: float, is_timely: bool):
        self.backchan.update(timestamp, is_timely)

    def log_turns(self, turns: List[DialogueTurn], duration_sec: float):
        self.ipu.update_from_turns(turns)
        self.backchan.set_duration(duration_sec)

    def log_motion_interrupt(self, interrupt_time: float,
                              frames: List[MotionFrame]):
        self.mot_sync.update(interrupt_time, frames)

    def log_mos(self, scores: List[float]):
        self.mos.update(scores)

    # ----- compute all -----

    def compute_all(self) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name, metric in [
            ("fted",     self.fted),
            ("barge_in", self.barge_in),
            ("turn_f1",  self.turn_f1),
            ("overlap",  self.overlap),
            ("backchan", self.backchan),
            ("ipu",      self.ipu),
            ("mot_sync", self.mot_sync),
            ("mos",      self.mos),
        ]:
            try:
                results.update(metric.compute())
            except Exception as e:
                warnings.warn(f"[{name}] compute failed: {e}")
        return results

    def report(self) -> str:
        results = self.compute_all()
        lines = ["=" * 60,
                 "  TTM Full-Duplex Evaluation Report",
                 "=" * 60]

        groups = {
            "★ Core Full-Duplex": [
                "FTED_mean_ms", "FTED_p90_ms", "FTED_pass_rate",
                "TOR", "barge_in_latency_mean_ms",
                "turn_taking_f1_macro",
                "turn_taking_f1_SHIFT", "turn_taking_f1_HOLD",
                "overlap_handling_overall_acc",
            ],
            "▶ Rhythm & Naturalness": [
                "backchannel_rate_per_min", "backchannel_timeliness",
                "gap_sec_mean", "overlap_sec_mean", "ipu_sec_mean",
            ],
            "★ TTM Motion-Interrupt Sync": [
                "motion_stop_latency_mean_ms", "motion_stop_success_rate",
                "motion_smoothness_mean",
            ],
            "▶ Subjective MOS": [
                "MOS_mean", "MOS_std", "MOS_95ci",
            ],
        }

        for group, keys in groups.items():
            lines.append(f"\n  [{group}]")
            for k in keys:
                if k in results:
                    v = results[k]
                    lines.append(f"    {k:<42s}: {v:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def reset_all(self):
        for metric in [self.fted, self.barge_in, self.turn_f1,
                       self.overlap, self.backchan, self.ipu,
                       self.mot_sync, self.mos]:
            metric.reset()


# ===========================================================================
# Demo / Smoke Test
# ===========================================================================

if __name__ == "__main__":
    import random
    rng = random.Random(0)
    np_rng = np.random.default_rng(0)

    evaluator = TTMFullDuplexEvaluator(
        fted_target_ms=200.0,
        barge_in_threshold_ms=500.0,
        motion_stop_target_ms=200.0,
    )

    # --- FTED: 模拟 50 个 chunk ---
    for _ in range(50):
        t0 = 0.0
        delay = np_rng.normal(loc=0.22, scale=0.05)   # ~220ms avg
        evaluator.log_chunk(t0, t0 + delay)

    # --- Barge-in: 模拟 30 次打断 ---
    for _ in range(30):
        t0 = 0.0
        latency = np_rng.normal(loc=0.48, scale=0.12)  # ~480ms avg
        evaluator.log_barge_in(t0, t0 + abs(latency))

    # --- Turn-taking labels ---
    labels = ["SHIFT", "HOLD", "BC"]
    y_true = [rng.choice(labels) for _ in range(200)]
    y_pred = [l if rng.random() > 0.15 else rng.choice(labels) for l in y_true]
    evaluator.log_turn_labels(y_true, y_pred)

    # --- Overlap events ---
    for otype in OverlapHandlingMetric.OVERLAP_TYPES:
        for _ in range(20):
            evaluator.log_overlap(otype, rng.random() > 0.1)

    # --- Backchannels ---
    evaluator.backchan.set_duration(600.0)  # 10-min dialogue
    for _ in range(60):
        evaluator.log_backchannel(rng.uniform(0, 600), rng.random() > 0.3)

    # --- IPU / Gap / Overlap via turns ---
    turns = []
    t = 0.0
    for i in range(40):
        speaker = "user" if i % 2 == 0 else "system"
        dur = np_rng.uniform(1.0, 5.0)
        turns.append(DialogueTurn(speaker, t, t + dur))
        t += dur + np_rng.uniform(-0.2, 0.8)  # some overlap, some gap
    evaluator.log_turns(turns, duration_sec=t)

    # --- Motion-Interrupt Sync ---
    J = 22  # joints
    for _ in range(20):
        interrupt_time = 0.0
        frames = []
        vel = 0.5
        for fi in range(30):
            ts = fi / 30.0
            vel = max(0.0, vel - np_rng.uniform(0.02, 0.06))
            pose = np_rng.normal(size=(J, 3))
            frames.append(MotionFrame(ts, pose))
        evaluator.log_motion_interrupt(interrupt_time, frames)

    # --- MOS scores ---
    evaluator.log_mos(list(np_rng.normal(loc=3.8, scale=0.4, size=50)))

    # --- Static TTM metrics ---
    print("\n  [TTM Original Metrics]")
    N, D = 256, 512
    real_f = np_rng.standard_normal((N, D))
    fake_f = real_f + np_rng.standard_normal((N, D)) * 0.3
    fid = compute_fid(real_f, fake_f)
    print(f"    {'FID / FGD':<42s}: {fid:.4f}")

    motion_f = np_rng.standard_normal((N, 256))
    text_f   = np_rng.standard_normal((N, 256))
    rp = compute_r_precision(motion_f, text_f)
    for k, v in rp.items():
        print(f"    {k:<42s}: {v:.4f}")

    div = compute_diversity(motion_f)
    print(f"    {'Diversity':<42s}: {div:.4f}")

    vels = np.abs(np_rng.standard_normal(900))
    beats = np.arange(0, 30, 0.5)
    bc = compute_beat_consistency(vels, beats)
    print(f"    {'BeatConsistency':<42s}: {bc:.4f}")

    # --- Full report ---
    print(evaluator.report())
