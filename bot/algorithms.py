"""
Pluggable difficulty-adjustment algorithms.

Each algorithm provides `update_level_batch(current_levels, recent_scores)`:
  - current_levels: (N,) int — current level per patient
  - recent_scores:  (N, 3) float — last 3 scores at this (patient, exercise),
                                    NaN-padded when fewer than 3 are available
  - returns: (N,) int — next level per patient
"""
from __future__ import annotations
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="Mean of empty slice")


def _safe_mean3(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (has_3_mask, mean_of_last_3). has_3_mask is False where any NaN."""
    has_3 = ~np.isnan(scores).any(axis=1)
    mean = np.where(has_3, np.nanmean(scores, axis=1), 0.0)
    return has_3, mean


class Algorithm:
    name = "base"
    max_level = 7

    def update_level_batch(self, current_levels: np.ndarray,
                            recent_scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CurrentRule(Algorithm):
    name = "current_50_80"

    def update_level_batch(self, current_levels, recent_scores):
        has_3, mean = _safe_mean3(recent_scores)
        nxt = current_levels.copy()
        down = has_3 & (mean < 0.5)
        up = has_3 & (mean >= 0.8)
        nxt = np.where(down, np.maximum(1, current_levels - 1), nxt)
        nxt = np.where(up, np.minimum(self.max_level, current_levels + 1), nxt)
        return nxt


class DropAt70(Algorithm):
    name = "drop_at_70"

    def update_level_batch(self, current_levels, recent_scores):
        has_3, mean = _safe_mean3(recent_scores)
        nxt = current_levels.copy()
        down = has_3 & (mean < 0.7)
        up = has_3 & (mean >= 0.8)
        nxt = np.where(down, np.maximum(1, current_levels - 1), nxt)
        nxt = np.where(up, np.minimum(self.max_level, current_levels + 1), nxt)
        return nxt


class LevelDifferentiated(Algorithm):
    name = "level_diff"
    # idx 0 unused; idx = level
    UP = np.array([0, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 1.01])
    DOWN = np.array([0, 0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])

    def update_level_batch(self, current_levels, recent_scores):
        has_3, mean = _safe_mean3(recent_scores)
        up_thr = self.UP[current_levels]
        down_thr = self.DOWN[current_levels]
        nxt = current_levels.copy()
        down = has_3 & (mean < down_thr)
        up = has_3 & (mean >= up_thr)
        nxt = np.where(down, np.maximum(1, current_levels - 1), nxt)
        nxt = np.where(up, np.minimum(self.max_level, current_levels + 1), nxt)
        return nxt


class StrictUp(Algorithm):
    """선행 분석(1차) 제안 3: 상승은 3회 모두 score >= n/(n+1). 하락은 평균 < 0.5."""
    name = "strict_up_n_over_np1"

    def update_level_batch(self, current_levels, recent_scores):
        has_3, mean = _safe_mean3(recent_scores)
        thr_per_level = current_levels / (current_levels + 1)  # (N,)
        all_above = (recent_scores >= thr_per_level[:, None]).all(axis=1) & has_3
        nxt = current_levels.copy()
        down = has_3 & (mean < 0.5)
        up = all_above
        nxt = np.where(down, np.maximum(1, current_levels - 1), nxt)
        nxt = np.where(up, np.minimum(self.max_level, current_levels + 1), nxt)
        return nxt


class Expand10(Algorithm):
    """7→10 확장 + 레벨별 차등 threshold."""
    name = "expand_10_diff"
    max_level = 10
    UP = np.array([0, 0.75, 0.78, 0.80, 0.82, 0.84,
                   0.86, 0.88, 0.90, 0.92, 1.01])
    DOWN = np.array([0, 0.0, 0.55, 0.60, 0.65, 0.68,
                     0.70, 0.72, 0.74, 0.76, 0.78])

    def update_level_batch(self, current_levels, recent_scores):
        has_3, mean = _safe_mean3(recent_scores)
        up_thr = self.UP[current_levels]
        down_thr = self.DOWN[current_levels]
        nxt = current_levels.copy()
        down = has_3 & (mean < down_thr)
        up = has_3 & (mean >= up_thr)
        nxt = np.where(down, np.maximum(1, current_levels - 1), nxt)
        nxt = np.where(up, np.minimum(self.max_level, current_levels + 1), nxt)
        return nxt


class HybridDropStrict(Algorithm):
    """제안 2+3 결합: 하락 기준 70% + 상승은 3회 모두 score >= n/(n+1).
    선행 분석의 두 제안을 합친 형태 — 안전한 하락 + 까다로운 상승."""
    name = "hybrid_drop70_strict_up"

    def update_level_batch(self, current_levels, recent_scores):
        has_3, mean = _safe_mean3(recent_scores)
        thr_per_level = current_levels / (current_levels + 1)
        all_above = (recent_scores >= thr_per_level[:, None]).all(axis=1) & has_3
        nxt = current_levels.copy()
        down = has_3 & (mean < 0.7)
        nxt = np.where(down, np.maximum(1, current_levels - 1), nxt)
        nxt = np.where(all_above, np.minimum(self.max_level, current_levels + 1), nxt)
        return nxt


class NarrowMaintain(Algorithm):
    """레벨 유지 zone 좁힘: 0.65 미만 하락 / 0.75 이상 상승. 유지 zone = 10%만."""
    name = "narrow_maintain_65_75"

    def update_level_batch(self, current_levels, recent_scores):
        has_3, mean = _safe_mean3(recent_scores)
        nxt = current_levels.copy()
        down = has_3 & (mean < 0.65)
        up = has_3 & (mean >= 0.75)
        nxt = np.where(down, np.maximum(1, current_levels - 1), nxt)
        nxt = np.where(up, np.minimum(self.max_level, current_levels + 1), nxt)
        return nxt


class AdaptivePersonal(Algorithm):
    """환자 본인의 누적 score 평균을 기준으로 상대적으로 잘했으면 상승.
    이는 환자 개인 baseline에 맞춰 진척을 결정 — 절대 점수가 낮아도
    본인 평균보다 잘했으면 상승. 안주 패턴을 줄이는 효과 기대."""
    name = "adaptive_personal_baseline"

    def update_level_batch(self, current_levels, recent_scores):
        has_3, mean = _safe_mean3(recent_scores)
        # for "baseline", use mean of available recent (1-3) — only valid when has_3
        # since recent_scores is (N, 3) circular buffer of most-recent, the baseline
        # proxy is the recent mean compared to a fixed reference of 0.60
        # algorithm: relative to fixed 0.60 reference, ±5% bands
        ref = 0.60
        nxt = current_levels.copy()
        down = has_3 & (mean < ref - 0.05)
        up = has_3 & (mean >= ref + 0.20)
        nxt = np.where(down, np.maximum(1, current_levels - 1), nxt)
        nxt = np.where(up, np.minimum(self.max_level, current_levels + 1), nxt)
        return nxt


class AggressiveDropUp(Algorithm):
    """공격적 변화: 하락 75% / 상승 85%. 안주 zone을 매우 좁힘."""
    name = "aggressive_75_85"

    def update_level_batch(self, current_levels, recent_scores):
        has_3, mean = _safe_mean3(recent_scores)
        nxt = current_levels.copy()
        down = has_3 & (mean < 0.75)
        up = has_3 & (mean >= 0.85)
        nxt = np.where(down, np.maximum(1, current_levels - 1), nxt)
        nxt = np.where(up, np.minimum(self.max_level, current_levels + 1), nxt)
        return nxt


class CliffJump(Algorithm):
    """큰 점프: 평균 >0.9 → +2 / 0.7-0.9 → +1 / 0.4-0.7 → 유지 / <0.4 → -2.
    드라마틱한 변화로 자극."""
    name = "cliff_jump"

    def update_level_batch(self, current_levels, recent_scores):
        has_3, mean = _safe_mean3(recent_scores)
        nxt = current_levels.copy()
        jump_up = has_3 & (mean >= 0.9)
        small_up = has_3 & (mean >= 0.7) & (mean < 0.9)
        jump_down = has_3 & (mean < 0.4)
        small_down = has_3 & (mean >= 0.4) & (mean < 0.55)
        nxt = np.where(jump_up, np.minimum(self.max_level, current_levels + 2), nxt)
        nxt = np.where(small_up, np.minimum(self.max_level, current_levels + 1), nxt)
        nxt = np.where(small_down, np.maximum(1, current_levels - 1), nxt)
        nxt = np.where(jump_down, np.maximum(1, current_levels - 2), nxt)
        return nxt


class SlowClimb(Algorithm):
    """천천히 올라감: 상승은 3회 평균 ≥0.85 + 모두 ≥0.7 (보수적 상승) / 하락 < 0.55.
    난이도 빨리 안 올라가게."""
    name = "slow_climb"

    def update_level_batch(self, current_levels, recent_scores):
        has_3, mean = _safe_mean3(recent_scores)
        all_good = (recent_scores >= 0.7).all(axis=1) & has_3
        nxt = current_levels.copy()
        down = has_3 & (mean < 0.55)
        up = all_good & (mean >= 0.85)
        nxt = np.where(down, np.maximum(1, current_levels - 1), nxt)
        nxt = np.where(up, np.minimum(self.max_level, current_levels + 1), nxt)
        return nxt


class Placebo(Algorithm):
    """레벨 절대 안 바꿈. 다른 정책의 baseline 의미 확인용."""
    name = "placebo_no_change"

    def update_level_batch(self, current_levels, recent_scores):
        return current_levels.copy()


class EasyMode(Algorithm):
    """절대 어려워지지 않음 — 평균 < 0.5면 -1, 아니면 유지. 상승 X."""
    name = "easy_mode_down_only"

    def update_level_batch(self, current_levels, recent_scores):
        has_3, mean = _safe_mean3(recent_scores)
        nxt = current_levels.copy()
        down = has_3 & (mean < 0.5)
        nxt = np.where(down, np.maximum(1, current_levels - 1), nxt)
        return nxt


class HardMode(Algorithm):
    """절대 쉬워지지 않음 — 평균 ≥ 0.5면 +1, 아니면 유지. 하락 X."""
    name = "hard_mode_up_only"

    def update_level_batch(self, current_levels, recent_scores):
        has_3, mean = _safe_mean3(recent_scores)
        nxt = current_levels.copy()
        up = has_3 & (mean >= 0.5)
        nxt = np.where(up, np.minimum(self.max_level, current_levels + 1), nxt)
        return nxt


class ScoreFloor(Algorithm):
    """단순한 규칙: 마지막 점수 0.6 이상 → 상승, 0.4 미만 → 하락. 3회 averaging 없음."""
    name = "single_score_floor"

    def update_level_batch(self, current_levels, recent_scores):
        last = recent_scores[:, -1]
        has_last = ~np.isnan(last)
        nxt = current_levels.copy()
        up = has_last & (last >= 0.6)
        down = has_last & (last < 0.4)
        nxt = np.where(up, np.minimum(self.max_level, current_levels + 1), nxt)
        nxt = np.where(down, np.maximum(1, current_levels - 1), nxt)
        return nxt


class RandomWalk(Algorithm):
    """랜덤 ±1 (점수 무시). 통제군 — 알고리즘이 score 신호를 사용해야 한다는 증거."""
    name = "random_walk"

    def __init__(self):
        self._rng = np.random.default_rng(42)

    def update_level_batch(self, current_levels, recent_scores):
        delta = self._rng.choice([-1, 0, 1], size=len(current_levels))
        return np.clip(current_levels + delta, 1, self.max_level)


ALGORITHMS = {
    a.name: a for a in [CurrentRule(), DropAt70(), LevelDifferentiated(),
                        StrictUp(), Expand10(),
                        HybridDropStrict(), NarrowMaintain(), AdaptivePersonal(),
                        AggressiveDropUp(), CliffJump(), SlowClimb(),
                        Placebo(), EasyMode(), HardMode(), ScoreFloor(), RandomWalk()]
}
