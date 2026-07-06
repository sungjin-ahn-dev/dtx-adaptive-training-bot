"""
Session composers — decide which exercises to prescribe in a sub-exercise session.

A composer is paired with a LevelAdjuster (in algorithms.py) to form a full
treatment Policy. The composer picks WHICH 4 of 9 sub-exercises to prescribe;
the adjuster decides the level for each.

Real data observation:
  - General sessions: always all 3 generals (g=3 s=0)
  - Sub sessions: 4 of 9 sub-exercises, with all 126 possible combos seen
    across the cohort (mean 22.6 distinct combos per patient)
  - General/sub session ratio ~ 1:1
"""
from __future__ import annotations
import numpy as np

SUB_EXERCISES = [
    "memorize_oddly", "word_throwing", "memorize_scene", "word_snake",
    "making_into_one", "clapping", "repeat_and_memorize",
    "word_finding", "memorize_orders",
]
N_SUB_PICK = 4


class SessionComposer:
    """Override `pick_subs` to define a new sub-exercise policy."""
    name = "base"

    def pick_subs(self, archetype: str, recent_scores: np.ndarray,
                  recent_levels: np.ndarray, rng: np.random.Generator) -> list[str]:
        """recent_scores / recent_levels: (9,) arrays for the 9 sub exercises
        (use np.nan when never played). Returns 4 exercise names."""
        raise NotImplementedError


class RandomBalanced(SessionComposer):
    """현행 baseline: 9개 중 4개 무작위. 실제 데이터 패턴(126조합 모두 등장)에 가장 가까움."""
    name = "random_balanced"

    def pick_subs(self, archetype, recent_scores, recent_levels, rng):
        idx = rng.choice(9, size=N_SUB_PICK, replace=False)
        return [SUB_EXERCISES[i] for i in idx]


class FixedRotation(SessionComposer):
    """결정적 순환: 환자별로 9->4 슬라이딩 윈도우. 모든 운동을 균등하게 cover."""
    name = "fixed_rotation"

    def __init__(self):
        self._counter = {}    # per-archetype rotation index

    def pick_subs(self, archetype, recent_scores, recent_levels, rng):
        i = self._counter.get(archetype, 0)
        idx = [(i + j) % 9 for j in range(N_SUB_PICK)]
        self._counter[archetype] = (i + 1) % 9
        return [SUB_EXERCISES[k] for k in idx]


class WeaknessFocused(SessionComposer):
    """환자가 점수 낮은 4개 운동에 집중. 미경험 운동(NaN)은 최우선."""
    name = "weakness_focused"

    def pick_subs(self, archetype, recent_scores, recent_levels, rng):
        # NaN (never played) gets lowest score effectively
        scores = np.where(np.isnan(recent_scores), -1.0, recent_scores)
        idx = np.argsort(scores)[:N_SUB_PICK]
        return [SUB_EXERCISES[i] for i in idx]


class ArchetypeAware(SessionComposer):
    """선행 분석(2차) 인사이트 직접 반영:
    - B (stalled_settled, 안주형): memorize_oddly 유지가 재처방↑ → 항상 포함
    - C (stalled_bored, 지루형): word_snake 정체가 재처방↓ → 빼고 변화 위주
    - A (early_dropout): 쉬운 운동 위주 (점수 높은 4개)
    - 그 외: balanced random
    """
    name = "archetype_aware"
    PRIORITY = {  # exercises to always include, per archetype
        "B_stalled_settled": ["memorize_oddly"],
        "A_early_dropout": [],  # use top-score logic
    }
    AVOID = {
        "C_stalled_bored": ["word_snake"],
    }

    def pick_subs(self, archetype, recent_scores, recent_levels, rng):
        priority = self.PRIORITY.get(archetype, [])
        avoid = self.AVOID.get(archetype, [])

        picked = list(priority)
        if archetype == "A_early_dropout":
            # easy-first: top-scored exercises
            scores = np.where(np.isnan(recent_scores), 0.5, recent_scores)
            order = np.argsort(-scores)
            for i in order:
                if SUB_EXERCISES[i] not in picked and len(picked) < N_SUB_PICK:
                    picked.append(SUB_EXERCISES[i])
        else:
            # otherwise random fill from allowed pool
            pool = [e for e in SUB_EXERCISES if e not in picked and e not in avoid]
            rng.shuffle(pool)
            for e in pool:
                if len(picked) >= N_SUB_PICK:
                    break
                picked.append(e)
            # if avoid pool emptied us short, allow avoid exercises
            if len(picked) < N_SUB_PICK:
                fallback = [e for e in SUB_EXERCISES if e not in picked]
                rng.shuffle(fallback)
                picked.extend(fallback[:N_SUB_PICK - len(picked)])
        return picked[:N_SUB_PICK]


class ScoreBalanced(SessionComposer):
    """균형 선택: top 2 잘하는 운동 + bottom 2 못하는 운동. 자신감 + 도전 mix."""
    name = "score_balanced"

    def pick_subs(self, archetype, recent_scores, recent_levels, rng):
        scores = np.where(np.isnan(recent_scores), 0.5, recent_scores)
        order = np.argsort(scores)        # ascending
        bottom2 = order[:2]                # weakest 2
        top2 = order[-2:]                  # strongest 2
        picked_idx = np.concatenate([bottom2, top2])
        return [SUB_EXERCISES[i] for i in picked_idx]


COMPOSERS = {
    c.name: c for c in [RandomBalanced(), FixedRotation(),
                        WeaknessFocused(), ArchetypeAware(),
                        ScoreBalanced()]
}
