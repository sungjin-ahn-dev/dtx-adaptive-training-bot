"""
Cohort-step DTx patient simulator (v3).

A treatment Policy = (SessionComposer, LevelAdjuster).
  - Composer picks WHICH sub-exercises (4 of 9) for sub sessions.
  - Adjuster picks the LEVEL per exercise (current 50/80 rule, etc.).

Each iteration is one session. General and sub sessions alternate ~1:1, matching
the real-data ratio. Predictions are batched across the alive cohort.

Run:
  python -m bot.simulator                  # compare all policies (composer × adjuster)
  python -m bot.simulator --adjuster drop_at_70 --composer weakness_focused
"""
from __future__ import annotations

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .algorithms import ALGORITHMS, Algorithm
from .composers import COMPOSERS, SessionComposer, SUB_EXERCISES, N_SUB_PICK
from . import calibration

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="Mean of empty slice|All-NaN slice encountered")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
MODELS = ROOT / "models"

GENERAL_EXERCISES = ["memorize_out_loud", "word_storming", "memorize_with_stories"]
ALL_EXERCISES = GENERAL_EXERCISES + SUB_EXERCISES                # 12 total
N_EX = len(ALL_EXERCISES)
EX_TO_IDX = {e: i for i, e in enumerate(ALL_EXERCISES)}
GENERAL_IDX = np.array([EX_TO_IDX[e] for e in GENERAL_EXERCISES])
SUB_IDX = np.array([EX_TO_IDX[e] for e in SUB_EXERCISES])
KIND_BY_IDX = ["general"] * len(GENERAL_EXERCISES) + ["sub"] * len(SUB_EXERCISES)

DEFAULT_WINDOW_DAYS = 83
SCORE_NOISE_SD = 0.18
MAX_ITERS = 800
HOUR_OF_DAY = 10
CATEGORICAL_COLS = {"exercise_name", "kind", "archetype", "prescribed_from", "hospital"}

# v4: Soft cap parameters
# When an algorithm pushes a virtual patient above their personal capability
# (target_max_level), score gets a linear penalty (=> they fail more, algorithm
# naturally drops them back). When session count exceeds target_max_sessions,
# dropout probability ramps up instead of a hard cut.
ABILITY_PENALTY_PER_LEVEL = 0.12   # score subtraction per level above ceiling
SESSION_CAP_SOFT_START = 0.8       # start ramping dropout at 80% of session cap
SESSION_CAP_SOFT_SLOPE = 1.5       # additional dropout prob per (frac - 0.8)


# --------------------------------------------------------------------------- #
# Policy = composer + adjuster
# --------------------------------------------------------------------------- #

class Policy:
    def __init__(self, composer: SessionComposer, adjuster: Algorithm):
        self.composer = composer
        self.adjuster = adjuster
        self.name = f"{adjuster.name} ⨯ {composer.name}"


def _to_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df.reindex(columns=feature_cols).copy()
    for c in CATEGORICAL_COLS & set(X.columns):
        X[c] = X[c].astype("category")
    return X


# --------------------------------------------------------------------------- #
# Cohort state
# --------------------------------------------------------------------------- #

class Cohort:
    def __init__(self, n: int, archetypes: np.ndarray, adherences: np.ndarray,
                 target_max_sessions: np.ndarray, target_max_level: np.ndarray,
                 max_level_algo: int):
        self.n = n
        self.archetype = archetypes
        self.adherence = adherences.astype(np.float32)
        self.target_max_sessions = target_max_sessions.astype(np.int32)
        self.target_max_level = target_max_level.astype(np.int32)
        self.levels = np.ones((n, N_EX), dtype=np.int32)
        self.recent_scores = np.full((n, N_EX, 3), np.nan, dtype=np.float32)
        self.n_at_exercise = np.zeros((n, N_EX), dtype=np.int32)
        self.n_at_ex_level = np.zeros((n, N_EX, max_level_algo + 2), dtype=np.int32)
        self.t_overall = np.zeros(n, dtype=np.int32)
        self.day_index = np.zeros(n, dtype=np.int32)
        self.last_day = np.full(n, -1, dtype=np.int32)
        self.session_idx = np.zeros(n, dtype=np.int32)
        self.gaps_history = [list() for _ in range(n)]
        self.alive = np.ones(n, dtype=bool)
        self.drop_day = np.full(n, -1, dtype=np.int32)
        # flat history: list of (day, ex_idx, level, score) per patient
        self.history = [list() for _ in range(n)]


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #

class DTxSimulator:
    def __init__(self, models_dir: Path = MODELS, rng_seed: int = 0):
        self.score_pack = joblib.load(models_dir / "score_model.joblib")
        self.dropout_pack = joblib.load(models_dir / "dropout_model.joblib")
        self.gap_pack = joblib.load(models_dir / "gap_model.joblib")
        self.term_pack = joblib.load(models_dir / "termination_model.joblib")
        self.rng = np.random.default_rng(rng_seed)
        patients = pd.read_parquet(OUT / "patients.parquet")
        eligible = patients[patients["archetype"] != "Z_never_played"]
        self._archetype_pool = eligible[["archetype", "adherence"]].reset_index(drop=True)

    def sample_cohort(self, n: int, max_level: int) -> Cohort:
        idx = self.rng.integers(0, len(self._archetype_pool), size=n)
        sampled = self._archetype_pool.iloc[idx]
        archetypes = sampled["archetype"].to_numpy()
        target_max = calibration.sample_target_max_sessions(archetypes, self.rng)
        target_lvl = calibration.sample_target_max_level(archetypes, self.rng)
        return Cohort(
            n=n,
            archetypes=archetypes,
            adherences=sampled["adherence"].to_numpy(),
            target_max_sessions=target_max,
            target_max_level=target_lvl,
            max_level_algo=max_level,
        )

    # ------------- score prediction (batched per exercise) ------------- #
    def _predict_scores(self, c: Cohort, ex_idx: int, new_levels: np.ndarray,
                         patient_indices: np.ndarray) -> np.ndarray:
        if len(patient_indices) == 0:
            return np.array([], dtype=np.float32)
        n = len(patient_indices)
        recent = c.recent_scores[patient_indices, ex_idx, :]
        df = pd.DataFrame({
            "exercise_name": np.full(n, ALL_EXERCISES[ex_idx]),
            "kind": np.full(n, KIND_BY_IDX[ex_idx]),
            "level": new_levels.astype(np.int32),
            "archetype": c.archetype[patient_indices],
            "adherence": c.adherence[patient_indices],
            "r3_mean": np.nanmean(recent, axis=1),
            "r3_min": np.nanmin(recent, axis=1),
            "r3_max": np.nanmax(recent, axis=1),
            "n_at_exercise": c.n_at_exercise[patient_indices, ex_idx],
            "n_at_exercise_level": c.n_at_ex_level[patient_indices, ex_idx, new_levels],
            "t_overall": c.t_overall[patient_indices],
            "hour": np.full(n, HOUR_OF_DAY, dtype=np.int32),
            "dow": (c.day_index[patient_indices] % 7).astype(np.int32),
            "days_into_window": c.day_index[patient_indices].astype(np.float32),
        })
        X = _to_X(df, self.score_pack["features"])
        pred = np.clip(self.score_pack["model"].predict(X), 0, 1)

        # SOFT ABILITY CAP: penalize scores when algorithm pushed patient above
        # their personal target_max_level. Penalty is linear in the overshoot.
        # This lets aggressive algorithms PUSH, but the patient fails more →
        # algorithm naturally pulls them back.
        cap = c.target_max_level[patient_indices]
        overshoot = np.maximum(0, new_levels - cap)
        pred = pred - ABILITY_PENALTY_PER_LEVEL * overshoot

        noise = self.rng.normal(0, SCORE_NOISE_SD, size=n)
        return np.clip(pred + noise, 0, 1).astype(np.float32)

    # ------------- dropout + gap (per session, batched over active) ------------- #
    def _build_session_features(self, c: Cohort, active: np.ndarray,
                                  window_days: int,
                                  sess_n_in: np.ndarray, sess_n_gen_in: np.ndarray,
                                  sess_mean_score: np.ndarray, sess_max_level: np.ndarray,
                                  sess_mean_level: np.ndarray) -> pd.DataFrame:
        n = len(active)
        days_since_prev = np.where(
            c.last_day[active] >= 0,
            (c.day_index[active] - c.last_day[active]).astype(np.float32),
            np.nan,
        )
        r3_gap = np.array([
            float(np.mean(c.gaps_history[i][-3:])) if c.gaps_history[i] else np.nan
            for i in active
        ], dtype=np.float32)
        days_into = c.day_index[active].astype(np.float32)
        return pd.DataFrame({
            "archetype": c.archetype[active],
            "adherence": c.adherence[active],
            "session_idx": c.session_idx[active].astype(np.int32),
            "n_in_session": sess_n_in.astype(np.int32),
            "n_general_in_session": sess_n_gen_in.astype(np.int32),
            "session_mean_score": sess_mean_score.astype(np.float32),
            "session_max_level": sess_max_level.astype(np.float32),
            "session_mean_level": sess_mean_level.astype(np.float32),
            "days_since_prev": days_since_prev,
            "r3_gap_mean": r3_gap,
            "days_into_window": days_into,
            "days_to_expiry": (window_days - days_into).astype(np.float32),
            "hour": np.full(n, HOUR_OF_DAY, dtype=np.int32),
            "dow": (c.day_index[active] % 7).astype(np.int32),
        })

    def _predict_dropout(self, df: pd.DataFrame) -> np.ndarray:
        X = _to_X(df, self.dropout_pack["features"])
        return self.dropout_pack["model"].predict_proba(X)[:, 1]

    def _predict_gap_days(self, df: pd.DataFrame) -> np.ndarray:
        X = _to_X(df, self.gap_pack["features"])
        return np.expm1(self.gap_pack["model"].predict(X))

    # ------------- termination (re-prescription) per patient ------------- #
    def _predict_represcribe(self, c: Cohort) -> np.ndarray:
        rows = []
        for i in range(c.n):
            gen_history = [h for h in c.history[i] if h[1] in GENERAL_IDX]
            scores = np.array([h[3] for h in gen_history]) if gen_history else np.array([])
            levels = np.array([h[2] for h in gen_history]) if gen_history else np.array([1])
            gaps = c.gaps_history[i]
            trend = 0.0
            if len(scores) >= 4:
                trend = float(scores[-10:].mean() - scores[: min(10, len(scores))].mean())
            n_sess = int(c.session_idx[i])
            rows.append({
                "archetype": c.archetype[i],
                "adherence": float(c.adherence[i]),
                "n_records": len(c.history[i]),
                "n_general": len(scores),
                "n_general_days": n_sess,
                "active_days": n_sess,
                "general_max_level": int(levels.max()) if levels.size else 1,
                "general_last_level": int(levels[-1]) if levels.size else 1,
                "general_median_level": float(np.median(levels)) if levels.size else 1.0,
                "general_mean_score": float(scores.mean()) if scores.size else np.nan,
                "score_trend10": trend,
                "median_gap_days": float(np.median(gaps)) if gaps else np.nan,
                "mean_gap_days": float(np.mean(gaps)) if gaps else np.nan,
                "n_sessions": n_sess,
                "n_terminal": 1 if not c.alive[i] else 0,
                "prescribed_from": "dtx-dashboard",
            })
        X = _to_X(pd.DataFrame(rows), self.term_pack["features"])
        return self.term_pack["model"].predict_proba(X)[:, 1]

    # ------------- play a set of exercises for the active patients ------------- #
    def _play_session(self, c: Cohort, active: np.ndarray, policy: Policy,
                      session_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (n_played, mean_score, max_level) per active patient."""
        max_lvl_algo = policy.adjuster.max_level
        n_active = len(active)
        if n_active == 0:
            return (np.zeros(0, dtype=np.int32),
                    np.zeros(0, dtype=np.float32),
                    np.zeros(0, dtype=np.float32))

        # build "(patient, exercise_idx)" list to play this session
        if session_type == "general":
            ex_indices_per_p = [GENERAL_IDX for _ in range(n_active)]
        else:
            # composer picks 4 of 9 subs PER PATIENT
            ex_indices_per_p = []
            for k, i in enumerate(active):
                recent = np.array([
                    np.nanmean(c.recent_scores[i, EX_TO_IDX[ex], :])
                    for ex in SUB_EXERCISES
                ], dtype=np.float32)
                lvls = np.array([
                    c.levels[i, EX_TO_IDX[ex]] for ex in SUB_EXERCISES
                ], dtype=np.int32)
                picks = policy.composer.pick_subs(
                    c.archetype[i], recent, lvls, self.rng)
                idx = np.array([EX_TO_IDX[ex] for ex in picks], dtype=np.int32)
                ex_indices_per_p.append(idx)

        # For each ex_idx that appears for any patient, batch-process its players.
        # Most general sessions: same 3 idx for all → 3 batches.
        # Sub sessions: variable 4 idx per patient → up to 9 batches.
        sess_score_sum = np.zeros(n_active, dtype=np.float64)
        sess_score_n = np.zeros(n_active, dtype=np.int32)
        sess_max_lvl = np.zeros(n_active, dtype=np.int32)

        # invert: ex_idx -> list of patient positions (k in active)
        plays_by_ex = defaultdict(list)   # ex_idx -> [k positions]
        for k, ex_idxs in enumerate(ex_indices_per_p):
            for ei in ex_idxs:
                plays_by_ex[int(ei)].append(k)

        for ex_idx, k_list in plays_by_ex.items():
            k_arr = np.array(k_list, dtype=np.int32)
            patient_idx = active[k_arr]
            cur_lvl = c.levels[patient_idx, ex_idx]
            recent_3 = c.recent_scores[patient_idx, ex_idx, :]
            new_lvl = policy.adjuster.update_level_batch(cur_lvl, recent_3)
            # only algorithm's own max_level is a hard cap; patient ability is
            # a SOFT cap enforced via score penalty in _predict_scores
            new_lvl = np.clip(new_lvl, 1, max_lvl_algo)

            scores = self._predict_scores(c, ex_idx, new_lvl, patient_idx)

            # update state
            c.levels[patient_idx, ex_idx] = new_lvl
            c.recent_scores[patient_idx, ex_idx, :-1] = c.recent_scores[patient_idx, ex_idx, 1:]
            c.recent_scores[patient_idx, ex_idx, -1] = scores
            c.n_at_exercise[patient_idx, ex_idx] += 1
            c.n_at_ex_level[patient_idx, ex_idx, new_lvl] += 1

            sess_score_sum[k_arr] += scores
            sess_score_n[k_arr] += 1
            sess_max_lvl[k_arr] = np.maximum(sess_max_lvl[k_arr], new_lvl)

            for pos, pid in enumerate(patient_idx):
                c.history[int(pid)].append(
                    (int(c.day_index[pid]), ex_idx, int(new_lvl[pos]), float(scores[pos]))
                )

        c.t_overall[active] += sess_score_n
        c.session_idx[active] += 1
        mean_score = sess_score_sum / np.maximum(1, sess_score_n)
        return sess_score_n, mean_score.astype(np.float32), sess_max_lvl.astype(np.float32)

    # ------------- main loop ------------- #
    def run(self, policy: Policy, n_patients: int = 400,
            window_days: int = DEFAULT_WINDOW_DAYS) -> dict:
        c = self.sample_cohort(n_patients, max_level=policy.adjuster.max_level)
        # alternate session types matching real-data ratio (~1:1)
        session_types = ["general", "sub"]

        for it in range(MAX_ITERS):
            active = np.where(c.alive & (c.day_index < window_days))[0]
            if len(active) == 0:
                break
            stype = session_types[it % 2]

            n_played, mean_score, max_lvl = self._play_session(c, active, policy, stype)
            n_gen_in = n_played if stype == "general" else np.zeros_like(n_played)

            sess_df = self._build_session_features(
                c, active, window_days,
                n_played, n_gen_in, mean_score, max_lvl, max_lvl.astype(np.float32))

            # SOFT session cap: dropout prob ramps up once patient passes
            # SESSION_CAP_SOFT_START * target_max_sessions, instead of hard kill.
            p_drop = self._predict_dropout(sess_df)
            sess_frac = c.session_idx[active] / np.maximum(1, c.target_max_sessions[active])
            extra_drop = np.maximum(0.0, sess_frac - SESSION_CAP_SOFT_START) * SESSION_CAP_SOFT_SLOPE
            p_drop_combined = np.minimum(1.0, p_drop + extra_drop)
            terminal = self.rng.random(len(active)) < p_drop_combined

            drop_idx = active[terminal]
            c.alive[drop_idx] = False
            c.drop_day[drop_idx] = c.day_index[drop_idx]

            keep_idx = active[~terminal]
            if len(keep_idx) == 0:
                continue
            gap_days = self._predict_gap_days(sess_df.iloc[~terminal])
            gap_days = np.clip(gap_days, 0.1, None)
            adv = np.maximum(1, np.round(gap_days).astype(np.int32))
            c.last_day[keep_idx] = c.day_index[keep_idx]
            c.day_index[keep_idx] = c.day_index[keep_idx] + adv
            for k, i in enumerate(keep_idx):
                c.gaps_history[i].append(float(gap_days[k]))

        p_rx_raw = self._predict_represcribe(c)
        # dynamic archetype: reclassify each patient by their observed behaviour
        dynamic_arch = np.array([
            _reclassify_archetype(c.history[i], window_days) for i in range(c.n)
        ])
        p_rx = calibration.calibrate_represcribe(p_rx_raw, dynamic_arch)
        rx = self.rng.random(c.n) < p_rx

        return self._summarize(c, p_rx, rx, policy, window_days, dynamic_arch)

    # ------------- summary ------------- #
    def _summarize(self, c: Cohort, p_rx: np.ndarray, rx: np.ndarray,
                   policy: Policy, window_days: int,
                   dynamic_arch: np.ndarray) -> dict:
        rows = []
        for i in range(c.n):
            gen_history = [h for h in c.history[i] if h[1] in GENERAL_IDX]
            sub_history = [h for h in c.history[i] if h[1] in SUB_IDX]
            gen_levels = [h[2] for h in gen_history] or [1]
            gen_scores = [h[3] for h in gen_history] or [float("nan")]
            rows.append({
                "patient_id": f"sim_{i:05d}",
                "archetype": c.archetype[i],
                "archetype_final": dynamic_arch[i],
                "n_sessions": int(c.session_idx[i]),
                "n_general_events": len(gen_history),
                "n_sub_events": len(sub_history),
                "max_level": int(max(gen_levels)),
                "last_level": int(gen_levels[-1]),
                "mean_score": float(np.nanmean(gen_scores)),
                "dropped": bool(not c.alive[i]),
                "p_represcribe": float(p_rx[i]),
                "represcribed": bool(rx[i]),
                "maintain_rate": _maintain_rate_general(c.history[i]),
            })
        df = pd.DataFrame(rows)
        summary = {
            "policy": policy.name,
            "composer": policy.composer.name,
            "adjuster": policy.adjuster.name,
            "n_patients": int(c.n),
            "window_days": int(window_days),
            "drop_rate": float(df["dropped"].mean()),
            "represcribe_rate": float(df["represcribed"].mean()),
            "expected_represcribe_rate": float(df["p_represcribe"].mean()),
            "mean_sessions": float(df["n_sessions"].mean()),
            "mean_general_events": float(df["n_general_events"].mean()),
            "mean_sub_events": float(df["n_sub_events"].mean()),
            "mean_max_level": float(df["max_level"].mean()),
            "mean_last_level": float(df["last_level"].mean()),
            "mean_score": float(df["mean_score"].mean()),
            "mean_maintain_rate": float(df["maintain_rate"].mean()),
            "max_level_hist": df["max_level"].value_counts().sort_index().to_dict(),
            "represcribe_by_archetype": (
                df.groupby("archetype")["represcribed"].mean().round(3).to_dict()),
            "final_archetype_dist": (
                df["archetype_final"].value_counts(normalize=True).round(3).to_dict()),
            "init_archetype_dist": (
                df["archetype"].value_counts(normalize=True).round(3).to_dict()),
        }
        return {"summary": summary, "patients": df}


def _reclassify_archetype(history: list[tuple], window_days: int) -> str:
    """Reclassify a virtual patient based on their *observed* trajectory, using
    the same rule set as label_archetypes.py. Lets algorithm effects bleed into
    termination calibration: a policy that pushes patients from C→B or F→E
    raises the cohort's expected re-prescription rate."""
    gen = [h for h in history if h[1] in GENERAL_IDX]
    if not gen:
        return "Z_never_played"
    levels = [h[2] for h in gen]
    n_gen = len(gen)
    max_lvl = max(levels)
    last_lvl = levels[-1]
    active_days = len(set(h[0] for h in gen))
    adherence = min(1.0, active_days / 90)

    if max_lvl == 1 and n_gen < 21:
        return "A_early_dropout"
    if max_lvl == 7:
        return "E_maxed_out"
    if last_lvl in (4, 5) and n_gen >= 21:
        return "D_mid_wall"
    if max_lvl in (2, 3):
        return "B_stalled_settled" if adherence >= 0.5 else "C_stalled_bored"
    return "F_steady_progress"


def _maintain_rate_general(history: list[tuple]) -> float:
    """Fraction of consecutive same-(general) events that stayed at the same level."""
    by_ex = defaultdict(list)
    for day, ex_idx, lvl, score in history:
        if ex_idx in GENERAL_IDX:
            by_ex[ex_idx].append(lvl)
    maint, total = 0, 0
    for levels in by_ex.values():
        for a, b in zip(levels[:-1], levels[1:]):
            total += 1
            if a == b:
                maint += 1
    return maint / total if total else float("nan")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def compare(composer_names: list[str], adjuster_names: list[str],
            n_patients: int, window_days: int, seed: int = 0) -> pd.DataFrame:
    rows = []
    for cn in composer_names:
        if cn not in COMPOSERS:
            raise SystemExit(f"unknown composer: {cn}; choose from {list(COMPOSERS)}")
        for an in adjuster_names:
            if an not in ALGORITHMS:
                raise SystemExit(f"unknown adjuster: {an}; choose from {list(ALGORITHMS)}")
            policy = Policy(COMPOSERS[cn], ALGORITHMS[an])
            print(f"  running {policy.name} ...", flush=True)
            sim = DTxSimulator(rng_seed=seed)
            res = sim.run(policy, n_patients=n_patients, window_days=window_days)
            s = res["summary"]
            rows.append({
                "composer": cn, "adjuster": an,
                "represcribe_rate": s["represcribe_rate"],
                "drop_rate": s["drop_rate"],
                "mean_max_level": s["mean_max_level"],
                "mean_last_level": s["mean_last_level"],
                "mean_general": s["mean_general_events"],
                "mean_sub": s["mean_sub_events"],
                "mean_score": s["mean_score"],
                "mean_maintain": s["mean_maintain_rate"],
            })
    return pd.DataFrame(rows).round(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--composer", nargs="*", default=list(COMPOSERS))
    p.add_argument("--adjuster", nargs="*", default=list(ALGORITHMS))
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--days", type=int, default=DEFAULT_WINDOW_DAYS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="simulation_compare.csv")
    args = p.parse_args()

    print(f"Simulating {args.n} patients × {args.days} days × "
          f"{len(args.composer)} composers × {len(args.adjuster)} adjusters "
          f"= {len(args.composer)*len(args.adjuster)} policies")
    df = compare(args.composer, args.adjuster, args.n, args.days, args.seed)
    print()
    print(df.to_string(index=False))
    out_path = OUT / args.out
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
