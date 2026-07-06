"""
Holdout validation: do the bot's trajectories look like real patient trajectories?

For each key metric, run a 2-sample Kolmogorov–Smirnov test between
  - real holdout patients (the 20% test split from training, by patient_id)
  - simulated patients (CurrentRule, since that's the rule the real data was generated under)

KS p > 0.05 means we cannot reject "same distribution" — i.e. the bot is
distributionally faithful on that metric.

Run:
  python -m bot.validate_bot
"""
from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import GroupShuffleSplit

from .simulator import DTxSimulator, GENERAL_EXERCISES, Policy, _maintain_rate_general
from .algorithms import CurrentRule
from .composers import RandomBalanced

_maintain_rate = _maintain_rate_general  # back-compat alias

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
RANDOM_STATE = 42


def real_holdout_metrics() -> pd.DataFrame:
    """Per-patient outcomes for the test-split patients only (same split as
    training). Metrics computed on GENERAL events only so they match the sim,
    which exposes general/sub events separately."""
    patients = pd.read_parquet(OUT / "patients.parquet")
    events = pd.read_parquet(OUT / "events.parquet")
    eligible = patients[patients["archetype"] != "Z_never_played"].reset_index(drop=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    _, test_idx = next(splitter.split(eligible, groups=eligible["patient_id"]))
    test_ids = set(eligible.iloc[test_idx]["patient_id"])

    from .simulator import GENERAL_EXERCISES, EX_TO_IDX

    rows = []
    for pid, grp in events[events["patient_id"].isin(test_ids)].groupby("patient_id"):
        grp = grp.sort_values("created_at")
        gen = grp[grp["kind"] == "general"]
        if len(gen) == 0:
            continue
        levels = gen["level"].dropna().astype(int).tolist()
        scores = gen["score"].dropna().tolist()
        # tuple form matching _maintain_rate_general signature
        history = [
            (0, EX_TO_IDX.get(row["exercise_name"], 0),
             int(row["level"]) if pd.notna(row["level"]) else 1,
             float(row["score"]) if pd.notna(row["score"]) else 0.0)
            for _, row in gen.iterrows()
        ]
        rows.append({
            "patient_id": pid,
            "n_general_events": len(gen),
            "max_level": int(max(levels)) if levels else 1,
            "last_level": int(levels[-1]) if levels else 1,
            "mean_score": float(np.mean(scores)) if scores else np.nan,
            "maintain_rate": _maintain_rate(history),
        })
    return pd.DataFrame(rows)


def main():
    print("Building real holdout metrics ...")
    real = real_holdout_metrics()
    print(f"  real n = {len(real)}")

    print("Simulating equivalent cohort under current rule ...")
    sim = DTxSimulator(rng_seed=RANDOM_STATE)
    res = sim.run(Policy(RandomBalanced(), CurrentRule()),
                  n_patients=max(800, len(real) * 3), window_days=83)
    sim_df = res["patients"]

    metrics = ["n_general_events", "max_level", "last_level", "mean_score", "maintain_rate"]
    # name mapping: sim_df uses 'n_general_events' too (added in summarizer)
    print()
    print(f"{'metric':16s} | real(mean,n) | sim(mean,n) | KS stat | p-value | pass?")
    print("-" * 88)
    out = {}
    for m in metrics:
        sim_col = m if m in sim_df.columns else m
        a = real[m].dropna().to_numpy()
        b = sim_df[sim_col].dropna().to_numpy()
        if len(a) < 5 or len(b) < 5:
            continue
        ks = ks_2samp(a, b)
        verdict = "✓" if ks.pvalue > 0.05 else "✗"
        print(f"{m:16s} | {a.mean():6.2f} (n={len(a):3d}) | {b.mean():6.2f} (n={len(b):3d}) "
              f"|  {ks.statistic:.3f}  | {ks.pvalue:.4f} |  {verdict}")
        out[m] = {"real_mean": float(a.mean()), "sim_mean": float(b.mean()),
                  "ks": float(ks.statistic), "p": float(ks.pvalue),
                  "pass": bool(ks.pvalue > 0.05)}

    # also compare re-prescription rate
    real_p = pd.read_parquet(OUT / "patients.parquet")
    real_rate = float(real_p.loc[real_p["patient_id"].isin(real["patient_id"]), "did_represcribe"].mean())
    sim_rate = float(sim_df["represcribed"].mean())
    print(f"{'represcribe%':16s} | {real_rate*100:6.2f}%       | {sim_rate*100:6.2f}%      | (rate only)")
    out["represcribe_rate"] = {"real": real_rate, "sim": sim_rate}

    out_path = OUT / "validation_ks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
