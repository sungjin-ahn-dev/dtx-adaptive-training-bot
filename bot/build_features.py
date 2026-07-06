"""
Build three feature tables — one per target our bot needs to predict.

Outputs in out/:
  score_features.parquet      — 1 row / event  → target = score
  session_features.parquet    — 1 row / session → targets = gap_to_next (days), is_terminal
  patient_features.parquet    — 1 row / patient → target = did_represcribe

Run:
  python -m bot.build_features
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

# how long with no activity before we call a session "terminal"
TERMINAL_GAP_DAYS = 14


def _rolling_recent(series: pd.Series, n: int) -> pd.DataFrame:
    """For each row, return mean/min/max of the previous N values (strictly prior)."""
    shifted = series.shift(1)
    return pd.DataFrame({
        f"r{n}_mean": shifted.rolling(n, min_periods=1).mean(),
        f"r{n}_min": shifted.rolling(n, min_periods=1).min(),
        f"r{n}_max": shifted.rolling(n, min_periods=1).max(),
    })


def build_score_features(events: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    df = events.sort_values(["patient_id", "exercise_name", "created_at"]).copy()

    # per (patient, exercise) rolling score features
    grp_pe = df.groupby(["patient_id", "exercise_name"], sort=False)
    df = df.reset_index(drop=True)
    rolled = (
        df.groupby(["patient_id", "exercise_name"], sort=False)["score"]
        .apply(lambda s: _rolling_recent(s, 3))
        .reset_index(level=[0, 1], drop=True)
    )
    df[["r3_mean", "r3_min", "r3_max"]] = rolled

    # count features
    df["n_at_exercise"] = grp_pe.cumcount()                           # prior count of this exercise
    df["n_at_exercise_level"] = df.groupby(
        ["patient_id", "exercise_name", "level"], sort=False
    ).cumcount()                                                       # prior count at same (ex, lvl)

    # overall count for the patient
    df["t_overall"] = df.sort_values("created_at").groupby("patient_id").cumcount()

    # time features
    df = df.sort_values("created_at")
    df["ts_local"] = df["created_at"].dt.tz_convert("Asia/Seoul")
    df["hour"] = df["ts_local"].dt.hour
    df["dow"] = df["ts_local"].dt.dayofweek

    # days into prescription window
    df = df.merge(
        patients[["patient_id", "first_activated", "archetype", "adherence",
                  "general_max_level", "general_last_level"]],
        on="patient_id", how="left",
    )
    df["days_into_window"] = (df["created_at"] - df["first_activated"]).dt.total_seconds() / 86400
    df["days_into_window"] = df["days_into_window"].fillna(-1)

    out_cols = ["patient_id", "assignment_id", "created_at",
                "exercise_name", "kind", "level", "score",
                "archetype", "adherence",
                "r3_mean", "r3_min", "r3_max",
                "n_at_exercise", "n_at_exercise_level", "t_overall",
                "hour", "dow", "days_into_window"]
    return df[out_cols]


def build_session_features(events: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """A session = events sharing the same (patient_id, created_at) minute bucket."""
    ev = events.copy()
    ev["session_key"] = ev["created_at"].dt.floor("min")
    sess = ev.groupby(["patient_id", "session_key"], sort=False).agg(
        n_in_session=("assignment_id", "count"),
        n_general_in_session=("kind", lambda s: (s == "general").sum()),
        session_mean_score=("score", "mean"),
        session_max_level=("level", "max"),
        session_mean_level=("level", "mean"),
    ).reset_index()

    sess = sess.sort_values(["patient_id", "session_key"]).reset_index(drop=True)
    sess["session_idx"] = sess.groupby("patient_id").cumcount()
    sess["n_sessions_so_far"] = sess["session_idx"]
    sess["prev_session"] = sess.groupby("patient_id")["session_key"].shift(1)
    sess["next_session"] = sess.groupby("patient_id")["session_key"].shift(-1)
    sess["days_since_prev"] = (sess["session_key"] - sess["prev_session"]).dt.total_seconds() / 86400
    sess["days_to_next"] = (sess["next_session"] - sess["session_key"]).dt.total_seconds() / 86400

    # rolling mean of prior gaps
    sess["r3_gap_mean"] = (
        sess.groupby("patient_id")["days_since_prev"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    # terminal session: no next session OR next session > TERMINAL_GAP_DAYS away
    sess["is_terminal"] = sess["days_to_next"].isna() | (sess["days_to_next"] > TERMINAL_GAP_DAYS)

    # merge patient info
    sess = sess.merge(
        patients[["patient_id", "first_activated", "last_expired",
                  "archetype", "adherence", "did_represcribe",
                  "general_max_level", "general_last_level"]],
        on="patient_id", how="left",
    )
    sess["days_into_window"] = (sess["session_key"] - sess["first_activated"]).dt.total_seconds() / 86400
    sess["days_to_expiry"] = (sess["last_expired"] - sess["session_key"]).dt.total_seconds() / 86400

    sess["ts_local"] = sess["session_key"].dt.tz_convert("Asia/Seoul")
    sess["hour"] = sess["ts_local"].dt.hour
    sess["dow"] = sess["ts_local"].dt.dayofweek

    cols = ["patient_id", "session_key", "session_idx",
            "n_in_session", "n_general_in_session",
            "session_mean_score", "session_max_level", "session_mean_level",
            "days_since_prev", "r3_gap_mean", "days_into_window", "days_to_expiry",
            "hour", "dow",
            "archetype", "adherence", "did_represcribe",
            "days_to_next", "is_terminal"]
    return sess[cols]


def build_patient_features(patients: pd.DataFrame, events: pd.DataFrame,
                           sessions: pd.DataFrame) -> pd.DataFrame:
    pf = patients.copy()

    # recent score trend per patient (across general exercises)
    gen = events[events["kind"] == "general"].sort_values("created_at")
    recent_score = gen.groupby("patient_id")["score"].apply(
        lambda s: s.tail(10).mean() - s.head(min(10, len(s))).mean()
    ).rename("score_trend10")
    pf = pf.merge(recent_score, on="patient_id", how="left")

    # mean session interval
    sess_int = (
        sessions.groupby("patient_id")["days_since_prev"]
        .agg(median_gap_days="median", mean_gap_days="mean")
        .reset_index()
    )
    pf = pf.merge(sess_int, on="patient_id", how="left")

    # number of sessions and terminal sessions
    sess_counts = (
        sessions.groupby("patient_id")
        .agg(n_sessions=("session_idx", "count"),
             n_terminal=("is_terminal", "sum"))
        .reset_index()
    )
    pf = pf.merge(sess_counts, on="patient_id", how="left")

    target = pf["did_represcribe"].astype("Int8")

    cols = ["patient_id", "archetype", "adherence",
            "n_records", "n_general", "n_general_days", "active_days",
            "general_max_level", "general_last_level", "general_median_level",
            "general_mean_score", "score_trend10",
            "median_gap_days", "mean_gap_days",
            "n_sessions", "n_terminal",
            "prescribed_from", "hospital",
            "did_represcribe"]
    out = pf[cols].copy()
    out["did_represcribe"] = target
    return out


def main():
    print("Loading events / patients ...")
    events = pd.read_parquet(OUT / "events.parquet")
    patients = pd.read_parquet(OUT / "patients.parquet")

    print("Building score features ...")
    score_feat = build_score_features(events, patients)
    score_feat.to_parquet(OUT / "score_features.parquet", index=False)
    print(f"  -> score_features: {len(score_feat):,} rows")

    print("Building session features ...")
    sess_feat = build_session_features(events, patients)
    sess_feat.to_parquet(OUT / "session_features.parquet", index=False)
    print(f"  -> session_features: {len(sess_feat):,} rows")

    print("Building patient features ...")
    pat_feat = build_patient_features(patients, events, sess_feat)
    pat_feat.to_parquet(OUT / "patient_features.parquet", index=False)
    print(f"  -> patient_features: {len(pat_feat):,} rows")

    print("\n=== quick sanity ===")
    print("score target stats:", score_feat["score"].describe().round(3).to_dict())
    print("session days_to_next stats:",
          sess_feat["days_to_next"].dropna().describe().round(2).to_dict())
    print("session is_terminal rate:", float(sess_feat["is_terminal"].mean().round(3)))
    print("patient did_represcribe rate:",
          float(pat_feat["did_represcribe"].mean().round(3)))


if __name__ == "__main__":
    main()
