"""
DTx patient trajectory builder.

Reads:
  data/assignment_record.csv
  data/prescription.csv

Writes:
  out/events.parquet     -- one row per training event
  out/patients.parquet   -- per-patient summary (for archetype labeling)

Run:
  python -m bot.build_dataset
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

GENERAL_EXERCISES = {"memorize_out_loud", "word_storming", "memorize_with_stories"}


def _parse_dt(s: pd.Series) -> pd.Series:
    """Robust datetime parse for the 0421 'clean' export, which contains some
    malformed values where the date/time separator was dropped, e.g.
    '2026-04-2003:45:44' -> '2026-04-20 03:45:44'. Insert the missing space,
    then let pandas infer (mixed ISO8601)."""
    s = s.astype("string")
    s = s.str.replace(r"(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})", r"\1 \2", regex=True)
    return pd.to_datetime(s, utc=True, format="mixed", errors="coerce")


def load_records() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data/assignment_record.csv")
    df["created_at"] = _parse_dt(df["created_at"])
    df["modified_at"] = _parse_dt(df["modified_at"])
    rec = df["record"].fillna("{}").apply(json.loads)
    df["level"] = rec.apply(lambda r: r.get("level"))
    df["score"] = rec.apply(lambda r: r.get("score"))
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["kind"] = np.where(df["exercise_name"].isin(GENERAL_EXERCISES), "general", "sub")
    return df


def load_prescriptions() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data/prescription.csv")
    for c in ("created_at", "activated_at", "expired_at"):
        df[c] = _parse_dt(df[c])
    return df


def build_events(rec: pd.DataFrame) -> pd.DataFrame:
    ev = rec[["assignment_id", "patient_id", "exercise_name", "kind",
              "level", "score", "created_at", "modified_at"]].copy()
    ev = ev.sort_values(["patient_id", "created_at"]).reset_index(drop=True)
    ev["t"] = ev.groupby("patient_id").cumcount()
    return ev


def per_patient(rec: pd.DataFrame, rx: pd.DataFrame) -> pd.DataFrame:
    rec = rec.sort_values("created_at")
    g_all = rec.groupby("patient_id", sort=False)
    summary = g_all.agg(
        n_records=("assignment_id", "count"),
        first_event=("created_at", "min"),
        last_event=("created_at", "max"),
        mean_score=("score", "mean"),
        all_max_level=("level", "max"),
    )
    summary["all_last_level"] = g_all["level"].last()
    summary["active_days"] = g_all["created_at"].apply(lambda s: s.dt.date.nunique())

    gen = rec[rec["kind"] == "general"].sort_values("created_at")
    gg = gen.groupby("patient_id", sort=False)
    g_summary = pd.DataFrame({
        "n_general": gg.size(),
        "general_max_level": gg["level"].max(),
        "general_last_level": gg["level"].last(),
        "general_median_level": gg["level"].median(),
        "general_mean_score": gg["score"].mean(),
        "n_general_days": gg["created_at"].apply(lambda s: s.dt.date.nunique()),
    })
    summary = summary.join(g_summary, how="left")

    # prescription join (latest prescription per patient drives window)
    rx_sorted = rx.sort_values("created_at")
    rxg = rx_sorted.groupby("patient_id", sort=False).agg(
        n_prescriptions=("prescription_id", "count"),
        first_prescription=("created_at", "min"),
        last_prescription=("created_at", "max"),
        first_activated=("activated_at", "min"),
        last_expired=("expired_at", "max"),
        prescribed_from=("prescribed_from", "last"),
        hospital=("hospital", "last"),
    )
    rxg["did_represcribe"] = rxg["n_prescriptions"] >= 2

    out = summary.join(rxg, how="outer")

    # adherence definitions
    out["window_days"] = (out["last_expired"] - out["first_activated"]).dt.days
    out["window_days"] = out["window_days"].fillna(90).clip(lower=30)

    # Adherence definition (decided after probing several candidates):
    # ratio of general-training days to a fixed 90-day window.
    # the prior analysis' absolute numbers don't reconcile to any single definition,
    # but this preserves the qualitative pattern (high adh -> higher re-prescription).
    out["adherence"] = (out["n_general_days"].fillna(0) / 90).clip(upper=1.0)

    return out.reset_index()


def main():
    print("Loading records...")
    rec = load_records()
    print(f"  records: {len(rec):,}")
    print(f"  unique patients (records): {rec['patient_id'].nunique():,}")

    print("Loading prescriptions...")
    rx = load_prescriptions()
    print(f"  prescriptions: {len(rx):,}")
    print(f"  unique patients (rx): {rx['patient_id'].nunique():,}")

    events = build_events(rec)
    patients = per_patient(rec, rx)

    events.to_parquet(OUT / "events.parquet", index=False)
    patients.to_parquet(OUT / "patients.parquet", index=False)
    print(f"\nWrote {len(events):,} events -> {OUT / 'events.parquet'}")
    print(f"Wrote {len(patients):,} patients -> {OUT / 'patients.parquet'}")

    print("\n=== patient summary describe ===")
    cols = ["n_records", "n_general", "active_days", "window_days",
            "adherence", "general_max_level", "general_last_level"]
    print(patients[cols].describe().round(2).to_string())

    print("\n=== adherence vs represcribe ===")
    patients["adh_band"] = pd.cut(
        patients["adherence"], [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", ">=0.8"])
    print(patients.groupby("adh_band", observed=True)["did_represcribe"]
          .agg(n="count", rx="sum", rate="mean").round(3).to_string())


if __name__ == "__main__":
    main()
