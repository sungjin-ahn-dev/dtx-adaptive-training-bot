"""
Label each patient with one of 6 behavioral archetypes derived from
prior internal EDA findings + our adherence definition.

A. early_dropout    : 사용 습관 미형성. general max_level=1 & n_general<21
B. stalled_settled  : 안주형. low-mid level 정체, 그래도 꾸준히 함 (adh>=0.5)
C. stalled_bored    : 지루형. low-mid level 정체 + 참여 떨어짐 (adh<0.5)
D. mid_wall         : 중간 벽 (레벨 4-5 정체기), n_general>=21
E. maxed_out        : 레벨 7 도달 (성실군의 "졸업" 패턴)
F. steady_progress  : 진행 중 (위 어디에도 안 들어감)
Z. never_played     : 처방만 받고 한 번도 안 함 (분석 제외용)

Run:
  python -m bot.label_archetypes
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"


def label_one(row) -> str:
    n = row["n_general"]
    m = row["general_max_level"]
    last = row["general_last_level"]
    adh = row["adherence"]

    if pd.isna(n) or n == 0:
        return "Z_never_played"
    if m == 1 and n < 21:
        return "A_early_dropout"
    if m == 7:
        return "E_maxed_out"
    if last in (4, 5) and n >= 21:
        return "D_mid_wall"
    if m in (2, 3):
        return "B_stalled_settled" if adh >= 0.5 else "C_stalled_bored"
    return "F_steady_progress"


def main():
    patients = pd.read_parquet(OUT / "patients.parquet")
    patients["archetype"] = patients.apply(label_one, axis=1)

    patients.to_parquet(OUT / "patients.parquet", index=False)

    print("=== archetype distribution ===")
    counts = patients["archetype"].value_counts().sort_index()
    print(counts.to_string())
    print(f"  total: {len(patients)}")

    print("\n=== represcribe rate by archetype ===")
    g = patients.groupby("archetype").agg(
        n=("patient_id", "count"),
        n_rx=("did_represcribe", "sum"),
        rx_rate=("did_represcribe", "mean"),
        mean_adh=("adherence", "mean"),
        mean_max_level=("general_max_level", "mean"),
        mean_last_level=("general_last_level", "mean"),
        mean_n_general=("n_general", "mean"),
    ).round(3)
    print(g.to_string())

    print("\n=== sanity: total re-prescription count ===")
    print(f"  patients with did_represcribe=True: {patients['did_represcribe'].sum()} (expected ~30)")


if __name__ == "__main__":
    main()
