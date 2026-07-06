# -*- coding: utf-8 -*-
"""
Synthetic sample-data generator.

이 저장소에는 실제 환자 데이터가 포함되어 있지 않습니다. 이 스크립트가
실데이터와 동일한 스키마의 '가짜' 데이터(assignment_record.csv,
prescription.csv)를 생성하며, 전체 파이프라인은 이 합성 데이터 위에서
그대로 동작합니다.

Design: each synthetic patient has a latent ability level and an engagement
rate. Scores follow score ~ base + slope*(ability - level) + noise, and the
in-app difficulty rule (mean of last 3 scores: <0.5 down / >=0.8 up) drives
level progression — so the 6 behavioral archetypes (early-dropout, stalled,
mid-wall, maxed-out, ...) emerge naturally, like in the real dataset.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(42)

N_PATIENTS = 500
MAX_LEVEL = 7
WINDOW_DAYS = 84  # 12-week prescription window

GENERAL = ["memorize_out_loud", "word_storming", "memorize_with_stories"]
SUB = ["clapping", "making_into_one", "memorize_oddly", "memorize_orders",
       "memorize_scene", "repeat_and_memorize", "word_finding",
       "word_snake", "word_throwing"]
HOSPITALS = [f"가상의료기관-{i:02d}" for i in range(1, 21)]
CHANNELS = ["channel-a", "channel-b", "channel-c"]

# archetype-shaping priors: (weight, ability_level, daily_play_prob, quit_after_days)
PROFILES = [
    ("early_quitter", 0.14, (1.0, 1.8), (0.15, 0.35), (3, 14)),
    ("stalled_engaged", 0.12, (2.0, 3.2), (0.65, 0.92), (70, 84)),
    ("stalled_bored", 0.10, (2.0, 3.2), (0.15, 0.35), (25, 55)),
    ("mid_wall", 0.16, (4.0, 5.4), (0.40, 0.75), (45, 84)),
    ("high_flyer", 0.10, (6.5, 8.0), (0.55, 0.90), (60, 84)),
    ("steady", 0.30, (3.5, 6.0), (0.35, 0.65), (30, 84)),
    ("never_played", 0.08, (1.0, 1.0), (0.0, 0.0), (0, 0)),
]


def _uuid() -> str:
    return str(uuid.UUID(bytes=RNG.bytes(16), version=4))


def _fmt_event(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S.") + f"{RNG.integers(0, 999):03d} +0900"


def simulate_patient(profile, start: pd.Timestamp):
    """Return list of event dicts for one patient."""
    _, _, ability_rng, play_rng, quit_rng = profile
    ability = RNG.uniform(*ability_rng)
    play_p = RNG.uniform(*play_rng)
    quit_after = RNG.integers(quit_rng[0], quit_rng[1] + 1) if quit_rng[1] > 0 else 0

    levels = {ex: 1 for ex in GENERAL + SUB}
    recent: dict[str, list[float]] = {ex: [] for ex in GENERAL + SUB}
    events = []

    for day in range(quit_after):
        if RNG.random() > play_p:
            continue
        t = start + pd.Timedelta(days=day, hours=int(RNG.integers(8, 21)),
                                 minutes=int(RNG.integers(0, 60)))
        n_ex = int(RNG.integers(2, 6))
        chosen = list(RNG.choice(GENERAL, size=min(2, n_ex), replace=False))
        if n_ex > 2:
            chosen += list(RNG.choice(SUB, size=n_ex - 2, replace=False))
        for ex in chosen:
            lv = levels[ex]
            # score peaks near the patient's latent ability level, so patients
            # naturally stall once the difficulty rule pushes them to ~ability
            mu = 0.66 + 0.15 * (ability - lv)
            score = float(np.clip(RNG.normal(mu, 0.13), 0.0, 1.0))
            score = round(score, 2)
            events.append({
                "assignment_id": _uuid(),
                "exercise_name": ex,
                "record": json.dumps({"level": int(lv), "score": score}),
                "created_at": _fmt_event(t),
            })
            t += pd.Timedelta(minutes=int(RNG.integers(3, 9)))
            # in-app difficulty rule: mean of last 3 -> <0.5 down / >=0.8 up
            recent[ex].append(score)
            if len(recent[ex]) >= 3:
                m = float(np.mean(recent[ex][-3:]))
                if m >= 0.8 and lv < MAX_LEVEL:
                    levels[ex] = lv + 1
                    recent[ex] = []
                elif m < 0.5 and lv > 1:
                    levels[ex] = lv - 1
                    recent[ex] = []
    return events, ability


def main():
    weights = np.array([p[1] for p in PROFILES])
    weights = weights / weights.sum()

    rx_rows, ev_rows = [], []
    base = pd.Timestamp("2025-09-01")

    for _ in range(N_PATIENTS):
        pid = _uuid()
        profile = PROFILES[int(RNG.choice(len(PROFILES), p=weights))]
        hospital = str(RNG.choice(HOSPITALS))
        channel = str(RNG.choice(CHANNELS, p=[0.7, 0.2, 0.1]))

        rx_created = base + pd.Timedelta(days=int(RNG.integers(0, 150)),
                                         hours=int(RNG.integers(0, 24)))
        activated = rx_created + pd.Timedelta(minutes=int(RNG.integers(2, 300)))
        expired = activated.normalize() + pd.Timedelta(days=WINDOW_DAYS, hours=15) \
            - pd.Timedelta(seconds=1)

        events, ability = simulate_patient(profile, activated.normalize()
                                           + pd.Timedelta(days=1))
        for e in events:
            e["patient_id"] = pid
            e["modified_at"] = e["created_at"]
        ev_rows += events

        rx_rows.append({
            "prescription_id": _uuid(), "prescribed_from": channel,
            "hospital": hospital, "patient_id": pid, "sequence": 1,
            "created_at": rx_created.strftime("%Y-%m-%d %H:%M:%S"),
            "activated_at": activated.strftime("%Y-%m-%d %H:%M:%S"),
            "expired_at": expired.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # re-prescription: engaged patients renew more often (adh -> rx pattern)
        n_days = len({e["created_at"][:10] for e in events})
        p_rerx = min(0.55, 0.02 + 0.006 * n_days)
        if RNG.random() < p_rerx:
            rx2 = expired + pd.Timedelta(days=int(RNG.integers(1, 20)))
            act2 = rx2 + pd.Timedelta(minutes=int(RNG.integers(2, 300)))
            rx_rows.append({
                "prescription_id": _uuid(), "prescribed_from": channel,
                "hospital": hospital, "patient_id": pid, "sequence": 2,
                "created_at": rx2.strftime("%Y-%m-%d %H:%M:%S"),
                "activated_at": act2.strftime("%Y-%m-%d %H:%M:%S"),
                "expired_at": (act2.normalize() + pd.Timedelta(days=WINDOW_DAYS, hours=15)
                               - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S"),
            })

    ev = pd.DataFrame(ev_rows)[["assignment_id", "exercise_name", "patient_id",
                                "record", "created_at", "modified_at"]]
    ev = ev.sort_values("created_at").reset_index(drop=True)
    rx = pd.DataFrame(rx_rows)

    ev.to_csv(HERE / "assignment_record.csv", index=False)
    rx.to_csv(HERE / "prescription.csv", index=False)
    print(f"wrote {len(ev):,} events / {ev.patient_id.nunique()} patients "
          f"-> {HERE / 'assignment_record.csv'}")
    print(f"wrote {len(rx):,} prescriptions -> {HERE / 'prescription.csv'}")


if __name__ == "__main__":
    main()
