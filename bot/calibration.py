"""
Per-archetype calibration tables, derived once from the real cohort and reused
by the simulator to keep cohort-level outcomes anchored to reality.

Computed at import time from out/patients.parquet so the values track the data.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

_patients = pd.read_parquet(OUT / "patients.parquet")
_eligible = _patients[_patients["archetype"] != "Z_never_played"]

# per-archetype re-prescription base rate
REPRESCRIBE_RATE: dict[str, float] = (
    _eligible.groupby("archetype")["did_represcribe"].mean().to_dict()
)

# per-archetype empirical n_general distribution
# we will sample max_sessions per virtual patient from this distribution
_N_GENERAL_BY_ARCH: dict[str, np.ndarray] = {
    arch: g["n_general"].dropna().astype(int).to_numpy()
    for arch, g in _eligible.groupby("archetype")
}

# per-archetype real session count distribution
# (session_idx from session_features represents minute-bucket sessions; the sim
# session is also one "I came to play" event, so this is the right comparison.)
_sessions_df = pd.read_parquet(OUT / "session_features.parquet")
_n_sessions_per_patient = _sessions_df.groupby("patient_id")["session_idx"].max() + 1
_pt_arch = _eligible.set_index("patient_id")["archetype"]
_joined = pd.DataFrame({
    "n_sessions": _n_sessions_per_patient,
    "archetype": _pt_arch,
}).dropna()
_N_SESSIONS_BY_ARCH: dict[str, np.ndarray] = {
    arch: g["n_sessions"].astype(int).to_numpy()
    for arch, g in _joined.groupby("archetype")
}

# per-archetype max_level distribution — each virtual patient draws their
# personal capability ceiling from here. This models inter-patient variance
# in cognitive baseline; algorithms can still affect *how* the patient progresses
# towards (or away from) their ceiling, but not the ceiling itself.
_MAX_LEVEL_BY_ARCH: dict[str, np.ndarray] = {
    arch: g["general_max_level"].dropna().astype(int).to_numpy()
    for arch, g in _eligible.groupby("archetype")
}


def sample_target_max_sessions(archetypes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """For each virtual patient, draw a session-count cap from the empirical
    distribution of total sessions (general + sub) within their archetype.
    This matches the simulator's one-session-per-iteration definition."""
    out = np.empty(len(archetypes), dtype=np.int32)
    for i, arch in enumerate(archetypes):
        pool = _N_SESSIONS_BY_ARCH.get(arch)
        if pool is None or len(pool) == 0:
            out[i] = 1
            continue
        out[i] = max(1, int(rng.choice(pool)))
    return out


def sample_target_max_level(archetypes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each virtual patient's personal level ceiling, drawn from the empirical
    archetype distribution of general_max_level."""
    out = np.empty(len(archetypes), dtype=np.int32)
    for i, arch in enumerate(archetypes):
        pool = _MAX_LEVEL_BY_ARCH.get(arch)
        out[i] = int(rng.choice(pool)) if pool is not None and len(pool) else 1
    return out


def calibrate_represcribe(p_raw: np.ndarray, archetypes: np.ndarray) -> np.ndarray:
    """Rescale termination probabilities so that, within each archetype, the
    expected re-prescription rate equals the real cohort base rate. Preserves
    the model's *ranking* of patients within an archetype while fixing the
    archetype-level mean to a calibrated value."""
    out = p_raw.copy().astype(np.float64)
    for arch, real_rate in REPRESCRIBE_RATE.items():
        mask = archetypes == arch
        if not mask.any():
            continue
        cur_mean = float(out[mask].mean())
        if cur_mean <= 1e-9:
            out[mask] = real_rate
        else:
            scale = real_rate / cur_mean
            out[mask] = np.clip(out[mask] * scale, 0, 1)
    return out
