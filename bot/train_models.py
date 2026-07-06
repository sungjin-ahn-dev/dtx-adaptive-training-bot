"""
Train three LightGBM models that, together, define the patient bot:

  score_model       — given (patient state, exercise, level) → score in [0, 1]
  gap_model         — given (session state) → log days until next session
  termination_model — given (patient state at end of window) → P(re-prescribe)

All three split by patient_id so the same patient never appears in train and test.

Run:
  python -m bot.train_models
"""
from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, brier_score_loss,
)
import lightgbm as lgb
import joblib

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

RANDOM_STATE = 42

CATEGORICAL = {
    "exercise_name", "kind", "archetype",
    "prescribed_from", "hospital",
}


def split_by_patient(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df["patient_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def _prep_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in feature_cols:
        if c in CATEGORICAL:
            X[c] = X[c].astype("category")
    return X


# ---------- 1. Score model ----------

def train_score_model(sf: pd.DataFrame):
    sf = sf.dropna(subset=["score"]).copy()
    feats = ["exercise_name", "kind", "level", "archetype", "adherence",
             "r3_mean", "r3_min", "r3_max",
             "n_at_exercise", "n_at_exercise_level", "t_overall",
             "hour", "dow", "days_into_window"]
    train, test = split_by_patient(sf)
    Xtr, ytr = _prep_X(train, feats), train["score"].values
    Xte, yte = _prep_X(test, feats), test["score"].values

    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=40,
        random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)],
              categorical_feature=[c for c in feats if c in CATEGORICAL],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    pred = np.clip(model.predict(Xte), 0, 1)

    metrics = {
        "n_train": len(train), "n_test": len(test),
        "MAE": float(mean_absolute_error(yte, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
        "R2": float(r2_score(yte, pred)),
        "baseline_MAE_mean": float(mean_absolute_error(yte, np.full_like(yte, ytr.mean()))),
    }
    return model, feats, metrics


# ---------- 2a. Dropout model ----------
# Per session: P(this session is the patient's last). Used together with the
# gap model in the simulator — dropout decides whether the bot ever returns,
# the gap model decides when (if it does).

def train_dropout_model(sess: pd.DataFrame):
    df = sess.copy()
    df["target"] = df["is_terminal"].astype(int)
    feats = ["archetype", "adherence", "session_idx",
             "n_in_session", "n_general_in_session",
             "session_mean_score", "session_max_level", "session_mean_level",
             "days_since_prev", "r3_gap_mean",
             "days_into_window", "days_to_expiry",
             "hour", "dow"]
    train, test = split_by_patient(df)
    Xtr, ytr = _prep_X(train, feats), train["target"].values
    Xte, yte = _prep_X(test, feats), test["target"].values

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=40,
        random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)],
              categorical_feature=[c for c in feats if c in CATEGORICAL],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    proba = model.predict_proba(Xte)[:, 1]

    metrics = {
        "n_train": len(train), "n_test": len(test),
        "pos_rate": float(ytr.mean()),
        "ROC_AUC": float(roc_auc_score(yte, proba)),
        "PR_AUC": float(average_precision_score(yte, proba)),
        "Brier": float(brier_score_loss(yte, proba)),
    }
    return model, feats, metrics


# ---------- 2b. Gap model ----------

def train_gap_model(sess: pd.DataFrame):
    df = sess.dropna(subset=["days_to_next"]).copy()
    # log-transform target to handle skew; predict in log-days
    df["target"] = np.log1p(df["days_to_next"])

    feats = ["archetype", "adherence", "session_idx",
             "n_in_session", "n_general_in_session",
             "session_mean_score", "session_max_level", "session_mean_level",
             "days_since_prev", "r3_gap_mean",
             "days_into_window", "hour", "dow"]
    train, test = split_by_patient(df)
    Xtr, ytr = _prep_X(train, feats), train["target"].values
    Xte, yte_log = _prep_X(test, feats), test["target"].values
    yte = test["days_to_next"].values

    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=31,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=80,
        random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte_log)],
              categorical_feature=[c for c in feats if c in CATEGORICAL],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    pred = np.expm1(model.predict(Xte)).clip(min=0)

    metrics = {
        "n_train": len(train), "n_test": len(test),
        "MAE_days": float(mean_absolute_error(yte, pred)),
        "MAE_log": float(mean_absolute_error(yte_log, np.log1p(pred))),
        "baseline_MAE_days": float(mean_absolute_error(
            yte, np.full_like(yte, np.median(train["days_to_next"])))),
    }
    return model, feats, metrics


# ---------- 3. Termination model ----------

def train_termination_model(pat: pd.DataFrame):
    df = pat.dropna(subset=["did_represcribe"]).copy()
    df = df[df["archetype"] != "Z_never_played"].copy()
    df["target"] = df["did_represcribe"].astype(int)

    feats = ["archetype", "adherence",
             "n_records", "n_general", "n_general_days", "active_days",
             "general_max_level", "general_last_level", "general_median_level",
             "general_mean_score", "score_trend10",
             "median_gap_days", "mean_gap_days",
             "n_sessions", "n_terminal",
             "prescribed_from"]
    train, test = split_by_patient(df)
    Xtr, ytr = _prep_X(train, feats), train["target"].values
    Xte, yte = _prep_X(test, feats), test["target"].values

    # only 30 positives total — heavy class imbalance
    pos = int(ytr.sum()); neg = len(ytr) - pos
    weight = neg / max(pos, 1)

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        subsample=0.9, colsample_bytree=0.9, min_child_samples=20,
        scale_pos_weight=weight,
        random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)],
              categorical_feature=[c for c in feats if c in CATEGORICAL],
              callbacks=[lgb.early_stopping(20, verbose=False)])
    proba = model.predict_proba(Xte)[:, 1]

    metrics = {
        "n_train": len(train), "n_test": len(test),
        "pos_train": int(ytr.sum()), "pos_test": int(yte.sum()),
        "ROC_AUC": float(roc_auc_score(yte, proba)) if yte.sum() > 0 else None,
        "PR_AUC": float(average_precision_score(yte, proba)) if yte.sum() > 0 else None,
        "Brier": float(brier_score_loss(yte, proba)),
        "baseline_Brier_prior": float(brier_score_loss(yte, np.full_like(yte, ytr.mean(), dtype=float))),
    }
    return model, feats, metrics


def main():
    print("Loading feature tables ...")
    score_feat = pd.read_parquet(OUT / "score_features.parquet")
    sess_feat = pd.read_parquet(OUT / "session_features.parquet")
    pat_feat = pd.read_parquet(OUT / "patient_features.parquet")

    print("\n[1/3] Training score model ...")
    sm, sm_feats, sm_metrics = train_score_model(score_feat)
    joblib.dump({"model": sm, "features": sm_feats}, MODELS / "score_model.joblib")
    print(json.dumps(sm_metrics, indent=2, ensure_ascii=False))

    print("\n[2/4] Training dropout model ...")
    dm, dm_feats, dm_metrics = train_dropout_model(sess_feat)
    joblib.dump({"model": dm, "features": dm_feats}, MODELS / "dropout_model.joblib")
    print(json.dumps(dm_metrics, indent=2, ensure_ascii=False))

    print("\n[3/4] Training gap model ...")
    gm, gm_feats, gm_metrics = train_gap_model(sess_feat)
    joblib.dump({"model": gm, "features": gm_feats}, MODELS / "gap_model.joblib")
    print(json.dumps(gm_metrics, indent=2, ensure_ascii=False))

    print("\n[4/4] Training termination (re-prescription) model ...")
    tm, tm_feats, tm_metrics = train_termination_model(pat_feat)
    joblib.dump({"model": tm, "features": tm_feats}, MODELS / "termination_model.joblib")
    print(json.dumps(tm_metrics, indent=2, ensure_ascii=False))

    # write a combined metrics file for the record
    with open(MODELS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"score": sm_metrics, "dropout": dm_metrics,
                   "gap": gm_metrics, "termination": tm_metrics},
                  f, indent=2, ensure_ascii=False)
    print("\nModels saved under", MODELS)


if __name__ == "__main__":
    main()
