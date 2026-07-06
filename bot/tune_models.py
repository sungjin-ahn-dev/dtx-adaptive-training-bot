"""
Optuna hyperparameter tuning + final retrain for all four models.

5-fold patient-level CV inside Optuna; final model is retrained on the full
data with the best hyperparams and the original train/test split (so audit
metrics stay comparable).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error, roc_auc_score, average_precision_score,
    brier_score_loss, r2_score, mean_squared_error,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
CAT = {"exercise_name", "kind", "archetype", "prescribed_from", "hospital"}
RS = 42

optuna.logging.set_verbosity(optuna.logging.WARNING)


def prep(df, feats):
    X = df.reindex(columns=feats).copy()
    for c in CAT & set(X.columns):
        X[c] = X[c].astype("category")
    return X


# --------------------------------------------------------------------------- #
# Tuning objective per model
# --------------------------------------------------------------------------- #

def _cv_iter(df, feats, k=5):
    gkf = GroupKFold(n_splits=k)
    for tr, te in gkf.split(df, groups=df["patient_id"]):
        yield df.iloc[tr], df.iloc[te]


def tune_score(trials: int):
    sf = pd.read_parquet(OUT / "score_features.parquet").dropna(subset=["score"])
    feats = ["exercise_name", "kind", "level", "archetype", "adherence",
             "r3_mean", "r3_min", "r3_max", "n_at_exercise", "n_at_exercise_level",
             "t_overall", "hour", "dow", "days_into_window"]

    def objective(trial):
        params = dict(
            n_estimators=1000,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            random_state=RS, verbose=-1,
        )
        maes = []
        for train, test in _cv_iter(sf, feats):
            m = lgb.LGBMRegressor(**params)
            m.fit(prep(train, feats), train["score"].values,
                  eval_set=[(prep(test, feats), test["score"].values)],
                  categorical_feature=[c for c in feats if c in CAT],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            pred = np.clip(m.predict(prep(test, feats)), 0, 1)
            maes.append(mean_absolute_error(test["score"].values, pred))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study, sf, feats


def tune_dropout(trials: int):
    sess = pd.read_parquet(OUT / "session_features.parquet")
    feats = ["archetype", "adherence", "session_idx", "n_in_session",
             "n_general_in_session", "session_mean_score", "session_max_level",
             "session_mean_level", "days_since_prev", "r3_gap_mean",
             "days_into_window", "days_to_expiry", "hour", "dow"]

    def objective(trial):
        params = dict(
            n_estimators=1000,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            min_child_samples=trial.suggest_int("min_child_samples", 30, 200),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            random_state=RS, verbose=-1,
        )
        aucs = []
        for train, test in _cv_iter(sess, feats):
            ytr = train["is_terminal"].astype(int).values
            yte = test["is_terminal"].astype(int).values
            pos, neg = ytr.sum(), len(ytr) - ytr.sum()
            m = lgb.LGBMClassifier(scale_pos_weight=neg/max(pos, 1), **params)
            m.fit(prep(train, feats), ytr,
                  eval_set=[(prep(test, feats), yte)],
                  categorical_feature=[c for c in feats if c in CAT],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            proba = m.predict_proba(prep(test, feats))[:, 1]
            if yte.sum() > 0:
                aucs.append(roc_auc_score(yte, proba))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study, sess, feats


def tune_gap(trials: int):
    sess = pd.read_parquet(OUT / "session_features.parquet")
    sg = sess.dropna(subset=["days_to_next"])
    feats = ["archetype", "adherence", "session_idx", "n_in_session",
             "n_general_in_session", "session_mean_score", "session_max_level",
             "session_mean_level", "days_since_prev", "r3_gap_mean",
             "days_into_window", "hour", "dow"]

    def objective(trial):
        params = dict(
            n_estimators=1000,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            min_child_samples=trial.suggest_int("min_child_samples", 40, 300),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            random_state=RS, verbose=-1,
        )
        maes = []
        for train, test in _cv_iter(sg, feats):
            ytr = np.log1p(train["days_to_next"].values)
            yte = np.log1p(test["days_to_next"].values)
            m = lgb.LGBMRegressor(**params)
            m.fit(prep(train, feats), ytr, eval_set=[(prep(test, feats), yte)],
                  categorical_feature=[c for c in feats if c in CAT],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            pred = np.expm1(m.predict(prep(test, feats))).clip(min=0)
            maes.append(mean_absolute_error(test["days_to_next"].values, pred))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study, sg, feats


def tune_termination(trials: int):
    """Strong regularization: small num_leaves, large min_child_samples,
    heavy reg_alpha/lambda. We're explicitly fighting the train ROC = 1.000
    memorization problem."""
    pf = pd.read_parquet(OUT / "patient_features.parquet").dropna(subset=["did_represcribe"])
    pf = pf[pf["archetype"] != "Z_never_played"].reset_index(drop=True)
    feats = ["archetype", "adherence",
             "n_records", "n_general", "n_general_days", "active_days",
             "general_max_level", "general_last_level", "general_median_level",
             "general_mean_score", "score_trend10",
             "median_gap_days", "mean_gap_days",
             "n_sessions", "n_terminal", "prescribed_from"]

    def objective(trial):
        params = dict(
            n_estimators=400,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
            num_leaves=trial.suggest_int("num_leaves", 4, 15),        # 작게
            min_child_samples=trial.suggest_int("min_child_samples", 30, 120),  # 크게
            max_depth=trial.suggest_int("max_depth", 3, 6),
            reg_alpha=trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),    # 강하게
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=RS, verbose=-1,
        )
        aucs, prs = [], []
        for train, test in _cv_iter(pf, feats):
            ytr = train["did_represcribe"].astype(int).values
            yte = test["did_represcribe"].astype(int).values
            if yte.sum() == 0:
                continue
            pos, neg = ytr.sum(), len(ytr) - ytr.sum()
            m = lgb.LGBMClassifier(scale_pos_weight=neg/max(pos, 1), **params)
            m.fit(prep(train, feats), ytr, eval_set=[(prep(test, feats), yte)],
                  categorical_feature=[c for c in feats if c in CAT],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
            proba = m.predict_proba(prep(test, feats))[:, 1]
            aucs.append(roc_auc_score(yte, proba))
            prs.append(average_precision_score(yte, proba))
        # primary objective: PR-AUC (more informative on tiny positive class)
        # secondary penalty: train/test gap
        return float(np.mean(prs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study, pf, feats


# --------------------------------------------------------------------------- #
# Final retrain with best params (use the same train/test split as before)
# --------------------------------------------------------------------------- #

def retrain_score(best_params, sf, feats):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS)
    tr, te = next(splitter.split(sf, groups=sf["patient_id"]))
    train, test = sf.iloc[tr], sf.iloc[te]
    m = lgb.LGBMRegressor(n_estimators=1500, random_state=RS, verbose=-1, **best_params)
    m.fit(prep(train, feats), train["score"].values,
          eval_set=[(prep(test, feats), test["score"].values)],
          categorical_feature=[c for c in feats if c in CAT],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    pred_tr = np.clip(m.predict(prep(train, feats)), 0, 1)
    pred = np.clip(m.predict(prep(test, feats)), 0, 1)
    metrics = {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "MAE": float(mean_absolute_error(test["score"].values, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(test["score"].values, pred))),
        "R2": float(r2_score(test["score"].values, pred)),
        "train_MAE": float(mean_absolute_error(train["score"].values, pred_tr)),
        "baseline_MAE_mean": float(mean_absolute_error(
            test["score"].values,
            np.full(len(test), train["score"].mean()))),
    }
    return m, metrics


def retrain_dropout(best_params, sess, feats):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS)
    tr, te = next(splitter.split(sess, groups=sess["patient_id"]))
    train, test = sess.iloc[tr], sess.iloc[te]
    ytr = train["is_terminal"].astype(int).values
    yte = test["is_terminal"].astype(int).values
    pos, neg = ytr.sum(), len(ytr) - ytr.sum()
    m = lgb.LGBMClassifier(n_estimators=1500, random_state=RS, verbose=-1,
                            scale_pos_weight=neg/max(pos, 1), **best_params)
    m.fit(prep(train, feats), ytr, eval_set=[(prep(test, feats), yte)],
          categorical_feature=[c for c in feats if c in CAT],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    proba = m.predict_proba(prep(test, feats))[:, 1]
    proba_tr = m.predict_proba(prep(train, feats))[:, 1]
    metrics = {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "pos_rate": float(ytr.mean()),
        "ROC_AUC": float(roc_auc_score(yte, proba)),
        "PR_AUC": float(average_precision_score(yte, proba)),
        "Brier": float(brier_score_loss(yte, proba)),
        "train_ROC_AUC": float(roc_auc_score(ytr, proba_tr)),
    }
    return m, metrics


def retrain_gap(best_params, sg, feats):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS)
    tr, te = next(splitter.split(sg, groups=sg["patient_id"]))
    train, test = sg.iloc[tr], sg.iloc[te]
    ytr = np.log1p(train["days_to_next"].values)
    yte = np.log1p(test["days_to_next"].values)
    m = lgb.LGBMRegressor(n_estimators=1500, random_state=RS, verbose=-1, **best_params)
    m.fit(prep(train, feats), ytr, eval_set=[(prep(test, feats), yte)],
          categorical_feature=[c for c in feats if c in CAT],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = np.expm1(m.predict(prep(test, feats))).clip(min=0)
    metrics = {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "MAE_days": float(mean_absolute_error(test["days_to_next"].values, pred)),
        "baseline_MAE_days": float(mean_absolute_error(
            test["days_to_next"].values,
            np.full(len(test), train["days_to_next"].median()))),
    }
    return m, metrics


def retrain_termination(best_params, pf, feats):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS)
    tr, te = next(splitter.split(pf, groups=pf["patient_id"]))
    train, test = pf.iloc[tr], pf.iloc[te]
    ytr = train["did_represcribe"].astype(int).values
    yte = test["did_represcribe"].astype(int).values
    pos, neg = ytr.sum(), len(ytr) - ytr.sum()
    m = lgb.LGBMClassifier(n_estimators=500, random_state=RS, verbose=-1,
                            scale_pos_weight=neg/max(pos, 1), **best_params)
    m.fit(prep(train, feats), ytr, eval_set=[(prep(test, feats), yte)],
          categorical_feature=[c for c in feats if c in CAT],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    proba = m.predict_proba(prep(test, feats))[:, 1]
    proba_tr = m.predict_proba(prep(train, feats))[:, 1]
    metrics = {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "pos_train": int(ytr.sum()), "pos_test": int(yte.sum()),
        "ROC_AUC": float(roc_auc_score(yte, proba)) if yte.sum() else None,
        "PR_AUC": float(average_precision_score(yte, proba)) if yte.sum() else None,
        "Brier": float(brier_score_loss(yte, proba)),
        "train_ROC_AUC": float(roc_auc_score(ytr, proba_tr)),
    }
    return m, metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--model", default="all",
                   choices=["all", "score", "dropout", "gap", "termination"])
    args = p.parse_args()

    summary = {}
    all_metrics = {}

    if args.model in ("all", "termination"):
        print(f"[1/4] Tuning TERMINATION (trials={args.trials})...")
        study, pf, feats = tune_termination(args.trials)
        print(f"  best CV PR-AUC: {study.best_value:.4f}")
        print(f"  best params: {study.best_params}")
        m, metrics = retrain_termination(study.best_params, pf, feats)
        joblib.dump({"model": m, "features": feats}, MODELS / "termination_model.joblib")
        summary["termination"] = {"best_cv": study.best_value,
                                  "best_params": study.best_params}
        all_metrics["termination"] = metrics
        print(f"  retrained: test ROC={metrics['ROC_AUC']:.3f} · train ROC={metrics['train_ROC_AUC']:.3f}"
              f" · gap={metrics['train_ROC_AUC']-metrics['ROC_AUC']:+.3f}")
        print(f"  PR-AUC: {metrics['PR_AUC']:.3f} · Brier: {metrics['Brier']:.4f}")

    if args.model in ("all", "dropout"):
        print(f"\n[2/4] Tuning DROPOUT (trials={args.trials})...")
        study, sess, feats = tune_dropout(args.trials)
        print(f"  best CV ROC: {study.best_value:.4f}")
        print(f"  best params: {study.best_params}")
        m, metrics = retrain_dropout(study.best_params, sess, feats)
        joblib.dump({"model": m, "features": feats}, MODELS / "dropout_model.joblib")
        summary["dropout"] = {"best_cv": study.best_value,
                              "best_params": study.best_params}
        all_metrics["dropout"] = metrics
        print(f"  retrained: test ROC={metrics['ROC_AUC']:.3f} · train ROC={metrics['train_ROC_AUC']:.3f}"
              f" · gap={metrics['train_ROC_AUC']-metrics['ROC_AUC']:+.3f}")

    if args.model in ("all", "gap"):
        print(f"\n[3/4] Tuning GAP (trials={args.trials})...")
        study, sg, feats = tune_gap(args.trials)
        print(f"  best CV MAE: {study.best_value:.4f} days")
        print(f"  best params: {study.best_params}")
        m, metrics = retrain_gap(study.best_params, sg, feats)
        joblib.dump({"model": m, "features": feats}, MODELS / "gap_model.joblib")
        summary["gap"] = {"best_cv": study.best_value,
                          "best_params": study.best_params}
        all_metrics["gap"] = metrics
        print(f"  retrained: test MAE={metrics['MAE_days']:.3f} days "
              f"(baseline {metrics['baseline_MAE_days']:.3f})")

    if args.model in ("all", "score"):
        print(f"\n[4/4] Tuning SCORE (trials={args.trials})...")
        study, sf, feats = tune_score(args.trials)
        print(f"  best CV MAE: {study.best_value:.4f}")
        print(f"  best params: {study.best_params}")
        m, metrics = retrain_score(study.best_params, sf, feats)
        joblib.dump({"model": m, "features": feats}, MODELS / "score_model.joblib")
        summary["score"] = {"best_cv": study.best_value,
                            "best_params": study.best_params}
        all_metrics["score"] = metrics
        print(f"  retrained: test MAE={metrics['MAE']:.4f} · train MAE={metrics['train_MAE']:.4f}"
              f" · gap={metrics['MAE']-metrics['train_MAE']:+.4f}")

    # save updated metrics.json (preserve any models that weren't tuned in this run)
    metrics_path = MODELS / "metrics.json"
    existing = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(all_metrics)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    with open(MODELS / "tuning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {MODELS / 'tuning_summary.json'}")


if __name__ == "__main__":
    main()
