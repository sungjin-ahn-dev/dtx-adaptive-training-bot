"""
모델 학습 감사. archetype leakage, 5-fold CV 안정성, train/test gap,
archetype별 subgroup 성능을 한 번에 찍어본다. 결과는 out/audit_results.json.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, brier_score_loss,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
MODELS = ROOT / "models"
CAT = {"exercise_name", "kind", "archetype", "prescribed_from", "hospital"}
RS = 42


def _best_params(model_name: str) -> dict:
    """Tuned hyperparams if we have them, else {} (LightGBM defaults)."""
    p = MODELS / "tuning_summary.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        s = json.load(f)
    return s.get(model_name, {}).get("best_params", {})


def prep(df, feats):
    X = df.reindex(columns=feats).copy()
    for c in CAT & set(X.columns):
        X[c] = X[c].astype("category")
    return X


# --------------------------------------------------------------------------- #
# 1. Archetype leakage check
# --------------------------------------------------------------------------- #

def archetype_ablation():
    """archetype feature 포함 vs 제외 한 번씩 학습. 큰 성능 갭은 leakage 신호."""
    print("=" * 70)
    print(" 1. ARCHETYPE LEAKAGE CHECK")
    print("=" * 70)

    results = {}

    # Score
    sf = pd.read_parquet(OUT / "score_features.parquet").dropna(subset=["score"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS)
    ti_tr, ti_te = next(splitter.split(sf, groups=sf["patient_id"]))
    train, test = sf.iloc[ti_tr], sf.iloc[ti_te]

    feats_full = ["exercise_name", "kind", "level", "archetype", "adherence",
                  "r3_mean", "r3_min", "r3_max",
                  "n_at_exercise", "n_at_exercise_level", "t_overall",
                  "hour", "dow", "days_into_window"]
    feats_no_arch = [f for f in feats_full if f != "archetype"]

    def score_run(feats):
        m = lgb.LGBMRegressor(n_estimators=600, random_state=RS, verbose=-1,
                               **_best_params("score"))
        m.fit(prep(train, feats), train["score"].values,
              eval_set=[(prep(test, feats), test["score"].values)],
              categorical_feature=[c for c in feats if c in CAT],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        pred = np.clip(m.predict(prep(test, feats)), 0, 1)
        return mean_absolute_error(test["score"].values, pred)

    mae_full = score_run(feats_full)
    mae_no = score_run(feats_no_arch)
    results["score"] = {"with_archetype_MAE": mae_full, "without_archetype_MAE": mae_no,
                        "delta": mae_no - mae_full}
    print(f"  Score MAE   — with arch: {mae_full:.4f} · without: {mae_no:.4f} · Δ={mae_no-mae_full:+.4f}")

    # Dropout
    sess = pd.read_parquet(OUT / "session_features.parquet")
    ti_tr, ti_te = next(splitter.split(sess, groups=sess["patient_id"]))
    train, test = sess.iloc[ti_tr], sess.iloc[ti_te]

    feats_full = ["archetype", "adherence", "session_idx",
                  "n_in_session", "n_general_in_session",
                  "session_mean_score", "session_max_level", "session_mean_level",
                  "days_since_prev", "r3_gap_mean", "days_into_window",
                  "days_to_expiry", "hour", "dow"]
    feats_no = [f for f in feats_full if f != "archetype"]

    def dropout_run(feats):
        y_tr = train["is_terminal"].astype(int).values
        y_te = test["is_terminal"].astype(int).values
        pos = y_tr.sum(); neg = len(y_tr) - pos
        m = lgb.LGBMClassifier(n_estimators=600, random_state=RS, verbose=-1,
                                scale_pos_weight=neg/max(pos, 1),
                                **_best_params("dropout"))
        m.fit(prep(train, feats), y_tr,
              eval_set=[(prep(test, feats), y_te)],
              categorical_feature=[c for c in feats if c in CAT],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        proba = m.predict_proba(prep(test, feats))[:, 1]
        return roc_auc_score(y_te, proba), average_precision_score(y_te, proba)

    auc_full, ap_full = dropout_run(feats_full)
    auc_no, ap_no = dropout_run(feats_no)
    results["dropout"] = {"with_archetype_ROC": auc_full, "without_archetype_ROC": auc_no,
                          "delta": auc_full - auc_no,
                          "with_PR": ap_full, "without_PR": ap_no}
    print(f"  Dropout ROC — with arch: {auc_full:.4f} · without: {auc_no:.4f} · Δ={auc_full-auc_no:+.4f}")
    print(f"  Dropout PR  — with arch: {ap_full:.4f} · without: {ap_no:.4f} · Δ={ap_full-ap_no:+.4f}")

    # Termination
    pf = pd.read_parquet(OUT / "patient_features.parquet").dropna(subset=["did_represcribe"])
    pf = pf[pf["archetype"] != "Z_never_played"]
    ti_tr, ti_te = next(splitter.split(pf, groups=pf["patient_id"]))
    train, test = pf.iloc[ti_tr], pf.iloc[ti_te]

    feats_full = ["archetype", "adherence",
                  "n_records", "n_general", "n_general_days", "active_days",
                  "general_max_level", "general_last_level", "general_median_level",
                  "general_mean_score", "score_trend10",
                  "median_gap_days", "mean_gap_days",
                  "n_sessions", "n_terminal", "prescribed_from"]
    feats_no = [f for f in feats_full if f != "archetype"]

    def term_run(feats):
        y_tr = train["did_represcribe"].astype(int).values
        y_te = test["did_represcribe"].astype(int).values
        pos = y_tr.sum(); neg = len(y_tr) - pos
        m = lgb.LGBMClassifier(n_estimators=400, random_state=RS, verbose=-1,
                                scale_pos_weight=neg/max(pos, 1),
                                **_best_params("termination"))
        m.fit(prep(train, feats), y_tr,
              eval_set=[(prep(test, feats), y_te)],
              categorical_feature=[c for c in feats if c in CAT],
              callbacks=[lgb.early_stopping(20, verbose=False)])
        proba = m.predict_proba(prep(test, feats))[:, 1]
        return (roc_auc_score(y_te, proba) if y_te.sum() else None,
                average_precision_score(y_te, proba) if y_te.sum() else None)

    auc_full, ap_full = term_run(feats_full)
    auc_no, ap_no = term_run(feats_no)
    results["termination"] = {"with_archetype_ROC": auc_full, "without_archetype_ROC": auc_no,
                              "delta": (auc_full or 0) - (auc_no or 0),
                              "with_PR": ap_full, "without_PR": ap_no}
    print(f"  Term ROC    — with arch: {auc_full:.4f} · without: {auc_no:.4f} · Δ={(auc_full or 0)-(auc_no or 0):+.4f}")
    print(f"  Term PR     — with arch: {ap_full:.4f} · without: {ap_no:.4f}")
    return results


# --------------------------------------------------------------------------- #
# 2. K-fold cross-validation
# --------------------------------------------------------------------------- #

def kfold_stability(k=5):
    print("\n" + "=" * 70)
    print(f" 2. {k}-FOLD CV STABILITY")
    print("=" * 70)
    out = {}

    # Score
    sf = pd.read_parquet(OUT / "score_features.parquet").dropna(subset=["score"])
    feats = ["exercise_name", "kind", "level", "archetype", "adherence",
             "r3_mean", "r3_min", "r3_max",
             "n_at_exercise", "n_at_exercise_level", "t_overall",
             "hour", "dow", "days_into_window"]
    gkf = GroupKFold(n_splits=k)
    maes = []
    for fold, (tr, te) in enumerate(gkf.split(sf, groups=sf["patient_id"])):
        train, test = sf.iloc[tr], sf.iloc[te]
        m = lgb.LGBMRegressor(n_estimators=600, random_state=RS, verbose=-1,
                               **_best_params("score"))
        m.fit(prep(train, feats), train["score"].values,
              eval_set=[(prep(test, feats), test["score"].values)],
              categorical_feature=[c for c in feats if c in CAT],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        pred = np.clip(m.predict(prep(test, feats)), 0, 1)
        mae = mean_absolute_error(test["score"].values, pred)
        maes.append(mae)
    out["score_MAE"] = {"folds": maes, "mean": np.mean(maes), "std": np.std(maes),
                        "cv": np.std(maes) / np.mean(maes)}
    print(f"  Score MAE 5-fold: {[f'{m:.3f}' for m in maes]} "
          f"→ {np.mean(maes):.4f} ± {np.std(maes):.4f}")

    # Dropout
    sess = pd.read_parquet(OUT / "session_features.parquet")
    feats = ["archetype", "adherence", "session_idx",
             "n_in_session", "n_general_in_session",
             "session_mean_score", "session_max_level", "session_mean_level",
             "days_since_prev", "r3_gap_mean", "days_into_window",
             "days_to_expiry", "hour", "dow"]
    aucs = []
    for fold, (tr, te) in enumerate(gkf.split(sess, groups=sess["patient_id"])):
        train, test = sess.iloc[tr], sess.iloc[te]
        y_tr = train["is_terminal"].astype(int).values
        y_te = test["is_terminal"].astype(int).values
        pos = y_tr.sum(); neg = len(y_tr) - pos
        m = lgb.LGBMClassifier(n_estimators=600, random_state=RS, verbose=-1,
                                scale_pos_weight=neg/max(pos, 1),
                                **_best_params("dropout"))
        m.fit(prep(train, feats), y_tr,
              eval_set=[(prep(test, feats), y_te)],
              categorical_feature=[c for c in feats if c in CAT],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        if y_te.sum() > 0:
            aucs.append(roc_auc_score(y_te, m.predict_proba(prep(test, feats))[:, 1]))
    out["dropout_ROC"] = {"folds": aucs, "mean": np.mean(aucs), "std": np.std(aucs)}
    print(f"  Dropout ROC 5-fold: {[f'{a:.3f}' for a in aucs]} "
          f"→ {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # Gap
    sg = sess.dropna(subset=["days_to_next"])
    feats = ["archetype", "adherence", "session_idx",
             "n_in_session", "n_general_in_session",
             "session_mean_score", "session_max_level", "session_mean_level",
             "days_since_prev", "r3_gap_mean", "days_into_window", "hour", "dow"]
    maes = []
    for tr, te in gkf.split(sg, groups=sg["patient_id"]):
        train, test = sg.iloc[tr], sg.iloc[te]
        m = lgb.LGBMRegressor(n_estimators=600, random_state=RS, verbose=-1,
                               **_best_params("gap"))
        ytr = np.log1p(train["days_to_next"].values)
        yte = np.log1p(test["days_to_next"].values)
        m.fit(prep(train, feats), ytr, eval_set=[(prep(test, feats), yte)],
              categorical_feature=[c for c in feats if c in CAT],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        pred = np.expm1(m.predict(prep(test, feats))).clip(min=0)
        maes.append(mean_absolute_error(test["days_to_next"].values, pred))
    out["gap_MAE"] = {"folds": maes, "mean": np.mean(maes), "std": np.std(maes)}
    print(f"  Gap MAE 5-fold (days): {[f'{m:.2f}' for m in maes]} "
          f"→ {np.mean(maes):.3f} ± {np.std(maes):.3f}")

    # Termination — heavy class imbalance, K-fold positive counts will be tiny
    pf = pd.read_parquet(OUT / "patient_features.parquet").dropna(subset=["did_represcribe"])
    pf = pf[pf["archetype"] != "Z_never_played"]
    feats = ["archetype", "adherence",
             "n_records", "n_general", "n_general_days", "active_days",
             "general_max_level", "general_last_level", "general_median_level",
             "general_mean_score", "score_trend10",
             "median_gap_days", "mean_gap_days",
             "n_sessions", "n_terminal", "prescribed_from"]
    aucs, pos_in_folds = [], []
    for tr, te in gkf.split(pf, groups=pf["patient_id"]):
        train, test = pf.iloc[tr], pf.iloc[te]
        y_tr = train["did_represcribe"].astype(int).values
        y_te = test["did_represcribe"].astype(int).values
        pos_in_folds.append(int(y_te.sum()))
        pos = y_tr.sum(); neg = len(y_tr) - pos
        m = lgb.LGBMClassifier(n_estimators=400, random_state=RS, verbose=-1,
                                scale_pos_weight=neg/max(pos, 1),
                                **_best_params("termination"))
        m.fit(prep(train, feats), y_tr,
              eval_set=[(prep(test, feats), y_te)],
              categorical_feature=[c for c in feats if c in CAT],
              callbacks=[lgb.early_stopping(20, verbose=False)])
        if y_te.sum() > 0:
            aucs.append(roc_auc_score(y_te, m.predict_proba(prep(test, feats))[:, 1]))
    out["termination_ROC"] = {"folds": aucs, "pos_per_fold": pos_in_folds,
                              "mean": float(np.mean(aucs)) if aucs else None,
                              "std": float(np.std(aucs)) if aucs else None}
    print(f"  Term ROC 5-fold: {[f'{a:.3f}' for a in aucs]} "
          f"→ {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Term positive count per fold: {pos_in_folds}")
    return out


# --------------------------------------------------------------------------- #
# 3. Train vs Test gap (overfitting check)
# --------------------------------------------------------------------------- #

def overfit_check():
    print("\n" + "=" * 70)
    print(" 3. TRAIN vs TEST GAP (OVERFITTING)")
    print("=" * 70)
    out = {}

    sf = pd.read_parquet(OUT / "score_features.parquet").dropna(subset=["score"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS)
    tr, te = next(splitter.split(sf, groups=sf["patient_id"]))
    train, test = sf.iloc[tr], sf.iloc[te]
    feats = ["exercise_name", "kind", "level", "archetype", "adherence",
             "r3_mean", "r3_min", "r3_max", "n_at_exercise", "n_at_exercise_level",
             "t_overall", "hour", "dow", "days_into_window"]
    m = lgb.LGBMRegressor(n_estimators=600, random_state=RS, verbose=-1,
                           **_best_params("score"))
    m.fit(prep(train, feats), train["score"].values,
          eval_set=[(prep(test, feats), test["score"].values)],
          categorical_feature=[c for c in feats if c in CAT],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    mae_tr = mean_absolute_error(train["score"].values,
                                  np.clip(m.predict(prep(train, feats)), 0, 1))
    mae_te = mean_absolute_error(test["score"].values,
                                  np.clip(m.predict(prep(test, feats)), 0, 1))
    out["score"] = {"train_MAE": mae_tr, "test_MAE": mae_te, "gap": mae_te - mae_tr}
    print(f"  Score MAE   — train: {mae_tr:.4f} · test: {mae_te:.4f} · gap={mae_te-mae_tr:+.4f}")

    # Dropout / Gap / Termination — similar but for ROC/MAE
    sess = pd.read_parquet(OUT / "session_features.parquet")
    tr, te = next(splitter.split(sess, groups=sess["patient_id"]))
    train, test = sess.iloc[tr], sess.iloc[te]
    feats = ["archetype", "adherence", "session_idx", "n_in_session",
             "n_general_in_session", "session_mean_score", "session_max_level",
             "session_mean_level", "days_since_prev", "r3_gap_mean",
             "days_into_window", "days_to_expiry", "hour", "dow"]
    y_tr = train["is_terminal"].astype(int).values
    y_te = test["is_terminal"].astype(int).values
    pos, neg = y_tr.sum(), len(y_tr) - y_tr.sum()
    m = lgb.LGBMClassifier(n_estimators=600, random_state=RS, verbose=-1,
                            scale_pos_weight=neg/max(pos, 1),
                            **_best_params("dropout"))
    m.fit(prep(train, feats), y_tr,
          eval_set=[(prep(test, feats), y_te)],
          categorical_feature=[c for c in feats if c in CAT],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    auc_tr = roc_auc_score(y_tr, m.predict_proba(prep(train, feats))[:, 1])
    auc_te = roc_auc_score(y_te, m.predict_proba(prep(test, feats))[:, 1])
    out["dropout"] = {"train_ROC": auc_tr, "test_ROC": auc_te, "gap": auc_tr - auc_te}
    print(f"  Dropout ROC — train: {auc_tr:.4f} · test: {auc_te:.4f} · gap={auc_tr-auc_te:+.4f}")

    pf = pd.read_parquet(OUT / "patient_features.parquet").dropna(subset=["did_represcribe"])
    pf = pf[pf["archetype"] != "Z_never_played"]
    tr, te = next(splitter.split(pf, groups=pf["patient_id"]))
    train, test = pf.iloc[tr], pf.iloc[te]
    feats = ["archetype", "adherence", "n_records", "n_general", "n_general_days",
             "active_days", "general_max_level", "general_last_level",
             "general_median_level", "general_mean_score", "score_trend10",
             "median_gap_days", "mean_gap_days", "n_sessions", "n_terminal",
             "prescribed_from"]
    y_tr = train["did_represcribe"].astype(int).values
    y_te = test["did_represcribe"].astype(int).values
    pos, neg = y_tr.sum(), len(y_tr) - y_tr.sum()
    m = lgb.LGBMClassifier(n_estimators=400, random_state=RS, verbose=-1,
                            scale_pos_weight=neg/max(pos, 1),
                            **_best_params("termination"))
    m.fit(prep(train, feats), y_tr,
          eval_set=[(prep(test, feats), y_te)],
          categorical_feature=[c for c in feats if c in CAT],
          callbacks=[lgb.early_stopping(20, verbose=False)])
    auc_tr = roc_auc_score(y_tr, m.predict_proba(prep(train, feats))[:, 1])
    auc_te = roc_auc_score(y_te, m.predict_proba(prep(test, feats))[:, 1])
    out["termination"] = {"train_ROC": auc_tr, "test_ROC": auc_te, "gap": auc_tr - auc_te}
    print(f"  Term ROC    — train: {auc_tr:.4f} · test: {auc_te:.4f} · gap={auc_tr-auc_te:+.4f}")
    return out


# --------------------------------------------------------------------------- #
# 4. Per-archetype subgroup
# --------------------------------------------------------------------------- #

def per_archetype_score():
    print("\n" + "=" * 70)
    print(" 4. PER-ARCHETYPE PERFORMANCE (Score 모델)")
    print("=" * 70)
    sf = pd.read_parquet(OUT / "score_features.parquet").dropna(subset=["score"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS)
    tr, te = next(splitter.split(sf, groups=sf["patient_id"]))
    train, test = sf.iloc[tr], sf.iloc[te]
    feats = ["exercise_name", "kind", "level", "archetype", "adherence",
             "r3_mean", "r3_min", "r3_max", "n_at_exercise", "n_at_exercise_level",
             "t_overall", "hour", "dow", "days_into_window"]
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                           min_child_samples=40, random_state=RS, verbose=-1)
    m.fit(prep(train, feats), train["score"].values,
          eval_set=[(prep(test, feats), test["score"].values)],
          categorical_feature=[c for c in feats if c in CAT],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    pred = np.clip(m.predict(prep(test, feats)), 0, 1)
    test = test.copy()
    test["pred"] = pred
    out = {}
    for arch, g in test.groupby("archetype"):
        out[arch] = {
            "n": len(g),
            "MAE": float(mean_absolute_error(g["score"].values, g["pred"].values)),
            "R2": float(r2_score(g["score"].values, g["pred"].values))
            if len(g) > 1 and g["score"].std() > 1e-6 else None,
        }
    for arch, v in sorted(out.items()):
        r2 = v["R2"]
        r2s = f"{r2:.3f}" if r2 is not None else "n/a"
        print(f"  {arch:25s}  n={v['n']:5d}  MAE={v['MAE']:.4f}  R²={r2s}")
    return out


def main():
    arch_res = archetype_ablation()
    kfold_res = kfold_stability(k=5)
    overfit_res = overfit_check()
    arch_score = per_archetype_score()

    out_path = OUT / "audit_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"archetype_ablation": arch_res, "kfold": kfold_res,
                   "overfit": overfit_res, "per_archetype_score": arch_score},
                  f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
