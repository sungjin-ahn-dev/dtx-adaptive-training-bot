"""
임팩트 있는 종합 deep-dive 리포트.

데이터 → archetype → ML 모델 → 시뮬 → 결과 전반을 풍부한 시각화로 정리.

Run:
  python -m bot.make_deep_report --out deep_report.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_curve, auc

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
MODELS = ROOT / "models"

for fname in ("Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"):
    try:
        matplotlib.rcParams["font.family"] = fname
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["axes.facecolor"] = "white"

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

C = {
    "primary": "#4f7cf7",
    "primary_dark": "#1a73e8",
    "success": "#1e8e3e",
    "warning": "#f9ab00",
    "danger": "#d93025",
    "purple": "#a142f4",
    "teal": "#00897b",
    "pink": "#e91e63",
    "ink": "#202124",
    "gray": "#9aa0a6",
    "light_gray": "#f1f3f4",
}

ARCH_ORDER = ["A_early_dropout", "B_stalled_settled", "C_stalled_bored",
              "D_mid_wall", "E_maxed_out", "F_steady_progress"]
ARCH_COLORS = {
    "A_early_dropout": "#d93025",
    "B_stalled_settled": "#f9ab00",
    "C_stalled_bored": "#fa7b17",
    "D_mid_wall": "#a142f4",
    "E_maxed_out": "#1e8e3e",
    "F_steady_progress": "#4f7cf7",
    "Z_never_played": "#9aa0a6",
}
ARCH_LABEL = {
    "A_early_dropout": "A. 초기 이탈",
    "B_stalled_settled": "B. 안주형",
    "C_stalled_bored": "C. 지루형",
    "D_mid_wall": "D. 중간 벽",
    "E_maxed_out": "E. 졸업자",
    "F_steady_progress": "F. 진행자",
    "Z_never_played": "Z. 미시작",
}
ARCH_TAGLINE = {
    "A_early_dropout": "사용 습관 미형성 — 학습 시작 전 이탈",
    "B_stalled_settled": "꾸준하지만 진척 없음 → 재처방률 최고",
    "C_stalled_bored": "레벨 정체 + 참여 떨어짐 → 이탈",
    "D_mid_wall": "중상위에서 막힘, 심리적 정체기",
    "E_maxed_out": "치료 목표 도달 = '끝났다' 신호",
    "F_steady_progress": "꾸준한 정상 진행",
    "Z_never_played": "처방만 받고 한 번도 안 함",
}


def fig_b64(fig, dpi=110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


# --------------------------------------------------------------------------- #
# Section 1: Data spotlight
# --------------------------------------------------------------------------- #

def chart_rx_timeline(rx: pd.DataFrame) -> str:
    rx = rx.copy()
    rx["month"] = pd.to_datetime(rx["created_at"], utc=True).dt.tz_convert("Asia/Seoul").dt.to_period("M").astype(str)
    monthly = rx.groupby(["month", "prescribed_from"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(13, 4))
    monthly.plot.area(ax=ax, alpha=0.75,
                       color=[C["primary"], C["success"], C["purple"]])
    ax.set_title("처방 채널별 월별 처방 추이", fontsize=14, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("처방 수")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    return fig_b64(fig)


def chart_top_hospitals(rx: pd.DataFrame) -> str:
    top = rx["hospital"].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.barh(top.index[::-1], top.values[::-1], color=C["primary"])
    bars[-1].set_color(C["success"])
    for b, v in zip(bars, top.values[::-1]):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2,
                str(v), va="center", fontsize=10)
    ax.set_title("처방 기관 TOP 12", fontsize=14, pad=10)
    ax.set_xlabel("처방 수")
    return fig_b64(fig)


def chart_patient_event_dist(patients: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12, 4))
    vals = patients["n_records"].dropna().values
    ax.hist(vals, bins=40, color=C["primary"], edgecolor="white", alpha=0.85)
    ax.axvline(np.median(vals), color=C["danger"], linestyle="--", linewidth=2,
                label=f"중앙값 {np.median(vals):.0f}")
    ax.axvline(np.mean(vals), color=C["success"], linestyle="--", linewidth=2,
                label=f"평균 {np.mean(vals):.0f}")
    ax.set_title(f"환자당 누적 이벤트 수 분포 (n={len(vals)})", fontsize=14, pad=10)
    ax.set_xlabel("이벤트 수")
    ax.set_ylabel("환자 수")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.3)
    return fig_b64(fig)


def chart_exercise_score_matrix(events: pd.DataFrame) -> str:
    g = events.dropna(subset=["level", "score"]).copy()
    g["level"] = g["level"].astype(int)
    pivot = g.pivot_table(index="exercise_name", columns="level",
                          values="score", aggfunc="mean")
    pivot = pivot.reindex(sorted(pivot.index)).sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if 0.45 < v < 0.85 else "white")
    ax.set_title("운동 × 레벨별 평균 점수", fontsize=14, pad=10)
    ax.set_xlabel("level")
    fig.colorbar(im, ax=ax).set_label("mean score")
    return fig_b64(fig)


def chart_sample_trajectories(events: pd.DataFrame, patients: pd.DataFrame) -> str:
    # one patient per archetype that has plenty of data
    eligible = patients[(patients["archetype"] != "Z_never_played") &
                        (patients["n_general"].fillna(0) >= 20)]
    sampled = (eligible.groupby("archetype")
                .apply(lambda g: g.sort_values("n_general", ascending=False).head(1))
                .reset_index(drop=True))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    axes = axes.flatten()
    for i, arch in enumerate(ARCH_ORDER):
        ax = axes[i]
        sub = sampled[sampled["archetype"] == arch]
        if not len(sub):
            ax.axis("off")
            continue
        pid = sub.iloc[0]["patient_id"]
        ev = events[(events["patient_id"] == pid) &
                    (events["kind"] == "general")].sort_values("created_at")
        if not len(ev):
            ax.axis("off")
            continue
        col = ARCH_COLORS[arch]
        ax.plot(range(len(ev)), ev["level"].values, color=col, lw=2,
                label="level")
        ax2 = ax.twinx()
        ax2.scatter(range(len(ev)), ev["score"].values, color=C["gray"], s=8, alpha=0.4,
                     label="score")
        ax2.set_ylim(0, 1.05)
        ax.set_title(f"{ARCH_LABEL[arch]} (n_gen={len(ev)})", fontsize=11)
        ax.set_xlabel("event index")
        if i % 3 == 0:
            ax.set_ylabel("level", color=col)
        ax.set_ylim(0, 7.5)
        ax.tick_params(axis="y", labelcolor=col)
    fig.suptitle("Archetype별 대표 환자 trajectory (선=level, 점=score)",
                  fontsize=14)
    fig.tight_layout()
    return fig_b64(fig)


# --------------------------------------------------------------------------- #
# Section 2: Archetype deep-dive
# --------------------------------------------------------------------------- #

def chart_archetype_donut(patients: pd.DataFrame) -> str:
    counts = patients["archetype"].value_counts().reindex(
        ARCH_ORDER + ["Z_never_played"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = [ARCH_COLORS[a] for a in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=[ARCH_LABEL[a] for a in counts.index],
        colors=colors, autopct="%.1f%%", startangle=90,
        pctdistance=0.78, wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2))
    for t in autotexts:
        t.set_fontsize(10); t.set_color("white"); t.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(11)
    ax.text(0, 0, f"{counts.sum()}\n환자", ha="center", va="center",
            fontsize=20, fontweight="bold")
    ax.set_title("Archetype 분포", fontsize=15, pad=10)
    return fig_b64(fig)


def chart_archetype_box(patients: pd.DataFrame, col: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    data, labels, colors = [], [], []
    for arch in ARCH_ORDER:
        d = patients.loc[patients["archetype"] == arch, col].dropna().values
        if not len(d):
            continue
        data.append(d)
        labels.append(ARCH_LABEL[arch])
        colors.append(ARCH_COLORS[arch])
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(2)
    ax.set_title(title, fontsize=14, pad=10)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    return fig_b64(fig)


def chart_archetype_stats(patients: pd.DataFrame) -> str:
    rates = (patients.groupby("archetype")["did_represcribe"].mean()
             .reindex(ARCH_ORDER + ["Z_never_played"], fill_value=0))
    counts = (patients["archetype"].value_counts()
              .reindex(ARCH_ORDER + ["Z_never_played"], fill_value=0))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = [ARCH_COLORS[a] for a in rates.index]
    bars = ax.bar([ARCH_LABEL[a] for a in rates.index], rates.values * 100,
                   color=colors)
    for b, v, n in zip(bars, rates.values, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f"{v*100:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=10)
    _b_rate = rates.get("B_stalled_settled", 0) * 100
    ax.set_title(f"Archetype별 실제 재처방률 — B 안주형이 {_b_rate:.1f}%로 최고", fontsize=14, pad=10)
    ax.set_ylabel("재처방률 %")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0, max(25, rates.max() * 100 * 1.2))
    ax.grid(axis="y", alpha=0.3)
    return fig_b64(fig)


def archetype_cards_html(patients: pd.DataFrame) -> str:
    cards = []
    for arch in ARCH_ORDER:
        g = patients[patients["archetype"] == arch]
        if not len(g):
            continue
        c = ARCH_COLORS[arch]
        n = len(g)
        pct = n / len(patients) * 100
        rx_rate = g["did_represcribe"].mean() * 100
        adh = g["adherence"].mean()
        max_lvl = g["general_max_level"].mean()
        n_gen = g["n_general"].mean()
        cards.append(f"""
<div class='arch-card' style='border-top: 5px solid {c}'>
  <div class='arch-head'><span class='arch-name' style='color:{c}'>{ARCH_LABEL[arch]}</span>
    <span class='arch-count'>{n}명 ({pct:.1f}%)</span></div>
  <div class='arch-tag'>{ARCH_TAGLINE[arch]}</div>
  <div class='arch-stats'>
    <div class='stat'><span class='num' style='color:{c}'>{rx_rate:.1f}%</span><span class='lab'>재처방률</span></div>
    <div class='stat'><span class='num'>{adh:.2f}</span><span class='lab'>adherence</span></div>
    <div class='stat'><span class='num'>{max_lvl:.1f}</span><span class='lab'>max level</span></div>
    <div class='stat'><span class='num'>{n_gen:.0f}</span><span class='lab'>n_general</span></div>
  </div>
</div>""")
    return f"<div class='arch-grid'>{''.join(cards)}</div>"


# --------------------------------------------------------------------------- #
# Section 3: ML models — re-evaluate on holdout for visuals
# --------------------------------------------------------------------------- #

def holdout_predictions():
    """Return dict with predictions on holdout for each model."""
    res = {}
    rs = 42
    cat_cols = {"exercise_name", "kind", "archetype", "prescribed_from", "hospital"}

    def prep(df, feats):
        X = df.reindex(columns=feats).copy()
        for c in cat_cols & set(X.columns):
            X[c] = X[c].astype("category")
        return X

    # Score
    pack = joblib.load(MODELS / "score_model.joblib")
    feats = pack["features"]
    sf = pd.read_parquet(OUT / "score_features.parquet").dropna(subset=["score"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
    _, ti = next(splitter.split(sf, groups=sf["patient_id"]))
    test = sf.iloc[ti]
    pred = np.clip(pack["model"].predict(prep(test, feats)), 0, 1)
    res["score"] = dict(y=test["score"].values, pred=pred,
                         feats=feats, importance=pack["model"].feature_importances_,
                         test_levels=test["level"].values.astype(int))

    # Dropout
    pack = joblib.load(MODELS / "dropout_model.joblib")
    feats = pack["features"]
    sess = pd.read_parquet(OUT / "session_features.parquet")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
    _, ti = next(splitter.split(sess, groups=sess["patient_id"]))
    test = sess.iloc[ti]
    proba = pack["model"].predict_proba(prep(test, feats))[:, 1]
    res["dropout"] = dict(y=test["is_terminal"].astype(int).values, proba=proba,
                           feats=feats, importance=pack["model"].feature_importances_)

    # Gap
    pack = joblib.load(MODELS / "gap_model.joblib")
    feats = pack["features"]
    sg = sess.dropna(subset=["days_to_next"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
    _, ti = next(splitter.split(sg, groups=sg["patient_id"]))
    test = sg.iloc[ti]
    log_pred = pack["model"].predict(prep(test, feats))
    res["gap"] = dict(y=test["days_to_next"].values,
                       pred=np.expm1(log_pred).clip(min=0),
                       feats=feats, importance=pack["model"].feature_importances_)

    # Termination
    pack = joblib.load(MODELS / "termination_model.joblib")
    feats = pack["features"]
    pf = pd.read_parquet(OUT / "patient_features.parquet").dropna(subset=["did_represcribe"])
    pf = pf[pf["archetype"] != "Z_never_played"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
    _, ti = next(splitter.split(pf, groups=pf["patient_id"]))
    test = pf.iloc[ti]
    proba = pack["model"].predict_proba(prep(test, feats))[:, 1]
    res["termination"] = dict(y=test["did_represcribe"].astype(int).values, proba=proba,
                                feats=feats, importance=pack["model"].feature_importances_)
    return res


def chart_score_pred_actual(d: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    sample = np.random.RandomState(0).choice(len(d["y"]), size=min(5000, len(d["y"])), replace=False)
    sc = ax.scatter(d["y"][sample], d["pred"][sample], alpha=0.15, s=8,
                     c=d["test_levels"][sample], cmap="viridis")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.set_xlabel("실제 score")
    ax.set_ylabel("예측 score")
    ax.set_title("Score 모델 — predicted vs actual (sample 5k)", fontsize=12)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    fig.colorbar(sc, ax=ax).set_label("level")
    ax.legend()

    ax = axes[1]
    err = d["pred"] - d["y"]
    ax.hist(err, bins=50, color=C["primary"], edgecolor="white", alpha=0.85)
    ax.axvline(0, color=C["danger"], linestyle="--", linewidth=2)
    ax.set_title(f"예측 오차 분포 (mean={err.mean():.3f}, std={err.std():.3f})", fontsize=12)
    ax.set_xlabel("pred - actual")
    fig.tight_layout()
    return fig_b64(fig)


def chart_roc_calibration(d: dict, title: str) -> str:
    y, proba = d["y"], d["proba"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    fpr, tpr, _ = roc_curve(y, proba)
    auc_val = auc(fpr, tpr)
    ax = axes[0]
    ax.plot(fpr, tpr, color=C["primary"], lw=2.5, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.fill_between(fpr, tpr, alpha=0.2, color=C["primary"])
    ax.set_title(f"{title} — ROC curve", fontsize=12)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # Calibration: bin predictions and compare empirical rate
    ax = axes[1]
    bins = np.linspace(0, 1, 11)
    digit = np.digitize(proba, bins) - 1
    digit = np.clip(digit, 0, 9)
    rates, centers, counts = [], [], []
    for i in range(10):
        m = digit == i
        if m.sum() < 5:
            continue
        rates.append(y[m].mean()); centers.append(proba[m].mean()); counts.append(m.sum())
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    sizes = np.array(counts) / max(counts) * 250 + 20
    ax.scatter(centers, rates, s=sizes, color=C["success"], alpha=0.7, edgecolor="white", lw=1.5)
    ax.plot(centers, rates, color=C["success"], lw=1.5, alpha=0.6)
    ax.set_title(f"{title} — calibration (size = bin count)", fontsize=12)
    ax.set_xlabel("예측 확률")
    ax.set_ylabel("실제 양성 비율")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig_b64(fig)


def chart_gap_pred_actual(d: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sample = np.random.RandomState(0).choice(len(d["y"]), size=min(4000, len(d["y"])), replace=False)
    y, pred = d["y"][sample], d["pred"][sample]
    ax = axes[0]
    ax.scatter(y, pred, alpha=0.15, s=8, color=C["primary"])
    mx = float(np.nanpercentile(y, 99))
    ax.plot([0, mx], [0, mx], "k--", alpha=0.5)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel("실제 gap (days)")
    ax.set_ylabel("예측 gap (days)")
    ax.set_title("Gap 모델 — predicted vs actual (sample 4k)", fontsize=12)

    ax = axes[1]
    err = pred - y
    ax.hist(err, bins=60, range=(-3, 3), color=C["warning"], edgecolor="white", alpha=0.85)
    ax.axvline(0, color=C["danger"], linestyle="--", linewidth=2)
    ax.set_title(f"오차 분포 (median |err|={np.median(np.abs(err)):.2f}일)", fontsize=12)
    ax.set_xlabel("pred - actual (days)")
    fig.tight_layout()
    return fig_b64(fig)


def chart_feature_importance(d: dict, title: str, top: int = 12) -> str:
    imp = pd.Series(d["importance"], index=d["feats"]).sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(10, 0.4 * top + 1))
    ax.barh(imp.index, imp.values, color=C["teal"])
    for i, v in enumerate(imp.values):
        ax.text(v + max(imp.values) * 0.01, i, str(int(v)), va="center", fontsize=9)
    ax.set_title(f"{title} — feature importance (top {top})", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    return fig_b64(fig)


# --------------------------------------------------------------------------- #
# Section 4: Simulator & policy results
# --------------------------------------------------------------------------- #

def chart_policy_ranking(agg: pd.DataFrame) -> str:
    df = agg.sort_values("represcribe_rate_mean", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 0.32 * len(df) + 1.5))
    labels = df.apply(lambda r: f"{r['adjuster']}  |  {r['composer']}", axis=1)
    means = df["represcribe_rate_mean"] * 100
    cis = df["represcribe_rate_ci95"] * 100
    norm = plt.Normalize(means.min(), means.max())
    colors = plt.cm.RdYlGn(norm(means.values))
    bars = ax.barh(labels, means, xerr=cis, color=colors,
                    capsize=3, error_kw={"ecolor": "#444", "lw": 1})
    for b, m, c in zip(bars, means, cis):
        ax.text(b.get_width() + c + 0.06, b.get_y() + b.get_height() / 2,
                f"{m:.1f}±{c:.1f}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("재처방률 % (mean ± 95% CI)")
    ax.set_title("정책별 재처방률 ranking (5-seed multi-seed)", fontsize=14, pad=10)
    ax.grid(axis="x", alpha=0.3)
    return fig_b64(fig)


def chart_policy_heatmap(agg: pd.DataFrame, metric: str, title: str,
                          fmt: str = "{:.1%}", cmap: str = "YlGn") -> str:
    pivot = agg.pivot(index="composer", columns="adjuster", values=f"{metric}_mean")
    fig, ax = plt.subplots(
        figsize=(1.5 + 1.4 * len(pivot.columns), 0.7 * len(pivot.index) + 2.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=14, pad=10)
    fig.colorbar(im, ax=ax)
    return fig_b64(fig)


def chart_marginal_effects(raw: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # composer effect
    ax = axes[0]
    g = raw.groupby("composer")["represcribe_rate"].agg(["mean", "std", "count"])
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
    g = g.sort_values("mean", ascending=False)
    bars = ax.bar(g.index, g["mean"] * 100, yerr=g["ci"] * 100,
                   color=C["primary"], capsize=4)
    bars[0].set_color(C["success"])
    for b, m, c in zip(bars, g["mean"], g["ci"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + c * 100 + 0.05,
                f"{m*100:.1f}±{c*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Composer 효과 (adjuster 모두 평균)", fontsize=13)
    ax.set_ylabel("재처방률 %")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)

    # adjuster effect
    ax = axes[1]
    g = raw.groupby("adjuster")["represcribe_rate"].agg(["mean", "std", "count"])
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
    g = g.sort_values("mean", ascending=False)
    bars = ax.bar(g.index, g["mean"] * 100, yerr=g["ci"] * 100,
                   color=C["purple"], capsize=4)
    bars[0].set_color(C["success"])
    for b, m, c in zip(bars, g["mean"], g["ci"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + c * 100 + 0.05,
                f"{m*100:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Adjuster 효과 (composer 모두 평균)", fontsize=13)
    ax.set_ylabel("재처방률 %")
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig_b64(fig)


def chart_maintain_vs_represcribe(raw: pd.DataFrame) -> str:
    g = raw.groupby(["composer", "adjuster"]).agg(
        rep=("represcribe_rate", "mean"),
        maintain=("mean_maintain", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    adj_set = sorted(g["adjuster"].unique())
    cmap = plt.cm.tab20
    for i, adj in enumerate(adj_set):
        sub = g[g["adjuster"] == adj]
        ax.scatter(sub["maintain"] * 100, sub["rep"] * 100, s=130,
                    color=cmap(i / max(1, len(adj_set))), label=adj,
                    edgecolor="white", linewidth=1.5, alpha=0.85)
    ax.set_xlabel("난이도 유지율 % (낮을수록 동적)")
    ax.set_ylabel("재처방률 %")
    ax.set_title("난이도 유지율 vs 재처방률 — 각 점 = 정책 (composer × adjuster)",
                  fontsize=13, pad=10)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9, frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig_b64(fig)


def chart_holdout_distribution() -> str:
    """Real holdout vs sim (baseline) for max_level distribution."""
    from .simulator import DTxSimulator, Policy
    from .algorithms import CurrentRule
    from .composers import RandomBalanced

    patients = pd.read_parquet(OUT / "patients.parquet")
    events = pd.read_parquet(OUT / "events.parquet")
    eligible = patients[patients["archetype"] != "Z_never_played"].reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, ti = next(splitter.split(eligible, groups=eligible["patient_id"]))
    test_ids = set(eligible.iloc[ti]["patient_id"])
    real_max = []
    for pid in test_ids:
        g = events[(events["patient_id"] == pid) & (events["kind"] == "general")]
        if len(g) > 0:
            real_max.append(int(g["level"].max()))

    sim = DTxSimulator(rng_seed=42)
    res = sim.run(Policy(RandomBalanced(), CurrentRule()),
                  n_patients=600, window_days=83)
    sim_max = res["patients"]["max_level"].values

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bins = np.arange(1, 9) - 0.5
    ax.hist([real_max, sim_max], bins=bins, label=["실제 holdout", "봇 시뮬"],
             color=[C["danger"], C["primary"]], alpha=0.75, edgecolor="white")
    ax.set_xticks(range(1, 8))
    ax.set_title("max_level 분포 — 실제 환자 vs 봇 시뮬 (⚠️ 시뮬이 더 높게 분포)",
                  fontsize=14, pad=10)
    ax.set_xlabel("도달 max_level")
    ax.set_ylabel("환자 수")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return fig_b64(fig)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

CSS = """
:root {
  --primary: #4f7cf7; --primary-dark: #1a73e8; --success: #1e8e3e;
  --warning: #f9ab00; --danger: #d93025; --ink: #202124; --gray: #5f6368;
  --bg: #fafbfc; --card: #ffffff; --border: #e8eaed;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, "Malgun Gothic", "Segoe UI", sans-serif;
       max-width: 1280px; margin: 0 auto; padding: 20px 28px 60px;
       color: var(--ink); line-height: 1.6; background: var(--bg); }
h1 { font-size: 32px; margin: 0 0 8px 0; }
h1 .sub { color: var(--gray); font-size: 16px; font-weight: 400; display: block; margin-top: 4px; }
h2 { margin-top: 56px; padding: 14px 18px;
     background: linear-gradient(135deg, #1a73e8 0%, #4f7cf7 100%);
     color: white; border-radius: 8px; font-size: 22px;
     box-shadow: 0 4px 12px rgba(26,115,232,0.2); }
h3 { color: var(--ink); margin-top: 32px; font-size: 18px;
     padding-left: 12px; border-left: 4px solid var(--primary); }
p, ul, li { font-size: 14.5px; }
ul { padding-left: 22px; }
img { max-width: 100%; height: auto; border-radius: 6px; }
code { background: #f1f3f4; padding: 2px 7px; border-radius: 4px;
       font-size: 0.9em; font-family: 'Cascadia Mono', 'Consolas', monospace; }

/* Top KPI strip */
.kpi-strip { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin: 22px 0 0; }
.kpi-card { background: var(--card); border-radius: 10px; padding: 18px 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-top: 4px solid var(--primary); }
.kpi-card.green { border-top-color: var(--success); }
.kpi-card.warn  { border-top-color: var(--warning); }
.kpi-card.red   { border-top-color: var(--danger); }
.kpi-card.purple{ border-top-color: #a142f4; }
.kpi-card.teal  { border-top-color: #00897b; }
.kpi-card .v { font-size: 26px; font-weight: 700; line-height: 1.1; }
.kpi-card .l { color: var(--gray); font-size: 12.5px; margin-top: 5px; }

/* Cards */
.card { background: var(--card); border: 1px solid var(--border);
        border-radius: 8px; padding: 16px 20px; margin: 14px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04); }

/* Archetype cards */
.arch-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 20px 0; }
.arch-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
             padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); transition: transform .15s; }
.arch-card:hover { transform: translateY(-2px); box-shadow: 0 4px 14px rgba(0,0,0,0.1); }
.arch-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.arch-name { font-size: 16px; font-weight: 700; }
.arch-count { color: var(--gray); font-size: 12.5px; }
.arch-tag { font-size: 12.5px; color: var(--gray); margin-bottom: 12px; font-style: italic; }
.arch-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
.stat { text-align: center; padding: 6px 4px; background: #f8f9fa; border-radius: 5px; }
.stat .num { display: block; font-size: 15px; font-weight: 700; }
.stat .lab { display: block; font-size: 10.5px; color: var(--gray); margin-top: 2px; }

/* Model cards */
.model-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 20px 0; }
.model-card { background: var(--card); border-radius: 8px; padding: 16px 18px;
              border: 1px solid var(--border); box-shadow: 0 2px 6px rgba(0,0,0,0.04); }
.model-card.green  { border-left: 5px solid var(--success); }
.model-card.yellow { border-left: 5px solid var(--warning); }
.model-card.red    { border-left: 5px solid var(--danger); }
.model-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.model-name { font-size: 17px; font-weight: 700; }
.verdict-tag { font-size: 13px; padding: 3px 9px; border-radius: 4px; font-weight: 600; }
.verdict-tag.ok { background: #e6f4ea; color: var(--success); }
.verdict-tag.warn { background: #fef7e0; color: #b06000; }
.verdict-tag.bad { background: #fce8e6; color: var(--danger); }
.model-row { font-size: 13.5px; margin: 4px 0; }
.model-metric { margin-top: 10px; padding: 8px 10px; background: #eef2ff;
                border-radius: 5px; font-family: monospace; font-size: 13px; }
.model-note { margin-top: 8px; font-size: 12.5px; color: var(--gray); font-style: italic; }

/* Callouts */
.note, .tip, .warn { padding: 14px 18px; margin: 18px 0; border-radius: 8px; font-size: 14px; }
.note { background: #fef7e0; border-left: 5px solid var(--warning); }
.tip  { background: #e6f4ea; border-left: 5px solid var(--success); }
.warn { background: #fce8e6; border-left: 5px solid var(--danger); }

/* Tables */
table { width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 13px;
        background: var(--card); border-radius: 6px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
th, td { border: 1px solid var(--border); padding: 8px 12px; text-align: right; }
th { background: #f1f3f4; font-weight: 600; }
td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) {
  text-align: left; font-family: 'Cascadia Mono', monospace; font-size: 12.5px; }
tr.best { background: #e6f4ea; font-weight: 600; }
.def-table td, .def-table th { text-align: left; font-family: inherit; font-size: 13px; }
.pass { color: var(--success); font-weight: 700; }
.fail { color: var(--danger); font-weight: 700; }

/* Pipeline ASCII */
.pipeline { background: #1f1f1f; color: #e6e6e6; padding: 16px 18px;
            border-radius: 8px; font-family: monospace; font-size: 12.5px;
            line-height: 1.5; white-space: pre; overflow-x: auto; }
.pipeline .step { color: #4fc3f7; font-weight: 700; }

.footer { color: var(--gray); font-size: 12px; margin-top: 60px; padding-top: 20px;
          border-top: 1px solid var(--border); text-align: center; }
.small { font-size: 12.5px; color: var(--gray); }

/* Executive Summary */
.exec-summary { background: linear-gradient(135deg, #f0fdf4 0%, #eff6ff 100%);
                border: 2px solid #1e8e3e; border-radius: 12px; padding: 24px 28px;
                margin: 28px 0; box-shadow: 0 4px 16px rgba(30,142,62,0.12); }
.exec-head { margin-bottom: 14px; }
.exec-tag { background: #1e8e3e; color: white; padding: 4px 12px; border-radius: 4px;
            font-size: 12px; font-weight: 700; letter-spacing: 1px; }
.exec-tldr { background: white; border-left: 5px solid #1e8e3e; padding: 14px 18px;
             margin: 14px 0 24px; border-radius: 6px; font-size: 15px; line-height: 1.7; }
.exec-h3 { color: #1a73e8; font-size: 17px; margin-top: 24px; padding-left: 0;
           border-left: none; border-bottom: 2px solid #4f7cf7; padding-bottom: 4px; }
.exec-mini-table { width: auto; min-width: 60%; }
.exec-mini-table td { padding: 7px 14px; }
.exec-mini-table td:first-child { background: #f8f9fa; font-weight: 600; min-width: 110px; }
.exec-rule { background: #fff; border: 1px solid #ddd; border-radius: 6px;
             padding: 14px 22px; font-size: 15px; line-height: 1.9; margin: 10px 0; }
.exec-rule code { background: #f1f3f4; padding: 2px 8px; }
.exec-issue { background: #fff8e1; border-left: 4px solid #f9ab00; padding: 12px 16px;
              border-radius: 6px; margin: 10px 0; font-size: 14px; }
.policy-card { background: white; border-radius: 8px; padding: 16px 20px; margin: 12px 0;
               border: 1px solid #e0e0e0; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }
.policy-label { display: inline-block; color: white; font-size: 12px; font-weight: 700;
                padding: 3px 10px; border-radius: 4px; margin-bottom: 8px; }
.policy-name { font-size: 16px; font-weight: 700; margin-bottom: 12px; }
.policy-name code { font-size: 15px; background: #eef2ff; }
.policy-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.pst { background: #f8f9fa; padding: 10px 12px; border-radius: 6px; text-align: center; }
.pst-v { display: block; font-size: 22px; font-weight: 700; color: #1a73e8; }
.pst-l { display: block; font-size: 11.5px; color: #5f6368; margin-top: 4px; line-height: 1.4; }
.exec-rank-table { width: 100%; max-width: 720px; }
.exec-rank-table td { padding: 6px 12px; }
tr.rank-best { background: #e6f4ea; font-weight: 700; }
tr.rank-current { background: #fff8e1; font-weight: 600; }
.exec-findings { font-size: 14px; line-height: 1.8; }
.exec-cta { margin-top: 22px; padding: 12px 16px; background: rgba(26,115,232,0.08);
            border-radius: 6px; text-align: center; font-size: 13px; color: #5f6368; }

.toc { background: var(--card); padding: 14px 20px; border-radius: 8px; margin: 18px 0;
       border: 1px solid var(--border); }
.toc ol { margin: 6px 0 0; padding-left: 22px; }
.toc li { margin: 3px 0; font-size: 14px; }
.toc a { color: var(--primary-dark); text-decoration: none; }
.toc a:hover { text-decoration: underline; }
"""


PIPELINE_ASCII = """[1] 원천 데이터 (0421)
    prescription 866건 · assignment 221,318건 · 환자 775명
                          ↓ build_dataset.py
[2] 환자 trajectory + archetype 라벨 (6+1 룰 기반)
    A 초기이탈 · B 안주 · C 지루 · D 중간벽 · E 졸업 · F 진행 · Z 미시작
                          ↓ build_features.py
[3] 학습용 피처 테이블 3종
    event-level 221k · session-level 221k · patient-level 775
                          ↓ train_models.py
[4] LightGBM 모델 4종 (환자단위 80/20 split)
    score · dropout · gap · termination
                          ↓ simulator.py + algorithms/composers
[5] 가상 환자 시뮬레이터 (cohort-step)
    입력: 정책 (composer × adjuster)
    출력: 재처방률 · 이탈률 · max_level · 유지율 · archetype 분포
                          ↓ make_report.py
[6] Multi-seed 정책 비교 + holdout KS 검증
    composer 5 × adjuster 11 × 5 seeds = 275 시뮬"""


def section_executive_summary(agg: pd.DataFrame) -> str:
    """Slack-friendly 요약 — 비기술 독자 대상 한눈 정리. 페이지 최상단에."""
    df = agg.copy()
    base = df[(df["composer"] == "random_balanced")
              & (df["adjuster"] == "current_50_80")]
    base_rep = float(base["represcribe_rate_mean"].iloc[0]) if len(base) else 0
    base_ci = float(base["represcribe_rate_ci95"].iloc[0]) if len(base) else 0
    base_drop = float(base["drop_rate_mean"].iloc[0]) if len(base) else 0

    # KEY METRIC: balance score = (rep_uplift) - (drop_uplift)
    # 둘 다 percentage point 단위로 같은 가중치. 양수면 baseline 대비 개선.
    df["balance_score"] = (df["represcribe_rate_mean"] - base_rep) - (df["drop_rate_mean"] - base_drop)

    # adjuster effect, averaged over composers — balance 기준 정렬
    adj_eff = (df.groupby("adjuster").agg(
        rep=("represcribe_rate_mean", "mean"),
        drop=("drop_rate_mean", "mean"),
        max_lvl=("mean_max_level_mean", "mean"),
    ))
    adj_eff["balance"] = (adj_eff["rep"] - base_rep) - (adj_eff["drop"] - base_drop)
    adj_eff = adj_eff.sort_values("balance", ascending=False)
    current_rank = list(adj_eff.index).index("current_50_80") + 1 if "current_50_80" in adj_eff.index else None

    # Top 3: 재처방 ↑ AND 이탈 ↓ 둘 다 개선된 정책 중 balance score 상위
    better_both = df[(df["represcribe_rate_mean"] > base_rep) & (df["drop_rate_mean"] < base_drop)]
    top3 = better_both.sort_values("balance_score", ascending=False).head(3) if len(better_both) >= 3 else df.sort_values("balance_score", ascending=False).head(3)
    best_balance = df.sort_values("balance_score", ascending=False).iloc[0]

    # rendered cards
    def policy_card(row, label, label_color):
        rep_delta = (row['represcribe_rate_mean'] - base_rep) * 100
        drop_delta = (row['drop_rate_mean'] - base_drop) * 100
        return f"""
<div class='policy-card'>
  <div class='policy-label' style='background:{label_color}'>{label}</div>
  <div class='policy-name'>
    <code>{row['adjuster']}</code> <span style='color:#9aa0a6'>×</span> <code>{row['composer']}</code>
  </div>
  <div class='policy-stats'>
    <div class='pst'><span class='pst-v' style='color:#1e8e3e'>{row['represcribe_rate_mean']*100:.1f}%</span>
      <span class='pst-l'>예상 재처방률 (↑좋음)<br>baseline {base_rep*100:.1f}% 대비 <b style='color:#1e8e3e'>{rep_delta:+.1f}%p</b></span></div>
    <div class='pst'><span class='pst-v' style='color:#1e8e3e'>{row['drop_rate_mean']*100:.1f}%</span>
      <span class='pst-l'>예상 이탈률 (↓좋음)<br>baseline {base_drop*100:.1f}% 대비 <b style='color:#1e8e3e'>{drop_delta:+.1f}%p</b></span></div>
    <div class='pst'><span class='pst-v'>{row['balance_score']*100:+.2f}p</span>
      <span class='pst-l'>균형 점수<br>(재처방↑ − 이탈↑)</span></div>
  </div>
</div>"""

    rank_rows = []
    for i, (adj, r) in enumerate(adj_eff.iterrows()):
        is_current = adj == "current_50_80"
        is_best = i == 0
        cls = "rank-best" if is_best else ("rank-current" if is_current else "")
        marker = "1위" if is_best else (f"{current_rank}위 (현행)" if is_current else f"{i+1}위")
        rank_rows.append(
            f"<tr class='{cls}'><td>{marker}</td><td><code>{adj}</code></td>"
            f"<td>{r['rep']*100:.2f}%</td><td>{r['drop']*100:.1f}%</td>"
            f"<td><b>{r['balance']*100:+.2f}p</b></td></tr>")

    # --- dynamic prose helpers (computed from THIS run, not hardcoded) ---
    _best_adj = adj_eff.index[0]
    _best_rep_d = (adj_eff.iloc[0]["rep"] - base_rep) * 100
    _best_drop_d = (adj_eff.iloc[0]["drop"] - base_drop) * 100
    _rep_top = adj_eff.sort_values("rep", ascending=False).index[0]      # 재처방만 최고
    _rep_top_drop_d = (adj_eff.loc[_rep_top, "drop"] - base_drop) * 100
    _proposed = [a for a in ("hybrid_drop70_strict_up", "level_diff", "drop_at_70")
                if a in adj_eff.index]
    _proposed_top = [a for a in adj_eff.index if a in _proposed][:3]       # balance 순
    _proposed_str = ", ".join(
        f"<code>{a}</code>({(adj_eff.loc[a,'balance'])*100:+.2f}p)" for a in _proposed_top)
    _best_drop_adj = adj_eff.sort_values("drop").index[0]                # 이탈 최소
    _best_drop_val = adj_eff.loc[_best_drop_adj, "drop"] * 100
    _n_adj = len(adj_eff)

    return f"""
<div class='exec-summary'>
  <div class='exec-head'>
    <span class='exec-tag'>EXECUTIVE SUMMARY</span>
    <h2 style='margin:6px 0 0 0; background:none; color:#202124; box-shadow:none; padding:0; font-size:24px'>
      어떻게 학습했고, 어떤 알고리즘이 효과적인가?
    </h2>
  </div>

  <div class='exec-tldr'>
    <b>TL;DR</b> &nbsp; <b>두 KPI(재처방률 ↑ + 이탈률 ↓)를 함께 보면 어느 규칙이 균형이 좋은지가 드러남</b>.
    이번 0421 데이터 기준 adjuster 균형 점수 1위는 <code>{_best_adj}</code>
    — baseline 대비 재처방률 <b>{_best_rep_d:+.1f}%p</b>, 이탈률 <b>{_best_drop_d:+.1f}%p</b>.
    현행 룰(<code>3회 평균 50/80%</code>)은 {_n_adj}개 중 <b>{current_rank}위</b>.
    실제 적용 전 작은 그룹 A/B 시험 필수.
  </div>

  <h3 class='exec-h3'>① 봇 학습 방법 (요약)</h3>
  <table class='exec-mini-table'>
    <tr><th>항목</th><th>내용</th></tr>
    <tr><td>학습 데이터</td><td>실제 DTx 제품 환자 775명 / 훈련 이벤트 221,318건 (운동 12종, 레벨 1~7) — <b>0421 데이터</b></td></tr>
    <tr><td>ML 모델</td><td>LightGBM 4종 — score(점수예측) · dropout(이탈) · gap(세션간격) · termination(재처방)</td></tr>
    <tr><td>검증</td><td>환자 단위 80/20 split + 5-fold CV + Holdout KS test</td></tr>
    <tr><td>봇 적합도</td><td>재처방률은 실제와 근접(real 15.7% vs sim 14.3%)이나, <b>레벨/점수 분포는 아직 KS 불일치</b> → 절대값 말고 상대 비교 권장 ⚠️</td></tr>
    <tr><td>시뮬 구조</td><td>200명 가상 환자 × 5 seed × 7 알고리즘 × 5 운동조합 = 175 시뮬</td></tr>
  </table>

  <h3 class='exec-h3'>② 현행 룰 (DTx 제품가 지금 운영 중)</h3>
  <div class='exec-rule'>
    <code>3회 연속 정답률 평균</code><br>
    &nbsp;&nbsp;• &lt; <b>50%</b> &nbsp;→ 난이도 <span style='color:#d93025;font-weight:bold'>하락 (-1)</span><br>
    &nbsp;&nbsp;• <b>50% ~ 80%</b> &nbsp;→ <span style='color:#9aa0a6;font-weight:bold'>유지</span><br>
    &nbsp;&nbsp;• ≥ <b>80%</b> &nbsp;→ 난이도 <span style='color:#1e8e3e;font-weight:bold'>상승 (+1)</span>
  </div>
  <div class='exec-issue'>
    <b>문제점 (실제 데이터로 측정)</b>: 난이도 유지율 평균 <b>약 86%</b> — 환자가 한 레벨에 너무 오래 머무름.
    유지 zone 30%p가 너무 넓고 3회 평균이 변화 신호를 약화시킴 → 환자 정체 → 동기부여↓ → 재처방↓.
  </div>

  <h3 class='exec-h3'>③ A/B 후보 TOP 3 — 재처방률 ↑ AND 이탈률 ↓ 둘 다 baseline 대비 개선</h3>
  <p style='font-size:13px;color:#5f6368;margin:6px 0 10px'>
    재처방률은 <b>높을수록 좋고</b>, 이탈률은 <b>낮을수록 좋음</b>. 두 KPI 모두 개선된 정책만 자동 필터.
    균형 점수 = (재처방률 uplift) − (이탈률 uplift) — 둘 다 percentage point.
  </p>
  {policy_card(top3.iloc[0], '1순위 추천', '#1e8e3e') if len(top3) >= 1 else ''}
  {policy_card(top3.iloc[1], '2순위', '#4f7cf7') if len(top3) >= 2 else ''}
  {policy_card(top3.iloc[2], '3순위', '#4f7cf7') if len(top3) >= 3 else ''}

  <h3 class='exec-h3'>④ {_n_adj}개 난이도 조정 알고리즘 ranking (균형 점수 기준)</h3>
  <p style='font-size:13px;color:#5f6368;margin:6px 0 10px'>
    composer 5개에 대한 평균. 재처방률만 보면 <code>{_rep_top}</code>이 1위지만,
    이탈률 변화({_rep_top_drop_d:+.1f}%p)를 함께 봐야 진짜 효과를 알 수 있음.
    <b>균형 점수 = (재처방률 uplift) − (이탈률 uplift)</b>.
  </p>
  <table class='exec-rank-table'>
    <tr><th>순위</th><th>알고리즘</th><th>재처방률 (↑좋음)</th><th>이탈률 (↓좋음)</th><th>균형점수</th></tr>
    {''.join(rank_rows)}
  </table>

  <h3 class='exec-h3'>⑤ 핵심 발견</h3>
  <ul class='exec-findings'>
    <li><b>두 KPI 동시 평가가 필수</b> — 재처방률만 보면 <code>{_rep_top}</code>이 위로 보여도
        이탈률을 함께 봐야 함. 단일 지표 의사결정은 위험.</li>
    <li><b>선행 분석 제안들이 상위권</b> — 균형 점수 기준 선행 제안 계열({_proposed_str})이
        현행 룰보다 위. PPTX의 방향(유지 zone을 좁히고 하락을 후하게)이 데이터에서도 유효.</li>
    <li><b>현행 룰은 {_n_adj}개 중 {current_rank}위</b> — 보수적이라 환자 정체. 유지율을
        74%→61~67%로 낮추는 제안들이 정체를 풀어줌.</li>
    <li><b>이탈을 가장 적게 만드는 규칙</b>은 <code>{_best_drop_adj}</code>(이탈률 {_best_drop_val:.0f}%)
        — 단, 너무 안 밀면 재처방도 안 늘어 균형을 함께 봐야 함.</li>
  </ul>

  <h3 class='exec-h3'>⑥ 한계 (정직하게)</h3>
  <ul class='exec-findings'>
    <li><b>시뮬 분포가 새 데이터에서 아직 덜 맞음</b> — 특히 시뮬이 레벨을 실제보다 높게 올림
        (sim max_level 5.6 vs real 3.45). <b>절대 수치 말고 알고리즘 간 상대 비교만</b> 결론으로.</li>
    <li>재처방 양성은 30→89명으로 늘어 모델 신뢰도↑. 그래도 절대 재처방률은 archetype calibration된 값이라 상대 ranking에 무게.</li>
    <li>봇 예측은 <i>방향성 prior</i>로 유용하나 정량 보장 X — <b>실제 환자 작은 그룹(30~50명) 단기 A/B 필수</b>.</li>
  </ul>

  <div class='exec-cta'>
    이하 9개 섹션은 위 결론의 <b>근거 데이터, 모델 학습 방법, 검증 결과</b>의 상세 버전입니다.
  </div>
</div>
"""


def section_data(patients, events, rx) -> str:
    return f"""
<h2 id="data">1. 학습에 사용한 데이터</h2>
<p>DTx 제품 운영 DB에서 2종 추출. 모든 ML 모델은 이 데이터 안에서 환자 단위 80/20 split으로 학습.</p>

<div class="kpi-strip">
  <div class="kpi-card"><div class="v">{len(rx):,}</div><div class="l">처방 건수</div></div>
  <div class="kpi-card green"><div class="v">{len(patients):,}</div><div class="l">고유 환자</div></div>
  <div class="kpi-card purple"><div class="v">{len(events):,}</div><div class="l">훈련 이벤트</div></div>
  <div class="kpi-card warn"><div class="v">{rx['hospital'].nunique()}</div><div class="l">처방 기관</div></div>
  <div class="kpi-card teal"><div class="v">12</div><div class="l">운동 종류</div></div>
  <div class="kpi-card red"><div class="v">{int(patients['did_represcribe'].sum())}</div><div class="l">재처방 양성</div></div>
</div>

<h3>처방 채널 × 월별 추이</h3>
{img(chart_rx_timeline(rx))}

<h3>처방 기관 TOP 12</h3>
{img(chart_top_hospitals(rx))}

<h3>환자당 누적 이벤트 분포 — 환자 활동량의 큰 편차</h3>
{img(chart_patient_event_dist(patients))}

<h3>운동 × 레벨별 평균 점수 — 학습 곡선 패턴</h3>
{img(chart_exercise_score_matrix(events))}
<div class="small">관찰: 운동별 점수가 0.5~0.95 사이 폭넓게 분포. 레벨이 높아져도 점수가 크게 떨어지지 않는 운동 (clapping)이 있고, 큰 차이 보이는 운동 (memorize_with_stories)이 있음.</div>

<h3>Archetype별 대표 환자 trajectory</h3>
{img(chart_sample_trajectories(events, patients))}
"""


def section_archetypes(patients) -> str:
    elig = patients[patients["archetype"] != "Z_never_played"]
    _rates = elig.groupby("archetype")["did_represcribe"].mean()
    b_rate = _rates.get("B_stalled_settled", 0) * 100
    c_rate = _rates.get("C_stalled_bored", 0) * 100
    return f"""
<h2 id="arch">2. 행동유형 (Archetype) 라벨링</h2>
<p>선행 EDA 분석 + 데이터 패턴 기반 룰 라벨링. 클러스터링이 아니라 해석 가능한 if/else 룰 — 시뮬레이터의 termination calibration anchor로 사용.</p>

<table class="def-table">
<tr><th>유형</th><th>분류 규칙 (위에서부터 먼저 걸리는 것)</th><th>의미</th></tr>
<tr><td><b>A 습관 미형성</b></td><td>general 최고레벨 = 1 그리고 general 횟수 &lt; 21</td><td>난이도 문제 아님 — 앱 쓰는 습관이 안 잡힘</td></tr>
<tr><td><b>B 안주형</b></td><td>general 최고레벨 ∈ {{2, 3}} 그리고 순응도 ≥ 0.5</td><td>쉬운 레벨에 자리잡고 꾸준히 함 → 재처방 최다</td></tr>
<tr><td><b>C 지루형</b></td><td>general 최고레벨 ∈ {{2, 3}} 그리고 순응도 &lt; 0.5</td><td>같은 레벨대인데 점점 안 옴 → 재처방 최소</td></tr>
<tr><td><b>D 중간 벽</b></td><td>general 횟수 ≥ 21 그리고 마지막 레벨 ∈ {{4, 5}}</td><td>레벨 4~5 심리적 정체기에 부딪힘</td></tr>
<tr><td><b>E 졸업</b></td><td>general 최고레벨 = 7 (최대)</td><td>끝까지 올라간 성실 그룹</td></tr>
<tr><td><b>F 성장 진행</b></td><td>위 어디에도 안 들어가는 나머지</td><td>정상적으로 진척 중</td></tr>
<tr><td><b>Z 미실행</b></td><td>general 훈련 기록 0건</td><td>분석/시뮬에서 제외</td></tr>
</table>
<p class="muted">순응도 = 90일 중 훈련한 날의 비율(0~1). '최고레벨대(2~3)'가 같아도 순응도 0.5를 기준으로 B(안주)와 C(지루)가 갈립니다.</p>

{img(chart_archetype_donut(patients))}
{img(chart_archetype_stats(patients))}

<h3>Archetype 카드 — 각 유형의 통계 한눈에</h3>
{archetype_cards_html(patients)}

<h3>Archetype × 핵심 지표 분포</h3>
{img(chart_archetype_box(patients, 'adherence', 'archetype별 adherence (일별 훈련일 비율)'))}
{img(chart_archetype_box(patients, 'general_max_level', 'archetype별 도달 max_level'))}

<div class="tip">
<b>핵심 발견</b>: B 안주형의 재처방률 <b>{b_rate:.1f}%</b>가 모든 유형 중 최고이고
C 지루형은 <b>{c_rate:.1f}%</b>로 최저 — 같은 레벨대라도 '꾸준함'이 운명을 가름.
선행 분석(2차)의 <i>"maintain ↑ → 재처방 ↑(안주) / 이탈(지루)"</i> 패턴과 일치하며,
archetype 라벨링이 실제 행동 시그널을 잡고 있다는 증거.
</div>
"""


def section_models(metrics, hp) -> str:
    return f"""
<h2 id="ml">3. ML 모델 4종 — 학습 결과와 holdout 평가</h2>
<p>모두 LightGBM (Gradient Boosting Decision Tree). 환자 단위 80/20 split (<code>GroupShuffleSplit</code>).
같은 환자가 train/test에 섞이지 않게 함 — data leakage 방지.</p>

<div class="model-grid">
  <div class="model-card yellow">
    <div class="model-head"><span class="model-name">① Score 모델</span><span class="verdict-tag warn">🟡 보통</span></div>
    <div class="model-row"><b>목적</b>: (환자, 운동, 레벨) → 점수 예측</div>
    <div class="model-row"><b>타겟</b>: continuous [0, 1] · <b>학습</b>: {metrics['score']['n_train']:,} · <b>holdout</b>: {metrics['score']['n_test']:,}</div>
    <div class="model-metric">
      MAE <b>{metrics['score']['MAE']:.3f}</b> &nbsp;(baseline {metrics['score']['baseline_MAE_mean']:.3f}, <b>{(1-metrics['score']['MAE']/metrics['score']['baseline_MAE_mean'])*100:.0f}%↓</b>)
      &nbsp;·&nbsp; R² <b>{metrics['score']['R2']:.2f}</b>
    </div>
    <div class="model-note">점수 자체가 noisy. R² {metrics['score']['R2']:.2f}가 한계지만 archetype·레벨·운동 패턴은 학습.</div>
  </div>

  <div class="model-card yellow">
    <div class="model-head"><span class="model-name">② Dropout 모델</span><span class="verdict-tag warn">🟡 CV 불안정</span></div>
    <div class="model-row"><b>목적</b>: 이 세션이 환자의 마지막일 확률</div>
    <div class="model-row"><b>타겟</b>: binary · <b>학습</b>: {metrics['dropout']['n_train']:,} · <b>holdout</b>: {metrics['dropout']['n_test']:,} · 양성 {metrics['dropout']['pos_rate']*100:.1f}%</div>
    <div class="model-metric">
      ROC-AUC <b>{metrics['dropout']['ROC_AUC']:.3f}</b> &nbsp;·&nbsp;
      PR-AUC {metrics['dropout']['PR_AUC']:.3f} &nbsp;·&nbsp;
      Brier {metrics['dropout']['Brier']:.4f}
    </div>
    <div class="model-note">⚠️ 0421 데이터에선 '이탈 세션'이 양성 {metrics['dropout']['pos_rate']*100:.1f}%로 매우 희귀해짐 → holdout ROC {metrics['dropout']['ROC_AUC']:.2f}는 좋아 보여도 5-fold CV 평균은 ~0.54로 <b>불안정</b>(아래 검증 표 참고). 시뮬엔 쓰되 절대 해석은 주의.</div>
  </div>

  <div class="model-card green">
    <div class="model-head"><span class="model-name">③ Gap 모델</span><span class="verdict-tag ok">🟢 잘 됨</span></div>
    <div class="model-row"><b>목적</b>: 다음 세션까지 며칠</div>
    <div class="model-row"><b>타겟</b>: log(days_to_next) · <b>학습</b>: {metrics['gap']['n_train']:,} · <b>holdout</b>: {metrics['gap']['n_test']:,}</div>
    <div class="model-metric">
      MAE <b>{metrics['gap']['MAE_days']:.2f}일</b> &nbsp;(baseline {metrics['gap']['baseline_MAE_days']:.2f}일, <b>{(1-metrics['gap']['MAE_days']/metrics['gap']['baseline_MAE_days'])*100:.0f}%↓</b>)
    </div>
    <div class="model-note">대부분 같은 날 재방문이라 간격 대부분 0 — baseline과 비슷한 수준, 시뮬에서 약한 고리.</div>
  </div>

  <div class="model-card yellow">
    <div class="model-head"><span class="model-name">④ Termination 모델</span><span class="verdict-tag warn">🟡 개선됨 (0421)</span></div>
    <div class="model-row"><b>목적</b>: 처방 끝나고 재처방 받을 확률</div>
    <div class="model-row"><b>타겟</b>: binary · <b>학습</b>: {metrics['termination']['n_train']:,} · <b>holdout</b>: {metrics['termination']['n_test']:,} · 학습 양성 {metrics['termination']['pos_train']}명 · holdout 양성 <b>{metrics['termination']['pos_test']}명</b></div>
    <div class="model-metric">
      ROC-AUC <b>{metrics['termination']['ROC_AUC']:.3f}</b> &nbsp;·&nbsp;
      PR-AUC {metrics['termination']['PR_AUC']:.3f} &nbsp;·&nbsp;
      Brier {metrics['termination']['Brier']:.4f}
    </div>
    <div class="model-note">0421 데이터로 재처방 양성이 30→{int(metrics['termination']['pos_train'])+int(metrics['termination']['pos_test'])}명으로 늘어 holdout 양성 {metrics['termination']['pos_test']}명 — 이전(4명)보다 신뢰도 크게↑. 그래도 시뮬은 안전하게 archetype calibration으로 anchor.</div>
  </div>
</div>

<h3>모델별 holdout 평가 — 예측 vs 실제</h3>

<h3>Score 모델 (LightGBM Regressor)</h3>
{img(chart_score_pred_actual(hp['score']))}
{img(chart_feature_importance(hp['score'], "Score 모델"))}

<h3>Dropout 모델 (LightGBM Classifier)</h3>
{img(chart_roc_calibration(hp['dropout'], "Dropout 모델"))}
{img(chart_feature_importance(hp['dropout'], "Dropout 모델"))}

<h3>Gap 모델 (LightGBM Regressor)</h3>
{img(chart_gap_pred_actual(hp['gap']))}
{img(chart_feature_importance(hp['gap'], "Gap 모델"))}

<h3>Termination 모델 (LightGBM Classifier, calibration 검증)</h3>
{img(chart_roc_calibration(hp['termination'], "Termination 모델"))}
{img(chart_feature_importance(hp['termination'], "Termination 모델"))}
<div class="warn">
<b>참고</b>: termination ROC-AUC <b>{metrics['termination']['ROC_AUC']:.3f}</b>은 holdout 양성 <b>{metrics['termination']['pos_test']}명</b> 기준
(0421 데이터로 이전 4명 → {metrics['termination']['pos_test']}명으로 확대되어 통계적 신뢰도가 올라감).
그래도 시뮬레이터는 보수적으로 raw 예측을 그대로 쓰지 않고 archetype별 실제 base rate로 rescale함.
</div>
"""


def section_simulator() -> str:
    return f"""
<h2 id="sim">4. 시뮬레이터 — 어떻게 분석하는가</h2>
<p>학습된 4개 모델을 결합한 cohort-step 시뮬레이터. 알고리즘 정책을 plug-in으로 받아
가상 환자 풀에 적용 → 출력 지표 비교.</p>

<div class="pipeline">{PIPELINE_ASCII}</div>

<h3>정책 = composer + adjuster (두 축 직교 비교)</h3>
<table>
  <tr><th>축</th><th>역할</th><th>현재 옵션</th></tr>
  <tr><td><b>composer</b></td><td>세부 운동 9개 중 4개 선택 (어떤 운동을 줄지)</td>
      <td><code>random_balanced</code>, <code>fixed_rotation</code>,
          <code>weakness_focused</code>, <code>archetype_aware</code>, <code>score_balanced</code></td></tr>
  <tr><td><b>adjuster</b></td><td>각 운동의 다음 레벨 결정 (어떻게 난이도 조정)</td>
      <td>11종: <code>current_50_80</code>(현행), <code>drop_at_70</code>, <code>level_diff</code>,
          <code>strict_up_n_over_np1</code>, <code>expand_10_diff</code>,
          <code>hybrid_drop70_strict_up</code>, <code>narrow_maintain_65_75</code>,
          <code>adaptive_personal_baseline</code>, <code>aggressive_75_85</code>,
          <code>cliff_jump</code>, <code>slow_climb</code></td></tr>
</table>
<p>새 정책 추가는 <code>bot/composers.py</code> 또는 <code>bot/algorithms.py</code>에 클래스 하나 추가 + dict 등록만 하면 끝.</p>

<h3>출력 지표</h3>
<table>
  <tr><th>지표</th><th>의미</th></tr>
  <tr><td>represcribe_rate</td><td>90일 끝 재처방률 (메인 KPI)</td></tr>
  <tr><td>drop_rate</td><td>90일 안 이탈률</td></tr>
  <tr><td>mean_max_level</td><td>도달한 최고 레벨</td></tr>
  <tr><td>mean_maintain</td><td>같은 레벨 유지 비율 (낮을수록 동적)</td></tr>
  <tr><td>final_archetype</td><td>봇 행동 결과로 재분류된 archetype (dynamic re-labeling)</td></tr>
</table>

<h3>Calibration 두 단계</h3>
<ul>
  <li><b>Session cap</b>: 각 가상 환자의 세션 수 상한을 archetype별 실제 분포에서 sampling
      → 환자 활동량 다양성 유지</li>
  <li><b>Max-level cap</b>: 각 가상 환자의 도달 가능 max_level도 archetype 분포에서 sampling
      → 개인 인지 능력 차이 모델링</li>
  <li><b>Termination calibration</b>: ML 모델 raw 예측 대신 archetype별 실제 재처방률에 anchor
      → 알고리즘이 환자를 다른 archetype으로 push하면 자연스럽게 효과 반영</li>
</ul>
"""


def section_audit() -> str:
    p = OUT / "audit_results.json"
    if not p.exists():
        return ""
    with open(p, encoding="utf-8") as f:
        a = json.load(f)
    tp = MODELS / "tuning_summary.json"
    tune = {}
    if tp.exists():
        with open(tp, encoding="utf-8") as f:
            tune = json.load(f)

    # Archetype ablation table
    al = a["archetype_ablation"]
    abl_rows = []
    for k, v in al.items():
        if k == "score":
            abl_rows.append(f"<tr><td>{k}</td><td>MAE {v['with_archetype_MAE']:.4f}</td>"
                            f"<td>MAE {v['without_archetype_MAE']:.4f}</td>"
                            f"<td>Δ {v['delta']:+.4f}</td></tr>")
        else:
            k1 = "with_archetype_ROC"; k2 = "without_archetype_ROC"
            abl_rows.append(f"<tr><td>{k}</td><td>ROC {v[k1]:.4f}</td>"
                            f"<td>ROC {v[k2]:.4f}</td>"
                            f"<td>Δ {v.get('delta', 0):+.4f}</td></tr>")

    # K-fold table
    kf = a["kfold"]
    kf_rows = []
    for k, v in kf.items():
        if v.get("mean") is None:
            continue
        folds = " · ".join(f"{x:.3f}" for x in v.get("folds", []))
        kf_rows.append(f"<tr><td>{k}</td><td>{folds}</td>"
                       f"<td><b>{v['mean']:.4f} ± {v['std']:.4f}</b></td></tr>")

    # Overfit table
    of = a["overfit"]
    of_rows = []
    for k, v in of.items():
        train_key = "train_MAE" if "MAE" in str(list(v.keys())) else "train_ROC"
        test_key = "test_MAE" if "MAE" in str(list(v.keys())) else "test_ROC"
        of_rows.append(f"<tr><td>{k}</td><td>{v[train_key]:.4f}</td>"
                       f"<td>{v[test_key]:.4f}</td>"
                       f"<td>{v['gap']:+.4f}</td></tr>")

    # Per-archetype score
    pas = a["per_archetype_score"]
    pas_rows = []
    for arch in ARCH_ORDER:
        if arch not in pas:
            continue
        v = pas[arch]
        r2 = f"{v['R2']:.3f}" if v.get("R2") is not None else "n/a"
        c = ARCH_COLORS[arch]
        pas_rows.append(f"<tr><td style='color:{c};font-weight:600'>{ARCH_LABEL[arch]}</td>"
                        f"<td>{v['n']}</td><td>{v['MAE']:.4f}</td><td>{r2}</td></tr>")

    # dynamic best/worst archetype by score MAE (for the conclusion note)
    _pas_valid = {k: v for k, v in pas.items() if k in ARCH_ORDER and v.get("MAE") is not None}
    _worst = max(_pas_valid, key=lambda k: _pas_valid[k]["MAE"]) if _pas_valid else None
    _best = min(_pas_valid, key=lambda k: _pas_valid[k]["MAE"]) if _pas_valid else None
    _worst_note = (f"{ARCH_LABEL[_worst]}(n={_pas_valid[_worst]['n']})에서 가장 부정확 "
                   f"(MAE {_pas_valid[_worst]['MAE']:.3f})") if _worst else ""
    _best_note = (f"{ARCH_LABEL[_best]}에서 가장 정확 (MAE {_pas_valid[_best]['MAE']:.3f}) "
                  f"— 활동량 많고 행동 안정적") if _best else ""

    # Tuning summary
    tune_html = ""
    if tune:
        tune_rows = []
        for k, v in tune.items():
            params = v.get("best_params", {})
            param_str = "<br>".join(f"<code>{pk}={pv:.4g}</code>" if isinstance(pv, float)
                                     else f"<code>{pk}={pv}</code>"
                                     for pk, pv in params.items())
            tune_rows.append(
                f"<tr><td>{k}</td><td>{v['best_cv']:.4f}</td><td>{param_str}</td></tr>")
        tune_html = f"""
<h3>Optuna 튜닝 결과 (best hyperparameters)</h3>
<table><tr><th>모델</th><th>best CV</th><th>hyperparameters</th></tr>
{''.join(tune_rows)}
</table>"""

    return f"""
<h2 id="audit">5. 모델 검증 (Audit)</h2>
<p>"학습이 제대로 됐는가"를 4개 관점에서 점검. 단일 holdout은 운에 좌우될 수 있어서
5-fold CV로 진짜 성능을 측정하고, train/test gap으로 overfitting 확인.</p>

<h3>5-1. Archetype feature leakage 검증</h3>
<p>archetype은 max_level·n_general에서 룰 라벨됐기 때문에 그걸 모델 input으로 쓰는 게
leakage일 가능성이 있음. → archetype feature를 빼고 학습해서 성능 갭 측정.</p>
<table><tr><th>모델</th><th>archetype 포함</th><th>archetype 제외</th><th>delta</th></tr>
{''.join(abl_rows)}
</table>
<div class="tip">
<b>결론</b>: leakage 우려는 무근. archetype을 빼도 성능 거의 동일 (Score MAE Δ{al['score']['delta']:+.4f}, 나머지 ~0)
— 다른 raw features(n_general, max_level, adherence 등)가 이미 같은 신호 담음.
</div>

<h3>5-2. 5-fold CV 안정성</h3>
<p>단일 holdout이 lucky/unlucky fold일 수 있어 5-fold로 모델 성능 분산 측정.</p>
<table><tr><th>모델</th><th>fold별 점수</th><th>mean ± std</th></tr>
{''.join(kf_rows)}
</table>
<div class="warn">
<b>발견 (0421)</b>: Score/Gap은 매우 안정(분산 &lt; 2%). <b>Dropout은 5-fold 평균 ROC ~0.54로 불안정</b>
— 이탈 세션이 0.4%로 너무 희귀해 fold마다 들쭉날쭉(0.44~0.65). 단일 holdout 0.96은 운 좋은 split.
Termination은 fold당 양성 15~20명으로 늘어 평균 0.98 ± 0.02로 <b>이전보다 신뢰도↑</b>.
</div>

<h3>5-3. Train vs Test gap (overfitting)</h3>
<p>train/test 차이가 작으면 일반화 잘 됨. 차이가 크면 모델이 외운 것.</p>
<table><tr><th>모델</th><th>train</th><th>test</th><th>gap</th></tr>
{''.join(of_rows)}
</table>
<div class="warn">
<b>주의</b>: Termination은 train ROC 1.000(양성 65명 완벽 memorize), test 0.99.
양성이 30→89명으로 늘어 일반화 신뢰도는 올랐지만 train=1.0 memorize 경향은 남음.
→ 시뮬에서 raw 예측 그대로 안 쓰고 archetype별 base rate로 calibration.
</div>

<h3>5-4. Per-archetype score 정확도</h3>
<p>Score 모델이 어떤 archetype에서 더 잘/못 잡는지 확인.</p>
<table><tr><th>Archetype</th><th>n (test)</th><th>MAE</th><th>R²</th></tr>
{''.join(pas_rows)}
</table>
<div class="note">
<b>관찰</b>: {_worst_note}. {_best_note}.
</div>

{tune_html}
"""


def section_validation() -> str:
    p = OUT / "validation_ks.json"
    if not p.exists():
        return ""
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    rows = ["<table><tr><th>지표</th><th>실제 환자</th><th>봇 시뮬</th>"
            "<th>KS</th><th>p-value</th><th>판정</th></tr>"]
    for k, d in data.items():
        if k == "represcribe_rate":
            rows.append(f"<tr><td>재처방률</td><td>{d['real']*100:.2f}%</td>"
                        f"<td>{d['sim']*100:.2f}%</td><td>-</td><td>-</td><td>비율 비교</td></tr>")
            continue
        cls = "pass" if d["pass"] else "fail"
        sym = "✓ 통과" if d["pass"] else "✗ fail"
        rows.append(f"<tr><td>{k}</td><td>{d['real_mean']:.2f}</td>"
                    f"<td>{d['sim_mean']:.2f}</td><td>{d['ks']:.3f}</td>"
                    f"<td>{d['p']:.4f}</td><td class='{cls}'>{sym}</td></tr>")
    rows.append("</table>")
    return f"""
<h2 id="val">6. Holdout 검증 — 봇은 실제 환자처럼 행동하는가?</h2>
<p>20% holdout 환자(153명)의 trajectory와 봇 시뮬(현행 룰, 약 600~800명) 분포를 KS test로 비교.</p>
{''.join(rows)}
{img(chart_holdout_distribution())}
<div class="warn">
<b>0421 데이터에선 분포가 아직 안 맞습니다.</b> 특히 <b>시뮬이 레벨을 실제보다 높게 올립니다</b>
(sim max_level ≈ 5.6 vs real ≈ 3.45). 재처방률만 실제와 근접(real 15.7% vs sim 14.3%).<br>
→ <b>절대 수치(평균 레벨 등)는 인용하지 말고, 알고리즘 간 상대 비교만</b> 결론으로 쓰세요.
이 격차를 줄이는 calibration 보정은 후속 과제입니다.
</div>
"""


def section_results(agg, raw) -> str:
    best = agg.sort_values("represcribe_rate_mean", ascending=False).iloc[0]
    base_row = agg[(agg["composer"] == "random_balanced")
                   & (agg["adjuster"] == "current_50_80")]
    base = base_row.iloc[0] if len(base_row) else None
    gap = best["represcribe_rate_mean"] - (base["represcribe_rate_mean"] if base is not None else 0)
    base_ci = float(base["represcribe_rate_ci95"]) if base is not None else 0
    significant = gap > (best["represcribe_rate_ci95"] + base_ci)
    sig_msg = "✓ <b>통계적 유의</b>" if significant else "⚠️ <b>noise 범위</b> (CI 겹침)"

    table_rows = []
    for i, r in agg.sort_values("represcribe_rate_mean", ascending=False).reset_index(drop=True).iterrows():
        cls = ' class="best"' if i == 0 else ""
        table_rows.append(
            f"<tr{cls}><td>{r['adjuster']}</td><td>{r['composer']}</td>"
            f"<td>{r['represcribe_rate_mean']:.1%} ± {r['represcribe_rate_ci95']:.1%}</td>"
            f"<td>{r['mean_max_level_mean']:.2f}</td>"
            f"<td>{r['mean_last_level_mean']:.2f}</td>"
            f"<td>{r['mean_maintain_mean']:.1%}</td>"
            f"<td>{r['drop_rate_mean']:.1%}</td>"
            f"<td>{r['mean_general_mean']:.1f}</td></tr>")

    return f"""
<h2 id="results">7. 정책 비교 결과 (Multi-seed K=5)</h2>

<div class="kpi-strip">
  <div class="kpi-card green"><div class="v">{best['represcribe_rate_mean']*100:.1f}%</div>
       <div class="l">최적 정책 재처방률<br>± {best['represcribe_rate_ci95']*100:.1f}%</div></div>
  <div class="kpi-card"><div class="v">{base['represcribe_rate_mean']*100:.1f}%</div>
       <div class="l">현행 baseline<br>(current × random_balanced)</div></div>
  <div class="kpi-card warn"><div class="v">{gap*100:+.1f}%p</div>
       <div class="l">절대 차이<br>{sig_msg}</div></div>
  <div class="kpi-card purple"><div class="v">{len(agg)}</div>
       <div class="l">비교 정책 수<br>(composer × adjuster)</div></div>
  <div class="kpi-card teal"><div class="v">{raw['seed'].nunique()}</div>
       <div class="l">시드 수<br>(평균 + 95% CI)</div></div>
  <div class="kpi-card red"><div class="v">{len(raw)}</div>
       <div class="l">총 시뮬 실행<br>(정책 × 시드)</div></div>
</div>

<h3>전체 정책 ranking (재처방률, 상위 ↓)</h3>
{img(chart_policy_ranking(agg))}

<h3>전체 정책 표</h3>
<table>
<tr><th>adjuster</th><th>composer</th><th>재처방률 ± CI</th>
    <th>max_lvl</th><th>last_lvl</th><th>유지율</th><th>이탈률</th><th>일반 events</th></tr>
{''.join(table_rows)}
</table>

<h3>Marginal effects — composer vs adjuster 분리</h3>
{img(chart_marginal_effects(raw))}
<div class="note">
<b>관찰</b>: composer 간 차이는 ±2-4%p로 좁고, adjuster 간 차이도 비슷한 수준.
→ 두 축 모두 단독으로는 baseline을 크게 능가하기 어려움.
</div>

<h3>난이도 유지율 vs 재처방률 — trade-off 산점도</h3>
{img(chart_maintain_vs_represcribe(raw))}

<h3>정책 매트릭스 — 재처방률</h3>
{img(chart_policy_heatmap(agg, 'represcribe_rate', '재처방률 (composer × adjuster)', '{:.1%}', 'YlGn'))}

<h3>정책 매트릭스 — 평균 max_level</h3>
{img(chart_policy_heatmap(agg, 'mean_max_level', 'mean_max_level', '{:.2f}', 'Blues'))}

<h3>정책 매트릭스 — 난이도 유지율</h3>
{img(chart_policy_heatmap(agg, 'mean_maintain', '유지율 (낮을수록 동적)', '{:.0%}', 'YlOrRd_r'))}
"""


def section_conclusion() -> str:
    return """
<h2 id="conclusion">8. 결론과 한계</h2>

<div class="tip">
<h3 style="margin-top:0;border:none;padding:0">🟢 봇이 신뢰성 있게 측정하는 것</h3>
<ul>
  <li><b>정책 간 상대 ranking</b> — 어떤 패턴이 일관적으로 위/아래인지</li>
  <li><b>재처방률 수준</b> — 실제 코호트와 근접 (real 15.7% vs sim 14.3%)</li>
  <li><b>유지율의 방향성</b> — 현행 대비 어느 규칙이 정체(유지율)를 더 풀어주는지</li>
  <li>알고리즘이 환자를 어떤 archetype으로 push하는지 (dynamic reclassification)</li>
  <li>운동 조합·난이도 룰 변경의 <b>방향성 효과</b> (어느 방향으로 환자 행동 바뀌는지)</li>
</ul>
</div>

<div class="warn">
<h3 style="margin-top:0;border:none;padding:0">🔴 봇이 신뢰성 떨어지는 것 (0421 데이터 기준)</h3>
<ul>
  <li><b>레벨/점수 분포 절대값</b> — 새 데이터에선 시뮬이 레벨을 실제보다 높게 올림
      (sim max_level 5.6 vs real 3.45, KS 불일치). 절대 수치 인용 금지.</li>
  <li><b>재처방률 절대값</b> — archetype별 base rate로 calibration된 결과
      (양성 30→89명으로 늘어 모델은 개선됐으나 절대값은 여전히 anchor된 값).</li>
  <li><b>이탈(dropout) 모델</b> — 이탈 세션이 0.4%로 희귀해 CV에서 불안정(ROC ~0.54).</li>
  <li><b>정책 간 재처방률 차이의 통계적 유의성</b> — CI 겹침, 1-2%p 차이는 noise.</li>
</ul>
</div>

<div class="note">
<h3 style="margin-top:0;border:none;padding:0">💡 실용적 권장</h3>
의사결정 prior로는 OK, 결정적 정량 근거로는 부족.<br>
"몇 % 재처방률 상승"이 아니라 "<b>어떤 방향으로 환자 행동이 바뀐다</b>"는 신호로 사용 권장.<br>
진짜 결정적 근거가 필요하면 <b>실제 환자에서 작은 A/B 단기 시험</b>이 필요.
</div>

<h3>다음 단계 후보</h3>
<ul>
  <li><b>재처방 양성 sample 확대</b>: 현재 30명이 statistical 한계. 데이터 누적 후 재학습</li>
  <li><b>새 알고리즘 가설 테스트</b>: DTx 제품 팀 아이디어를 <code>algorithms.py</code>에 추가 → 즉시 비교</li>
  <li><b>archetype 정의 refinement</b>: 현재 6개 + Z. 환자 코호트가 커지면 세분화 검토</li>
  <li><b>실제 A/B 디자인 가이드</b>: 봇이 가장 차이 보이는 정책 페어 → 작은 그룹 시험</li>
</ul>
"""


def build_html(patients, events, rx, metrics, hp, agg, raw) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_p = len(patients)
    n_e = len(events)
    rx_rate = patients["did_represcribe"].mean() * 100
    eligible = (patients["archetype"] != "Z_never_played").sum()

    return f"""<!doctype html>
<html lang="ko"><head><meta charset='utf-8'>
<title>DTx Bot — Deep Dive Report</title>
<style>{CSS}</style></head><body>

<h1>DTx Patient Twin Bot
<span class="sub">실제 환자 데이터로 학습된 가상 환자 시뮬레이터 — 정책 변경 효과 사전 예측</span></h1>

<div class="kpi-strip">
  <div class="kpi-card"><div class="v">{n_p}</div><div class="l">학습 환자</div></div>
  <div class="kpi-card green"><div class="v">{n_e:,}</div><div class="l">학습 이벤트</div></div>
  <div class="kpi-card purple"><div class="v">4</div><div class="l">LightGBM 모델</div></div>
  <div class="kpi-card teal"><div class="v">{len(agg)}</div><div class="l">비교 정책</div></div>
  <div class="kpi-card warn"><div class="v">{raw['seed'].nunique()}×</div><div class="l">multi-seed</div></div>
  <div class="kpi-card red"><div class="v">{rx_rate:.1f}%</div><div class="l">실제 재처방률</div></div>
</div>

{section_executive_summary(agg)}

<div class="toc">
<b>목차 (상세 분석)</b>
<ol>
  <li><a href="#data">학습에 사용한 데이터</a> — 5개 chart</li>
  <li><a href="#arch">행동유형(Archetype) 라벨링</a> — 6+1개 유형 deep dive</li>
  <li><a href="#ml">ML 모델 4종</a> — holdout 평가 + feature importance</li>
  <li><a href="#sim">시뮬레이터 구조</a> — composer × adjuster 정책</li>
  <li><a href="#audit">모델 검증 (Audit)</a> — leakage / 5-fold CV / overfitting / Optuna 튜닝</li>
  <li><a href="#val">Holdout 검증 (KS test)</a></li>
  <li><a href="#results">정책 비교 결과 (Multi-seed)</a></li>
  <li><a href="#conclusion">결론과 한계</a></li>
</ol>
</div>

{section_data(patients, events, rx)}
{section_archetypes(patients)}
{section_models(metrics, hp)}
{section_simulator()}
{section_audit()}
{section_validation()}
{section_results(agg, raw)}
{section_conclusion()}

<div class="footer">
DTx Patient Twin Bot · LightGBM × cohort-step simulator · 생성 {ts}<br>
재실행: <code>python -m bot.make_deep_report</code>
</div>

</body></html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="deep_report.html")
    args = p.parse_args()

    print("Loading data + models ...")
    patients = pd.read_parquet(OUT / "patients.parquet")
    events = pd.read_parquet(OUT / "events.parquet")
    from .build_dataset import load_prescriptions
    rx = load_prescriptions()   # 0421 data + robust datetime parsing
    with open(MODELS / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    agg = pd.read_csv(OUT / "simulation_compare.csv")
    raw = pd.read_csv(OUT / "simulation_raw_multiseed.csv")

    print("Computing holdout predictions ...")
    hp = holdout_predictions()

    print("Building HTML ...")
    html = build_html(patients, events, rx, metrics, hp, agg, raw)
    out_path = OUT / args.out
    out_path.write_text(html, encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
