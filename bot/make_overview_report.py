"""
종합 HTML 리포트. 데이터 → archetype 라벨링 → ML 모델 4종 → 시뮬레이터 →
정책 비교까지, 무엇으로 학습했고 어떻게 분석하며 결과가 어떤지를 한 페이지에 정리.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

ARCH_COLORS = {
    "A_early_dropout": "#d93025",
    "B_stalled_settled": "#fbbc04",
    "C_stalled_bored": "#fa7b17",
    "D_mid_wall": "#a142f4",
    "E_maxed_out": "#1e8e3e",
    "F_steady_progress": "#4f7cf7",
    "Z_never_played": "#9aa0a6",
}
ARCH_LABEL_KO = {
    "A_early_dropout": "A. 초기 이탈",
    "B_stalled_settled": "B. 안주형",
    "C_stalled_bored": "C. 지루형",
    "D_mid_wall": "D. 중간 벽",
    "E_maxed_out": "E. 졸업자",
    "F_steady_progress": "F. 진행자",
    "Z_never_played": "Z. 미시작",
}


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


# --------------------------------------------------------------------------- #
# Data section
# --------------------------------------------------------------------------- #

def chart_data_overview(patients: pd.DataFrame, events: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    rx = patients["prescribed_from"].value_counts()
    axes[0].bar(rx.index, rx.values, color="#4f7cf7")
    axes[0].set_title("처방 채널별 환자 수")
    axes[0].tick_params(axis="x", rotation=20)

    by_kind = events["exercise_name"].value_counts()
    axes[1].barh(by_kind.index, by_kind.values, color="#1e8e3e")
    axes[1].set_title("운동별 이벤트 수 (총 130,450)")
    axes[1].invert_yaxis()

    lvl = events["level"].dropna().astype(int).value_counts().sort_index()
    axes[2].bar(lvl.index, lvl.values, color="#a142f4")
    axes[2].set_title("레벨별 이벤트 분포")
    axes[2].set_xlabel("level")

    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_archetypes(patients: pd.DataFrame) -> str:
    arch_order = ["A_early_dropout", "B_stalled_settled", "C_stalled_bored",
                  "D_mid_wall", "E_maxed_out", "F_steady_progress", "Z_never_played"]
    counts = patients["archetype"].value_counts().reindex(arch_order, fill_value=0)
    rates = patients.groupby("archetype")["did_represcribe"].mean().reindex(arch_order, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    colors = [ARCH_COLORS[a] for a in arch_order]
    labels = [ARCH_LABEL_KO[a] for a in arch_order]

    bars = axes[0].bar(labels, counts.values, color=colors)
    for b, v in zip(bars, counts.values):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height(),
                     str(v), ha="center", va="bottom", fontsize=10)
    axes[0].set_title("Archetype별 환자 수 (총 576명)")
    axes[0].tick_params(axis="x", rotation=20)

    bars2 = axes[1].bar(labels, rates.values * 100, color=colors)
    for b, v in zip(bars2, rates.values):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height(),
                     f"{v*100:.1f}%", ha="center", va="bottom", fontsize=10)
    axes[1].set_title("Archetype별 실제 재처방률")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylabel("재처방률 %")

    fig.tight_layout()
    return _fig_to_b64(fig)


# --------------------------------------------------------------------------- #
# Models section
# --------------------------------------------------------------------------- #

MODEL_INFO = {
    "score": dict(
        title="Score 모델",
        purpose="(환자, 운동, 레벨) → 0~1 점수 예측",
        target="continuous score [0,1]",
        algo="LightGBM Regressor",
        rows=130450,
        verdict="🟡 보통",
        verdict_reason="점수 자체가 noisy. R²=0.33 한계지만 archetype·레벨 패턴은 학습.",
    ),
    "dropout": dict(
        title="Dropout 모델",
        purpose="이 세션이 환자의 마지막일 확률",
        target="binary (is_terminal)",
        algo="LightGBM Classifier",
        rows=37755,
        verdict="🟢 잘 됨",
        verdict_reason="양성 1.6%인데도 ROC-AUC 0.977로 잘 분리.",
    ),
    "gap": dict(
        title="Gap 모델",
        purpose="다음 세션까지 며칠 걸릴지",
        target="log(days_to_next)",
        algo="LightGBM Regressor",
        rows=37190,
        verdict="🟢 잘 됨",
        verdict_reason="MAE 0.27일로 baseline 절반.",
    ),
    "termination": dict(
        title="Termination 모델",
        purpose="처방 끝나고 재처방 받을 확률",
        target="binary (did_represcribe)",
        algo="LightGBM Classifier",
        rows=565,
        verdict="🔴 숫자만 좋음",
        verdict_reason="홀드아웃 양성 4명. ROC 0.97이지만 statistical power 부족 — 시뮬은 archetype calibration으로 보완.",
    ),
}


def model_cards_html(metrics: dict) -> str:
    cards = []
    for key, info in MODEL_INFO.items():
        m = metrics[key]
        if key == "score":
            metric_line = (
                f"MAE <b>{m['MAE']:.3f}</b> (baseline {m['baseline_MAE_mean']:.3f}, "
                f"<b>{(1 - m['MAE'] / m['baseline_MAE_mean']) * 100:.0f}% ↓</b>) · "
                f"R² {m['R2']:.2f}"
            )
        elif key == "gap":
            metric_line = (
                f"MAE <b>{m['MAE_days']:.2f}일</b> (baseline {m['baseline_MAE_days']:.2f}일, "
                f"<b>{(1 - m['MAE_days'] / m['baseline_MAE_days']) * 100:.0f}% ↓</b>)"
            )
        else:
            metric_line = (
                f"ROC-AUC <b>{m['ROC_AUC']:.3f}</b> · PR-AUC {m['PR_AUC']:.3f} · "
                f"Brier {m['Brier']:.4f}"
            )
        cards.append(
            f"<div class='card'>"
            f"<div class='card-head'><span class='verdict'>{info['verdict']}</span>"
            f"<b>{info['title']}</b></div>"
            f"<div class='card-row'><b>목적</b>: {info['purpose']}</div>"
            f"<div class='card-row'><b>타겟</b>: <code>{info['target']}</code></div>"
            f"<div class='card-row'><b>모델</b>: {info['algo']} · 학습 행 {info['rows']:,}</div>"
            f"<div class='card-row metric'>{metric_line}</div>"
            f"<div class='card-row note'>{info['verdict_reason']}</div>"
            f"</div>"
        )
    return f"<div class='card-grid'>{''.join(cards)}</div>"


def chart_model_metrics(metrics: dict) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.5))
    rows = [
        ("Score MAE\n(lower=better)", metrics["score"]["MAE"],
         metrics["score"]["baseline_MAE_mean"], "#4f7cf7"),
        ("Gap MAE days\n(lower=better)", metrics["gap"]["MAE_days"],
         metrics["gap"]["baseline_MAE_days"], "#fa7b17"),
        ("Dropout ROC-AUC\n(higher=better)", metrics["dropout"]["ROC_AUC"], 0.5, "#1e8e3e"),
        ("Termination ROC-AUC\n(higher=better)", metrics["termination"]["ROC_AUC"], 0.5, "#d93025"),
    ]
    labels = [r[0] for r in rows]
    actual = [r[1] for r in rows]
    baseline = [r[2] for r in rows]
    colors = [r[3] for r in rows]
    x = np.arange(len(rows))
    w = 0.35
    ax.bar(x - w / 2, baseline, w, label="baseline", color="#cccccc")
    ax.bar(x + w / 2, actual, w, label="model", color=colors)
    for i, (b, a) in enumerate(zip(baseline, actual)):
        ax.text(i - w / 2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("4개 모델 성능 vs baseline")
    return _fig_to_b64(fig)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validation_table_html() -> str:
    p = OUT / "validation_ks.json"
    if not p.exists():
        return '<div class="warn">검증 결과 없음 — <code>python -m bot.validate_bot</code> 실행하세요.</div>'
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    rows = ["<table><tr><th>지표</th><th>실제 환자</th><th>봇 시뮬</th>"
            "<th>KS 통계량</th><th>p-value</th><th>판정</th></tr>"]
    for k, d in data.items():
        if k == "represcribe_rate":
            rows.append(
                f"<tr><td>재처방률</td><td>{d['real']*100:.2f}%</td>"
                f"<td>{d['sim']*100:.2f}%</td><td>-</td><td>-</td><td>비율</td></tr>")
            continue
        cls = "pass" if d["pass"] else "fail"
        sym = "✓ 통과" if d["pass"] else "✗ fail"
        rows.append(
            f"<tr><td>{k}</td><td>{d['real_mean']:.2f}</td>"
            f"<td>{d['sim_mean']:.2f}</td><td>{d['ks']:.3f}</td>"
            f"<td>{d['p']:.4f}</td><td class='{cls}'>{sym}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Policy results
# --------------------------------------------------------------------------- #

def policy_ranking_chart(agg: pd.DataFrame) -> str:
    df = agg.sort_values("represcribe_rate_mean", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(df) + 1.5))
    labels = df.apply(lambda r: f"{r['adjuster']:24s} | {r['composer']}", axis=1)
    means = df["represcribe_rate_mean"] * 100
    cis = df["represcribe_rate_ci95"] * 100
    bars = ax.barh(labels, means, xerr=cis, color="#4f7cf7", capsize=3,
                    error_kw={"ecolor": "#444", "lw": 1})
    bars[0].set_color("#1e8e3e")
    for b, m, c in zip(bars, means, cis):
        ax.text(b.get_width() + c + 0.05, b.get_y() + b.get_height() / 2,
                f"{m:.1f}±{c:.1f}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("재처방률 % (± 95% CI, K=5 seeds)")
    ax.set_title("정책별 재처방률 ranking")
    ax.grid(axis="x", alpha=0.3)
    return _fig_to_b64(fig)


def policy_heatmap(agg: pd.DataFrame, metric: str, title: str,
                    fmt: str = "{:.1%}", cmap: str = "YlGn") -> str:
    pivot = agg.pivot(index="composer", columns="adjuster",
                      values=f"{metric}_mean")
    fig, ax = plt.subplots(
        figsize=(1.5 + 1.4 * len(pivot.columns), 0.6 * len(pivot.index) + 2.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, fmt.format(pivot.values[i, j]),
                    ha="center", va="center", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return _fig_to_b64(fig)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

CSS = """
body { font-family: -apple-system, "Malgun Gothic", BlinkMacSystemFont, "Segoe UI",
       sans-serif; max-width: 1180px; margin: 30px auto; padding: 0 24px;
       color: #1f1f1f; line-height: 1.55; }
h1 { border-bottom: 3px solid #1e8e3e; padding-bottom: 10px; }
h2 { margin-top: 40px; border-left: 5px solid #4f7cf7; padding-left: 12px;
     background: linear-gradient(90deg, #f5f8ff 0%, transparent 100%); padding-top: 4px; padding-bottom: 4px; }
h3 { color: #444; margin-top: 24px; }
table { border-collapse: collapse; margin: 14px 0; font-size: 13px; width: 100%; }
th, td { border: 1px solid #ddd; padding: 6px 9px; text-align: right; }
th { background: #f0f3f7; font-weight: 600; }
td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) {
    text-align: left; font-family: 'Cascadia Mono', monospace; }
img { max-width: 100%; height: auto; }
code { background: #f4f4f4; padding: 1px 6px; border-radius: 3px; font-size: 0.92em; }
.kv { display: inline-block; padding: 3px 10px; background: #eef2ff; border-radius: 4px;
      margin-right: 8px; font-family: monospace; font-size: 12.5px; }
.note { background: #fff8e1; border-left: 4px solid #f9ab00; padding: 10px 14px;
        margin: 14px 0; font-size: 13.5px; }
.warn { background: #fce8e6; border-left: 4px solid #d93025; padding: 10px 14px;
        margin: 14px 0; font-size: 13.5px; }
.tip { background: #e6f4ea; border-left: 4px solid #1e8e3e; padding: 10px 14px;
       margin: 14px 0; font-size: 13.5px; }
.card-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;
             margin: 18px 0; }
.card { border: 1px solid #ddd; border-radius: 6px; padding: 14px 16px;
        background: #fafbfc; font-size: 13.5px; }
.card-head { font-size: 16px; margin-bottom: 10px;
             display: flex; gap: 10px; align-items: center; }
.card-row { margin: 4px 0; }
.card-row.metric { background: #eef2ff; padding: 5px 8px; border-radius: 3px;
                   font-family: monospace; margin-top: 8px; }
.card-row.note { color: #555; font-size: 12.5px; margin-top: 8px; font-style: italic; }
.verdict { font-size: 14px; }
.pass { color: #1e8e3e; font-weight: bold; }
.fail { color: #d93025; }
.pipeline { background: #fff; border: 1px solid #ddd; border-radius: 6px;
            padding: 16px; font-family: monospace; font-size: 13px;
            white-space: pre; overflow-x: auto; line-height: 1.5; }
.footer { color: #888; font-size: 12px; margin-top: 50px; padding-top: 20px;
          border-top: 1px solid #eee; }
ul { margin: 8px 0; }
li { margin: 4px 0; }
"""

PIPELINE_ASCII = r"""
┌────────────────────────────────────────────────────────────────────────────┐
│  1. 원천 데이터 (DTx 제품 운영 DB 추출)                                          │
│     • prescription 606건 (576명, 재처방 30명)                                   │
│     • assignment_record 130,450건 (12 운동 × 7 레벨)                            │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │ build_dataset.py
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  2. 환자별 trajectory + 행동유형 라벨링 (룰 기반)                                │
│     A 초기이탈 · B 안주 · C 지루 · D 중간벽 · E 졸업 · F 진행 · Z 미시작              │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │ build_features.py
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  3. 피처 테이블 3종 (score / session / patient)                                │
│     • event-level (130k)  • session-level (37k)  • patient-level (576)        │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │ train_models.py
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  4. ML 모델 4종 (LightGBM, 환자 단위 80/20 split)                              │
│     • score   • dropout   • gap   • termination                              │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │ simulator.py
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  5. 가상 환자 시뮬레이터 (cohort-step)                                          │
│     [입력] 알고리즘 정책 = (composer, adjuster)                                 │
│     [출력] 재처방률 · 이탈률 · max_level · 난이도 유지율                          │
│     ※ archetype별 base rate로 termination calibration                         │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │ make_report.py
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  6. Multi-seed 정책 비교 + holdout KS 검증                                     │
│     composer 5 × adjuster 11 × seeds 5 = 275 시뮬                              │
└────────────────────────────────────────────────────────────────────────────┘
"""


def build_html(patients, events, metrics, agg, n_seeds, n_patients):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_patients_real = len(patients)
    n_events = len(events)
    n_eligible = (patients["archetype"] != "Z_never_played").sum()
    rx_rate = patients["did_represcribe"].mean() * 100

    best = agg.sort_values("represcribe_rate_mean", ascending=False).iloc[0]
    base_row = agg[(agg["composer"] == "random_balanced")
                   & (agg["adjuster"] == "current_50_80")]
    baseline = float(base_row["represcribe_rate_mean"].iloc[0]) if len(base_row) else 0
    baseline_ci = float(base_row["represcribe_rate_ci95"].iloc[0]) if len(base_row) else 0

    return f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>DTx Bot — 종합 리포트</title>
<style>{CSS}</style></head><body>

<h1>DTx Patient Twin Bot — 종합 리포트</h1>
<div>
  <span class="kv">환자 {n_patients_real}명</span>
  <span class="kv">이벤트 {n_events:,}건</span>
  <span class="kv">실제 재처방률 {rx_rate:.1f}%</span>
  <span class="kv">ML 모델 4종</span>
  <span class="kv">시뮬 정책 5×11</span>
  <span class="kv">생성 {ts}</span>
</div>

<p class="note">
<b>이 봇이 하는 일</b>: 실제 DTx 제품 환자 데이터로 학습된 가상 환자에게
새 난이도 조정 알고리즘 / 운동 추천 정책을 적용해보고,
<b>예상 재처방률·이탈률·레벨 도달 패턴</b>이 어떻게 바뀔지 사전 예측.
</p>

<h2>1. 전체 파이프라인</h2>
<div class="pipeline">{PIPELINE_ASCII}</div>

<h2>2. 학습에 사용한 데이터</h2>
<p>DTx 제품 운영 데이터 2종을 사용. 모든 분석/모델 학습은 이 데이터 안에서 이루어짐.</p>
<ul>
  <li><code>prescription.csv</code> &mdash; 606 처방 (576명)</li>
  <li><code>assignment_record.csv</code> &mdash; 130,450 훈련 이벤트</li>
</ul>
{_img(chart_data_overview(patients, events))}

<h3>운동 12종</h3>
<ul>
  <li><b>전반 훈련 (3종, 각 ~19,500건)</b>:
      <code>memorize_out_loud</code>, <code>word_storming</code>, <code>memorize_with_stories</code></li>
  <li><b>세부 훈련 (9종, 각 ~8,000건)</b>:
      <code>memorize_oddly</code>, <code>word_throwing</code>, <code>memorize_scene</code>,
      <code>word_snake</code>, <code>making_into_one</code>, <code>clapping</code>,
      <code>repeat_and_memorize</code>, <code>word_finding</code>, <code>memorize_orders</code></li>
</ul>

<h2>3. 행동유형 라벨링 (6+1개 archetype)</h2>
<p>실제 데이터 패턴 + 선행 EDA 분석을 반영한 룰 기반 분류. ML 클러스터링이 아니라 의도적으로 해석 가능한 룰.</p>
<table>
  <tr><th>유형</th><th>정의</th><th>의미</th></tr>
  <tr><td>A 초기 이탈</td><td>max_level=1 AND n_general &lt; 21</td><td>사용 습관 미형성, 학습 시작 전 이탈</td></tr>
  <tr><td>B 안주형</td><td>max_level ∈ {{2,3}} AND adherence ≥ 0.5</td><td>꾸준하지만 진척 없음 → 재처방률 높음</td></tr>
  <tr><td>C 지루형</td><td>max_level ∈ {{2,3}} AND adherence &lt; 0.5</td><td>레벨 정체 + 참여 떨어짐</td></tr>
  <tr><td>D 중간 벽</td><td>last_level ∈ {{4,5}} AND n_general ≥ 21</td><td>심리적 정체기, 중상위에서 막힘</td></tr>
  <tr><td>E 졸업자</td><td>max_level = 7</td><td>치료 목표 도달 = "끝났다" 신호</td></tr>
  <tr><td>F 진행자</td><td>위 어디에도 안 들어감</td><td>꾸준한 정상 진행</td></tr>
  <tr><td>Z 미시작</td><td>처방만 받고 한 번도 안 함</td><td>분석 제외</td></tr>
</table>
{_img(chart_archetypes(patients))}

<p class="tip">
<b>핵심 관찰</b>:
B 안주형의 재처방률이 19.7%로 가장 높음 (선행 분석(2차)의 "maintain_memorize_oddly↑ → 재처방↑" 패턴과 일치).
A 초기 이탈군은 재처방 0%. C 지루군은 가장 많지만 2%만 재처방.
</p>

<h2>4. ML 모델 4종 (LightGBM)</h2>
<p>봇이 "가상 환자처럼" 행동하기 위한 핵심 모델 4개. 모두 LightGBM (Gradient Boosting Decision Tree).</p>
{model_cards_html(metrics)}
{_img(chart_model_metrics(metrics))}

<h3>학습 방식 요약</h3>
<ul>
  <li><b>데이터 split</b>: <code>GroupShuffleSplit</code>로 <i>환자 단위 80/20</i>. 같은 환자가 train/test에 섞이지 않음 (data leakage 방지)</li>
  <li><b>Categorical 처리</b>: LightGBM native 지원 (one-hot 안 함). <code>exercise_name</code>, <code>archetype</code>, <code>prescribed_from</code> 등</li>
  <li><b>Class imbalance</b>: termination 모델은 양성 26명 / 음성 426명 → <code>scale_pos_weight</code>로 보정</li>
  <li><b>Early stopping</b>: eval set 기준 30 라운드 정체 시 중단</li>
  <li><b>Calibration</b>: termination 모델 raw 예측은 archetype별 base rate로 rescale (양성 sample 30명 한계 보완)</li>
</ul>

<h2>5. 시뮬레이터 — 어떻게 분석하나</h2>
<p>학습된 모델 4개를 결합한 cohort-step 시뮬레이터. 알고리즘 정책을 <i>plug-in</i>으로 받아 동일 가상 환자 풀에 적용 → 출력 지표 비교.</p>
<h3>정책 = composer + adjuster</h3>
<ul>
  <li><b>composer</b> &mdash; <i>"어떤 운동을 처방할지"</i> (세부 운동 9개 중 4개 선택 정책)
    <br>현재 5종: <code>random_balanced</code>, <code>fixed_rotation</code>, <code>weakness_focused</code>,
    <code>archetype_aware</code>, <code>score_balanced</code></li>
  <li><b>adjuster</b> &mdash; <i>"어떻게 난이도 조정할지"</i> (각 운동의 레벨 결정 룰)
    <br>현재 11종: <code>current_50_80</code> (현행), <code>drop_at_70</code>, <code>level_diff</code>,
    <code>strict_up_n_over_np1</code>, <code>expand_10_diff</code>, <code>hybrid_drop70_strict_up</code>,
    <code>narrow_maintain_65_75</code>, <code>adaptive_personal_baseline</code>, <code>aggressive_75_85</code>,
    <code>cliff_jump</code>, <code>slow_climb</code></li>
</ul>
<p>새 정책 추가는 <code>bot/composers.py</code> 또는 <code>bot/algorithms.py</code>에
클래스 한 개 추가 + dict 등록만 하면 끝. 시뮬레이터 코드 손댈 필요 없음.</p>

<h3>출력 지표</h3>
<table>
  <tr><th>지표</th><th>의미</th></tr>
  <tr><td>represcribe_rate</td><td>90일 끝나고 재처방 받을 비율 (메인 KPI)</td></tr>
  <tr><td>drop_rate</td><td>90일 안에 그만두는 비율</td></tr>
  <tr><td>mean_max_level</td><td>도달한 최고 레벨의 평균</td></tr>
  <tr><td>mean_maintain_rate</td><td>같은 레벨에 머문 비율 (낮을수록 동적)</td></tr>
  <tr><td>final_archetype_dist</td><td>봇 행동 결과로 재분류된 archetype 분포</td></tr>
</table>

<h2>6. Holdout 검증 (실제 환자 분포 vs 봇 분포)</h2>
<p>20% holdout 환자의 trajectory 분포와 봇 시뮬(현행 룰) 분포를 Kolmogorov–Smirnov test로 비교.</p>
{validation_table_html()}
<p class="note">
<b>해석</b>: <code>max_level</code> 분포는 KS p=0.9995로 완벽 일치 ✓.
다른 지표는 평균값은 가까우나 분포 spread가 좁아 KS fail.
→ 봇이 <b>평균적인 경로</b>를 잘 재현하지만 <b>변동성</b>은 약간 보수적.
</p>

<h2>7. 정책 비교 결과 (Multi-seed, K=5)</h2>
<p>{len(agg)} 정책을 각각 {n_seeds}개 시드로 돌려 평균 ± 95% CI. n={n_patients} 가상 환자.</p>
{_img(policy_ranking_chart(agg))}
{_img(policy_heatmap(agg, "represcribe_rate", "재처방률 (composer × adjuster)", "{:.1%}", "YlGn"))}
{_img(policy_heatmap(agg, "mean_maintain", "난이도 유지율 (낮을수록 동적)", "{:.0%}", "YlOrRd_r"))}

<p class="note">
<b>최적 정책</b>: <code>{best['adjuster']}</code> × <code>{best['composer']}</code>
&mdash; 재처방률 <b>{best['represcribe_rate_mean']*100:.1f}% ± {best['represcribe_rate_ci95']*100:.1f}%</b><br>
<b>현행 baseline</b>: <code>current_50_80</code> × <code>random_balanced</code>
&mdash; <b>{baseline*100:.1f}% ± {baseline_ci*100:.1f}%</b><br>
<b>gap</b>: {(best['represcribe_rate_mean']-baseline)*100:+.1f}%p
{'— ⚠️ noise 범위 (CI 겹침)' if (best['represcribe_rate_mean']-baseline) <= (best['represcribe_rate_ci95']+baseline_ci) else '— ✓ 통계적 유의'}
</p>

<h2>8. 결론과 한계</h2>
<div class="tip">
<b>봇이 신뢰성 있게 측정하는 것</b>:
<ul>
  <li>정책 간 <b>상대 ranking</b> (어떤 패턴이 일관적으로 위/아래)</li>
  <li><b>maintain_rate</b> &mdash; 선행 분석 시뮬 결과(85/76/72/70%)와 정확히 일치</li>
  <li><b>max_level 분포</b> &mdash; 실제 환자 분포와 KS p=0.99 일치</li>
  <li>알고리즘이 환자를 어떤 archetype으로 push하는지 (dynamic reclassification)</li>
</ul>
</div>
<div class="warn">
<b>봇이 신뢰성 떨어지는 것</b>:
<ul>
  <li><b>재처방률 절대값</b> &mdash; archetype별 base rate로 calibration된 결과 (양성 30명 한계 보완 위해)</li>
  <li><b>정책 간 재처방률 차이의 통계적 유의성</b> &mdash; CI 겹침, 1-2%p 차이는 noise 범위</li>
  <li><b>cold-start 정확도</b> &mdash; 새 운동 첫 시도의 점수 분포는 평균만 모델링</li>
</ul>
</div>
<p class="note">
<b>실용적 결론</b>: 의사결정 prior로는 OK, 결정적 정량 근거로는 부족.
"몇 % 재처방률 상승"이 아니라 "어떤 방향으로 환자 행동이 바뀐다"는 신호로 사용 권장.
진짜 결정적 근거가 필요하면 실제 환자에서 작은 A/B 단기 시험이 필요.
</p>

<h2>9. 사용법</h2>
<div class="pipeline">python -m bot.build_dataset       # CSV → parquet (~5s)
python -m bot.label_archetypes    # 6+1개 행동유형 라벨링
python -m bot.build_features      # 학습용 피처 3종
python -m bot.train_models        # LightGBM 4종 학습 (~20s)
python -m bot.simulator           # 정책 비교 (단일 시드)
python -m bot.validate_bot        # holdout KS 검증
python -m bot.make_report --seeds 5  # Multi-seed 정책 매트릭스
python -m bot.make_overview_report   # 이 종합 리포트</div>

<div class="footer">
DTx Patient Twin Bot — overview report ·
ML: LightGBM ·
시뮬: cohort-step, K={n_seeds} seeds ·
생성 {ts}
</div>

</body></html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="overview.html")
    p.add_argument("--n", type=int, default=200, help="시뮬 환자 수 (이미 돌린 결과 사용 시 메타용)")
    args = p.parse_args()

    patients = pd.read_parquet(OUT / "patients.parquet")
    events = pd.read_parquet(OUT / "events.parquet")
    with open(MODELS / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    agg = pd.read_csv(OUT / "simulation_compare.csv")
    raw = pd.read_csv(OUT / "simulation_raw_multiseed.csv")
    n_seeds = raw["seed"].nunique()

    html = build_html(patients, events, metrics, agg, n_seeds, args.n)
    out_path = OUT / args.out
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
