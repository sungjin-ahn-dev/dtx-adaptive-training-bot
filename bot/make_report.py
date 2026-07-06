"""
Multi-seed policy matrix report. Runs each (composer, adjuster) policy over K
seeds and reports mean ± 95% CI, so real algorithm effect is separable from
per-run noise. Renders to a standalone HTML.
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

from .algorithms import ALGORITHMS
from .composers import COMPOSERS
from .simulator import DTxSimulator, Policy

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

for fname in ("Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"):
    try:
        matplotlib.rcParams["font.family"] = fname
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False
Z95 = 1.96


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


# --------------------------------------------------------------------------- #
# Run + aggregate
# --------------------------------------------------------------------------- #

def run_multiseed(composer_names: list[str], adjuster_names: list[str],
                   n: int, days: int, seeds: list[int]) -> pd.DataFrame:
    rows = []
    total = len(composer_names) * len(adjuster_names) * len(seeds)
    i = 0
    for cn in composer_names:
        for an in adjuster_names:
            for seed in seeds:
                i += 1
                policy = Policy(COMPOSERS[cn], ALGORITHMS[an])
                print(f"  [{i:3d}/{total}] {policy.name} seed={seed}", flush=True)
                sim = DTxSimulator(rng_seed=seed)
                res = sim.run(policy, n_patients=n, window_days=days)
                s = res["summary"]
                rows.append({
                    "composer": cn, "adjuster": an, "seed": seed,
                    "represcribe_rate": s["represcribe_rate"],
                    "drop_rate": s["drop_rate"],
                    "mean_max_level": s["mean_max_level"],
                    "mean_last_level": s["mean_last_level"],
                    "mean_general": s["mean_general_events"],
                    "mean_sub": s["mean_sub_events"],
                    "mean_score": s["mean_score"],
                    "mean_maintain": s["mean_maintain_rate"],
                })
    return pd.DataFrame(rows)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["represcribe_rate", "drop_rate", "mean_max_level",
                   "mean_last_level", "mean_general", "mean_sub",
                   "mean_score", "mean_maintain"]
    n_seeds = raw["seed"].nunique()
    g = raw.groupby(["composer", "adjuster"])
    out = pd.DataFrame()
    for col in metric_cols:
        out[f"{col}_mean"] = g[col].mean()
        out[f"{col}_std"] = g[col].std()
        out[f"{col}_ci95"] = Z95 * g[col].std() / np.sqrt(n_seeds)
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

def heatmap_mean(df: pd.DataFrame, metric: str, title: str,
                  fmt: str = "{:.1%}", cmap: str = "YlGn") -> str:
    mean_col = f"{metric}_mean"
    pivot = df.pivot(index="composer", columns="adjuster", values=mean_col)
    fig, ax = plt.subplots(
        figsize=(1.5 + 1.4 * len(pivot.columns), 0.6 * len(pivot.index) + 2.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax).set_label(metric)
    return _fig_to_b64(fig)


def ranking_with_ci(df: pd.DataFrame, metric: str, title: str,
                     pct: bool = True) -> str:
    mean_col = f"{metric}_mean"
    ci_col = f"{metric}_ci95"
    df2 = df.sort_values(mean_col, ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 0.35 * len(df2) + 1.5))
    labels = df2.apply(lambda r: f"{r['adjuster']:24s} | {r['composer']}", axis=1)
    means = df2[mean_col] * (100 if pct else 1)
    cis = df2[ci_col] * (100 if pct else 1)
    bars = ax.barh(labels, means, xerr=cis, color="#4f7cf7",
                    capsize=3, error_kw={"ecolor": "#444", "lw": 1})
    bars[0].set_color("#1e8e3e")
    for b, m, c in zip(bars, means, cis):
        ax.text(b.get_width() + c + (0.05 if pct else 0.005),
                b.get_y() + b.get_height() / 2,
                f"{m:.1f}{'%' if pct else ''} ±{c:.1f}",
                va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric} ({'%' if pct else ''}, ± 95% CI)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    return _fig_to_b64(fig)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

CSS = """
body { font-family: -apple-system, "Malgun Gothic", BlinkMacSystemFont, "Segoe UI",
       sans-serif; max-width: 1200px; margin: 30px auto; padding: 0 20px;
       color: #222; line-height: 1.5; }
h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { margin-top: 36px; border-left: 4px solid #4f7cf7; padding-left: 10px; }
table { border-collapse: collapse; margin: 16px 0; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 5px 8px; text-align: right; }
th { background: #f5f5f5; }
td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) {
    text-align: left; font-family: monospace; }
img { max-width: 100%; height: auto; }
.kv { display: inline-block; padding: 2px 8px; background: #eef; border-radius: 4px;
      margin-right: 8px; font-family: monospace; font-size: 13px; }
.note { background: #fff8e1; border-left: 4px solid #ffa000; padding: 8px 12px;
        margin: 12px 0; font-size: 13px; }
.green { background: #e7f5ec; font-weight: bold; }
.footer { color: #888; font-size: 12px; margin-top: 40px; }
"""


def policy_table_html(df: pd.DataFrame) -> str:
    df = df.sort_values("represcribe_rate_mean", ascending=False).reset_index(drop=True)
    rows = ["<table><tr>"
            "<th>adjuster</th><th>composer</th>"
            "<th>재처방률 ± CI</th>"
            "<th>max_lvl ± CI</th>"
            "<th>last_lvl</th>"
            "<th>general events</th>"
            "<th>유지율 ± CI</th>"
            "<th>drop_rate</th></tr>"]
    for i, r in df.iterrows():
        cls = ' class="green"' if i == 0 else ""
        rows.append(
            f"<tr{cls}>"
            f"<td>{r['adjuster']}</td><td>{r['composer']}</td>"
            f"<td>{r['represcribe_rate_mean']:.1%} ± {r['represcribe_rate_ci95']:.1%}</td>"
            f"<td>{r['mean_max_level_mean']:.2f} ± {r['mean_max_level_ci95']:.2f}</td>"
            f"<td>{r['mean_last_level_mean']:.2f}</td>"
            f"<td>{r['mean_general_mean']:.1f}</td>"
            f"<td>{r['mean_maintain_mean']:.1%} ± {r['mean_maintain_ci95']:.1%}</td>"
            f"<td>{r['drop_rate_mean']:.1%}</td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def significance_note(df: pd.DataFrame) -> str:
    """Identify the best policy and whether its CI overlaps with the runner-up."""
    df2 = df.sort_values("represcribe_rate_mean", ascending=False).reset_index(drop=True)
    best = df2.iloc[0]
    second = df2.iloc[1] if len(df2) > 1 else None
    baseline = df2[(df2["composer"] == "random_balanced")
                   & (df2["adjuster"] == "current_50_80")]

    parts = [
        f"<b>최적 정책</b>: <code>{best['adjuster']}</code> ⨯ <code>{best['composer']}</code>"
        f" &mdash; 재처방률 <b>{best['represcribe_rate_mean']:.1%}</b>"
        f" (95% CI ±{best['represcribe_rate_ci95']:.1%})"
    ]
    if second is not None:
        gap = best['represcribe_rate_mean'] - second['represcribe_rate_mean']
        ci_combined = best['represcribe_rate_ci95'] + second['represcribe_rate_ci95']
        sig = "✓ 통계적으로 유의" if gap > ci_combined else "⚠️ 차이가 noise 범위 — 시드/N 늘려서 재확인 권장"
        parts.append(
            f"<br>2위 (<code>{second['adjuster']}</code> ⨯ <code>{second['composer']}</code>): "
            f"{second['represcribe_rate_mean']:.1%}, 갭 {gap*100:.1f}%p — {sig}"
        )
    if len(baseline):
        b = baseline.iloc[0]
        uplift = (best['represcribe_rate_mean'] - b['represcribe_rate_mean']) / max(b['represcribe_rate_mean'], 1e-6)
        parts.append(
            f"<br>현행 baseline <code>current_50_80 ⨯ random_balanced</code>: "
            f"{b['represcribe_rate_mean']:.1%} → <b>{uplift:+.0%}</b>"
        )
    return " ".join(parts)


def validation_summary_html() -> str:
    ks_path = OUT / "validation_ks.json"
    if not ks_path.exists():
        return '<div class="note">Run <code>python -m bot.validate_bot</code> first.</div>'
    with open(ks_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = ["<table><tr><th>metric</th><th>real mean</th><th>sim mean</th>"
            "<th>KS</th><th>p-value</th><th>pass</th></tr>"]
    for metric, d in data.items():
        if metric == "represcribe_rate":
            rows.append(
                f"<tr><td>represcribe%</td><td>{d['real']*100:.2f}%</td>"
                f"<td>{d['sim']*100:.2f}%</td><td>-</td><td>-</td><td>(rate)</td></tr>")
            continue
        passed = "✓" if d["pass"] else "✗"
        rows.append(
            f"<tr><td>{metric}</td><td>{d['real_mean']:.2f}</td>"
            f"<td>{d['sim_mean']:.2f}</td><td>{d['ks']:.3f}</td>"
            f"<td>{d['p']:.4f}</td><td>{passed}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


def build_html(df: pd.DataFrame, n: int, days: int, n_seeds: int,
               composers: list[str], adjusters: list[str]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>DTx Bot Policy Matrix (multi-seed)</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>DTx Bot — Policy Matrix (multi-seed)</h1>",
        f'<div><span class="kv">N = {n}</span>'
        f'<span class="kv">window = {days}d</span>'
        f'<span class="kv">composers = {len(composers)}</span>'
        f'<span class="kv">adjusters = {len(adjusters)}</span>'
        f'<span class="kv">seeds = {n_seeds}</span>'
        f'<span class="kv">{ts}</span></div>',
        f'<p class="note">{significance_note(df)}</p>',
        '<p class="note"><b>해석 주의:</b> '
        '재처방률 절대값은 archetype별 실제 base rate로 calibration된 결과 — '
        '정책 간 <b>상대 비교</b>가 신뢰 신호. CI가 겹치면 차이가 noise.</p>',

        "<h2>1. 정책 정렬 표 (재처방률 내림차순)</h2>",
        policy_table_html(df),

        "<h2>2. 재처방률 ranking (± 95% CI)</h2>",
        _img(ranking_with_ci(df, "represcribe_rate",
                              "Re-prescription rate per policy (5-seed average)")),

        "<h2>3. 재처방률 매트릭스 (mean)</h2>",
        _img(heatmap_mean(df, "represcribe_rate",
                           "Re-prescription rate by composer × adjuster")),

        "<h2>4. 평균 max_level 매트릭스</h2>",
        _img(heatmap_mean(df, "mean_max_level", "Mean max_level",
                           fmt="{:.2f}", cmap="Blues")),

        "<h2>5. 난이도 유지율 매트릭스</h2>",
        "<p>낮을수록 환자가 정체 안 됨.</p>",
        _img(heatmap_mean(df, "mean_maintain", "Difficulty-maintenance rate",
                           fmt="{:.0%}", cmap="YlOrRd_r")),

        "<h2>6. Holdout 검증 (baseline 정책)</h2>",
        validation_summary_html(),

        "<h2>7. 정책 구성 요소</h2>",
        "<h3>Adjuster (난이도 조정 룰)</h3><ul>",
        "<li><code>current_50_80</code> — 현행</li>",
        "<li><code>drop_at_70</code> — 하락 50→70%</li>",
        "<li><code>level_diff</code> — 레벨별 차등</li>",
        "<li><code>strict_up_n_over_np1</code> — 상승 strict</li>",
        "<li><code>expand_10_diff</code> — 7→10 확장</li>",
        "<li><code>hybrid_drop70_strict_up</code> — 70% 하락 + strict 상승</li>",
        "<li><code>narrow_maintain_65_75</code> — 유지 zone 10%</li>",
        "<li><code>adaptive_personal_baseline</code> — baseline 대비 상대 평가</li>",
        "<li><code>aggressive_75_85</code> — 신규: 안주 zone 매우 좁힘</li>",
        "<li><code>cliff_jump</code> — 신규: 점수 따라 ±2 점프</li>",
        "<li><code>slow_climb</code> — 신규: 보수적 상승 (3회 모두 ≥0.7)</li>",
        "</ul>",
        "<h3>Composer (세부 운동 4개 선택)</h3><ul>",
        "<li><code>random_balanced</code> — 9개 중 4개 무작위 (현행)</li>",
        "<li><code>fixed_rotation</code> — 결정적 순환</li>",
        "<li><code>weakness_focused</code> — 점수 낮은 4개</li>",
        "<li><code>archetype_aware</code> — 선행 분석(2차) 인사이트 반영</li>",
        "<li><code>score_balanced</code> — 신규: top2 + bottom2</li>",
        "</ul>",

        '<div class="footer">'
        f"DTx Patient Twin Bot v3 — multi-seed report (K={n_seeds})"
        "</div>",
        "</body></html>",
    ]
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="patients per simulation")
    p.add_argument("--days", type=int, default=83)
    p.add_argument("--seeds", type=int, default=5, help="number of seeds per policy")
    p.add_argument("--composer", nargs="*",
                   default=["random_balanced", "fixed_rotation",
                            "weakness_focused", "archetype_aware", "score_balanced"])
    p.add_argument("--adjuster", nargs="*",
                   default=["current_50_80", "drop_at_70", "level_diff",
                            "hybrid_drop70_strict_up", "aggressive_75_85",
                            "cliff_jump", "slow_climb"])
    p.add_argument("--out", type=str, default="report.html")
    args = p.parse_args()

    seeds = list(range(args.seeds))
    n_policy = len(args.composer) * len(args.adjuster)
    print(f"Simulating {args.n} patients × {args.days} days × "
          f"{n_policy} policies × {args.seeds} seeds = {n_policy * args.seeds} runs ...")
    raw = run_multiseed(args.composer, args.adjuster, args.n, args.days, seeds)
    raw.to_csv(OUT / "simulation_raw_multiseed.csv", index=False)

    agg = aggregate(raw)
    agg.to_csv(OUT / "simulation_compare.csv", index=False)

    html = build_html(agg, args.n, args.days, args.seeds, args.composer, args.adjuster)
    out_path = OUT / args.out
    out_path.write_text(html, encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Wrote {OUT / 'simulation_compare.csv'} (aggregated)")
    print(f"Wrote {OUT / 'simulation_raw_multiseed.csv'} (raw per-seed)")


if __name__ == "__main__":
    main()
