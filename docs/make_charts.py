# -*- coding: utf-8 -*-
"""
README demo charts, rendered from the pipeline outputs into docs/img/*.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

for fname in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    try:
        matplotlib.font_manager.findfont(fname, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = fname
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

# reference palette (validated)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
          "#e87ba4", "#eb6834"]
BLUE = SERIES[0]

ARCH_LABEL = {
    "A_early_dropout": "A 습관 미형성",
    "B_stalled_settled": "B 안주형",
    "C_stalled_bored": "C 지루형",
    "D_mid_wall": "D 중간 벽",
    "E_maxed_out": "E 만렙 졸업",
    "F_steady_progress": "F 꾸준 진행",
    "Z_never_played": "Z 미사용",
}


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_archetypes():
    p = pd.read_parquet(OUT / "patients.parquet")
    order = sorted(ARCH_LABEL)
    g = p.groupby("archetype").agg(n=("patient_id", "count"),
                                   rx=("did_represcribe", "mean")).reindex(order)
    labels = [ARCH_LABEL[a] for a in order]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    ax = axes[0]
    style_ax(ax)
    bars = ax.bar(labels, g["n"], color=BLUE, width=0.62, zorder=3)
    for b, v in zip(bars, g["n"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}", ha="center",
                fontsize=9, color=INK2)
    ax.set_title("행동 아키타입 분포 (규칙 기반 라벨링)", fontsize=11,
                 color=INK, loc="left", pad=10)
    ax.set_ylabel("환자 수", fontsize=9, color=INK2)
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1]
    style_ax(ax)
    bars = ax.bar(labels, g["rx"] * 100, color=BLUE, width=0.62, zorder=3)
    for b, v in zip(bars, g["rx"] * 100):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.0f}%", ha="center",
                fontsize=9, color=INK2)
    ax.set_title("아키타입별 재처방률 — 참여 유지가 재처방을 이끈다", fontsize=11,
                 color=INK, loc="left", pad=10)
    ax.set_ylabel("재처방률 (%)", fontsize=9, color=INK2)
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(IMG / "archetypes.png", dpi=150, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print("wrote archetypes.png")


def chart_simulation():
    sim = pd.read_csv(OUT / "simulation_compare.csv")
    sim = sim.sort_values("represcribe_rate")
    names = sim["adjuster"].tolist()
    colors = [MUTED if n == "placebo_no_change" else BLUE for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURFACE)
    for ax, col, title, fmt in (
        (axes[0], "represcribe_rate", "정책별 예측 재처방률 (디지털 트윈 시뮬레이션)", "{:.1%}"),
        (axes[1], "mean_maintain", "정책별 레벨 유지율 — 난이도 반응성", "{:.0%}"),
    ):
        style_ax(ax)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.yaxis.grid(False)
        bars = ax.barh(names, sim[col], color=colors, height=0.62, zorder=3)
        for b, v in zip(bars, sim[col]):
            ax.text(v + sim[col].max() * 0.02, b.get_y() + b.get_height() / 2,
                    fmt.format(v), va="center", fontsize=9, color=INK2)
        ax.set_xlim(0, sim[col].max() * 1.18)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(IMG / "simulation.png", dpi=150, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print("wrote simulation.png")


def chart_trajectories():
    ev = pd.read_parquet(OUT / "events.parquet")
    p = pd.read_parquet(OUT / "patients.parquet")[["patient_id", "archetype"]]
    ev = ev[ev["kind"] == "general"].merge(p, on="patient_id")
    ev["tg"] = ev.groupby("patient_id").cumcount()
    ev = ev[ev["tg"] < 100]
    order = ["A_early_dropout", "B_stalled_settled", "C_stalled_bored",
             "D_mid_wall", "E_maxed_out", "F_steady_progress"]

    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    style_ax(ax)
    for arch, color in zip(order, SERIES):
        traj = (ev[ev["archetype"] == arch]
                .groupby("tg")["level"].mean())
        traj = traj[traj.index <= ev[ev["archetype"] == arch]
                    .groupby("patient_id")["tg"].max().median()]
        if len(traj) < 3:
            continue
        ax.plot(traj.index, traj.values, color=color, linewidth=2, zorder=3)
        ax.annotate(ARCH_LABEL[arch], (traj.index[-1], traj.values[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=color, va="center", fontweight="bold")
    ax.set_xlim(0, 118)
    ax.set_ylim(0.8, 7.4)
    ax.set_yticks(range(1, 8))
    ax.set_title("아키타입별 평균 레벨 궤적 (general 훈련, 이벤트 순서 기준)",
                 fontsize=11, color=INK, loc="left", pad=10)
    ax.set_xlabel("n번째 general 훈련", fontsize=9, color=INK2)
    ax.set_ylabel("평균 레벨", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(IMG / "trajectories.png", dpi=150, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print("wrote trajectories.png")


if __name__ == "__main__":
    chart_archetypes()
    chart_simulation()
    chart_trajectories()
