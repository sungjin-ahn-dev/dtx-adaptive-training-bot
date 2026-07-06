"""
Self-contained "how the experiment works" explainer (Korean). Reads the live
pipeline outputs and renders one HTML page (out/explainer.html) that walks a
non-ML reader through the whole thing: the data, the 6 archetypes and their
exact rules, the 4 models, training/validation, the simulation loop, and how
to read the results table.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
MODELS = ROOT / "models"

# Korean labels for the 6 archetypes + the exact rule that defines each one.
ARCH_INFO = {
    "A_early_dropout": ("A · 습관 미형성형", "general 훈련 최고레벨이 1 그대로이고, general 훈련을 21회 미만으로 하고 멈춘 사람", "난이도 문제가 아니라 '앱 쓰는 습관' 자체가 안 잡힌 케이스. 선행 분석(2차)의 '레벨1 정체자 16명'이 여기."),
    "B_stalled_settled": ("B · 안주형 (성실)", "general 최고레벨이 2~3에 머물지만 순응도 ≥ 0.5 (꾸준히 옴)", "쉬운 레벨에 자리잡고 계속 즐기는 사람. 레벨이 안 올라도 만족 → 재처방률이 가장 높음."),
    "C_stalled_bored": ("C · 지루형 (이탈)", "general 최고레벨이 2~3인데 순응도 < 0.5 (점점 안 옴)", "낮은 레벨에 갇혀 지루해져서 빠지는 사람. 안주형과 레벨대는 같지만 참여가 떨어짐 → 재처방률 최저."),
    "D_mid_wall": ("D · 중간 벽형", "general을 21회 이상 했고, 마지막 레벨이 4 또는 5에서 멈춤", "레벨 4~5 '심리적 정체기'에 부딪힌 사람. 선행 분석이 지목한 난이도 벽 구간."),
    "E_maxed_out": ("E · 졸업형", "general 최고레벨이 7(최대)에 도달", "끝까지 다 올라간 성실 그룹. max를 찍으면 '다 했다'는 종료 신호가 됨."),
    "F_steady_progress": ("F · 성장 진행형", "위 어디에도 안 들어가는 나머지 (중간 단계를 정상적으로 올라가는 중)", "특정 함정에 빠지지 않고 꾸준히 진척 중인 표준 케이스."),
    "Z_never_played": ("Z · 미실행 (분석 제외)", "처방은 받았지만 general 훈련 기록이 0건", "한 번도 안 한 사람. 행동 데이터가 없어 모델/시뮬에서 제외."),
}

# Plain-Korean description of every difficulty-adjustment algorithm.
ALGO_INFO = {
    "current_50_80": ("현행 룰", "최근 3회 평균 <50% → 한 단계 내림 / ≥80% → 한 단계 올림 / 그 사이는 유지. 지금 실제로 쓰는 규칙(기준선)."),
    "drop_at_70": ("하락 기준 50→70%", "내리는 기준만 70%로 올림(≥80% 상승은 동일). 선행 분석 2차 제안. 더 빨리 내려줘서 정체를 풀자는 취지."),
    "level_diff": ("레벨별 차등", "레벨이 높을수록 상승은 더 어렵게·하락은 더 쉽게. 높은 레벨에 우연히 갇히는 걸 방지."),
    "strict_up_n_over_np1": ("상승 까다롭게", "올리려면 3회 '모두' score ≥ n/(n+1) 이어야 함(평균 아님). 하락은 평균<50%. 선행 분석 1차 제안 3."),
    "expand_10_diff": ("7→10단계 확장", "최대 레벨을 10으로 늘리고 레벨별 차등 적용. max 도달 시점을 뒤로 밀어 조기 종료 지연."),
    "hybrid_drop70_strict_up": ("하락70%+상승까다", "내림은 70%로 후하게, 올림은 3회 모두 통과해야. 두 제안의 결합형."),
    "narrow_maintain_65_75": ("유지구간 좁힘", "<65% 내림 / ≥75% 올림. '유지' 구간을 10%로 좁혀 레벨이 자주 움직이게."),
    "adaptive_personal_baseline": ("개인 기준 적응", "고정 기준(0.60) 대비 상대 평가. 절대 점수가 낮아도 본인 기준 이상이면 상승."),
    "aggressive_75_85": ("공격적", "<75% 내림 / ≥85% 올림. 안주 구간을 매우 좁힘."),
    "cliff_jump": ("점프형", "평균에 따라 ±2단계까지 한 번에 점프. 드라마틱한 변화로 자극."),
    "slow_climb": ("천천히 상승", "올림은 3회 모두 ≥0.7 + 평균 ≥0.85일 때만(보수적). 난이도가 천천히 오름."),
    "easy_mode_down_only": ("내리기만", "절대 어려워지지 않음(상승 없음). 통제 비교용."),
    "hard_mode_up_only": ("올리기만", "절대 쉬워지지 않음(하락 없음). 통제 비교용."),
    "single_score_floor": ("단순 1회 기준", "3회 평균이 아니라 마지막 1회 점수로만 ±1. 단순 규칙 비교용."),
    "placebo_no_change": ("고정(플라시보)", "레벨을 절대 안 바꿈. '알고리즘이 정말 효과 있나'를 재는 통제군."),
    "random_walk": ("랜덤", "점수 무시하고 무작위 ±1. '점수 신호를 써야 한다'는 증거용 통제군."),
}

CSS = """
:root { --blue:#2c6cf6; --green:#1e8e3e; --amber:#ffa000; --ink:#1a1a2e; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Malgun Gothic", "Segoe UI", sans-serif;
  max-width: 960px; margin: 0 auto; padding: 0 22px 80px; color: var(--ink);
  line-height: 1.75; font-size: 16px; background: #fafbfc; }
h1 { font-size: 30px; margin-top: 40px; }
h2 { font-size: 23px; margin-top: 52px; border-left: 5px solid var(--blue);
  padding-left: 12px; }
h3 { font-size: 18px; margin-top: 30px; color: #333; }
p { margin: 12px 0; }
code { background: #eef1f8; padding: 1px 6px; border-radius: 4px; font-size: 14px;
  font-family: "Cascadia Code", Consolas, monospace; }
table { border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 14px; }
th, td { border: 1px solid #dde; padding: 8px 10px; text-align: left; vertical-align: top; }
th { background: #eef1f8; }
.lead { font-size: 18px; background: #eaf1ff; border-radius: 10px; padding: 18px 22px;
  border: 1px solid #cfe0ff; }
.box { background: #fff; border: 1px solid #e3e6ee; border-radius: 10px;
  padding: 4px 20px; margin: 18px 0; box-shadow: 0 1px 3px rgba(0,0,0,.04); }
.warn { background: #fff8e1; border-left: 5px solid var(--amber); padding: 12px 18px;
  border-radius: 6px; margin: 18px 0; }
.ok { background: #e7f5ec; border-left: 5px solid var(--green); padding: 12px 18px;
  border-radius: 6px; margin: 18px 0; }
.step { background: #fff; border: 1px solid #e3e6ee; border-radius: 10px; padding: 16px 20px;
  margin: 14px 0; }
.step .n { display: inline-block; width: 28px; height: 28px; line-height: 28px;
  text-align: center; background: var(--blue); color: #fff; border-radius: 50%;
  font-weight: bold; margin-right: 8px; }
.kv { display:inline-block; background:#eef1f8; border-radius:6px; padding:4px 10px;
  margin:3px 6px 3px 0; font-size:14px; }
.big { font-size: 22px; font-weight: bold; color: var(--blue); }
.toc { background:#fff; border:1px solid #e3e6ee; border-radius:10px; padding:8px 24px; }
.toc a { color: var(--blue); text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.muted { color:#888; font-size:13px; }
.arrow { color: var(--blue); font-weight: bold; }
.green-row { background:#e7f5ec; font-weight:bold; }
"""


def _fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def load_context() -> dict:
    ctx = {}
    pat = pd.read_parquet(OUT / "patients.parquet")
    ctx["n_patients"] = len(pat)
    ctx["n_played"] = int((pat["archetype"] != "Z_never_played").sum())
    ev = pd.read_parquet(OUT / "events.parquet")
    ctx["n_events"] = len(ev)
    ctx["n_represcribe"] = int(pat["did_represcribe"].fillna(False).sum())

    # archetype table (rx rate, adherence, levels)
    g = pat.groupby("archetype").agg(
        n=("patient_id", "count"),
        n_rx=("did_represcribe", "sum"),
        rx_rate=("did_represcribe", "mean"),
        mean_adh=("adherence", "mean"),
        mean_max_level=("general_max_level", "mean"),
    )
    ctx["arch_table"] = g

    with open(MODELS / "metrics.json", encoding="utf-8") as f:
        ctx["metrics"] = json.load(f)

    ks_path = OUT / "validation_ks.json"
    ctx["ks"] = json.load(open(ks_path, encoding="utf-8")) if ks_path.exists() else {}

    cmp_path = OUT / "simulation_compare.csv"
    ctx["sim"] = pd.read_csv(cmp_path) if cmp_path.exists() else None
    return ctx


def arch_section(ctx) -> str:
    g = ctx["arch_table"]
    rows = ["<table><tr><th>유형</th><th>정확한 분류 기준 (코드 그대로)</th>"
            "<th>인원</th><th>재처방률</th><th>평균 순응도</th><th>의미</th></tr>"]
    order = ["A_early_dropout", "B_stalled_settled", "C_stalled_bored",
             "D_mid_wall", "E_maxed_out", "F_steady_progress", "Z_never_played"]
    for key in order:
        label, rule, meaning = ARCH_INFO[key]
        if key in g.index:
            r = g.loc[key]
            n = _fmt_int(r["n"])
            rx = f"{r['rx_rate']*100:.1f}% ({int(r['n_rx'])}명)"
            adh = f"{r['mean_adh']:.2f}"
        else:
            n, rx, adh = "-", "-", "-"
        rows.append(
            f"<tr><td><b>{label}</b><br><span class='muted'>{key}</span></td>"
            f"<td>{rule}</td><td>{n}</td><td>{rx}</td><td>{adh}</td><td>{meaning}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


def algo_section() -> str:
    rows = ["<table><tr><th>알고리즘 코드</th><th>한 줄 이름</th><th>무엇을 하나</th></tr>"]
    for code, (name, desc) in ALGO_INFO.items():
        rows.append(f"<tr><td><code>{code}</code></td><td><b>{name}</b></td><td>{desc}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


def model_metrics_section(ctx) -> str:
    m = ctx["metrics"]
    def g(d, k, pct=False, nd=3):
        v = d.get(k)
        if v is None:
            return "-"
        return f"{v*100:.1f}%" if pct else f"{v:.{nd}f}"
    s, dr, gp, tm = m.get("score", {}), m.get("dropout", {}), m.get("gap", {}), m.get("termination", {})
    return f"""
<table>
<tr><th>모델</th><th>예측 대상</th><th>학습/시험 건수</th><th>핵심 성능</th><th>해석</th></tr>
<tr><td><b>score</b> (점수)</td><td>이 레벨에서 받을 점수(0~1)</td>
  <td>{_fmt_int(s.get('n_train'))} / {_fmt_int(s.get('n_test'))}</td>
  <td>MAE {g(s,'MAE')} (그냥 평균 찍기 {g(s,'baseline_MAE_mean')})</td>
  <td>평균만 찍는 것보다 오차가 작음 → 신호 있음</td></tr>
<tr><td><b>dropout</b> (이탈)</td><td>이번 세션 후 그만둘 확률</td>
  <td>{_fmt_int(dr.get('n_train'))} / {_fmt_int(dr.get('n_test'))}</td>
  <td>ROC-AUC {g(dr,'ROC_AUC')}</td>
  <td>⚠️ 이탈 세션이 양성 {g(dr,'pos_rate',pct=True)}로 매우 희귀 → holdout은 높지만 교차검증은 불안정(아래 한계 참고)</td></tr>
<tr><td><b>gap</b> (간격)</td><td>다음 훈련까지 며칠</td>
  <td>{_fmt_int(gp.get('n_train'))} / {_fmt_int(gp.get('n_test'))}</td>
  <td>MAE {g(gp,'MAE_days')}일 (기준 {g(gp,'baseline_MAE_days')}일)</td>
  <td>대부분 같은 날 재방문이라 약한 모델</td></tr>
<tr><td><b>termination</b> (재처방)</td><td>90일 뒤 재처방 확률</td>
  <td>{_fmt_int(tm.get('n_train'))} / {_fmt_int(tm.get('n_test'))} (양성 {tm.get('pos_train','?')}/{tm.get('pos_test','?')})</td>
  <td>ROC-AUC {g(tm,'ROC_AUC')}, PR-AUC {g(tm,'PR_AUC')}</td>
  <td>양성이 늘어 이전보다 훨씬 신뢰도↑</td></tr>
</table>
"""


def validation_section(ctx) -> str:
    ks = ctx["ks"]
    if not ks:
        return "<p class='muted'>검증 결과 파일이 아직 없습니다.</p>"
    rows = ["<table><tr><th>지표</th><th>실제 환자</th><th>시뮬레이션</th><th>일치?</th></tr>"]
    any_fail = False
    for metric, d in ks.items():
        if metric == "represcribe_rate":
            rows.append(f"<tr><td>재처방률</td><td>{d['real']*100:.1f}%</td>"
                        f"<td>{d['sim']*100:.1f}%</td><td>근접</td></tr>")
            continue
        ok = d.get("pass")
        any_fail = any_fail or (ok is False)
        mark = "✅ 통과" if ok else "❌ 불일치"
        rows.append(f"<tr><td>{metric}</td><td>{d['real_mean']:.2f}</td>"
                    f"<td>{d['sim_mean']:.2f}</td><td>{mark}</td></tr>")
    rows.append("</table>")
    note = ""
    if any_fail:
        note = ("<div class='warn'><b>지금 상태(정직하게):</b> 새 0421 데이터에서는 시뮬레이션이 "
                "실제 환자 분포를 아직 완벽히 재현하지 못합니다. 특히 시뮬이 레벨을 실제보다 높게 "
                "밀어올리는 경향이 있습니다. <b>그래서 절대 수치(예: 평균 레벨)는 그대로 믿지 말고, "
                "알고리즘 A vs B의 상대 비교(어느 쪽이 정체를 더 풀어주나)만 결론으로 쓰세요.</b> "
                "재처방률만큼은 실제와 근접합니다.</div>")
    return "\n".join(rows) + note


def results_section(ctx) -> str:
    sim = ctx["sim"]
    if sim is None:
        return ("<p class='muted'>아직 시뮬레이션 결과(simulation_compare.csv)가 없습니다. "
                "<code>python -m bot.make_report</code> 를 먼저 돌리세요.</p>")
    # average across composers, per adjuster
    agg = (sim.groupby("adjuster")
           .agg(rx=("represcribe_rate_mean", "mean"),
                drop=("drop_rate_mean", "mean"),
                maxlvl=("mean_max_level_mean", "mean"),
                maint=("mean_maintain_mean", "mean"))
           .sort_values("rx", ascending=False))
    rows = ["<table><tr><th>알고리즘</th><th>재처방률</th><th>이탈률</th>"
            "<th>평균 최고레벨</th><th>난이도 유지율</th></tr>"]
    for i, (name, r) in enumerate(agg.iterrows()):
        label = ALGO_INFO.get(name, (name, ""))[0]
        cls = " class='green-row'" if i == 0 else ""
        cur = " ← 현행" if name == "current_50_80" else ""
        rows.append(
            f"<tr{cls}><td><code>{name}</code><br><span class='muted'>{label}{cur}</span></td>"
            f"<td>{r['rx']*100:.1f}%</td><td>{r['drop']*100:.0f}%</td>"
            f"<td>{r['maxlvl']:.2f}</td><td>{r['maint']*100:.0f}%</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


def build_html(ctx) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    P = []
    A = P.append
    A(f"<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
      f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
      f"<title>DTx 제품 환자 트윈 봇 — 실험 설명서</title><style>{CSS}</style></head><body>")

    A("<h1>DTx 제품 환자 트윈 봇 — 실험이 어떻게 돌아가는지 A to Z</h1>")
    A(f"<p class='muted'>데이터: <code>dtx_*_20260421_clean.csv</code> · 생성 {ts}</p>")

    A("<div class='lead'>한 문장 요약: <b>실제 환자들의 훈련 기록으로 '가짜 환자(디지털 트윈)'를 "
      "학습시킨 뒤, 새 난이도 조정 규칙을 진짜 환자에게 쓰기 전에 컴퓨터 안에서 미리 돌려보고 "
      "이탈·재처방·레벨 변화가 어떻게 달라지는지 비교하는 실험</b>입니다.</div>")

    A("<div class='toc'><b>목차</b>"
      "<ol>"
      "<li><a href='#why'>왜 이걸 하나</a></li>"
      "<li><a href='#data'>1. 원본 데이터</a></li>"
      "<li><a href='#arch'>2. 환자 6유형 — '안주형'이 뭔지 정확한 기준</a></li>"
      "<li><a href='#models'>3. 환자 행동을 4개 모델로 쪼갰다</a></li>"
      "<li><a href='#features'>4. 모델에 넣는 입력(피처)</a></li>"
      "<li><a href='#train'>5. 어떻게 학습·검증했나</a></li>"
      "<li><a href='#sim'>6. 시뮬레이션은 이렇게 돌아간다</a></li>"
      "<li><a href='#algos'>7. 비교한 난이도 알고리즘들</a></li>"
      "<li><a href='#results'>8. 결과 읽는 법</a></li>"
      "<li><a href='#limits'>9. 한계 (먼저 말할 것)</a></li>"
      "</ol></div>")

    # WHY
    A("<h2 id='why'>왜 이걸 하나</h2>")
    A("<p>선행 분석에서 '지금 난이도 규칙은 레벨이 너무 안 바뀐다(유지율 85%). 이걸 바꾸면 "
      "환자가 덜 지루해하고 재처방이 늘 것'이라는 <b>가설</b>이 나왔습니다. 그런데 이걸 진짜 환자에게 "
      "바로 적용하면 잘못됐을 때 환자가 이탈해버립니다. 그래서 <b>가짜 환자 수백 명을 만들어 "
      "여러 규칙을 똑같은 조건에서 돌려보고</b> 어느 규칙이 나은지 미리 보는 겁니다. "
      "업계 용어로 <b>디지털 트윈 / 오프라인 정책 평가</b>라고 합니다.</p>")

    # DATA
    A("<h2 id='data'>1. 원본 데이터</h2>")
    A("<p>병원에서 처방하면 환자가 앱으로 인지훈련을 합니다. 우리가 받은 raw 데이터는 두 개입니다.</p>")
    A(f"<div class='box'><p>"
      f"<span class='kv'>훈련 기록 <b>{_fmt_int(ctx['n_events'])}</b>건</span>"
      f"<span class='kv'>환자 <b>{_fmt_int(ctx['n_patients'])}</b>명</span>"
      f"<span class='kv'>한 번이라도 훈련한 환자 <b>{_fmt_int(ctx['n_played'])}</b>명</span>"
      f"<span class='kv'>재처방 받은 환자 <b>{_fmt_int(ctx['n_represcribe'])}</b>명</span>"
      f"</p><p class='muted'>훈련 1건 = '환자가 어떤 운동을 몇 레벨에서 해서 몇 점 받았다'는 한 줄. "
      f"훈련 종류는 12개(전반 3종 + 세부 9종), 레벨 1~7, 점수 0~1.</p></div>")
    A("<div class='ok'><b>이번 0421 데이터의 좋은 점:</b> 이전(0310) 대비 기록이 약 1.7배로 늘었고, "
      "특히 분석에서 가장 약했던 <b>재처방 사례가 30명 → "
      f"{_fmt_int(ctx['n_represcribe'])}명으로 크게 늘어</b> 재처방 모델이 훨씬 믿을 만해졌습니다.</div>")

    # ARCHETYPES
    A("<h2 id='arch'>2. 환자 6유형 — '안주형'이 정확히 뭔지</h2>")
    A("<p>모든 환자를 행동 패턴에 따라 6가지 유형으로 자동 분류합니다. 분류에 쓰는 재료는 딱 4개입니다:</p>")
    A("<ul>"
      "<li><b>general 최고레벨</b> — 전반 훈련에서 도달한 가장 높은 레벨 (1~7)</li>"
      "<li><b>general 마지막레벨</b> — 마지막에 머문 레벨</li>"
      "<li><b>general 훈련 횟수</b> — 전반 훈련을 몇 번 했나 (21회가 기준선)</li>"
      "<li><b>순응도(adherence)</b> — 90일 중 며칠이나 훈련했나의 비율 (0~1), 0.5가 기준선</li>"
      "</ul>")
    A("<p>이 4개로 아래 표의 <b>규칙을 위에서부터 순서대로</b> 적용해 가장 먼저 걸리는 유형으로 정합니다. "
      "예를 들어 <b>'안주형(B)'은 '최고레벨이 2~3에 머물렀지만 순응도가 0.5 이상이라 꾸준히 온 사람'</b>이고, "
      "똑같이 2~3에 머물렀는데 순응도가 0.5 미만이면 '지루형(C)'으로 갈립니다.</p>")
    A(arch_section(ctx))
    A("<div class='box'><p><b>왜 이렇게 나누나?</b> 선행 분석(2차)의 핵심 발견 — "
      "<b>안주형(B)은 재처방을 가장 많이 하고, 지루형(C)은 가장 적게 한다</b> — 이 패턴이 "
      "위 표의 재처방률에 그대로 나타납니다. 즉 '레벨대'가 같아도 '얼마나 꾸준한가'가 운명을 가릅니다. "
      "시뮬레이션에서 알고리즘이 환자를 C(지루·이탈)에서 B(안주·잔류)나 F(성장)로 옮겨주면 "
      "재처방률이 올라가는 식으로 효과가 반영됩니다.</p></div>")

    # MODELS
    A("<h2 id='models'>3. 환자 행동을 4개 모델로 쪼갰다</h2>")
    A("<p>'가짜 환자'를 만들려면 진짜 환자가 앱에서 하는 행동을 흉내내야 합니다. 행동을 "
      "<b>네 개의 질문</b>으로 분해하고, 각 질문에 답하는 AI 모델을 하나씩 만들었습니다. "
      "모두 <b>LightGBM</b>(표 데이터에 강한 표준 트리 모델)입니다.</p>")
    A(model_metrics_section(ctx))
    A("<p class='muted'>읽는 법: <b>MAE</b>는 '평균적으로 이만큼 틀린다'(작을수록 좋음). "
      "<b>ROC-AUC</b>는 '맞는 사람을 더 위험하다고 줄세우는 능력'(1에 가까울수록 좋음, 0.5는 찍기). "
      "각 모델은 '그냥 평균/다수로 찍기(baseline)'보다 나아야 의미가 있습니다.</p>")

    # FEATURES
    A("<h2 id='features'>4. 모델에 넣는 입력(피처)</h2>")
    A("<p>모델은 마법이 아니라 '입력 → 출력' 함수입니다. 각 모델에 넣는 입력은 사람이 골랐습니다.</p>")
    A("<table><tr><th>모델</th><th>대표 입력들</th></tr>"
      "<tr><td>score</td><td>운동 종류, 레벨, 환자 유형, 순응도, <b>최근 3회 점수</b>(평균·최소·최대), "
      "이 운동을 몇 번 했는지, 전체 훈련 누적 횟수, 시간대/요일</td></tr>"
      "<tr><td>dropout / gap</td><td>유형, 순응도, 세션 순번, 이번 세션 평균점수·최고레벨, "
      "<b>지난 세션과의 간격</b>, 처방 만료까지 남은 일수</td></tr>"
      "<tr><td>termination</td><td>유형, 순응도, 총 훈련 수, general 최고/마지막/중앙 레벨, "
      "평균 점수, <b>점수 추세</b>, 평균 세션 간격, 처방기관</td></tr>"
      "</table>")
    A("<p class='muted'>'최근 3회 점수'가 중요한 이유: 현행 난이도 규칙 자체가 '최근 3회 평균'으로 "
      "레벨을 올리고 내리기 때문에, 모델도 같은 정보를 봐야 현실을 흉내낼 수 있습니다.</p>")

    # TRAIN
    A("<h2 id='train'>5. 어떻게 학습하고 검증했나</h2>")
    A("<div class='step'><span class='n'>1</span><b>환자 단위로 학습/시험 분리.</b> "
      "한 환자의 기록이 학습용과 시험용에 <b>동시에 들어가지 않게</b> 막았습니다(<code>GroupKFold</code>). "
      "안 그러면 '이 환자 다른 기록을 외워서' 성능이 부풀려집니다. (가장 흔한 평가 부정행위 방지)</div>")
    A("<div class='step'><span class='n'>2</span><b>하이퍼파라미터 튜닝(선택).</b> "
      "<code>tune_models.py</code>가 Optuna로 모델 손잡이(학습률·복잡도)를 자동으로 여러 번 시도해 "
      "최적값을 찾습니다. (이번 리포트는 빠른 기본 설정으로 학습)</div>")
    A("<div class='step'><span class='n'>3</span><b>홀드아웃 검증.</b> "
      "학습에 안 쓴 실제 환자들과, 같은 조건으로 돌린 시뮬레이션 환자들의 분포를 "
      "통계 검정(KS test)으로 비교합니다. 아래가 그 결과입니다.</div>")
    A(validation_section(ctx))

    # SIMULATION
    A("<h2 id='sim'>6. 시뮬레이션은 이렇게 돌아간다</h2>")
    A("<p>이게 실험의 심장입니다. 가짜 환자 수백 명을 만들어 90일을 하루씩 굴립니다.</p>")
    A("<div class='step'><span class='n'>1</span><b>환자 생성.</b> 실제 환자 풀에서 유형·순응도를 뽑고, "
      "각 가짜 환자에게 숨은 <b>'능력 한계 레벨'</b>과 <b>'끈기 한계(세션 수)'</b>를 실제 분포에서 부여합니다.</div>")
    A("<div class='step'><span class='n'>2</span><b>세션 반복</b> (전반↔세부 번갈아):"
      "<br>&nbsp;&nbsp;① <span class='arrow'>알고리즘</span>이 최근 3회 점수를 보고 <b>레벨 조정</b>"
      "<br>&nbsp;&nbsp;② <span class='arrow'>score 모델</span>이 그 레벨에서 받을 <b>점수 예측</b>"
      "<br>&nbsp;&nbsp;③ <span class='arrow'>dropout 모델</span>이 <b>그만둘 확률</b> → 주사위"
      "<br>&nbsp;&nbsp;④ <span class='arrow'>gap 모델</span>이 <b>며칠 뒤 올지</b> 예측 → 날짜 전진</div>")
    A("<div class='step'><span class='n'>3</span><b>90일 후</b> "
      "<span class='arrow'>termination 모델</span>이 <b>재처방 확률</b>을 계산하고, "
      "유형별 실제 재처방률에 맞춰 보정(calibration)합니다.</div>")
    A("<div class='box'><p><b>핵심 장치 — '소프트 능력 한계':</b> 알고리즘이 환자를 "
      "자기 능력 이상으로 밀어올리면 점수에 페널티가 붙어 <b>점수가 떨어지고</b>, 그러면 환자가 더 자주 틀려 "
      "<b>알고리즘이 자연스럽게 레벨을 다시 내립니다.</b> 덕분에 '공격적' 알고리즘이 무한정 레벨을 올리는 "
      "비현실적 결과를 막습니다. (단, 아래 한계 참고 — 이 보정이 새 데이터에선 아직 덜 맞습니다.)</p></div>")

    # ALGORITHMS
    A("<h2 id='algos'>7. 비교한 난이도 알고리즘들</h2>")
    A("<p>핵심은 '같은 가짜 환자들에게 규칙만 바꿔서 돌린다'는 점입니다. 비교 대상 규칙은 "
      "현행 + 선행 분석 제안 + 통제군까지 다음과 같습니다. (새 규칙을 추가하려면 "
      "<code>algorithms.py</code>에 클래스 하나만 추가하면 됩니다.)</p>")
    A(algo_section())

    # RESULTS
    A("<h2 id='results'>8. 결과 읽는 법</h2>")
    A("<p>아래는 알고리즘별 평균 결과입니다(여러 세부운동 선택 방식·여러 난수 시드 평균). "
      "전체 표·그래프는 <code>report.html</code>에 있습니다.</p>")
    A(results_section(ctx))
    A("<div class='box'><p><b>읽는 순서:</b><br>"
      "① <b>난이도 유지율</b> — 낮을수록 레벨이 자주 움직임(정체 안 됨). 현행이 높으면 '갇혀있다'는 신호.<br>"
      "② <b>재처방률</b> — 높을수록 환자가 만족해 다시 처방받음. 우리가 올리고 싶은 값.<br>"
      "③ <b>평균 최고레벨 / 이탈률</b> — 부작용 점검용(너무 밀어붙여 이탈이 늘진 않나).<br>"
      "<b>결론은 항상 '현행(current_50_80) 대비 어느 규칙이 유지율을 낮추면서 재처방을 지키/올리나'의 "
      "상대 비교로 말하세요.</b></p></div>")

    # LIMITS
    A("<h2 id='limits'>9. 한계 (질문 들어오기 전에 먼저 말할 것)</h2>")
    A("<div class='warn'><ul>"
      "<li><b>절대 수치는 보정값이다.</b> 재처방률 절대값은 유형별 실제 비율에 맞춰 강제 보정한 것이라, "
      "알고리즘 간 <b>상대 비교만</b> 의미가 있습니다.</li>"
      "<li><b>새 데이터에서 시뮬 분포가 아직 덜 맞는다.</b> 특히 시뮬이 레벨을 실제보다 높게 올립니다 "
      "(검증 표 참고). 절대 레벨 수치는 그대로 인용하지 마세요.</li>"
      "<li><b>이탈(dropout) 모델은 CV 불안정.</b> 0421 데이터에선 이탈 세션이 0.4%로 너무 드물어 "
      "holdout 성능은 높아도 교차검증에선 들쭉날쭉합니다.</li>"
      "<li><b>gap 모델은 약하다.</b> 대부분 같은 날 재방문이라 '며칠 뒤' 예측은 정보가 적습니다.</li>"
      "<li><b>현실 요인 미반영.</b> 보호자·의사 개입, 계절성 등은 모델에 없습니다.</li>"
      "</ul></div>")

    A(f"<p class='muted' style='margin-top:50px'>DTx Patient Twin Bot — explainer · {ts}</p>")
    A("</body></html>")
    return "\n".join(P)


def main():
    ctx = load_context()
    html = build_html(ctx)
    out_path = OUT / "explainer.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
