"""Plotly charts used by the Streamlit pages."""
import pandas as pd
import plotly.graph_objects as go

from config import LEVELS, PASS_MARK
from lesson_content import TOPICS

LEVEL_COLORS = {"Weak": "#EF6B6B", "Average": "#F2B84B", "Strong": "#46D2A0"}
RISK_COLORS = {"Low": "#46D2A0", "Medium": "#F2B84B", "High": "#EF6B6B"}
ACCENT = "#7C9CFF"


def _base(fig, height=320):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#E8EBF5"), showlegend=False)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    return fig


def topic_scores(results: pd.DataFrame):
    avg = results.groupby("topic")["score"].mean().reindex(TOPICS)
    fig = go.Figure(go.Bar(x=avg.index, y=avg.values, marker_color=ACCENT,
                           text=[f"{v:.0f}%" if pd.notna(v) else "" for v in avg.values],
                           textposition="outside"))
    fig.add_hline(y=PASS_MARK, line_dash="dash", line_color="rgba(255,255,255,0.35)",
                  annotation_text=f"pass mark {PASS_MARK}%", annotation_position="top left")
    fig.update_yaxes(range=[0, 115], title="Average score (%)")
    fig.update_layout(title="Performance by topic")
    return _base(fig)


def score_progress(results: pd.DataFrame):
    n = list(range(1, len(results) + 1))
    labels = [f"{t} ({d})" for t, d in zip(results["topic"], results["difficulty"])]
    fig = go.Figure(go.Scatter(x=n, y=results["score"], mode="lines+markers", text=labels,
                               line=dict(color=ACCENT, width=3), marker=dict(size=9),
                               hovertemplate="Quiz %{x}: %{text}<br>Score %{y}%<extra></extra>"))
    fig.add_hline(y=PASS_MARK, line_dash="dash", line_color="rgba(255,255,255,0.35)")
    fig.update_yaxes(range=[0, 105], title="Score (%)")
    fig.update_xaxes(title="Quiz number", dtick=1)
    fig.update_layout(title="Score progress")
    return _base(fig)


def probabilities(probs: dict):
    fig = go.Figure(go.Bar(x=[probs[l] * 100 for l in LEVELS], y=LEVELS, orientation="h",
                           marker_color=[LEVEL_COLORS[l] for l in LEVELS],
                           text=[f"{probs[l] * 100:.0f}%" for l in LEVELS], textposition="outside"))
    fig.update_xaxes(range=[0, 118], title="Probability (%)")
    fig.update_layout(title="ANN output (Softmax probabilities)")
    return _base(fig, 250)


def factors(factors: dict):
    """Normalised input factors: positive = better than the typical trainee."""
    items = sorted(factors.items(), key=lambda kv: kv[1])
    vals = [max(min(v, 3), -3) for _, v in items]
    fig = go.Figure(go.Bar(x=vals, y=[k for k, _ in items], orientation="h",
                           marker_color=[LEVEL_COLORS["Strong"] if v >= 0 else LEVEL_COLORS["Weak"] for v in vals]))
    fig.update_xaxes(range=[-3, 3], title="Below typical  <  0  >  above typical")
    fig.update_layout(title="Normalised input factors")
    return _base(fig, 340)


def risk_gauge(prob_weak: float, risk: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob_weak * 100, number=dict(suffix="%", font=dict(size=38)),
        title=dict(text=f"Risk of falling behind: {risk}", font=dict(size=16)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=RISK_COLORS[risk]),
                   bgcolor="rgba(255,255,255,0.05)", borderwidth=0,
                   steps=[dict(range=[0, 25], color="rgba(70,210,160,0.20)"),
                          dict(range=[25, 50], color="rgba(242,184,75,0.20)"),
                          dict(range=[50, 100], color="rgba(239,107,107,0.20)")])))
    return _base(fig, 260)


def level_counts(counts: dict):
    fig = go.Figure(go.Bar(x=LEVELS, y=[counts.get(l, 0) for l in LEVELS],
                           marker_color=[LEVEL_COLORS[l] for l in LEVELS],
                           text=[counts.get(l, 0) for l in LEVELS], textposition="outside"))
    fig.update_yaxes(title="Trainees", dtick=1)
    fig.update_layout(title="Trainees by predicted level")
    return _base(fig, 300)
