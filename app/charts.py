import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Theme Constants ───────────────────────────────────────────────────────────
NAVY_900  = "#0a0e1a"
NAVY_800  = "#0f1628"
NAVY_700  = "#151d36"
NAVY_600  = "#1c2844"
GOLD_400  = "#f0c040"
GOLD_300  = "#f5d070"
MUTED     = "#8892a4"
OFF_WHITE = "#f0ede8"
SUCCESS   = "#2dd4a0"
DANGER    = "#f06060"
WARNING   = "#f0a040"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=OFF_WHITE),
    margin=dict(l=20, r=20, t=40, b=20),
)


# ── Helper: bias severity color ───────────────────────────────────────────────
def _metric_color(value: float, metric: str) -> str:
    """Returns a color based on how biased a metric value is."""
    if metric == "disparate_impact":
        # ideal ~1.0; <0.8 is concerning
        if value >= 0.8:
            return SUCCESS
        elif value >= 0.5:
            return WARNING
        else:
            return DANGER
    else:
        # ideal = 0; further from 0 = worse
        abs_val = abs(value)
        if abs_val <= 0.05:
            return SUCCESS
        elif abs_val <= 0.15:
            return WARNING
        else:
            return DANGER


# ── Chart 1: Bar Chart — Bias Metrics Overview ────────────────────────────────
def render_bias_bar_chart(audit_result: dict):
    """
    Horizontal bar chart showing all 4 bias metrics side by side.
    Color-coded by severity.
    """
    metrics = {
    "Disparate Impact":         audit_result.get("disparate_impact", 0),
    "Stat. Parity Diff.":       abs(audit_result.get("statistical_parity_difference", 0)),
    "Equal Opportunity Diff.":  abs(audit_result.get("equal_opportunity_difference", 0)),
    "Demographic Parity Diff.": abs(audit_result.get("demographic_parity_difference", 0)),
    }

    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = [
        _metric_color(values[0], "disparate_impact"),
        _metric_color(values[1], "other"),
        _metric_color(values[2], "other"),
        _metric_color(values[3], "other"),
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(color="rgba(255,255,255,0.05)", width=1)
        ),
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(size=12, color=OFF_WHITE, family="DM Mono, monospace"),
        hovertemplate="<b>%{y}</b><br>Value: %{x:.4f}<extra></extra>",
    ))

    # Ideal reference line at 0
    fig.add_vline(
        x=0,
        line_dash="dot",
        line_color=GOLD_400,
        line_width=1,
        annotation_text="ideal",
        annotation_font_color=GOLD_400,
        annotation_font_size=10,
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="Bias Metrics Overview",
            font=dict(size=14, color=GOLD_300, family="DM Mono, monospace"),
            x=0
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
            tickfont=dict(size=11, color=MUTED),
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(size=11, color=OFF_WHITE),
        ),
        height=350,
        bargap=0.35,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Chart 2: Gauge — Disparate Impact ────────────────────────────────────────
def render_disparate_impact_gauge(audit_result: dict):
    """
    Gauge chart showing Disparate Impact score.
    Green zone: 0.8–1.2 (fair), Red zone: outside that.
    """
    value = audit_result.get("disparate_impact", 0)
    color = _metric_color(value, "disparate_impact")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={
            "reference": 1.0,
            "increasing": {"color": SUCCESS},
            "decreasing": {"color": DANGER},
            "font": {"size": 13, "color": MUTED}
        },
        number={
            "font": {"size": 38, "color": OFF_WHITE, "family": "Playfair Display, serif"},
            "suffix": ""
        },
        gauge={
            "axis": {
                "range": [0, 2],
                "tickwidth": 1,
                "tickcolor": MUTED,
                "tickfont": {"size": 10, "color": MUTED},
                "nticks": 5,
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": NAVY_600,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 0.5],    "color": "rgba(240,96,96,0.15)"},
                {"range": [0.5, 0.8],  "color": "rgba(240,160,64,0.12)"},
                {"range": [0.8, 1.2],  "color": "rgba(45,212,160,0.12)"},
                {"range": [1.2, 2.0],  "color": "rgba(240,160,64,0.10)"},
            ],
            "threshold": {
                "line": {"color": GOLD_400, "width": 2},
                "thickness": 0.75,
                "value": 1.0,
            },
        },
        title={
            "text": "DISPARATE IMPACT",
            "font": {"size": 11, "color": GOLD_300, "family": "DM Mono, monospace"},
        },
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Legend below gauge
    st.markdown(
        """
        <div style="display:flex; gap:1.5rem; justify-content:center;
                    font-size:0.68rem; font-family:'DM Mono',monospace;
                    color:#8892a4; margin-top:-1rem;">
            <span style="color:#f06060">■ Biased (&lt;0.5)</span>
            <span style="color:#f0a040">■ Concerning (0.5–0.8)</span>
            <span style="color:#2dd4a0">■ Fair (0.8–1.2)</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ── Chart 3: Before vs After Accuracy ────────────────────────────────────────
def render_before_after_chart(audit_result: dict):
    """
    Grouped bar chart comparing accuracy and demographic parity
    before and after bias mitigation.
    """
    before_acc = round(audit_result.get("before_accuracy", 0) * 100, 2)
    after_acc  = round(audit_result.get("after_accuracy", 0) * 100, 2)
    before_dpd = round(abs(audit_result.get("before_demographic_parity", 0)), 4)
    after_dpd  = round(abs(audit_result.get("after_demographic_parity", 0)), 4)

    fig = go.Figure()

    # Accuracy bars
    fig.add_trace(go.Bar(
        name="Accuracy (%)",
        x=["Before Mitigation", "After Mitigation"],
        y=[before_acc, after_acc],
        marker_color=[DANGER, SUCCESS],
        marker_opacity=0.85,
        text=[f"{before_acc}%", f"{after_acc}%"],
        textposition="outside",
        textfont=dict(size=12, color=OFF_WHITE, family="DM Mono, monospace"),
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>",
        yaxis="y1",
    ))

    # Demographic parity line
    fig.add_trace(go.Scatter(
        name="Demographic Parity Diff.",
        x=["Before Mitigation", "After Mitigation"],
        y=[before_dpd, after_dpd],
        mode="lines+markers+text",
        line=dict(color=GOLD_400, width=2, dash="dot"),
        marker=dict(size=8, color=GOLD_400, symbol="diamond"),
        text=[f"{before_dpd}", f"{after_dpd}"],
        textposition="top center",
        textfont=dict(size=11, color=GOLD_300, family="DM Mono, monospace"),
        hovertemplate="<b>%{x}</b><br>DPD: %{y:.4f}<extra></extra>",
        yaxis="y2",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="Before vs After Mitigation",
            font=dict(size=14, color=GOLD_300, family="DM Mono, monospace"),
            x=0
        ),
        xaxis=dict(
            tickfont=dict(size=12, color=OFF_WHITE),
            gridcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            title=dict(text="Accuracy (%)", font=dict(size=11, color=MUTED)),
            tickfont=dict(size=11, color=MUTED),
            gridcolor="rgba(255,255,255,0.05)",
            range=[max(0, min(before_acc, after_acc) - 5), 100],
        ),
        yaxis2=dict(
            title=dict(text="Demographic Parity Diff.", font=dict(size=11, color=GOLD_300)),
            tickfont=dict(size=11, color=GOLD_300),
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            range=[0, max(before_dpd, after_dpd) * 2.5],
        ),
        legend=dict(
            font=dict(size=11, color=OFF_WHITE),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=-0.15,
        ),
        height=320,
        bargap=0.4,
        barmode="group",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Main Render Function ──────────────────────────────────────────────────────
def render_charts_section(audit_result: dict):
    """
    Call this from main.py after audit completes.
    Renders all charts in a clean spacious layout.
    """
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="font-family:'DM Mono',monospace; font-size:0.68rem;
                    letter-spacing:0.2em; color:#f0c040; text-transform:uppercase;
                    margin-bottom:1.5rem; padding-bottom:0.6rem;
                    border-bottom:1px solid rgba(240,192,64,0.15);">
            Visual Analysis
        </div>
        """,
        unsafe_allow_html=True
    )

    # Row 1 — Gauge full width centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_disparate_impact_gauge(audit_result)

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    # Row 2 — Bias metrics bar chart full width
    render_bias_bar_chart(audit_result)

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    # Row 3 — Before vs After full width
    render_before_after_chart(audit_result)

if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Charts Test", layout="wide")
    
    dummy_result = {
        "dataset": "adult",
        "protected_attribute": "sex",
        "disparate_impact": 0.3635,
        "statistical_parity_difference": -0.1989,
        "equal_opportunity_difference": 0.0221,
        "demographic_parity_difference": 0.0481,
        "before_accuracy": 0.7914,
        "after_accuracy": 0.7886,
        "before_demographic_parity": 0.0481,
        "after_demographic_parity": 0.0002
    }
    
    render_charts_section(dummy_result)