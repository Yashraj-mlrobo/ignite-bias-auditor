import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from app.upload import render_upload_section
from app.charts import render_charts_section
from app.report import render_report_section
from app.gemini import get_ai_summary

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IGNITE — Bias Auditor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
def load_css(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("assets/style.css")

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom: 0.2rem;">
        <span style="font-family: 'DM Mono', monospace; font-size: 0.68rem;
                     letter-spacing: 0.2em; color: #f0c040; text-transform: uppercase;">
            Google Solution Challenge 2026 · Team IGNITE
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("IGNITE Bias Auditor")

st.markdown(
    """
    <p style="color: #8892a4; font-size: 1rem; font-weight: 300;
              margin-top: -0.5rem; margin-bottom: 2rem;">
        Detect, explain, and fix bias in your machine learning models.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────────────────────────
render_upload_section()

# ── Results Section (only shown after audit runs) ─────────────────────────────
if "audit_result" in st.session_state:
    result = st.session_state["audit_result"]

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    render_charts_section(result)

    # ── AI Ethics Report ──────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-family:'DM Mono',monospace; font-size:0.68rem;
                    letter-spacing:0.2em; color:#f0c040; text-transform:uppercase;
                    margin-bottom:1rem; padding-bottom:0.6rem;
                    border-bottom:1px solid rgba(240,192,64,0.15);">
            AI Ethics Report
        </div>
        """,
        unsafe_allow_html=True
    )

    # Generate and cache AI report in session state
    if "ai_report" not in st.session_state:
        with st.spinner("Generating AI ethics report via Gemini..."):
            ai_report = get_ai_summary(result)
            st.session_state["ai_report"] = ai_report
    else:
        ai_report = st.session_state["ai_report"]

    st.markdown(
        f"""
        <div style="background:rgba(240,192,64,0.05);
                    border:1px solid rgba(240,192,64,0.2);
                    border-radius:12px; padding:1.5rem 2rem;
                    font-size:0.9rem; line-height:1.8;
                    color:#f0ede8;">
            {ai_report}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── PDF Export ────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    render_report_section()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center; padding:1rem 0 2rem;
                    font-family:'DM Mono',monospace; font-size:0.65rem;
                    letter-spacing:0.15em; color:#8892a4;">
            IGNITE BIAS AUDITOR · GOOGLE SOLUTION CHALLENGE 2026 · TEAM IGNITE · AMITY UNIVERSITY MUMBAI
        </div>
        """,
        unsafe_allow_html=True
    )