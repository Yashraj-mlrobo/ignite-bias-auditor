import streamlit as st
from upload import render_upload_section
from charts import render_charts_section
from report import render_report_section

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IGNITE — Bias Auditor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
def load_css(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

# ── Charts Section (only after audit runs) ────────────────────────────────────
if "audit_result" in st.session_state:
    render_charts_section(st.session_state["audit_result"])
    render_report_section()