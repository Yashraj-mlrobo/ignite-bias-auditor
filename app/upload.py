import streamlit as st
import pandas as pd
from audit import run_full_audit

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 50

ADULT_COLUMNS = {
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
}

COMPAS_COLUMNS = {
    "sex", "age", "race", "juv_fel_count", "juv_misd_count",
    "priors_count", "is_recid", "two_year_recid", "decile_score"
}


# ── Dataset Detection ─────────────────────────────────────────────────────────
def detect_dataset_type(df: pd.DataFrame) -> str:
    """
    Detects whether uploaded CSV is 'adult' or 'compas' based on column overlap.
    Falls back to 'adult' if unclear.
    """
    cols = set(df.columns.str.strip().str.lower())
    adult_score = len(cols & {c.lower() for c in ADULT_COLUMNS})
    compas_score = len(cols & {c.lower() for c in COMPAS_COLUMNS})
    return "compas" if compas_score > adult_score else "adult"


# ── Validation ────────────────────────────────────────────────────────────────
def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """
    Sanity checks on uploaded data.
    Returns list of warning strings (empty = all good).
    """
    issues = []
    if df.empty:
        issues.append("The uploaded file is empty.")
    if df.shape[1] < 2:
        issues.append("File has fewer than 2 columns — the model needs more features.")
    if df.isnull().values.any():
        null_cols = df.columns[df.isnull().any()].tolist()
        issues.append(f"Missing values found in: {', '.join(null_cols)}")
    return issues


# ── Backend Bridge ────────────────────────────────────────────────────────────
def run_audit_for_dataset(df: pd.DataFrame, dataset_type: str) -> dict:
    """
    Calls Yash's audit.py run_full_audit() with detected dataset type.
    Returns the full result dictionary.
    """
    try:
        result = run_full_audit(dataset=dataset_type)
        result["status"] = "ok"
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ── Upload UI ─────────────────────────────────────────────────────────────────
def render_upload_section():
    """
    Main upload UI component. Call this from main.py.

    Stores in st.session_state:
        uploaded_df        → pd.DataFrame
        uploaded_filename  → str
        dataset_type       → 'adult' | 'compas'
        audit_result       → dict from run_full_audit()
    """

    # ── Section header
    st.markdown(
        """
        <div class="upload-header">
            <span class="upload-label">DATASET UPLOAD</span>
            <p class="upload-sub">Upload your CSV to begin the bias audit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── File uploader
    uploaded_file = st.file_uploader(
        label="Upload CSV",
        type=["csv"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    # Clear stale state when file removed
    if uploaded_file is None:
        for key in ("uploaded_df", "uploaded_filename", "dataset_type", "audit_result"):
            st.session_state.pop(key, None)
        _render_drop_hint()
        return

    # ── Size guard
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        st.error(f"⚠ File too large ({size_mb:.1f} MB). Maximum allowed is {MAX_FILE_SIZE_MB} MB.")
        return

    # ── Parse CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return

    # ── Validate
    issues = validate_dataframe(df)
    for issue in issues:
        st.warning(issue)

    # ── Detect dataset type
    dataset_type = detect_dataset_type(df)

    # ── File info bar
    st.markdown(
        f"""
        <div class="file-info-bar">
            <div class="file-info-item">
                <span class="info-label">FILE</span>
                <span class="info-value">{uploaded_file.name}</span>
            </div>
            <div class="file-info-item">
                <span class="info-label">ROWS</span>
                <span class="info-value">{df.shape[0]:,}</span>
            </div>
            <div class="file-info-item">
                <span class="info-label">COLUMNS</span>
                <span class="info-value">{df.shape[1]}</span>
            </div>
            <div class="file-info-item">
                <span class="info-label">DETECTED AS</span>
                <span class="info-value dataset-tag">{dataset_type.upper()}</span>
            </div>
            <div class="file-info-item">
                <span class="info-label">SIZE</span>
                <span class="info-value">{size_mb:.2f} MB</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Data preview
    with st.expander("Preview data", expanded=True):
        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=False
        )

    # ── Run audit button
    st.markdown("<div style='height: 1.2rem'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_clicked = st.button(
            "▶  RUN BIAS AUDIT",
            type="primary",
            use_container_width=True,
        )

    if run_clicked:
        with st.spinner("Running full bias audit…"):
            result = run_audit_for_dataset(df, dataset_type)

        if result.get("status") == "ok":
            # Persist everything in session
            st.session_state["uploaded_df"] = df
            st.session_state["uploaded_filename"] = uploaded_file.name
            st.session_state["dataset_type"] = dataset_type
            st.session_state["audit_result"] = result

            st.success("✓ Audit complete — scroll down to view results.")
            _render_audit_summary(result)
        else:
            st.error(f"Audit failed: {result.get('message', 'Unknown error')}")


# ── Audit Summary Card ────────────────────────────────────────────────────────
def _render_audit_summary(result: dict):
    """Renders a clean summary of key audit metrics after analysis."""

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="audit-summary-title">AUDIT SUMMARY</div>', unsafe_allow_html=True)

    metrics = [
        ("Disparate Impact",           result.get("disparate_impact", "—"),                "ideal ≈ 1.0"),
        ("Stat. Parity Diff.",         result.get("statistical_parity_difference", "—"),   "ideal = 0"),
        ("Equal Opportunity Diff.",    result.get("equal_opportunity_difference", "—"),    "ideal = 0"),
        ("Demographic Parity Diff.",   result.get("demographic_parity_difference", "—"),   "ideal = 0"),
    ]

    cols = st.columns(4)
    for col, (label, value, note) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Before / After accuracy
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            <div class="metric-card accent-before">
                <div class="metric-label">ACCURACY — BEFORE MITIGATION</div>
                <div class="metric-value">{round(result.get('before_accuracy', 0) * 100, 2)}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_b:
        st.markdown(
            f"""
            <div class="metric-card accent-after">
                <div class="metric-label">ACCURACY — AFTER MITIGATION</div>
                <div class="metric-value">{round(result.get('after_accuracy', 0) * 100, 2)}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ── Drop Hint ─────────────────────────────────────────────────────────────────
def _render_drop_hint():
    """Placeholder shown before any file is uploaded."""
    st.markdown(
        """
        <div class="drop-hint">
            <div class="drop-icon">↑</div>
            <div class="drop-text">Drop your CSV file above to begin</div>
            <div class="drop-sub">Supports Adult Income &amp; COMPAS datasets</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(page_title="IGNITE — Upload", layout="wide")
    render_upload_section()