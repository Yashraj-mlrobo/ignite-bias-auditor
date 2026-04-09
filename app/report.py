import io
from datetime import datetime
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Color Palette ─────────────────────────────────────────────────────────────
NAVY      = colors.HexColor("#0a0e1a")
NAVY_700  = colors.HexColor("#151d36")
NAVY_600  = colors.HexColor("#1c2844")
GOLD      = colors.HexColor("#f0c040")
GOLD_DARK = colors.HexColor("#c9960c")
OFF_WHITE = colors.HexColor("#f0ede8")
MUTED     = colors.HexColor("#8892a4")
SUCCESS   = colors.HexColor("#2dd4a0")
DANGER    = colors.HexColor("#f06060")
WARNING   = colors.HexColor("#f0a040")
WHITE     = colors.white

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm


# ── Custom Styles ─────────────────────────────────────────────────────────────
def _styles():
    return {
        "cover_tag": ParagraphStyle(
            "cover_tag",
            fontName="Helvetica",
            fontSize=7,
            textColor=GOLD,
            spaceAfter=4,
            spaceBefore=0,
            leading=10,
            wordWrap="LTR",
        ),
        "cover_title": ParagraphStyle(
            "cover_title",
            fontName="Helvetica-Bold",
            fontSize=28,
            textColor=WHITE,
            spaceAfter=6,
            leading=34,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            fontName="Helvetica",
            fontSize=11,
            textColor=MUTED,
            spaceAfter=4,
            leading=16,
        ),
        "section_label": ParagraphStyle(
            "section_label",
            fontName="Helvetica",
            fontSize=7,
            textColor=GOLD,
            spaceBefore=18,
            spaceAfter=6,
            leading=10,
            wordWrap="LTR",
        ),
        "section_title": ParagraphStyle(
            "section_title",
            fontName="Helvetica-Bold",
            fontSize=15,
            textColor=OFF_WHITE,
            spaceAfter=10,
            leading=20,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9,
            textColor=MUTED,
            spaceAfter=6,
            leading=14,
        ),
        "metric_label": ParagraphStyle(
            "metric_label",
            fontName="Helvetica",
            fontSize=7,
            textColor=MUTED,
            spaceAfter=2,
            leading=10,
            wordWrap="LTR",
        ),
        "metric_value": ParagraphStyle(
            "metric_value",
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=WHITE,
            spaceAfter=2,
            leading=22,
        ),
        "metric_note": ParagraphStyle(
            "metric_note",
            fontName="Helvetica-Oblique",
            fontSize=7,
            textColor=MUTED,
            leading=10,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            fontName="Helvetica-Bold",
            fontSize=7,
            textColor=GOLD,
            leading=10,
            wordWrap="LTR",
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            fontName="Helvetica",
            fontSize=8,
            textColor=OFF_WHITE,
            leading=11,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica",
            fontSize=7,
            textColor=MUTED,
            alignment=TA_CENTER,
            leading=10,
        ),
    }


# ── Bias Severity Helper ──────────────────────────────────────────────────────
def _bias_color(value: float, metric: str) -> colors.Color:
    if metric == "disparate_impact":
        if value >= 0.8: return SUCCESS
        elif value >= 0.5: return WARNING
        else: return DANGER
    else:
        abs_val = abs(value)
        if abs_val <= 0.05: return SUCCESS
        elif abs_val <= 0.15: return WARNING
        else: return DANGER


def _bias_label(value: float, metric: str) -> str:
    if metric == "disparate_impact":
        if value >= 0.8: return "FAIR"
        elif value >= 0.5: return "CONCERNING"
        else: return "BIASED"
    else:
        abs_val = abs(value)
        if abs_val <= 0.05: return "FAIR"
        elif abs_val <= 0.15: return "CONCERNING"
        else: return "BIASED"


# ── Page Background Canvas ────────────────────────────────────────────────────
def _dark_page(canvas, doc):
    """Draws dark navy background and gold footer line on every page."""
    canvas.saveState()
    # Background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # Gold top bar
    canvas.setFillColor(GOLD)
    canvas.rect(0, PAGE_H - 3, PAGE_W, 3, fill=1, stroke=0)
    # Footer line
    canvas.setStrokeColor(colors.HexColor("#1c2844"))
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 18 * mm, PAGE_W - MARGIN, 18 * mm)
    # Footer text
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 12 * mm, "IGNITE Bias Auditor  |  Google Solution Challenge 2026  |  Team IGNITE")
    canvas.drawRightString(
        PAGE_W - MARGIN, 12 * mm,
        f"Page {doc.page}  |  Generated {datetime.now().strftime('%d %b %Y')}"
    )
    canvas.restoreState()


# ── Section Divider ───────────────────────────────────────────────────────────
def _divider():
    return HRFlowable(
        width="100%",
        thickness=0.5,
        color=colors.HexColor("#1c2844"),
        spaceAfter=10,
        spaceBefore=4,
    )


# ── Metric Card Row ───────────────────────────────────────────────────────────
def _metric_card_table(metrics: list[tuple], styles: dict) -> Table:
    """
    Creates a row of metric cards as a ReportLab Table.
    metrics: list of (label, value, note, metric_key)
    """
    header_row = []
    value_row  = []
    note_row   = []
    status_row = []

    for label, value, note, key in metrics:
        c = _bias_color(value, key)
        header_row.append(Paragraph(label.upper(), styles["metric_label"]))
        value_row.append(Paragraph(f"{value}", styles["metric_value"]))
        note_row.append(Paragraph(note, styles["metric_note"]))
        status_row.append(Paragraph(_bias_label(value, key), ParagraphStyle(
            "status", fontName="Helvetica-Bold", fontSize=7,
            textColor=c, leading=10
        )))

    col_w = (PAGE_W - 2 * MARGIN) / len(metrics)
    t = Table(
        [header_row, value_row, note_row, status_row],
        colWidths=[col_w] * len(metrics),
        rowHeights=[12, 26, 12, 14]
    )
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY_700),
        ("ROWBACKGROUND", (0, 0), (-1, -1), NAVY_700),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#1c2844")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#1c2844")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ── Main Report Generator ─────────────────────────────────────────────────────
def generate_pdf_report(
    audit_result: dict,
    filename: str = "unknown.csv",
    row_count: int = 0,
    col_count: int = 0,
) -> bytes:
    """
    Generates a full PDF audit report.
    Returns PDF as bytes for Streamlit download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN + 6 * mm,
        bottomMargin=26 * mm,
    )

    s = _styles()
    story = []
    now = datetime.now().strftime("%d %B %Y, %H:%M")
    dataset = audit_result.get("dataset", "unknown").upper()
    protected = audit_result.get("protected_attribute", "—")

    # ── COVER SECTION ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("GOOGLE SOLUTION CHALLENGE 2026  ·  TEAM IGNITE", s["cover_tag"]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("IGNITE Bias Auditor", s["cover_title"]))
    story.append(Paragraph("AI Fairness Audit Report", s["cover_sub"]))
    story.append(Spacer(1, 4 * mm))

    # Cover meta table
    cover_data = [
        [
            Paragraph("DATASET", s["metric_label"]),
            Paragraph("PROTECTED ATTRIBUTE", s["metric_label"]),
            Paragraph("GENERATED ON", s["metric_label"]),
            Paragraph("FILE", s["metric_label"]),
        ],
        [
            Paragraph(dataset, s["table_cell"]),
            Paragraph(protected.upper(), s["table_cell"]),
            Paragraph(now, s["table_cell"]),
            Paragraph(filename, s["table_cell"]),
        ],
    ]
    col_w = (PAGE_W - 2 * MARGIN) / 4
    cover_table = Table(cover_data, colWidths=[col_w] * 4, rowHeights=[14, 18])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY_700),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#1c2844")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#1c2844")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 8 * mm))
    story.append(_divider())

    # ── SECTION 1: DATASET INFO ───────────────────────────────────────────────
    story.append(Paragraph("01  ·  DATASET OVERVIEW", s["section_label"]))
    story.append(Paragraph("Dataset Summary", s["section_title"]))

    ds_data = [
        [
            Paragraph("ROWS", s["metric_label"]),
            Paragraph("COLUMNS", s["metric_label"]),
            Paragraph("DATASET TYPE", s["metric_label"]),
            Paragraph("PROTECTED ATTRIBUTE", s["metric_label"]),
        ],
        [
            Paragraph(f"{row_count:,}", s["metric_value"]),
            Paragraph(str(col_count), s["metric_value"]),
            Paragraph(dataset, s["metric_value"]),
            Paragraph(protected.upper(), s["metric_value"]),
        ],
    ]
    col_w = (PAGE_W - 2 * MARGIN) / 4
    ds_table = Table(ds_data, colWidths=[col_w] * 4, rowHeights=[14, 30])
    ds_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY_700),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#1c2844")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#1c2844")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(ds_table)
    story.append(Spacer(1, 6 * mm))
    story.append(_divider())

    # ── SECTION 2: BIAS METRICS ───────────────────────────────────────────────
    story.append(Paragraph("02  ·  BIAS METRICS", s["section_label"]))
    story.append(Paragraph("Fairness Analysis", s["section_title"]))
    story.append(Paragraph(
        "The following metrics were computed using IBM AIF360 and Microsoft Fairlearn. "
        "Values are compared against ideal thresholds to determine fairness.",
        s["body"]
    ))
    story.append(Spacer(1, 3 * mm))

    metrics_data = [
        ("Disparate Impact",       round(audit_result.get("disparate_impact", 0), 4),          "ideal ~ 1.0",  "disparate_impact"),
        ("Stat. Parity Diff.",     round(audit_result.get("statistical_parity_difference", 0), 4), "ideal = 0", "other"),
        ("Equal Opp. Diff.",       round(audit_result.get("equal_opportunity_difference", 0), 4),  "ideal = 0", "other"),
        ("Demographic Parity Diff.", round(audit_result.get("demographic_parity_difference", 0), 4), "ideal = 0", "other"),
    ]
    story.append(_metric_card_table(metrics_data, s))
    story.append(Spacer(1, 6 * mm))

    # Metrics explanation table
    story.append(Paragraph("METRIC DEFINITIONS", s["section_label"]))
    definitions = [
        [Paragraph("METRIC", s["table_header"]),       Paragraph("DEFINITION", s["table_header"]),            Paragraph("IDEAL", s["table_header"])],
        [Paragraph("Disparate Impact", s["table_cell"]),        Paragraph("Ratio of positive outcomes for unprivileged vs privileged group", s["table_cell"]),  Paragraph("~ 1.0", s["table_cell"])],
        [Paragraph("Stat. Parity Diff.", s["table_cell"]),      Paragraph("Difference in positive prediction rates between groups", s["table_cell"]),            Paragraph("0", s["table_cell"])],
        [Paragraph("Equal Opp. Diff.", s["table_cell"]),        Paragraph("Difference in true positive rates between groups", s["table_cell"]),                  Paragraph("0", s["table_cell"])],
        [Paragraph("Demographic Parity Diff.", s["table_cell"]), Paragraph("Difference in predicted positive rates across sensitive groups", s["table_cell"]),   Paragraph("0", s["table_cell"])],
    ]
    col_ws = [(PAGE_W - 2 * MARGIN) * r for r in [0.28, 0.52, 0.2]]
    def_table = Table(definitions, colWidths=col_ws)
    def_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY_600),
        ("BACKGROUND",    (0, 1), (-1, -1), NAVY_700),
        ("ROWBACKGROUND", (0, 2), (-1, -1), NAVY),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#1c2844")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#1c2844")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(def_table)
    story.append(Spacer(1, 6 * mm))
    story.append(_divider())

    # ── SECTION 3: BEFORE / AFTER MITIGATION ─────────────────────────────────
    story.append(Paragraph("03  ·  MITIGATION RESULTS", s["section_label"]))
    story.append(Paragraph("Before vs After Bias Mitigation", s["section_title"]))
    story.append(Paragraph(
        "Fairlearn ThresholdOptimizer was applied with demographic parity constraints. "
        "The table below shows the impact on model accuracy and fairness.",
        s["body"]
    ))
    story.append(Spacer(1, 3 * mm))

    before_acc  = round(audit_result.get("before_accuracy", 0) * 100, 2)
    after_acc   = round(audit_result.get("after_accuracy", 0) * 100, 2)
    before_dpd  = round(abs(audit_result.get("before_demographic_parity", 0)), 4)
    after_dpd   = round(abs(audit_result.get("after_demographic_parity", 0)), 4)
    acc_delta   = round(after_acc - before_acc, 2)
    dpd_delta   = round(after_dpd - before_dpd, 4)

    mitigation_data = [
        [
            Paragraph("METRIC", s["table_header"]),
            Paragraph("BEFORE MITIGATION", s["table_header"]),
            Paragraph("AFTER MITIGATION", s["table_header"]),
            Paragraph("CHANGE", s["table_header"]),
        ],
        [
            Paragraph("Model Accuracy", s["table_cell"]),
            Paragraph(f"{before_acc}%", s["table_cell"]),
            Paragraph(f"{after_acc}%", s["table_cell"]),
            Paragraph(f"{'+' if acc_delta >= 0 else ''}{acc_delta}%", ParagraphStyle(
                "delta", fontName="Helvetica-Bold", fontSize=8,
                textColor=SUCCESS if acc_delta >= 0 else DANGER, leading=11
            )),
        ],
        [
            Paragraph("Demographic Parity Diff.", s["table_cell"]),
            Paragraph(str(before_dpd), s["table_cell"]),
            Paragraph(str(after_dpd), s["table_cell"]),
            Paragraph(f"{'+' if dpd_delta >= 0 else ''}{dpd_delta}", ParagraphStyle(
                "delta2", fontName="Helvetica-Bold", fontSize=8,
                textColor=SUCCESS if dpd_delta <= 0 else DANGER, leading=11
            )),
        ],
    ]
    col_ws2 = [(PAGE_W - 2 * MARGIN) * r for r in [0.34, 0.22, 0.22, 0.22]]
    mit_table = Table(mitigation_data, colWidths=col_ws2, rowHeights=[18, 22, 22])
    mit_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY_600),
        ("BACKGROUND",    (0, 1), (-1, -1), NAVY_700),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#1c2844")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#1c2844")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(mit_table)
    story.append(Spacer(1, 6 * mm))
    story.append(_divider())

    # ── SECTION 4: CONCLUSION ─────────────────────────────────────────────────
    story.append(Paragraph("04  ·  CONCLUSION", s["section_label"]))
    story.append(Paragraph("Audit Findings", s["section_title"]))

    di = audit_result.get("disparate_impact", 0)
    if di < 0.5:
        finding = (
            "The model exhibits significant bias against the unprivileged group. "
            "A Disparate Impact score below 0.5 indicates the unprivileged group "
            "receives favorable outcomes at less than half the rate of the privileged group. "
            "Immediate remediation is strongly recommended before deployment."
        )
    elif di < 0.8:
        finding = (
            "The model shows moderate bias. The Disparate Impact score falls below the "
            "commonly accepted 0.8 threshold (the '4/5ths rule'). While mitigation has "
            "been applied, further review of the training data and feature selection is advised."
        )
    else:
        finding = (
            "The model meets basic fairness thresholds. The Disparate Impact score is "
            "within acceptable range. Continued monitoring is recommended as the model "
            "is deployed to ensure fairness is maintained over time."
        )

    story.append(Paragraph(finding, s["body"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "This report was generated automatically by IGNITE Bias Auditor using IBM AIF360 "
        "and Microsoft Fairlearn. Results should be reviewed by a qualified ML fairness "
        "practitioner before making deployment decisions.",
        ParagraphStyle("disclaimer", fontName="Helvetica-Oblique", fontSize=8,
                       textColor=MUTED, leading=12)
    ))

    # ── BUILD ─────────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_dark_page, onLaterPages=_dark_page)
    return buffer.getvalue()


# ── Streamlit Render Function ─────────────────────────────────────────────────
def render_report_section():
    """
    Call this from main.py after audit and charts.
    Shows a download button for the PDF report.
    """
    if "audit_result" not in st.session_state:
        return

    audit_result = st.session_state["audit_result"]
    filename     = st.session_state.get("uploaded_filename", "dataset.csv")
    df           = st.session_state.get("uploaded_df")
    row_count    = df.shape[0] if df is not None else 0
    col_count    = df.shape[1] if df is not None else 0

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-family:'DM Mono',monospace; font-size:0.68rem;
                    letter-spacing:0.2em; color:#f0c040; text-transform:uppercase;
                    margin-bottom:1rem; padding-bottom:0.6rem;
                    border-bottom:1px solid rgba(240,192,64,0.15);">
            Export Report
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style="color:#8892a4; font-size:0.9rem; font-weight:300; margin-bottom:1.2rem;">
            Download a full PDF audit report with all metrics, definitions, and mitigation results.
        </p>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Generating report..."):
                pdf_bytes = generate_pdf_report(
                    audit_result=audit_result,
                    filename=filename,
                    row_count=row_count,
                    col_count=col_count,
                )
            st.download_button(
                label="Download Audit Report (PDF)",
                data=pdf_bytes,
                file_name=f"IGNITE_Bias_Audit_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )