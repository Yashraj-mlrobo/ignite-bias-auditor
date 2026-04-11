import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.gemini import get_ai_summary, get_shap_summary


# ── Dummy Data ──────────────────────────────────────────────────

DUMMY_AUDIT = {
    "dataset": "adult",
    "protected_attribute": "sex",
    "disparate_impact": 0.3635,
    "before_demographic_parity": 0.1989,
    "before_accuracy": 0.7914,
    "after_demographic_parity": 0.0002,
    "after_accuracy": 0.7886,
    "equal_opportunity_difference": 0.0221
}

DUMMY_FEATURES = """
- education-num: 0.35
- age: 0.28
- hours-per-week: 0.21
- occupation: 0.18
- capital-gain: 0.12
"""


# ── API Key Tests ───────────────────────────────────────────────

def test_gemini_api_key_exists():
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    assert key is not None, "GEMINI_API_KEY not found in .env"
    assert len(key) > 10, "GEMINI_API_KEY seems invalid"


# ── Report Generation Tests ─────────────────────────────────────

def test_get_ai_summary_returns_string():
    result = get_ai_summary(DUMMY_AUDIT)
    assert isinstance(result, str)

def test_get_ai_summary_not_empty():
    result = get_ai_summary(DUMMY_AUDIT)
    assert len(result) > 100, "Report is too short to be meaningful"

def test_get_ai_summary_mentions_bias():
    result = get_ai_summary(DUMMY_AUDIT)
    keywords = ["bias", "fair", "parity", "accuracy", "model"]
    found = any(word in result.lower() for word in keywords)
    assert found, "Report does not mention any bias-related keywords"

def test_get_ai_summary_mentions_dataset():
    result = get_ai_summary(DUMMY_AUDIT)
    assert "adult" in result.lower(), "Report does not mention the dataset"

def test_get_ai_summary_no_error_string():
    result = get_ai_summary(DUMMY_AUDIT)
    assert not result.startswith("An error"), f"API returned error: {result}"


# ── SHAP Summary Tests ──────────────────────────────────────────

def test_get_shap_summary_returns_string():
    result = get_shap_summary(DUMMY_FEATURES, "sex", "adult")
    assert isinstance(result, str)

def test_get_shap_summary_not_empty():
    result = get_shap_summary(DUMMY_FEATURES, "sex", "adult")
    assert len(result) > 50, "SHAP summary too short"

def test_get_shap_summary_no_error_string():
    result = get_shap_summary(DUMMY_FEATURES, "sex", "adult")
    assert not result.startswith("Error"), f"API returned error: {result}"


# ── Edge Case Tests ─────────────────────────────────────────────

def test_get_ai_summary_missing_key_handled():
    incomplete_data = {"dataset": "adult"}
    result = get_ai_summary(incomplete_data)
    assert isinstance(result, str)
    assert len(result) > 0

def test_get_ai_summary_compas_dataset():
    compas_data = {
        "dataset": "compas",
        "protected_attribute": "race",
        "disparate_impact": 1.2195,
        "before_demographic_parity": 0.096,
        "before_accuracy": 0.9694,
        "after_demographic_parity": 0.0012,
        "after_accuracy": 0.9364,
        "equal_opportunity_difference": -0.0004
    }
    result = get_ai_summary(compas_data)
    assert isinstance(result, str)
    assert len(result) > 100

def test_get_ai_summary_mentions_dataset():
    result = get_ai_summary(DUMMY_AUDIT)
    if any(code in result for code in ["503", "429", "quota", "unavailable", "exhausted"]):
        pytest.skip("API quota/availability issue — skipping")
    assert "adult" in result.lower()

def test_get_ai_summary_no_error_string():
    result = get_ai_summary(DUMMY_AUDIT)
    if any(code in result for code in ["503", "429", "quota", "unavailable", "exhausted"]):
        pytest.skip("API quota/availability issue — skipping")
    assert not result.startswith("An error")

def test_get_shap_summary_no_error_string():
    result = get_shap_summary(DUMMY_FEATURES, "sex", "adult")
    if any(code in result for code in ["503", "429", "quota", "unavailable", "exhausted"]):
        pytest.skip("API quota/availability issue — skipping")
    assert not result.startswith("Error")