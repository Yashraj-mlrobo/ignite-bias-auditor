import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.audit import (
    load_adult_data,
    load_compas_data,
    preprocess_adult,
    preprocess_compas,
    train_model,
    run_aif360_audit,
    run_fairlearn_audit,
    run_mitigation,
    run_full_audit
)


# ── Data Loading Tests ──────────────────────────────────────────

def test_load_adult_data_returns_dataframe():
    df = load_adult_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_load_adult_data_has_correct_columns():
    df = load_adult_data()
    required_cols = ["age", "sex", "race", "income", "education"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

def test_load_compas_data_returns_dataframe():
    df = load_compas_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_load_compas_data_has_correct_columns():
    df = load_compas_data()
    required_cols = ["race", "sex", "two_year_recid", "decile_score"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


# ── Preprocessing Tests ─────────────────────────────────────────

def test_preprocess_adult_no_nulls():
    df = load_adult_data()
    df = preprocess_adult(df)
    assert df.isnull().sum().sum() == 0

def test_preprocess_adult_binary_sex():
    df = load_adult_data()
    df = preprocess_adult(df)
    assert set(df["sex"].unique()).issubset({0, 1})

def test_preprocess_adult_binary_race():
    df = load_adult_data()
    df = preprocess_adult(df)
    assert set(df["race"].unique()).issubset({0, 1})

def test_preprocess_adult_binary_income():
    df = load_adult_data()
    df = preprocess_adult(df)
    assert set(df["income"].unique()).issubset({0, 1})

def test_preprocess_compas_binary_sex():
    df = load_compas_data()
    df = preprocess_compas(df)
    assert set(df["sex"].unique()).issubset({0, 1})

def test_preprocess_compas_binary_race():
    df = load_compas_data()
    df = preprocess_compas(df)
    assert set(df["race"].unique()).issubset({0, 1})


# ── Model Training Tests ────────────────────────────────────────

def test_train_model_returns_model_and_columns():
    df = load_adult_data()
    df = preprocess_adult(df)
    X = pd.get_dummies(df.drop(columns=["income"]), drop_first=True)
    y = df["income"]
    model, cols = train_model(X, y)
    assert model is not None
    assert isinstance(cols, list)
    assert len(cols) > 0

def test_train_model_accuracy_above_threshold():
    df = load_adult_data()
    df = preprocess_adult(df)
    X = pd.get_dummies(df.drop(columns=["income"]), drop_first=True)
    y = df["income"]
    model, cols = train_model(X, y)
    X = X.reindex(columns=cols, fill_value=0)
    accuracy = model.score(X, y)
    assert accuracy > 0.75, f"Accuracy too low: {accuracy}"


# ── Bias Audit Tests ────────────────────────────────────────────

def test_run_full_audit_adult_returns_dict():
    results = run_full_audit(dataset="adult")
    assert isinstance(results, dict)

def test_run_full_audit_adult_has_required_keys():
    results = run_full_audit(dataset="adult")
    required_keys = [
        "dataset",
        "protected_attribute",
        "disparate_impact",
        "statistical_parity_difference",
        "demographic_parity_difference",
        "before_accuracy",
        "after_accuracy",
        "before_demographic_parity",
        "after_demographic_parity"
    ]
    for key in required_keys:
        assert key in results, f"Missing key: {key}"

def test_run_full_audit_adult_disparate_impact_is_biased():
    results = run_full_audit(dataset="adult")
    # Disparate impact below 0.8 means legally biased
    assert results["disparate_impact"] < 0.8, "Expected bias not detected"

def test_run_full_audit_mitigation_reduces_bias():
    results = run_full_audit(dataset="adult")
    assert results["after_demographic_parity"] < results["before_demographic_parity"], \
        "Mitigation did not reduce bias"

def test_run_full_audit_accuracy_drop_is_acceptable():
    results = run_full_audit(dataset="adult")
    accuracy_drop = results["before_accuracy"] - results["after_accuracy"]
    assert accuracy_drop < 0.05, f"Accuracy drop too large: {accuracy_drop}"

def test_run_full_audit_compas_returns_dict():
    results = run_full_audit(dataset="compas")
    assert isinstance(results, dict)
    assert results["dataset"] == "compas"
    assert results["protected_attribute"] == "race"

def test_run_full_audit_invalid_dataset():
    with pytest.raises(ValueError):
        run_full_audit(dataset="invalid_dataset")


# ── Fairlearn Tests ─────────────────────────────────────────────

def test_fairlearn_audit_returns_metrics():
    df = load_adult_data()
    df = preprocess_adult(df)
    X = pd.get_dummies(df.drop(columns=["income"]), drop_first=True)
    y = df["income"]
    model, cols = train_model(X, y)
    X = X.reindex(columns=cols, fill_value=0)
    y_pred = model.predict(X)
    sensitive = df["sex"]
    metrics = run_fairlearn_audit(X, y, y_pred, sensitive)
    assert "demographic_parity_difference" in metrics
    assert "equalized_odds_difference" in metrics