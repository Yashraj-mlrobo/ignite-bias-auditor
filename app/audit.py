import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer

warnings.filterwarnings('ignore')

def load_adult_data():
    """loads both adult CSV files with correct column names"""
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num", 
        "marital-status", "occupation", "relationship", "race", "sex", 
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    df_list = []
    for file_name in ["adult_income.csv", "adult_test.csv"]:
        path = os.path.join("data", file_name)
        if os.path.exists(path):
            # Read mapped columns directly to handle both header and no-header scenarios
            df = pd.read_csv(path, header=None, names=columns, skipinitialspace=True)
            # Handle if the first row was practically a column header
            if not df.empty and str(df.iloc[0]['age']).strip() == 'age':
                df = df.iloc[1:].reset_index(drop=True)
            df_list.append(df)
            
    if not df_list:
        return pd.DataFrame(columns=columns)
        
    return pd.concat(df_list, ignore_index=True)


def load_compas_data():
    """loads COMPAS with only the selected columns, drops nulls"""
    columns_to_keep = [
        "id", "sex", "age", "age_cat", "race", "juv_fel_count", 
        "juv_misd_count", "juv_other_count", "priors_count", 
        "c_charge_degree", "is_recid", "two_year_recid", 
        "decile_score", "score_text"
    ]
    path = os.path.join("data", "compas.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, skipinitialspace=True)
        # Select existing columns from the requested list
        keep_cols = [c for c in columns_to_keep if c in df.columns]
        df = df[keep_cols]
        df = df.dropna()
        return df
    else:
        return pd.DataFrame(columns=columns_to_keep)


def preprocess_adult(df):
    """cleans ?, strips whitespace, fixes trailing dots, encodes sex and race as binary, creates target column"""
    if df.empty:
        return df
        
    # Strip whitespace for string columns
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
    
    # Clean ?
    df = df.replace('?', np.nan)
    df = df.dropna()
    
    # Fix trailing dots in income
    if 'income' in df.columns:
        df['income'] = df['income'].astype(str).str.replace('.', '', regex=False)
        # creates target column: >50K = 1, <=50K = 0
        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
        
    # Encodes sex and race as binary
    # sex: Male = 1, Female = 0
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).apply(lambda x: 1 if 'Male' in x else 0)
        
    # race: White = 1, Non-White = 0
    if 'race' in df.columns:
        df['race'] = df['race'].astype(str).apply(lambda x: 1 if 'White' in x else 0)
        
    return df


def preprocess_compas(df):
    """encodes sex and race as binary, creates target from two_year_recid"""
    if df.empty:
        return df
        
    # creates target from two_year_recid
    if 'two_year_recid' in df.columns:
        df['two_year_recid'] = df['two_year_recid'].astype(int)
        
    # Encodes sex and race as binary
    # sex: Male = 1, Female = 0
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).apply(lambda x: 1 if 'Male' in x else 0)
        
    # race: Caucasian = 1, Non-Caucasian = 0 (standard for compas binary split)
    if 'race' in df.columns:
        df['race'] = df['race'].astype(str).apply(lambda x: 1 if 'Caucasian' in x else 0)
        
    return df


def train_model(X, y):
    X_encoded = pd.get_dummies(X, drop_first=True)
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_encoded, y)
    return model, X_encoded.columns.tolist()


def run_aif360_audit(df, protected_attribute, target_col='income'):
    """uses IBM AIF360 BinaryLabelDataset to compute: Disparate Impact, Statistical Parity Difference, Equal Opportunity Difference, Average Odds Difference"""
    try:
        # Requires everything to be numeric
        df_numeric = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
        df_numeric[target_col] = df[target_col]
        
        dataset = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=df_numeric,
            label_names=[target_col],
            protected_attribute_names=[protected_attribute]
        )
        
        privileged_groups = [{protected_attribute: 1}]
        unprivileged_groups = [{protected_attribute: 0}]
        
        metric = BinaryLabelDatasetMetric(
            dataset, 
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        
        # To get ClassificationMetrics, we generate predictions using a surrogate model
        X = df_numeric.drop(columns=[target_col])
        y = df_numeric[target_col]
        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        dataset_pred = dataset.copy()
        dataset_pred.labels = y_pred.reshape(-1, 1)
        
        class_metric = ClassificationMetric(
            dataset, 
            dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        
        return {
            "disparate_impact": metric.disparate_impact(),
            "statistical_parity_difference": metric.statistical_parity_difference(),
            "equal_opportunity_difference": class_metric.equal_opportunity_difference(),
            "average_odds_difference": class_metric.average_odds_difference()
        }
    except Exception as e:
        print(f"Warning: AIF360 Audit failed due to {e}")
        return {
            "disparate_impact": 0.0,
            "statistical_parity_difference": 0.0,
            "equal_opportunity_difference": 0.0,
            "average_odds_difference": 0.0
        }


def run_fairlearn_audit(X, y, y_pred, sensitive_features):
    """uses Microsoft Fairlearn MetricFrame to compute: Demographic Parity Difference, Equalized Odds Difference"""
    try:
        dpd = demographic_parity_difference(y_true=y, y_pred=y_pred, sensitive_features=sensitive_features)
        eod = equalized_odds_difference(y_true=y, y_pred=y_pred, sensitive_features=sensitive_features)
        return {
            "demographic_parity_difference": dpd,
            "equalized_odds_difference": eod
        }
    except Exception as e:
        print(f"Warning: Fairlearn Audit failed due to {e}")
        return {
            "demographic_parity_difference": 0.0,
            "equalized_odds_difference": 0.0
        }


def run_mitigation(X, y, sensitive_features, model):
    """applies Fairlearn ThresholdOptimizer, returns mitigated predictions"""
    try:
        X_encoded = pd.get_dummies(X, drop_first=True)
        optimizer = ThresholdOptimizer(
            estimator=model,
            constraints="demographic_parity",
            prefit=True
        )
        optimizer.fit(X_encoded, y, sensitive_features=sensitive_features)
        mitigated_pred = optimizer.predict(X_encoded, sensitive_features=sensitive_features)
        return mitigated_pred
    except Exception as e:
        print(f"Warning: Fairlearn Mitigation failed due to {e}")
        X_encoded = pd.get_dummies(X, drop_first=True)
        return model.predict(X_encoded)


def run_full_audit(dataset="adult"):
    """orchestrates everything, returns a clean dictionary with all bias metrics and before/after mitigation scores"""
    if dataset == "adult":
        df = load_adult_data()
        df = preprocess_adult(df)
        protected_attribute = "sex"
        target_col = "income"
    elif dataset == "compas":
        df = load_compas_data()
        df = preprocess_compas(df)
        protected_attribute = "race"
        target_col = "two_year_recid"
    else:
        raise ValueError("dataset must be 'adult' or 'compas'")
        
    if df.empty:
        print(f"Warning: Dataset '{dataset}' is empty or not found.")
        return {
            "dataset": dataset,
            "protected_attribute": protected_attribute,
            "disparate_impact": 0.0,
            "statistical_parity_difference": 0.0,
            "equal_opportunity_difference": 0.0,
            "demographic_parity_difference": 0.0,
            "before_accuracy": 0.0,
            "after_accuracy": 0.0,
            "before_demographic_parity": 0.0,
            "after_demographic_parity": 0.0
        }

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_encoded = pd.get_dummies(X, drop_first=True)
    sensitive_features = df[protected_attribute]
    
    # 1. Train model Before Mitigation
    model, encoded_cols = train_model(X, y)
    X_encoded = X_encoded.reindex(columns=encoded_cols, fill_value=0)
    y_pred = model.predict(X_encoded)
    before_accuracy = accuracy_score(y, y_pred)
    
    # 2. AIF360 Audit
    aif_metrics = run_aif360_audit(df, protected_attribute, target_col=target_col)
    
    # 3. Fairlearn Audit Before Mitigation
    fl_metrics = run_fairlearn_audit(X, y, y_pred, sensitive_features)
    before_dpd = fl_metrics["demographic_parity_difference"]
    
    # 4. Mitigation
    y_pred_mitigated = run_mitigation(X, y, sensitive_features, model)
    after_accuracy = accuracy_score(y, y_pred_mitigated)
    
    # 5. Fairlearn Audit After Mitigation
    fl_metrics_after = run_fairlearn_audit(X, y, y_pred_mitigated, sensitive_features)
    after_dpd = fl_metrics_after["demographic_parity_difference"]
    
    result = {
        "dataset": dataset,
        "protected_attribute": protected_attribute,
        "disparate_impact": round(aif_metrics.get("disparate_impact", 0.0), 4),
        "statistical_parity_difference": round(aif_metrics.get("statistical_parity_difference", 0.0), 4),
        "equal_opportunity_difference": round(aif_metrics.get("equal_opportunity_difference", 0.0), 4),
        "demographic_parity_difference": round(before_dpd, 4),
        "before_accuracy": round(before_accuracy, 4),
        "after_accuracy": round(after_accuracy, 4),
        "before_demographic_parity": round(before_dpd, 4),
        "after_demographic_parity": round(after_dpd, 4)
    }
    
    return result


if __name__ == "__main__":
    print("=== Running Audit for Adult Dataset ===")
    adult_results = run_full_audit(dataset="adult")
    print(json.dumps(adult_results, indent=4))
    
    print("\n=== Running Audit for COMPAS Dataset ===")
    compas_results = run_full_audit(dataset="compas")
    print(json.dumps(compas_results, indent=4))
    # Auto-generate sample_upload.csv
    df_sample = load_adult_data()
    df_sample = preprocess_adult(df_sample)
    df_sample.sample(200, random_state=42).to_csv("data/sample_upload.csv", index=False)
    print("\n✅ sample_upload.csv generated in data/ folder!")