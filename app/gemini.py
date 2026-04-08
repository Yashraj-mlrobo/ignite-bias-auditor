import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def get_ai_summary(audit_results):
    """
    Takes the output dictionary from run_full_audit() and generates
    a plain English bias audit report using Gemini.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an AI Ethics Auditor for Team IGNITE, competing in the Google Solution Challenge 2026.
    Analyze the following machine learning bias audit results.

    DATASET CONTEXT:
    - Dataset: {audit_results.get('dataset', 'Unknown').upper()}
    - Protected Attribute: {audit_results.get('protected_attribute', 'Unknown')}

    BEFORE MITIGATION:
    - Accuracy: {audit_results.get('before_accuracy', 'N/A')}
    - Demographic Parity Difference: {audit_results.get('before_demographic_parity', 'N/A')}
    - Disparate Impact: {audit_results.get('disparate_impact', 'N/A')}
    - Equal Opportunity Difference: {audit_results.get('equal_opportunity_difference', 'N/A')}

    AFTER MITIGATION (Fairlearn ThresholdOptimizer):
    - Accuracy: {audit_results.get('after_accuracy', 'N/A')}
    - Demographic Parity Difference: {audit_results.get('after_demographic_parity', 'N/A')}

    INSTRUCTIONS:
    1. Assess the initial model — was it biased? Explain Disparate Impact and Demographic Parity simply.
    2. Evaluate the mitigation — did ThresholdOptimizer successfully reduce bias?
    3. Discuss the trade-off — was the accuracy drop worth the fairness gain?
    4. Keep tone professional and easy to understand for a non-technical judge.
    5. Format with Markdown — use bolding and bullet points.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"An error occurred while generating the AI report: {e}"


def get_shap_summary(feature_impacts, protected_attribute, dataset):
    """
    Takes top SHAP features and explains in plain English
    why the model is discriminating against the unprivileged group.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found."

    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an AI Ethics Auditor for Team IGNITE, Google Solution Challenge 2026.

    A SHAP analysis was run on a machine learning model trained on the {dataset} dataset.
    The protected attribute being analyzed is: {protected_attribute}

    The top features driving bias against the unprivileged group are:
    {feature_impacts}

    INSTRUCTIONS:
    1. Explain in plain English why each feature may be causing discrimination
    2. Suggest which features the organization should investigate or remove
    3. Keep it under 150 words
    4. Use simple language a non-technical judge can understand
    5. Format with bullet points
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating SHAP summary: {e}"


if __name__ == "__main__":
    dummy_data = {
        "dataset": "adult",
        "protected_attribute": "sex",
        "disparate_impact": 0.3635,
        "before_demographic_parity": 0.1989,
        "before_accuracy": 0.7914,
        "after_demographic_parity": 0.0002,
        "after_accuracy": 0.7886,
        "equal_opportunity_difference": 0.0221
    }
    print("Testing Gemini API...")
    report = get_ai_summary(dummy_data)
    print(report)