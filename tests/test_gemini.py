import pytest
from unittest.mock import patch, MagicMock
from app.gemini import get_ai_summary, BiasAuditorAI

# Dummy data from Antigravity's engine
dummy_metrics = {
    "dataset": "test_dataset",
    "protected_attribute": "age",
    "disparate_impact": 0.75,
    "statistical_parity_difference": -0.15,
    "equal_opportunity_difference": 0.05,
    "before_accuracy": 0.85,
    "after_accuracy": 0.82,
    "before_demographic_parity": 0.15,
    "after_demographic_parity": 0.02
}

@patch('app.gemini.genai.GenerativeModel')
def test_generate_report_success(mock_model):
    # Setup the Mock API Response
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is a mocked AI audit report. The model shows bias."
    mock_instance.generate_content.return_value = mock_response
    
    # Apply the mock to the class
    mock_model.return_value = mock_instance

    # Execute our function
    report = get_ai_summary(dummy_metrics)

    # Assertions
    assert report is not None
    assert "mocked AI audit report" in report
    
    # Verify the API was called exactly once
    mock_instance.generate_content.assert_called_once()
    
    # Verify the prompt contained our critical metrics
    called_prompt = mock_instance.generate_content.call_args[0][0]
    assert "test_dataset" in called_prompt
    assert "0.75" in called_prompt # Disparate impact value