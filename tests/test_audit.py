import pytest
import pandas as pd
from app.audit import get_fairness_metrics # Assuming this is Antigravity's function name

def test_disparate_impact_calculation():
    # Create a dummy dataset with obvious bias
    # Privileged group (1) gets favorable outcome (1) more often than Unprivileged (0)
    data = {
        'sex': [1, 1, 1, 1, 0, 0, 0, 0],  # 1: Male, 0: Female
        'income': [1, 1, 1, 0, 1, 0, 0, 0] # 1: >50K, 0: <=50K
    }
    df = pd.DataFrame(data)
    
    # Run the audit function
    # Expected: Privileged success rate = 3/4 (0.75)
    # Expected: Unprivileged success rate = 1/4 (0.25)
    # Expected Disparate Impact = 0.25 / 0.75 = 0.333
    
    results = get_fairness_metrics(
        df=df, 
        protected_attribute='sex', 
        target='income',
        privileged_class=1,
        unprivileged_class=0,
        favorable_outcome=1
    )
    
    # Assertions
    assert "disparate_impact" in results
    assert "statistical_parity_difference" in results
    
    # Check if the math is roughly correct (allowing for floating point rounding)
    assert round(results["disparate_impact"], 2) == 0.33
    assert round(results["statistical_parity_difference"], 2) == -0.50