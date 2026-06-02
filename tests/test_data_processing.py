import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ensure system path resolution for modular routing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import AdvancedFeatureExtractor, generate_proxy_target_variable

@pytest.fixture
def sample_transaction_dataframe():
    """Generates synthetic historical transactions to validate processing nodes."""
    data = {
        'CustomerId': ['C01', 'C01', 'C02', 'C03'],
        'TransactionId': ['T01', 'T02', 'T03', 'T04'],
        'Amount': [150.00, 350.00, 20.00, 5000.00],
        'Value': [150.00, 350.00, 20.00, 5000.00],
        'TransactionStartTime': [
            '2026-01-01 10:00:00',
            '2026-01-02 11:30:00',
            '2026-01-15 14:00:00',
            '2026-02-01 09:15:00'
        ],
        'FraudResult': [0, 0, 0, 0]
    }
    return pd.DataFrame(data)

def test_advanced_feature_extractor_columns(sample_transaction_dataframe):
    """Test 1: Confirms feature extraction correctly Appends time-velocity attributes."""
    extractor = AdvancedFeatureExtractor()
    extractor.fit(sample_transaction_dataframe)
    transformed_df = extractor.transform(sample_transaction_dataframe)
    
    # Assert existence of engineered temporal signatures
    assert 'Transaction_Hour' in transformed_df.columns
    assert 'Transaction_Day' in transformed_df.columns
    assert 'Transaction_Month' in transformed_df.columns
    assert 'Transaction_Year' in transformed_df.columns
    
    # Assert structural customer profile velocity transformations exist
    assert 'Total_Amount' in transformed_df.columns
    assert 'Avg_Amount' in transformed_df.columns

def test_generate_proxy_target_variable_bounds(sample_transaction_dataframe):
    """Test 2: Verifies that proxy target mapping builds strict binary array layouts."""
    output_df = generate_proxy_target_variable(sample_transaction_dataframe, n_clusters=2, random_state=42)
    
    assert 'is_high_risk' in output_df.columns
    # Ensure binary target bounds consist uniquely of 1s and 0s
    unique_values = output_df['is_high_risk'].unique()
    for value in unique_values:
        assert value in [0, 1]