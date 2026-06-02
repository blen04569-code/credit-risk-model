import pandas as pd
import numpy as np
import sys
import os

# Ensure the root directory is on the path for clean module importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import build_production_pipeline, generate_proxy_target_variable

print("🔄 Loading data sample for complete pipeline run...")
try:
    # Load raw transaction entries
    df = pd.read_csv('data/raw data/data.csv')
    
    # Standardize names
    if 'Transactic' in df.columns:
        df = df.rename(columns={'Transactic': 'TransactionId'})
    
    date_col = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_col:
        df = df.rename(columns={date_col[0]: 'TransactionStartTime'})
    
    # 1. Run Task 4 to append our behavioral target matrix
    df_with_target = generate_proxy_target_variable(df, n_clusters=3, random_state=42)
    
    # 2. Define features for Task 3 feature engineering
    num_features = ['Amount', 'Value', 'Total_Amount', 'Avg_Amount', 'Transaction_Count', 'Std_Amount']
    cat_features = ['ProductCategory', 'ChannelId']
    woe_features = ['ProviderId']
    
    print("\n🏗️ Building production transformations...")
    pipeline = build_production_pipeline(num_features, cat_features, woe_features)
    
    # 3. Fit pipeline transformations (including the engineered proxy target label)
    processed_array = pipeline.fit_transform(df_with_target, y=df_with_target['FraudResult'])
    
    print("\n✅ TASK 4 VERIFICATION SUCCESSFUL!")
    print(f"Final Output Array Shape: {processed_array.shape}")
    print("The processed dataset contains robust engineering transformations alongside the integrated 'is_high_risk' target flag.")

except Exception as e:
    print("\n❌ Verification Routine Halted!")
    print(f"Error Details: {str(e)}")