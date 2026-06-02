import pandas as pd
import numpy as np
from src.data_processing import build_production_pipeline

print("🔄 Loading data sample for pipeline verification...")
try:
    # Load just the first 100 rows to test the pipeline speed and logic
    df = pd.read_csv('data/raw data/data.csv', nrows=100)
    
    # Quick structural fix for potential column truncation in data stream views
    if 'Transactic' in df.columns:
        df = df.rename(columns={'Transactic': 'TransactionId'})
    
    # Locate and rename the date column
    date_col = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_col:
        df = df.rename(columns={date_col[0]: 'TransactionStartTime'})
    else:
        # Fallback if no date string match is found
        df['TransactionStartTime'] = pd.date_range(start='2026-01-01', periods=len(df), freq='H')

    # Define the exact features your pipeline is expecting
    # (Note: Extractor builds Total_Amount, Avg_Amount, etc., so they must be listed in numerical features)
    num_features = ['Amount', 'Value', 'Total_Amount', 'Avg_Amount', 'Transaction_Count', 'Std_Amount']
    cat_features = ['ProductCategory', 'ChannelId']
    woe_features = ['ProviderId']

    print("🏗️ Initializing production pipeline architecture...")
    pipeline = build_production_pipeline(
        numerical_cols=num_features, 
        categorical_cols=cat_features, 
        woe_cols=woe_features
    )

    print("⚙️ Fitting transformers and executing feature mapping...")
    # Run the full pipeline (passing FraudResult as the target vector for WoE calculations)
    processed_array = pipeline.fit_transform(df, y=df['FraudResult'])
    
    print("\n✅ PIPELINE VERIFICATION SUCCESSFUL!")
    print(f"Processed Output Matrix Shape: {processed_array.shape}")
    print("Your raw data has been successfully engineered, encoded, scaled, and is model-ready.")

except Exception as e:
    print("\n❌ Pipeline Verification Failed!")
    print(f"Error Details: {str(e)}")