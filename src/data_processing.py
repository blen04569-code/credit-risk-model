import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class AdvancedFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to handle DateTime parsing and customer velocity aggregations."""
    def __init__(self, date_col='TransactionStartTime', customer_col='CustomerId'):
        self.date_col = date_col
        self.customer_col = customer_col
        self.customer_aggregates_ = {}

    def fit(self, X, y=None):
        # Calculate historical aggregate profiles per customer from training baseline
        X_copy = X.copy()
        if 'Amount' in X_copy.columns and self.customer_col in X_copy.columns:
            aggs = X_copy.groupby(self.customer_col)['Amount'].agg(['sum', 'mean', 'count', 'std']).reset_index()
            aggs.columns = [self.customer_col, 'Total_Amount', 'Avg_Amount', 'Transaction_Count', 'Std_Amount']
            # Fill missing standard deviation values (for single-transaction profiles) with 0
            aggs['Std_Amount'] = aggs['Std_Amount'].fillna(0)
            self.customer_aggregates_ = aggs.set_index(self.customer_col).to_dict(orient='index')
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # 1. Extract Time-Velocity Features
        if self.date_col in X_out.columns:
            X_out[self.date_col] = pd.to_datetime(X_out[self.date_col])
            X_out['Transaction_Hour'] = X_out[self.date_col].dt.hour
            X_out['Transaction_Day'] = X_out[self.date_col].dt.day
            X_out['Transaction_Month'] = X_out[self.date_col].dt.month
            X_out['Transaction_Year'] = X_out[self.date_col].dt.year
            X_out = X_out.drop(columns=[self.date_col])
            
        # 2. Map Customer Historical Aggregate Features
        if self.customer_col in X_out.columns:
            # Map aggregated states; fallback to population defaults if new customer is encountered
            global_mean = X_out['Amount'].mean() if 'Amount' in X_out.columns else 0
            
            total_amt, avg_amt, tx_count, std_amt = [], [], [], []
            for cust in X_out[self.customer_col]:
                profile = self.customer_aggregates_.get(cust, None)
                if profile:
                    total_amt.append(profile['Total_Amount'])
                    avg_amt.append(profile['Avg_Amount'])
                    tx_count.append(profile['Transaction_Count'])
                    std_amt.append(profile['Std_Amount'])
                else:
                    total_amt.append(global_mean)
                    avg_amt.append(global_mean)
                    tx_count.append(1)
                    std_amt.append(0)
                    
            X_out['Total_Amount'] = total_amt
            X_out['Avg_Amount'] = avg_amt
            X_out['Transaction_Count'] = tx_count
            X_out['Std_Amount'] = std_amt
            
        return X_out

class WeightOfEvidenceTransformer(BaseEstimator, TransformerMixin):
    """Applies Weight of Evidence (WoE) mapping on high-cardinality targets."""
    def __init__(self, columns_to_encode, target_col='FraudResult'):
        self.columns_to_encode = columns_to_encode
        self.target_col = target_col
        self.woe_maps_ = {}

    def fit(self, X, y=None):
        if y is None or self.target_col not in X.columns:
            return self
            
        X_copy = X.copy()
        total_pos = X_copy[self.target_col].sum()
        total_neg = len(X_copy) - total_pos
        
        # Add a tiny smoothing factor to prevent division-by-zero errors (Laplace smoothing)
        epsilon = 1e-5
        
        for col in self.columns_to_encode:
            if col in X_copy.columns:
                woe_dict = {}
                grouped = X_copy.groupby(col)[self.target_col].agg(['sum', 'count'])
                for category, row in grouped.iterrows():
                    pos_count = row['sum']
                    neg_count = row['count'] - pos_count
                    
                    p_pos = (pos_count + epsilon) / (total_pos + epsilon)
                    p_neg = (neg_count + epsilon) / (total_neg + epsilon)
                    
                    woe_dict[category] = np.log(p_pos / p_neg)
                self.woe_maps_[col] = woe_dict
        return self

    def transform(self, X):
        X_out = X.copy()
        for col, woe_map in self.woe_maps_.items():
            if col in X_out.columns:
                # Map categories to calculated WoE values; fallback default to 0 (neutral risk impact)
                X_out[f'{col}_WoE'] = X_out[col].map(woe_map).fillna(0)
                X_out = X_out.drop(columns=[col])
        return X_out

def build_production_pipeline(numerical_cols, categorical_cols, woe_cols):
    """Chains transformation transformations using clean pipeline styling."""
    
    # Base preprocessing sub-assembly
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='passthrough')
    
    # Combined automation architecture
    full_pipeline = Pipeline(steps=[
        ('extractor', AdvancedFeatureExtractor()),
        ('woe_encoder', WeightOfEvidenceTransformer(columns_to_encode=woe_cols)),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

if __name__ == "__main__":
    print("Pipeline compilation successful. Ready to run downstream processing routines.")