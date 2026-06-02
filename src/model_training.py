import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_and_log_metrics(y_true, y_pred, y_prob, model_name):
    """Calculates model metrics and returns an MLflow-ready dictionary."""
    metrics = {
        f"{model_name}_accuracy": accuracy_score(y_true, y_pred),
        f"{model_name}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{model_name}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{model_name}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{model_name}_auc": roc_auc_score(y_true, y_prob)
    }
    return metrics

def run_model_training_pipeline(processed_data_path, target_column='is_high_risk'):
    """Loads engineered data, performs tuned training, and tracks via MLflow."""
    print("⏳ Initializing MLOps Training Data Streams...")
    
    # Load dataset
    df = pd.read_csv(processed_data_path)
    
    # Isolate Target Vector from Feature Space
    X = df.drop(columns=[target_column, 'CustomerId'], errors='ignore')
    # Dropping non-numeric structural leftovers if present after array transformations
    X = X.select_dtypes(include=[np.number])
    y = df[target_column]
    
    # Stratified Split for robust structural validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Set up local MLflow experiment tracker session
    mlflow.set_experiment("Bati_Bank_Credit_Risk_Assessment")
    
    # ----------------------------------------------------
    # MODEL 1: Optimized Random Forest Classifier
    # ----------------------------------------------------
    with mlflow.start_run(run_name="Random_Forest_Tuning"):
        print("\n🌲 Initializing Random Forest Tuning Hyperparameters...")
        rf_param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
        
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions=rf_param_dist,
            n_iter=4, cv=3, scoring='f1', random_state=42, n_jobs=-1
        )
        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_
        
        # Log Hyperparameters
        mlflow.log_params(rf_search.best_params_)
        
        # Performance Evaluation
        rf_preds = best_rf.predict(X_test)
        rf_probs = best_rf.predict_proba(X_test)[:, 1]
        rf_metrics = evaluate_and_log_metrics(y_test, rf_preds, rf_probs, "RF")
        
        # Log Metrics & Save Artifact
        mlflow.log_metrics(rf_metrics)
        mlflow.sklearn.log_model(best_rf, "random_forest_model")
        print("✅ Random Forest tracking run finalized inside MLflow Registry.")
        print(rf_metrics)

    # ----------------------------------------------------
    # MODEL 2: Gradient Boosting (XGBoost Classifier)
    # ----------------------------------------------------
    with mlflow.start_run(run_name="XGBoost_Tuning"):
        print("\n🚀 Initializing XGBoost Tuning Hyperparameters...")
        xgb_param_dist = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
        xgb_search = RandomizedSearchCV(
            XGBClassifier(random_state=42, eval_metric='logloss'),
            param_distributions=xgb_param_dist,
            n_iter=4, cv=3, scoring='f1', random_state=42, n_jobs=-1
        )
        xgb_search.fit(X_train, y_train)
        best_xgb = xgb_search.best_estimator_
        
        # Log Hyperparameters
        mlflow.log_params(xgb_search.best_params_)
        
        # Performance Evaluation
        xgb_preds = best_xgb.predict(X_test)
        xgb_probs = best_xgb.predict_proba(X_test)[:, 1]
        xgb_metrics = evaluate_and_log_metrics(y_test, xgb_preds, xgb_probs, "XGBoost")
        
        # Log Metrics & Save Artifact
        mlflow.log_metrics(xgb_metrics)
        mlflow.xgboost.log_model(best_xgb, "xgboost_model")
        print("✅ XGBoost tracking run finalized inside MLflow Registry.")
        print(xgb_metrics)

    print("\n🏁 Execution complete. Run 'mlflow ui' in your terminal to compare results.")

if __name__ == "__main__":
    # Point directly to your engineered matrix output path
    # For testing, you can pass a path containing your processed DataFrame
    pass