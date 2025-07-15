import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import joblib
from config import Config
import os

def prepare_target_variable(df):
    """
    Create target variable for signal prediction.
    Target: 1 for buy signal, 0 for sell signal, based on future price movement.
    """
    # Detect correct column names
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    # Calculate future price change (next 4 hours)
    df['future_return'] = df[close_col].shift(-4) / df[close_col] - 1
    # Create binary target: 1 if price goes up by more than 0.1%, 0 otherwise
    threshold = 0.001  # 0.1% threshold
    df['target'] = (df['future_return'] > threshold).astype(int)
    return df

def train_signal_model(features_df, model_path='models/saved_models/signal_model.pkl'):
    """
    Train XGBoost model for signal prediction.
    Args:
        features_df (pd.DataFrame): DataFrame with features and target
        model_path (str): Path to save the trained model
    Returns:
        xgb.XGBClassifier: Trained model
    """
    print('DEBUG: features_df columns before prepare_target_variable:', features_df.columns)
    print(features_df.head())
    # Prepare target variable
    df = prepare_target_variable(features_df.copy())
    print('DEBUG: df columns after prepare_target_variable:', df.columns)
    print(df.head())
    
    if 'target' not in df.columns:
        print('DEBUG: target column missing! Columns:', df.columns)
        print(df.head())
        raise ValueError('target column missing after prepare_target_variable')
    
    # Use the same exclusion logic as in predict_signal
    exclude_cols = [col for col in df.columns if col.startswith(('Close', 'Open', 'High', 'Low', 'Volume'))] + ['future_return', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    df = df.dropna(subset=['target'])
    print('DEBUG: df columns after dropna on target:', df.columns)
    print(df.head())
    
    if df.empty:
        print('DEBUG: DataFrame is empty after dropna on target!')
        print(df.head())
        raise ValueError('No data left after dropping NaN targets.')
    
    if len(df) < 100:
        raise ValueError("Not enough data for training. Need at least 100 samples.")
    
    X = df[feature_cols]
    y = df['target']
    
    if len(y.unique()) < 2:
        raise ValueError("Need both buy and sell signals for training.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Base models
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
    lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    # Ensemble with soft voting
    ensemble = VotingClassifier(estimators=[('xgb', xgb_model), ('lgb', lgb_model)], voting='soft')
    # Probability calibration
    calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
    calibrated.fit(X_train_scaled, y_train)
    
    y_pred = calibrated.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(calibrated, model_path)
    joblib.dump(scaler, model_path.replace('.pkl', '_scaler.pkl'))
    
    print(f"Model saved to {model_path}")
    
    return calibrated, scaler, feature_cols

if __name__ == "__main__":
    # This will be called when training the model
    print("Training signal prediction model...")
