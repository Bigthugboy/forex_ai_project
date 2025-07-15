import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib
import os

# You may need to adjust this path or loading logic for your real data pipeline
DATA_PATH = 'data/features_targets.csv'  # Placeholder for demonstration
MODEL_DIR = 'models/saved_models/'

# Example parameter grids (can be expanded)
xgb_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
}
lgb_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
}

def load_features_and_target():
    """
    Load features and target for tuning. Replace with your real pipeline.
    Returns: X (DataFrame), y (Series)
    """
    # For demo, generate random data
    n_samples = 500
    feature_cols = [
        'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 'rsi_14',
        'macd', 'macd_signal', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d',
        'atr_14', 'news_sentiment'
    ]
    X = pd.DataFrame(np.random.randn(n_samples, len(feature_cols)), columns=feature_cols)
    y = np.random.randint(0, 2, n_samples)
    return X, y

def tune_model(model, param_grid, X, y, model_name):
    print(f"Tuning {model_name}...")
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid.fit(X, y)
    print(f"Best params for {model_name}: {grid.best_params_}")
    print(f"Best score for {model_name}: {grid.best_score_}")
    # Save best model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f'{model_name}_best.pkl')
    joblib.dump(grid.best_estimator_, model_path)
    print(f"Best {model_name} model saved to {model_path}")
    return grid.best_estimator_

def main():
    X, y = load_features_and_target()
    # Optionally scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Tune XGBoost
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    best_xgb = tune_model(xgb_model, xgb_param_grid, X_scaled, y, 'xgboost')
    # Tune LightGBM
    lgb_model = lgb.LGBMClassifier(random_state=42)
    best_lgb = tune_model(lgb_model, lgb_param_grid, X_scaled, y, 'lightgbm')
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, 'tuning_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    main() 