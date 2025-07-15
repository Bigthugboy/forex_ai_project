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
from sklearn.utils import resample

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

def audit_data_quality(df, feature_cols, target_col='target'):
    import matplotlib.pyplot as plt
    print('--- DATA AUDIT ---')
    print('Shape:', df.shape)
    print('Missing values per column:')
    print(df.isnull().sum())
    print('Feature summary:')
    print(df[feature_cols].describe())
    print('Target distribution:')
    print(df[target_col].value_counts())
    # Visualize feature distributions
    df[feature_cols].hist(bins=30, figsize=(15, 10))
    plt.tight_layout()
    plt.show()
    # Visualize target distribution
    df[target_col].value_counts().plot(kind='bar', title='Target Distribution')
    plt.show()

def test_label_thresholds(df, close_col, thresholds=[0.001, 0.002, 0.003], lookaheads=[4, 8, 12]):
    import matplotlib.pyplot as plt
    for lookahead in lookaheads:
        for thresh in thresholds:
            future_return = df[close_col].shift(-lookahead) / df[close_col] - 1
            target = (future_return > thresh).astype(int)
            plt.figure()
            target.value_counts().plot(kind='bar')
            plt.title(f'Target Distribution (thresh={thresh}, lookahead={lookahead})')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.show()

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
    
    print(f"Class distribution: {np.bincount(y)}")

    # Visualize class balance before upsampling
    print('Class balance before upsampling:')
    print(y.value_counts())
    # Always initialize to valid defaults
    X_bal, y_bal = X, y
    # Upsample minority class
    Xy = pd.concat([X, y], axis=1)
    class_counts = y.value_counts()
    if len(class_counts) == 2:
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        majority = Xy[Xy['target'] == majority_class]
        minority = Xy[Xy['target'] == minority_class]
        if isinstance(majority, pd.DataFrame) and isinstance(minority, pd.DataFrame) and not majority.empty and not minority.empty:
            minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
            if isinstance(minority_upsampled, pd.DataFrame) and not minority_upsampled.empty:
                Xy_balanced = pd.concat([majority, minority_upsampled], axis=0).reset_index(drop=True)
                X_bal = Xy_balanced[feature_cols]
                y_bal = Xy_balanced['target']
                print('Class balance after upsampling:')
                print(y_bal.value_counts())
                from sklearn.utils import shuffle
                X_bal, y_bal = shuffle(X_bal, y_bal, random_state=42)
            else:
                print('Warning: Upsampled minority is not a valid DataFrame or is empty, skipping upsampling.')
        else:
            print('Warning: One of the classes is not a valid DataFrame or is empty, skipping upsampling.')
    else:
        print('Warning: Only one class present, skipping upsampling.')
    
    # Final check before train_test_split
    if X_bal is None or y_bal is None:
        raise ValueError('X_bal or y_bal is None before train_test_split. Check data pipeline and upsampling logic.')
    if not hasattr(X_bal, '__iter__') or not hasattr(y_bal, '__iter__'):
        raise ValueError('X_bal or y_bal is not iterable before train_test_split. Check data pipeline and upsampling logic.')

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Base models
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', scale_pos_weight=(y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1)
    lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, class_weight='balanced')
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
    
    # Print feature distributions for debugging
    print('DEBUG: Feature distributions before training:')
    for col in feature_cols:
        col_data = df[col]
        print(f"{col}: mean={col_data.mean():.4f}, std={col_data.std():.4f}, min={col_data.min():.4f}, max={col_data.max():.4f}, unique={col_data.nunique()}")
    
    # Audit data quality before training
    audit_data_quality(df, feature_cols, target_col='target')

    # Remove constant features
    constant_features = [col for col in feature_cols if df[col].nunique() <= 1]
    for col in constant_features:
        print(f'Removing constant feature: {col}')
    feature_cols = [col for col in feature_cols if col not in constant_features]
    if not feature_cols:
        raise ValueError('No features left after removing constant features!')
    # Re-run audit with updated feature_cols
    audit_data_quality(df, feature_cols, target_col='target')

    # SHAP and feature importance analysis (robust)
    try:
        import shap
        import matplotlib.pyplot as plt
        # For CalibratedClassifierCV, get the underlying VotingClassifier if available
        if hasattr(calibrated, 'base_estimator_'):
            base_ensemble = calibrated.base_estimator_
            # Use the first estimator (e.g., xgb or lgb) for SHAP and feature importances
            tree_model = base_ensemble.estimators_[0][1]
            explainer = shap.Explainer(tree_model, X_train)
            shap_values = explainer(X_train)
            print('SHAP summary plot:')
            shap.summary_plot(shap_values, X_train, show=False)
            plt.tight_layout()
            plt.savefig('logs/shap_summary.png')
            plt.close()
            # Model feature importances (tree-based)
            importances = tree_model.feature_importances_
            plt.figure(figsize=(10, 6))
            plt.barh(feature_cols, importances)
            plt.title('Model Feature Importances')
            plt.tight_layout()
            plt.savefig('logs/feature_importances.png')
            plt.close()
        else:
            print('[Warning] CalibratedClassifierCV has no base_estimator_ attribute after fitting.')
    except Exception as e:
        print(f'[Warning] SHAP/feature importance analysis failed: {e}')

    return calibrated, scaler, feature_cols

if __name__ == "__main__":
    # This will be called when training the model
    print("Training signal prediction model...")
