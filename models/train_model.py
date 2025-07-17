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
from utils.logger import get_logger
logger = get_logger('train_model', log_file='logs/train_model.log')

def prepare_target_variable(df):
    """
    Create target variable for signal prediction.
    Target: 1 for buy signal, 0 for sell signal, based on future price movement.
    """
    logger.info('Entering prepare_target_variable')
    # Detect correct column names
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    # Calculate future price change (next 5 hours)
    df['future_return'] = df[close_col].shift(-5) / df[close_col] - 1
    # Create binary target: 1 if price goes up by more than 0.1%, 0 otherwise
    threshold = 0.001  # 0.1% threshold
    df['target'] = (df['future_return'] > threshold).astype(int)
    logger.info('Exiting prepare_target_variable')
    return df

def audit_data_quality(df, feature_cols, target_col='target'):
    import matplotlib.pyplot as plt
    logger.info('Entering audit_data_quality')
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
    logger.info('Exiting audit_data_quality')

def test_label_thresholds(df, close_col, thresholds=[0.001, 0.002, 0.003], lookaheads=[5, 8, 12]):
    import matplotlib.pyplot as plt
    logger.info('Entering test_label_thresholds')
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
    logger.info('Exiting test_label_thresholds')

def train_signal_model(features_df, model_path='models/saved_models/signal_model.pkl', top_n_features=10):
    logger.info('Model is starting a new training session...')
    logger.info('Model is cleaning and preparing data...')
    logger.info(f'features_df columns before prepare_target_variable: {features_df.columns}')
    print(features_df.head())
    # Prepare target variable
    logger.info('Preparing target variable...')
    df = prepare_target_variable(features_df.copy())
    logger.info(f'df columns after prepare_target_variable: {df.columns}')
    print(df.head())
    
    if 'target' not in df.columns:
        logger.error('target column missing after prepare_target_variable')
        print('DEBUG: target column missing! Columns:', df.columns)
        print(df.head())
        raise ValueError('target column missing after prepare_target_variable')
    
    # Use the same exclusion logic as in predict_signal
    exclude_cols = [col for col in df.columns if col.startswith(('Close', 'Open', 'High', 'Low', 'Volume'))] + ['future_return', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    # Only keep numeric columns
    feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    logger.info(f"Final feature columns used for training: {feature_cols}")
    
    logger.info('Dropping NaN targets...')
    df = df.dropna(subset=['target'])
    print('DEBUG: df columns after dropna on target:', df.columns)
    print(df.head())
    
    if df.empty:
        logger.error('DataFrame is empty after dropna on target!')
        print('DEBUG: DataFrame is empty after dropna on target!')
        print(df.head())
        raise ValueError('No data left after dropping NaN targets.')
    
    if len(df) < 100:
        logger.error('Not enough data for training. Need at least 100 samples.')
        raise ValueError("Not enough data for training. Need at least 100 samples.")
    
    X = df[feature_cols]
    # Filter out non-numeric columns
    X = X.select_dtypes(include=[np.number])
    y = df['target']
    
    if len(y.unique()) < 2:
        logger.error('Need both buy and sell signals for training.')
        raise ValueError("Need both buy and sell signals for training.")
    
    logger.info(f"Class distribution: {np.bincount(y)}")

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
                # Ensure all feature columns are present after upsampling
                for col in feature_cols:
                    if col not in Xy_balanced.columns:
                        Xy_balanced[col] = 0
                X_bal = Xy_balanced[feature_cols]
                y_bal = Xy_balanced['target']
                print('Class balance after upsampling:')
                print(y_bal.value_counts())
                from sklearn.utils import shuffle
                X_bal, y_bal = shuffle(X_bal, y_bal, random_state=42)
                logger.info('Upsampling successful.')
            else:
                print('Warning: Upsampled minority is not a valid DataFrame or is empty, skipping upsampling.')
                logger.warning('Upsampled minority is not a valid DataFrame or is empty, skipping upsampling.')
        else:
            print('Warning: One of the classes is not a valid DataFrame or is empty, skipping upsampling.')
            logger.warning('One of the classes is not a valid DataFrame or is empty, skipping upsampling.')
    else:
        print('Warning: Only one class present, skipping upsampling.')
        logger.warning('Only one class present, skipping upsampling.')
    
    # After upsampling, ensure X_bal is numeric only
    if isinstance(X_bal, pd.DataFrame):
        non_numeric_cols = [col for col in X_bal.columns if not pd.api.types.is_numeric_dtype(X_bal[col])]
        if non_numeric_cols:
            logger.warning(f"Dropping non-numeric columns after upsampling: {non_numeric_cols}")
            X_bal = X_bal.select_dtypes(include=[np.number])
    
    # Final check before train_test_split
    if X_bal is None or y_bal is None:
        raise ValueError('X_bal or y_bal is None before train_test_split. Check data pipeline and upsampling logic.')
    # No need for __iter__ check; pandas DataFrame/Series are always iterable if not None

    logger.info('Splitting train/test...')
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )
    
    logger.info('Scaling features...')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Base models
    logger.info('Initializing models (XGBoost, LightGBM, VotingClassifier)...')
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', scale_pos_weight=(y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1)
    lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, class_weight='balanced')
    # Ensemble with soft voting
    ensemble = VotingClassifier(estimators=[('xgb', xgb_model), ('lgb', lgb_model)], voting='soft')
    # Probability calibration
    calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
    logger.info('Model is fitting ensemble (XGBoost + LightGBM) and calibrating probabilities...')
    calibrated.fit(X_train_scaled, y_train)
    logger.info('Model fit complete.')
    
    y_pred = calibrated.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(calibrated, model_path)
    joblib.dump(scaler, model_path.replace('.pkl', '_scaler.pkl'))
    
    logger.info('Model is saving trained model and scaler...')
    logger.info(f"Model saved to {model_path}")
    
    # Print feature distributions for debugging
    logger.info('Feature distributions before training:')
    for col in feature_cols:
        col_data = df[col]
        logger.info(f"{col}: mean={col_data.mean():.4f}, std={col_data.std():.4f}, min={col_data.min():.4f}, max={col_data.max():.4f}, unique={col_data.nunique()}")
    
    # Audit data quality before training
    logger.info('Auditing data quality before training...')
    audit_data_quality(df, feature_cols, target_col='target')

    # Remove constant features
    constant_features = [col for col in feature_cols if df[col].nunique() <= 1]
    for col in constant_features:
        print(f'Removing constant feature: {col}')
        logger.warning(f'Removing constant feature: {col}')
    feature_cols = [col for col in feature_cols if col not in constant_features]
    if not feature_cols:
        raise ValueError('No features left after removing constant features!')
        logger.error('No features left after removing constant features!')
    # Re-run audit with updated feature_cols
    logger.info('Auditing data quality after removing constant features...')
    audit_data_quality(df, feature_cols, target_col='target')

    # SHAP and feature importance analysis (robust)
    logger.info('Model is engineering features and studying candlestick patterns...')
    try:
        import shap
        import matplotlib.pyplot as plt
        # For CalibratedClassifierCV, get the underlying VotingClassifier if available
        base_ensemble = getattr(calibrated, 'base_estimator_', None)
        if base_ensemble is not None:
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
            logger.info('Feature importance plot saved.')
        else:
            print('[Warning] CalibratedClassifierCV has no base_estimator_ attribute after fitting.')
            logger.warning('CalibratedClassifierCV has no base_estimator_ attribute after fitting.')
    except Exception as e:
        print(f'[Warning] SHAP/feature importance analysis failed: {e}')
        logger.warning(f'SHAP/feature importance analysis failed: {e}')
    logger.info('Model is analyzing feature importance and explainability (SHAP)...')
    # --- SHAP/Feature Importance Filtering ---
    try:
        import shap
        import matplotlib.pyplot as plt
        base_ensemble = getattr(calibrated, 'base_estimator_', None)
        if base_ensemble is not None:
            tree_model = base_ensemble.estimators_[0][1]
            explainer = shap.Explainer(tree_model, X_train)
            shap_values = explainer(X_train)
            shap_importance = np.abs(shap_values.values).mean(axis=0)
            feature_importance = tree_model.feature_importances_
            feature_ranking = sorted(zip(feature_cols, shap_importance, feature_importance), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in feature_ranking[:top_n_features]]
            logger.info(f"Top {top_n_features} features by SHAP: {top_features}")
            dropped_features = [f for f in feature_cols if f not in top_features]
            logger.info(f"Dropping uninformative features: {dropped_features}")
            feature_cols = top_features
            # Save feature list for prediction
            import json
            with open(model_path.replace('.pkl', '_features.json'), 'w') as f:
                json.dump(feature_cols, f)
        else:
            logger.warning('CalibratedClassifierCV has no base_estimator_ attribute after fitting.')
    except Exception as e:
        logger.warning(f'SHAP/feature importance analysis failed: {e}')
    logger.info('Model training complete! Ready for new predictions.')
    return calibrated, scaler, feature_cols

if __name__ == "__main__":
    # This will be called when training the model
    print("Training signal prediction model...")
    logger.info("Training signal prediction model...")
