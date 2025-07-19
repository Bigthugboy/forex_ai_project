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

import os
from sklearn.utils import resample
from utils.logger import get_logger
import json
logger = get_logger('train_model', log_file='logs/train_model.log')
from models.model_registry import ModelRegistry
from config import Config, REMOVE_CONSTANT_FEATURES, ALWAYS_KEEP_FEATURES

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

def train_signal_model(features_df, pair, model_dir='models/saved_models/', top_n_features=10):
    import os
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'signal_model_{pair}.pkl')
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')
    features_path = model_path.replace('.pkl', '_features.json')
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
    
    # --- Ensure all possible pattern/structure columns are present before selecting feature_cols ---
    all_pattern_cols = [
        'double_bottom', 'double_top', 'fakeout_down', 'fakeout_up',
        'wyckoff_accumulation', 'wyckoff_distribution', 'wyckoff_markup', 'wyckoff_markdown', 'wyckoff_unknown',
        'head_shoulders', 'inv_head_shoulders', 'rising_wedge', 'falling_wedge',
        # Add any others you use
    ]
    for col in all_pattern_cols:
        if col not in df.columns:
            df[col] = 0

    # --- Select features: use all columns except target, future_return, and OHLCV columns ---
    exclude_cols = [col for col in df.columns if col.startswith(('Close', 'Open', 'High', 'Low', 'Volume'))] + ['future_return', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"[DEBUG] Final feature columns used for training: {feature_cols}")
    
    # Assign X before any use
    X = df[feature_cols]
    
    logger.info('Dropping NaN targets...')
    df = df.dropna(subset=['target'])
    print('DEBUG: df columns after dropna on target:', df.columns)
    print(df.head())
    # Ensure y is defined before use
    y = df['target']
    if len(df) < 100:
        logger.error(f'Not enough data for training {pair}. Need at least 100 samples, got {len(df)}.')
        raise ValueError(f"Not enough data for training {pair}. Need at least 100 samples, got {len(df)}.")
    if len(y.unique()) < 2:
        logger.error(f'Need both buy and sell signals for training {pair}.')
        raise ValueError(f"Need both buy and sell signals for training {pair}.")
    logger.info(f"Class balance before upsampling: {y.value_counts().to_dict()}")
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
    
    joblib.dump(calibrated, model_path)
    joblib.dump(scaler, scaler_path)
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f)
    logger.info(f"Model, scaler, and features saved for {pair}")

    # Register model in ModelRegistry
    try:
        registry = ModelRegistry()
        # Compute version as hash of model file and features
        import hashlib
        with open(model_path, 'rb') as mf:
            model_bytes = mf.read()
        version = hashlib.md5(model_bytes).hexdigest()
        metrics = {
            'accuracy': float(accuracy),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        registry.register_model(
            model_type=f'signal_model_{pair}',
            version=version,
            features=feature_cols,
            scaler_path=scaler_path,
            model_path=model_path,
            metrics=metrics,
            data_range=f"{df.index.min()} to {df.index.max()}",
            notes=None,
            retrain_source='train_signal_model',
            status='active'
        )
        logger.info(f"Model registered in ModelRegistry for {pair}, version {version}")
    except Exception as e:
        logger.error(f"Failed to register model in ModelRegistry: {e}")
    
    # Print feature distributions for debugging
    logger.info('Feature distributions before training:')
    for col in feature_cols:
        col_data = df[col]
        # Check if col_data is a Series (single column) or DataFrame (multiple columns)
        if isinstance(col_data, pd.Series):
            # Check if column is numeric before trying to format as float
            if pd.api.types.is_numeric_dtype(col_data):
                logger.info(f"{col}: mean={col_data.mean():.4f}, std={col_data.std():.4f}, min={col_data.min():.4f}, max={col_data.max():.4f}, unique={col_data.nunique()}")
            else:
                logger.info(f"{col}: dtype={col_data.dtype}, unique={col_data.nunique()}, sample_values={col_data.value_counts().head(3).to_dict()}")
        else:
            # Handle case where col_data is a DataFrame (multiple columns with same name)
            logger.info(f"{col}: DataFrame with {len(col_data.columns)} columns, shape={col_data.shape}")
    
    # Audit data quality before training
    logger.info('Auditing data quality before training...')
    audit_data_quality(df, feature_cols, target_col='target')

    # Remove constant features if enabled in config
    if REMOVE_CONSTANT_FEATURES:
        if feature_cols is None or not isinstance(feature_cols, list):
            logger.error('feature_cols is None or not a list before constant feature removal!')
            raise ValueError('feature_cols is None or not a list before constant feature removal!')
        if df is None or not hasattr(df, 'columns'):
            logger.error('df is None or not a DataFrame before constant feature removal!')
            raise ValueError('df is None or not a DataFrame before constant feature removal!')
        constant_features = [col for col in feature_cols if df[col].nunique() <= 1 and col not in ALWAYS_KEEP_FEATURES]
        for col in constant_features:
            print(f'Removing constant feature: {col}')
            logger.warning(f'Removing constant feature: {col}')
        feature_cols = [col for col in feature_cols if col not in constant_features]
        if not feature_cols:
            logger.error('No features left after removing constant features!')
            raise ValueError('No features left after removing constant features!')
        # Re-run audit with updated feature_cols
        logger.info('Auditing data quality after removing constant features...')
        audit_data_quality(df, feature_cols, target_col='target')

    # --- PATCH: Robustness check for feature_cols ---
    # Ensure feature_cols matches columns in X_bal/X_train
    if X_bal is not None and isinstance(X_bal, pd.DataFrame):
        final_train_cols = list(X_bal.columns)
        missing_in_train = [col for col in feature_cols if col not in final_train_cols]
        extra_in_train = [col for col in final_train_cols if col not in feature_cols]
        if missing_in_train:
            logger.warning(f"[Feature Consistency] Features in feature_cols but missing in training data: {missing_in_train}")
        if extra_in_train:
            logger.warning(f"[Feature Consistency] Features in training data but not in feature_cols: {extra_in_train}")
    else:
        final_train_cols = []
        logger.warning("[Feature Consistency] X_bal is None or not a DataFrame when checking final training columns. Skipping feature consistency checks.")
    # Always update feature_cols to match final training columns
    feature_cols = final_train_cols
    # Save feature list for prediction (always save, regardless of SHAP analysis)
    logger.info(f"[DEBUG] About to save feature columns: {feature_cols}")
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f)
    logger.info(f"[DEBUG] Feature columns saved to {features_path}")
    # --- END PATCH ---

    # --- PATCH: Ensure DataFrame with feature names for LightGBM/ensemble ---
    # Convert scaled arrays back to DataFrames with correct column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    # --- END PATCH ---

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
    # (Disabled: always use full feature set for all pairs)
    # try:
    #     import shap
    #     import matplotlib.pyplot as plt
    #     base_ensemble = getattr(calibrated, 'base_estimator_', None)
    #     if base_ensemble is not None:
    #         tree_model = base_ensemble.estimators_[0][1]
    #         explainer = shap.Explainer(tree_model, X_train)
    #         shap_values = explainer(X_train)
    #         shap_importance = np.abs(shap_values.values).mean(axis=0)
    #         feature_importance = tree_model.feature_importances_
    #         feature_ranking = sorted(zip(feature_cols, shap_importance, feature_importance), key=lambda x: x[1], reverse=True)
    #         top_features = [f[0] for f in feature_ranking[:top_n_features]]
    #         logger.info(f"Top {top_n_features} features by SHAP: {top_features}")
    #         dropped_features = [f for f in feature_cols if f not in top_features]
    #         logger.info(f"Dropping uninformative features: {dropped_features}")
    #         feature_cols = top_features
    #         # Update feature list file with top features
    #         with open(features_path, 'w') as f:
    #             json.dump(feature_cols, f)
    #         logger.info(f"Updated feature columns with top {top_n_features} features")
    #     else:
    #         logger.warning('CalibratedClassifierCV has no base_estimator_ attribute after fitting.')
    # except Exception as e:
    #     logger.warning(f'SHAP/feature importance analysis failed: {e}')
    logger.info('Model training complete! Ready for new predictions.')
    return {'model': calibrated, 'scaler': scaler, 'feature_cols': feature_cols, 'pair': pair}

if __name__ == "__main__":
    from config import Config
    from data.fetch_market import get_price_data
    from data.preprocess import preprocess_features
    from data.fetch_news import get_news_sentiment_with_cache
    import sys
    logger.info("Batch retraining all models for all pairs in Config.TRADING_PAIRS...")
    for pair in Config.TRADING_PAIRS:
        logger.info(f"Retraining model for {pair}...")
        price_df = get_price_data(pair, interval='1h', lookback=Config.LOOKBACK_PERIOD)
        if price_df is None or price_df.empty:
            logger.error(f"No price data for {pair}, skipping.")
            continue
        keywords = [pair]
        from_date = price_df.index[-Config.LOOKBACK_PERIOD].strftime('%Y-%m-%d')
        to_date = price_df.index[-1].strftime('%Y-%m-%d')
        sentiment = get_news_sentiment_with_cache(keywords, from_date, to_date, pair)
        features_df = preprocess_features(price_df, sentiment, use_multi_timeframe=True)
        try:
            train_signal_model(features_df, pair)
            logger.info(f"Retrained and saved model for {pair}.")
        except Exception as e:
            logger.error(f"Error retraining model for {pair}: {e}")
    logger.info("Batch retraining complete.")
