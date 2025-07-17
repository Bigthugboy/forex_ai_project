import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample, shuffle
import xgboost as xgb
import lightgbm as lgb
from utils.logger import get_logger

logger = get_logger('model_training_service', log_file='logs/model_training_service.log')

class ModelTrainingService:
    def __init__(self, model_dir='models/saved_models/', min_samples=100, min_classes=2):
        self.model_dir = model_dir
        self.min_samples = min_samples
        self.min_classes = min_classes
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_target(self, df, close_col=None, threshold=0.001, horizon=5):
        """
        Add a binary target column to df based on future price movement.
        """
        if close_col is None:
            close_col = [col for col in df.columns if col.lower().startswith('close')][0]
        df = df.copy()
        df['future_return'] = df[close_col].shift(-horizon) / df[close_col] - 1
        df['target'] = (df['future_return'] > threshold).astype(int)
        return df

    def select_features(self, df, required_features=None, exclude_cols=None):
        if required_features is None:
            required_features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'future_return', 'target']]
        if exclude_cols is None:
            exclude_cols = [col for col in df.columns if col.startswith(('Close', 'Open', 'High', 'Low', 'Volume'))] + ['future_return', 'target']
        feature_cols = [col for col in required_features if col in df.columns and col not in exclude_cols]
        return feature_cols

    def upsample(self, X, y):
        # Ensure X is a DataFrame and y is a Series
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            logger.error('X must be a DataFrame and y must be a Series for upsampling.')
            return X, y
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
                    # Ensure X.columns is a list of column names
                    if not hasattr(X, 'columns') or X.columns is None:
                        logger.error('X.columns is None in upsample, cannot reindex Xy_balanced.')
                        return X, y
                    X_bal = Xy_balanced[X.columns]
                    y_bal = Xy_balanced['target']
                    X_bal, y_bal = shuffle(X_bal, y_bal, random_state=42)
                    return X_bal, y_bal
        return X, y

    def train(self, df, pair, required_features=None, save=True):
        logger.info(f"Starting training for {pair}...")
        df = self.prepare_target(df)
        if 'target' not in df.columns:
            logger.error('Target column missing after prepare_target')
            raise ValueError('Target column missing after prepare_target')
        if len(df) < self.min_samples:
            logger.error(f'Not enough data for training {pair}. Need at least {self.min_samples}, got {len(df)}.')
            raise ValueError(f'Not enough data for training {pair}.')
        if len(df["target"].unique()) < self.min_classes:
            logger.error(f'Need at least {self.min_classes} classes for training {pair}.')
            raise ValueError(f'Need at least {self.min_classes} classes for training {pair}.')
        feature_cols = self.select_features(df, required_features)
        X = df[feature_cols]
        y = df['target']
        X, y = self.upsample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Fix scale_pos_weight calculation
        if not isinstance(y, pd.Series):
            logger.error('y must be a pandas Series for scale_pos_weight calculation.')
            n_pos = 1
            n_neg = 1
        else:
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
        lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, class_weight='balanced')
        ensemble = VotingClassifier(estimators=[('xgb', xgb_model), ('lgb', lgb_model)], voting='soft')
        calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
        logger.info('Fitting ensemble and calibrating probabilities...')
        calibrated.fit(X_train_scaled, y_train)
        y_pred = calibrated.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:\n%s" % classification_report(y_test, y_pred))
        model_path = os.path.join(self.model_dir, f'signal_model_{pair}.pkl')
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        features_path = model_path.replace('.pkl', '_features.json')
        if save:
            joblib.dump(calibrated, model_path)
            joblib.dump(scaler, scaler_path)
            with open(features_path, 'w') as f:
                json.dump(feature_cols, f)
            logger.info(f"Model, scaler, and features saved for {pair}")
        return {
            'model': calibrated,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'accuracy': accuracy
        }

    def load(self, pair):
        model_path = os.path.join(self.model_dir, f'signal_model_{pair}.pkl')
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        features_path = model_path.replace('.pkl', '_features.json')
        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
            logger.error(f"Model, scaler, or features not found for {pair}")
            return None, None, None
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(features_path, 'r') as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols 