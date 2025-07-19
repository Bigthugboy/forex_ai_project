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
import hashlib
from pymongo import MongoClient
import gridfs
from config import Config

logger = get_logger('model_training_service', log_file='logs/model_training_service.log')

class ModelTrainingService:
    def __init__(self, model_dir='models/saved_models/', min_samples=100, min_classes=2, mongo_uri='mongodb://localhost:27017/', mongo_db='forex_ai', mongo_collection='models'):
        self.model_dir = model_dir
        self.min_samples = min_samples
        self.min_classes = min_classes
        os.makedirs(self.model_dir, exist_ok=True)
        # MongoDB setup
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.mongo_collection = mongo_collection
        self.mongo_client = MongoClient(self.mongo_uri)
        self.db = self.mongo_client[self.mongo_db]
        self.fs = gridfs.GridFS(self.db)
        self.collection = self.db[self.mongo_collection]
        logger.info(f"MongoDB connected: {self.mongo_uri}, db: {self.mongo_db}, collection: {self.mongo_collection}")

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

    def get_model_version(self, pair, config_hash=None):
        """Compute a version hash based on model code and config."""
        # Use the hash of this file and config hash
        code_path = os.path.abspath(__file__)
        with open(code_path, 'rb') as f:
            code_bytes = f.read()
        code_hash = hashlib.md5(code_bytes).hexdigest()
        version_str = code_hash
        if config_hash:
            version_str += str(config_hash)
        return hashlib.md5(version_str.encode()).hexdigest()

    def save_version(self, pair, version):
        version_path = os.path.join(self.model_dir, f'signal_model_{pair}_version.txt')
        with open(version_path, 'w') as f:
            f.write(version)
        logger.info(f"Saved model version for {pair}: {version}")

    def load_version(self, pair):
        version_path = os.path.join(self.model_dir, f'signal_model_{pair}_version.txt')
        if not os.path.exists(version_path):
            return None
        with open(version_path, 'r') as f:
            return f.read().strip()

    def save_to_mongo(self, pair, model, scaler, feature_cols, version, accuracy):
        import pickle
        # Remove old entry for this pair/version
        self.collection.delete_many({'pair': pair, 'version': version})
        # Save model and scaler as binary
        model_bin = pickle.dumps(model)
        scaler_bin = pickle.dumps(scaler)
        model_id = self.fs.put(model_bin, filename=f'{pair}_model_{version}')
        scaler_id = self.fs.put(scaler_bin, filename=f'{pair}_scaler_{version}')
        doc = {
            'pair': pair,
            'version': version,
            'feature_cols': feature_cols,
            'accuracy': accuracy,
            'model_id': model_id,
            'scaler_id': scaler_id,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self.collection.insert_one(doc)
        logger.info(f"Model for {pair} (version {version}) saved to MongoDB.")

    def load_from_mongo(self, pair, version):
        import pickle
        doc = self.collection.find_one({'pair': pair, 'version': version})
        if not doc:
            logger.warning(f"No model found in MongoDB for {pair} version {version}")
            return None, None, None
        model_bin = self.fs.get(doc['model_id']).read()
        scaler_bin = self.fs.get(doc['scaler_id']).read()
        model = pickle.loads(model_bin)
        scaler = pickle.loads(scaler_bin)
        feature_cols = doc['feature_cols']
        logger.info(f"Model for {pair} (version {version}) loaded from MongoDB.")
        return model, scaler, feature_cols

    def train(self, df, pair, required_features=None, save=True, config_hash=None, force_retrain=False):
        logger.info(f"Starting training for {pair}...")
        current_version = self.get_model_version(pair, config_hash)
        saved_version = self.load_version(pair)
        # Try to load from Mongo first
        if not force_retrain and saved_version == current_version:
            logger.info(f"Model for {pair} is up to date (version: {current_version}). Loading from MongoDB.")
            model, scaler, feature_cols = self.load_from_mongo(pair, current_version)
            if model and scaler and feature_cols:
                return {'model': model, 'scaler': scaler, 'feature_cols': feature_cols, 'accuracy': None}
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
            self.save_version(pair, current_version)
            logger.info(f"Model, scaler, features, and version saved for {pair}")
            # Save to MongoDB
            self.save_to_mongo(pair, calibrated, scaler, feature_cols, current_version, accuracy)
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

    def retrain_model_for_pair(self, pair):
        if pair not in Config.TRADING_PAIRS:
            logger.warning(f"[TRAIN] Skipping {pair}: not in config.TRADING_PAIRS.")
            return
        """Retrain the model for a single pair, using the current feature set."""
        logger.info(f"[Auto-Retrain] Retraining model for {pair} due to feature mismatch or update.")
        from data.fetch_market import get_price_data
        from data.preprocess import preprocess_features
        price_df = get_price_data(pair, interval='1h', lookback=200)
        if price_df is None or price_df.empty:
            logger.error(f"[Auto-Retrain] No price data for {pair}. Cannot retrain.")
            return
        # Use dummy sentiment score (0.0) for retrain
        features_df = preprocess_features(price_df, sentiment_score=0.0, use_multi_timeframe=True)
        from models.train_model import train_signal_model
        try:
            train_signal_model(features_df, pair, model_dir=self.model_dir)
            logger.info(f"[Auto-Retrain] Model retrained and saved for {pair}.")
        except Exception as e:
            logger.error(f"[Auto-Retrain] Model retrain failed for {pair}: {e}")

    def retrain_all_models(self):
        for pair in Config.TRADING_PAIRS:
            self.retrain_model_for_pair(pair)
        # Optionally, clean up model files for pairs not in config
        model_files = os.listdir(self.model_dir)
        for fname in model_files:
            for ext in ['.pkl', '_scaler.pkl', '_features.json', '_version.txt']:
                if fname.endswith(ext):
                    pair_name = fname.replace('signal_model_', '').replace(ext, '').replace('_scaler', '').replace('_features', '').replace('_version', '')
                    if pair_name not in Config.TRADING_PAIRS:
                        try:
                            os.remove(os.path.join(self.model_dir, fname))
                            logger.info(f"[CLEANUP] Removed model file for non-config pair: {fname}")
                        except Exception as e:
                            logger.warning(f"[CLEANUP] Failed to remove {fname}: {e}")

# --- Unit test stub for MongoDB model persistence ---
def test_mongo_model_persistence():
    mts = ModelTrainingService()
    import numpy as np
    import pandas as pd
    # Dummy data with all required features
    feature_cols = [
        'Close', 'Open', 'High', 'Low', 'Volume',
        'supply_zone', 'demand_zone', 'wyckoff_markdown', 'wyckoff_markup', 'wyckoff_unknown',
        'wyckoff_accumulation', 'wyckoff_distribution',
        'head_shoulders', 'inv_head_shoulders', 'double_top', 'double_bottom',
        'rising_wedge', 'falling_wedge', 'fakeout_up', 'fakeout_down',
        'sma_20_4h', 'sma_50_4h', 'ema_12_4h', 'ema_26_4h', 'rsi_14_4h',
        'macd_4h', 'macd_signal_4h', 'bb_high_4h', 'bb_low_4h', 'atr_14_4h', 'volatility_20_4h',
        'price_vs_sma20_4h', 'price_vs_sma50_4h', 'trend_strength_4h',
        'atr_14', 'volatility_20'
    ]
    n = 120
    data = {col: np.random.rand(n) for col in feature_cols}
    df = pd.DataFrame(data)
    df['target'] = np.random.randint(0, 2, size=n)
    pair = 'TESTPAIR'
    mts.train(df, pair, save=True, force_retrain=True)
    version = mts.get_model_version(pair)
    model, scaler, feature_cols = mts.load_from_mongo(pair, version)
    assert model is not None and scaler is not None and feature_cols is not None
    print("MongoDB model persistence test passed.") 