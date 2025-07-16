import pandas as pd
import os
from models.train_model import train_signal_model
from config import Config
from data.preprocess import preprocess_features
from utils.logger import get_logger
import joblib

logger = get_logger('retrain', log_file='logs/retrain.log')

SIGNALS_CSV = 'logs/signals.csv'
MODEL_PATH = 'models/saved_models/signal_model.pkl'
SCALER_PATH = 'models/saved_models/signal_model_scaler.pkl'

# 1. Load signal log
if not os.path.exists(SIGNALS_CSV):
    logger.error(f"Signal log not found at {SIGNALS_CSV}")
    exit(1)
df = pd.read_csv(SIGNALS_CSV)

# 2. Filter for resolved signals
resolved = df[df['outcome'].isin(['TP1', 'TP2', 'TP3', 'SL'])].copy()
if resolved.empty:
    logger.error("No resolved signals with outcomes (TP1, TP2, TP3, SL) found.")
    exit(1)

# 3. Prepare features and target
def extract_features(row):
    # If features are stored as JSON string, parse them
    if 'features' in row and isinstance(row['features'], str):
        try:
            import json
            return pd.Series(json.loads(row['features']))
        except Exception:
            return pd.Series()
    return pd.Series()

features_df = resolved.apply(extract_features, axis=1)

# Drop rows with no features
features_df = features_df.dropna(axis=0, how='all')
resolved = resolved.loc[features_df.index]

# 4. Create target: 1 for TP*, 0 for SL
resolved['target'] = resolved['outcome'].apply(lambda x: 1 if x in ['TP1', 'TP2', 'TP3'] else 0)

# 5. Align features and target
y = resolved['target']

# 6. Retrain model
logger.info(f"Retraining model on {len(features_df)} samples from signal log...")
model, scaler, feature_cols = train_signal_model(features_df, model_path=MODEL_PATH)

# 7. Save model and scaler
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
logger.info(f"Retrained model and scaler saved to {MODEL_PATH} and {SCALER_PATH}")

print(f"Retraining complete. Model saved to {MODEL_PATH}") 