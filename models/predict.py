import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
import json
import logging
from models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

def load_model(pair, model_dir='models/saved_models/'):
    """
    Load the trained model and scaler.
    Args:
        model_path (str): Path to the saved model
    Returns:
        tuple: (model, scaler, feature_cols)
    """
    import os, json
    model_path = os.path.join(model_dir, f'signal_model_{pair}.pkl')
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')
    features_path = model_path.replace('.pkl', '_features.json')
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                feature_cols = json.load(f)
        else:
            feature_cols = []
        return model, scaler, feature_cols
    except FileNotFoundError:
        print(f"Model not found for {pair} at {model_path}. Please train the model first.")
        return None, None, None

def load_model_registry_aware(pair, model_dir='models/saved_models/'):
    """
    Load the latest/active model for a pair using ModelRegistry. Fallback to disk if not found.
    Returns: (model, scaler, feature_cols)
    """
    registry = ModelRegistry()
    entry = registry.get_latest_model(f'signal_model_{pair}')
    if entry and entry.get('model_path') and entry.get('scaler_path') and entry.get('features'):
        try:
            import joblib
            import json
            model = joblib.load(entry['model_path'])
            scaler = joblib.load(entry['scaler_path'])
            feature_cols = entry['features']
            return model, scaler, feature_cols
        except Exception as e:
            logger.warning(f"[ModelRegistry] Failed to load model from registry for {pair}: {e}. Falling back to disk.")
    # Fallback to disk
    return load_model(pair, model_dir)

def predict_signal(features_df, pair, model_dir='models/saved_models/', mode='live'):
    """
    Predict trading signal for the latest data point.
    Args:
        features_df (pd.DataFrame): DataFrame with features
        model_path (str): Path to the saved model
        mode (str): 'live' or 'study_only'
    Returns:
        dict: Prediction results with signal, confidence, and probabilities
    """
    model, scaler, feature_cols = load_model_registry_aware(pair, model_dir)
    logger.info(f"[DEBUG] Loaded model for {pair} in mode={mode}")
    if model is None or scaler is None or not feature_cols:
        logger.warning(f"[DEBUG] Model/scaler/feature_cols missing for {pair}. Returning None.")
        return None
    # --- PATCH: Enforce feature consistency ---
    missing_cols = [col for col in feature_cols if col not in features_df.columns]
    extra_cols = [col for col in features_df.columns if col not in feature_cols]
    if missing_cols:
        logger.warning(f"[Feature Consistency] Missing columns for prediction: {missing_cols}. Filling with 0.0.")
        for col in missing_cols:
            features_df[col] = 0.0
    if extra_cols:
        logger.warning(f"[Feature Consistency] Extra columns in features_df: {extra_cols}. These will be dropped.")
    # Reindex to match feature_cols exactly
    features_df = features_df.reindex(columns=feature_cols)
    latest_data = features_df.iloc[-1:].copy()
    latest_data = latest_data.reindex(columns=feature_cols)
    latest_data = latest_data.astype(float)
    # --- END PATCH ---
    # Debug: print column differences
    missing = [col for col in feature_cols if col not in latest_data.columns]
    extra = [col for col in latest_data.columns if col not in feature_cols]
    logger.info(f"[DEBUG] Missing in latest_data: {missing}")
    logger.info(f"[DEBUG] Extra in latest_data: {extra}")
    logger.info(f"[DEBUG] latest_data columns: {list(latest_data.columns)}")
    logger.info(f"[DEBUG] feature_cols: {feature_cols}")
    logger.info(f"[DEBUG] Final columns for prediction: {list(latest_data.columns)}")
    logger.info(f"[DEBUG] Model expects: {feature_cols}")
    logger.info(f"[DEBUG] Columns in latest_data before scaling: {list(latest_data.columns)}")
    logger.info(f"[DEBUG] Model expects: {feature_cols}")
    # Check for duplicate columns
    duplicates = latest_data.columns[latest_data.columns.duplicated()].tolist()
    logger.info(f"[DEBUG] Duplicate columns in latest_data: {duplicates}")
    logger.info(f"[DEBUG] latest_data dtypes: {latest_data.dtypes}")
    logger.info(f"[DEBUG] latest_data values going into scaler:\n{latest_data.values}")
    # Now scale
    latest_scaled = scaler.transform(latest_data)
    if not isinstance(latest_scaled, pd.DataFrame):
        latest_scaled = pd.DataFrame(latest_scaled, columns=feature_cols, index=latest_data.index)
    logger.info(f"[DEBUG] Output from scaler:\n{latest_scaled}")
    prediction = model.predict(latest_scaled)[0]
    probabilities = model.predict_proba(latest_scaled)[0]
    confidence = max(probabilities)
    # Determine signal type
    if prediction == 1:
        signal_type = "BUY"
    else:
        signal_type = "SELL"
    # Study mode: allow signals at lower confidence, log as study
    min_conf = 0.75 if mode == 'live' else 0.65
    if confidence < min_conf:
        logger.info(f"[FILTERED] {pair} | Mode={mode} | Confidence {confidence:.2f} below threshold {min_conf}. No signal generated.")
        return None
    logger.info(f"[SIGNAL] {pair} | Mode={mode} | Signal: {signal_type}, Confidence: {confidence:.2f}")
    return {
        'signal': signal_type,
        'confidence': confidence,
        'probabilities': probabilities,
        'prediction': prediction,
        'mode': mode
    }

def get_signal_strength(confidence):
    """
    Convert confidence score to signal strength description.
    """
    if confidence >= 0.8:
        return "Strong"
    elif confidence >= 0.6:
        return "Moderate"
    else:
        return "Weak"

if __name__ == "__main__":
    # Test prediction functionality
    print("Signal prediction module loaded.")
