import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
import json
import logging

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

def predict_signal(features_df, pair, model_dir='models/saved_models/'):
    """
    Predict trading signal for the latest data point.
    Args:
        features_df (pd.DataFrame): DataFrame with features
        model_path (str): Path to the saved model
    Returns:
        dict: Prediction results with signal, confidence, and probabilities
    """
    model, scaler, feature_cols = load_model(pair, model_dir)
    
    if model is None or scaler is None or not feature_cols:
        return None
    
    # Only use columns in feature_cols (do not add/fill columns not in feature_cols)
    features_df = features_df.reindex(columns=feature_cols)
    latest_data = features_df.iloc[-1:].copy()
    # Final guarantee: reindex latest_data to feature_cols
    latest_data = latest_data.reindex(columns=feature_cols)
    latest_data = latest_data.astype(float)
    # Debug: print column differences
    missing = [col for col in feature_cols if col not in latest_data.columns]
    extra = [col for col in latest_data.columns if col not in feature_cols]
    print(f"[DEBUG] Missing in latest_data: {missing}")
    print(f"[DEBUG] Extra in latest_data: {extra}")
    print(f"[DEBUG] latest_data columns: {list(latest_data.columns)}")
    print(f"[DEBUG] feature_cols: {feature_cols}")
    print(f"[DEBUG] Final columns for prediction: {list(latest_data.columns)}")
    print(f"[DEBUG] Model expects: {feature_cols}")
    print("[DEBUG] Columns in latest_data before scaling:", list(latest_data.columns))
    print("[DEBUG] Model expects:", feature_cols)

    # Check for duplicate columns
    duplicates = latest_data.columns[latest_data.columns.duplicated()].tolist()
    print(f"[DEBUG] Duplicate columns in latest_data: {duplicates}")
    # Print dtypes
    print(f"[DEBUG] latest_data dtypes: {latest_data.dtypes}")
    # Print values going into scaler
    print(f"[DEBUG] latest_data values going into scaler:\n{latest_data.values}")
    # Now scale
    latest_scaled = scaler.transform(latest_data)
    print(f"[DEBUG] Output from scaler:\n{latest_scaled}")
    
    # Make prediction
    prediction = model.predict(latest_scaled)[0]
    probabilities = model.predict_proba(latest_scaled)[0]
    confidence = max(probabilities)
    
    # Determine signal type
    if prediction == 1:
        signal_type = "BUY"
    else:
        signal_type = "SELL"
    
    return {
        'signal': signal_type,
        'confidence': confidence,
        'probabilities': probabilities,
        'prediction': prediction
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
