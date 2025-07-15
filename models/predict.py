import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def load_model(model_path='models/saved_models/signal_model.pkl'):
    """
    Load the trained model and scaler.
    Args:
        model_path (str): Path to the saved model
    Returns:
        tuple: (model, scaler, feature_cols)
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
        
        # Define feature columns (same as in training)
        feature_cols = [
            'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 'rsi_14',
            'macd', 'macd_signal', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d', 
            'atr_14', 'news_sentiment'
        ]
        
        return model, scaler, feature_cols
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Please train the model first.")
        return None, None, None

def predict_signal(features_df, model_path='models/saved_models/signal_model.pkl'):
    """
    Predict trading signal for the latest data point.
    Args:
        features_df (pd.DataFrame): DataFrame with features
        model_path (str): Path to the saved model
    Returns:
        dict: Prediction results with signal, confidence, and probabilities
    """
    model, scaler, _ = load_model(model_path)
    
    if model is None:
        return None
    
    # Dynamically select feature columns (exclude price columns and target)
    exclude_cols = [col for col in features_df.columns if col.startswith(('Close', 'Open', 'High', 'Low', 'Volume'))] + ['future_return', 'target']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    # Get the latest data point
    latest_data = features_df[feature_cols].iloc[-1:].copy()
    
    # Scale the features
    latest_scaled = scaler.transform(latest_data)
    
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
