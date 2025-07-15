import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.predict import load_model
import xgboost as xgb
import lightgbm as lgb
import os
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

def load_base_models(model_path='models/saved_models/signal_model.pkl'):
    """
    Load the VotingClassifier ensemble and extract base XGBoost and LightGBM models.
    Returns: (xgb_model, lgb_model, scaler, feature_cols)
    """
    model, scaler, feature_cols = load_model(model_path)
    if model is None or scaler is None or feature_cols is None:
        raise FileNotFoundError(f"Model, scaler, or feature_cols not found at {model_path}. Please train the model first.")
    # Extract fitted VotingClassifier from CalibratedClassifierCV
    if isinstance(model, CalibratedClassifierCV):
        # Use the first calibrated classifier (should be sufficient for SHAP)
        ensemble = model.calibrated_classifiers_[0].estimator
    else:
        ensemble = model
    xgb_model = None
    lgb_model = None
    if hasattr(ensemble, 'estimators_'):
        for name, est in ensemble.estimators_:
            if isinstance(est, xgb.XGBClassifier):
                xgb_model = est
            elif isinstance(est, lgb.LGBMClassifier):
                lgb_model = est
    if xgb_model is None or lgb_model is None:
        raise ValueError("Could not extract XGBoost or LightGBM base models from ensemble.")
    return xgb_model, lgb_model, scaler, feature_cols

def explain_with_shap(features_df, model_type='xgb', model_path='models/saved_models/signal_model.pkl', plot=True, max_display=15):
    """
    Compute and plot SHAP values for the given features using the specified base model.
    Args:
        features_df (pd.DataFrame): DataFrame of features (unscaled)
        model_type (str): 'xgb' or 'lgb'
        model_path (str): Path to the saved ensemble model
        plot (bool): Whether to show SHAP summary plot
        max_display (int): Max features to display in plot
    Returns:
        shap_values, explainer
    """
    xgb_model, lgb_model, scaler, feature_cols = load_base_models(model_path)
    # Select model
    if model_type == 'xgb':
        model = xgb_model
    elif model_type == 'lgb':
        model = lgb_model
    else:
        raise ValueError("model_type must be 'xgb' or 'lgb'")
    # Select and scale features
    if not isinstance(feature_cols, list) or not all(isinstance(f, str) for f in feature_cols):
        raise ValueError("feature_cols must be a list of strings.")
    X = features_df[feature_cols]
    if scaler is None:
        raise ValueError("Scaler is None. Cannot scale features for SHAP explanation.")
    X_scaled = scaler.transform(X)
    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    if plot:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_cols, max_display=max_display, show=True)
    return shap_values, explainer

def explain_single_sample(features_df, idx=-1, model_type='xgb', model_path='models/saved_models/signal_model.pkl'):
    """
    Show SHAP force plot for a single sample (default: latest row).
    """
    xgb_model, lgb_model, scaler, feature_cols = load_base_models(model_path)
    if model_type == 'xgb':
        model = xgb_model
    elif model_type == 'lgb':
        model = lgb_model
    else:
        raise ValueError("model_type must be 'xgb' or 'lgb'")
    if not isinstance(feature_cols, list) or not all(isinstance(f, str) for f in feature_cols):
        raise ValueError("feature_cols must be a list of strings.")
    X = features_df[feature_cols]
    if scaler is None:
        raise ValueError("Scaler is None. Cannot scale features for SHAP explanation.")
    X_scaled = scaler.transform(X)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    # Use SHAP's force plot for a single sample
    shap.initjs()
    return shap.force_plot(explainer.expected_value, shap_values[idx], X.iloc[idx], feature_names=feature_cols)

if __name__ == "__main__":
    # Example usage: run SHAP on a batch of features
    import sys
    try:
        model, scaler, feature_cols = load_model()
        if model is None or scaler is None or feature_cols is None:
            raise ValueError("Model, scaler, or feature_cols is None. Please check your model and feature engineering.")
        if not isinstance(feature_cols, list) or not all(isinstance(f, str) for f in feature_cols):
            raise ValueError("feature_cols must be a list of strings.")
        n_samples = 100
        X_demo = pd.DataFrame(np.random.randn(n_samples, len(feature_cols)), columns=feature_cols)
        print("Running SHAP summary for XGBoost base model...")
        explain_with_shap(X_demo, model_type='xgb', plot=True)
        print("Running SHAP summary for LightGBM base model...")
        explain_with_shap(X_demo, model_type='lgb', plot=True)
        # Show force plot for last sample (requires Jupyter/HTML)
        # force_plot = explain_single_sample(X_demo, idx=-1, model_type='xgb')
        # shap.save_html('shap_force_plot.html', force_plot)
    except Exception as e:
        print(f"Error running SHAP analysis: {e}") 