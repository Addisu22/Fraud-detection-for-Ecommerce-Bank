import shap
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError


def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def compute_shap_values(model, X_sample):
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)
        return explainer, shap_values
    except NotFittedError:
        raise RuntimeError("Model is not fitted")
    except Exception as e:
        raise RuntimeError(f"SHAP computation failed: {e}")
