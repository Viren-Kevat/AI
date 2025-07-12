"""
decision_engine.py
Core prioritization functions for Vinrox Technologies order prioritization.
Loads the trained model and provides a function to score/prioritize new orders.
"""

import joblib
import pandas as pd
import numpy as np

# Load the trained model (ensure the path is correct)
MODEL_PATH = 'priority_model.joblib'
model = joblib.load(MODEL_PATH)

# List of features expected by the model (must match training)
RELEVANT_FEATURES = [
    'quantity_scaled',
    'product_type_enc',
    'customer_segment_enc',
    'machine_id_enc',
    'status_enc',
    'days_to_due_scaled',
    'production_load_scaled',
    'urgency_score_scaled',
    'is_delayed'
]

def prioritize_orders(order_df: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame of (preprocessed & feature-engineered) orders, return predicted priority scores.
    Args:
        order_df (pd.DataFrame): DataFrame with all required features/columns.
    Returns:
        np.ndarray: Predicted priority scores.
    """
    # Ensure all required features are present
    missing = [f for f in RELEVANT_FEATURES if f not in order_df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    X = order_df[RELEVANT_FEATURES]
    return model.predict(X)

# Example usage (uncomment for testing):
# new_orders = pd.read_csv('new_orders.csv')
# preds = prioritize_orders(new_orders)
# print(preds)
