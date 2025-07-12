"""
preprocessing.py
Reusable feature engineering and preprocessing functions for Vinrox Technologies order prioritization.
These functions transform raw order data into the features required by the trained model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Paths for encoders and scaler (fit on training data and saved for reuse)
SCALER_PATH = 'scaler.joblib'
ENCODERS_PATH = 'encoders.joblib'

# Load or fit encoders/scaler as needed
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    scaler = None  # Will fit on the fly if not found

if os.path.exists(ENCODERS_PATH):
    encoders = joblib.load(ENCODERS_PATH)
else:
    encoders = {}

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['due_date'] = pd.to_datetime(df['due_date'])
    df['days_to_due'] = (df['due_date'] - df['order_date']).dt.days
    df['production_load'] = df['quantity'] / (df['resource_available'] + 1)
    df['urgency_score'] = df['days_to_due'] / (df['estimated_production_time'] + 1e-3)
    df['is_delayed'] = (df['status'] == 'delayed').astype(int)
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Use or fit LabelEncoders for each categorical column
    cat_cols = ['product_type', 'customer_segment', 'machine_id', 'status']
    for col in cat_cols:
        if col in df.columns:
            if col in encoders:
                le = encoders[col]
            else:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                encoders[col] = le
            df[col + '_enc'] = le.transform(df[col].astype(str))
    return df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = ['quantity', 'days_to_due', 'production_load', 'urgency_score']
    # Fit scaler if not loaded
    global scaler
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[num_cols])
    scaled = scaler.transform(df[num_cols])
    for i, col in enumerate(num_cols):
        df[col + '_scaled'] = scaled[:, i]
    return df

def preprocess_order_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: feature engineering, encoding, scaling.
    Args:
        df (pd.DataFrame): Raw order data.
    Returns:
        pd.DataFrame: DataFrame with all features required by the model.
    """
    df = feature_engineer(df)
    df = encode_features(df)
    df = scale_features(df)
    return df

# Save encoders and scaler for reuse (call after fitting on training data)
def save_preprocessing_artifacts():
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
