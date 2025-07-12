# Vinrox Technologies Order Prioritization AI

## Overview
This project provides an end-to-end AI-driven solution for order prioritization in manufacturing or supply chain environments. It includes:
- Data preprocessing and feature engineering
- Model training and evaluation (Random Forest Regressor)
- Model explainability (SHAP)
- A reusable decision engine for batch or real-time scoring
- A Streamlit web UI for both manual and batch predictions

## Project Structure
```
├── app.py                  # Streamlit web application for predictions
├── decision_engine.py      # Core model loading and prediction logic
├── preprocessing.py        # Feature engineering and preprocessing functions
├── priority_model_training.ipynb  # Jupyter notebook for model development
├── orders.csv              # Example dataset (raw orders)
├── priority_model.joblib   # Trained model artifact
├── scaler.joblib           # Saved StandardScaler from training
├── encoders.joblib         # Saved LabelEncoders from training
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

## How It Works
1. **Data Preparation:**
   - Raw order data is loaded from `orders.csv`.
   - Feature engineering and preprocessing are performed (see `preprocessing.py`).
2. **Model Training:**
   - The notebook (`priority_model_training.ipynb`) guides you through EDA, feature engineering, temporal train-test split, model training, evaluation, and SHAP analysis.
   - The trained model, scaler, and encoders are saved for deployment.
3. **Prediction Engine:**
   - `decision_engine.py` loads the trained model and provides a function to score new orders using the same features as in training.
4. **Web UI:**
   - `app.py` provides a Streamlit interface for users to either upload a CSV or manually enter order details.
   - All preprocessing and feature engineering are automated.
   - Results are displayed and can be downloaded.

## Setup Instructions
1. **Clone the repository or copy the project files.**
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Train the model (if not already trained):**
   - Open and run all cells in `priority_model_training.ipynb`.
   - This will save the model, scaler, and encoders for use in the app.
4. **Run the Streamlit app:**
   ```
   streamlit run app.py
   ```
5. **Use the app:**
   - Upload a CSV file with raw order data, or
   - Enter order details manually in the sidebar.
   - The app will preprocess, predict, and display/download results.

## File Descriptions
- **app.py:** Streamlit UI for predictions.
- **decision_engine.py:** Loads the trained model and predicts priority scores for new orders.
- **preprocessing.py:** Contains all feature engineering, encoding, and scaling logic. Loads saved encoders/scaler for consistency.
- **priority_model_training.ipynb:** Full ML workflow, including EDA, feature engineering, model training, evaluation, and explainability.
- **orders.csv:** Example input data (raw orders).
- **priority_model.joblib:** Trained Random Forest Regressor model.
- **scaler.joblib:** StandardScaler fitted on training data.
- **encoders.joblib:** LabelEncoders fitted on training data for categorical features.

## Notes
- If you retrain the model or add new categories, rerun the notebook and re-save the encoders/scaler.
- The app expects the same columns and data types as in the training set.
- For best results, use the provided preprocessing pipeline for all new data.

## Contact
For questions or support, contact the Vinrox Technologies data team.
# AI
