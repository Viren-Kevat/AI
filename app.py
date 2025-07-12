import streamlit as st
import pandas as pd
from decision_engine import prioritize_orders
from preprocessing import preprocess_order_data

st.title("Order Priority Prediction (Vinrox Technologies)")

st.sidebar.header("Manual Order Entry (Single Prediction)")

# Manual input widgets for raw order features
order_date = st.sidebar.date_input("Order Date")
due_date = st.sidebar.date_input("Due Date")
quantity = st.sidebar.number_input("Quantity", min_value=1, value=10)
resource_available = st.sidebar.number_input("Resource Available", min_value=0, value=5)
estimated_production_time = st.sidebar.number_input("Estimated Production Time", min_value=1, value=3)
product_type = st.sidebar.text_input("Product Type", value="A")
customer_segment = st.sidebar.text_input("Customer Segment", value="Retail")
machine_id = st.sidebar.text_input("Machine ID", value="M1")
status = st.sidebar.text_input("Status", value="on_time")

st.sidebar.markdown("---")
st.sidebar.markdown("Or upload a CSV file below for batch prediction.")

uploaded_file = st.file_uploader("Upload order data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    try:
        processed_df = preprocess_order_data(df)
        preds = prioritize_orders(processed_df)
        df['Predicted Priority'] = preds
        st.write("Predicted Priorities:")
        st.dataframe(df)
        st.download_button("Download Results", df.to_csv(index=False), "prioritized_orders.csv")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    if st.sidebar.button("Predict for Manual Entry"):
        # Build a DataFrame from manual input
        input_dict = {
            'order_date': [order_date],
            'due_date': [due_date],
            'quantity': [quantity],
            'resource_available': [resource_available],
            'estimated_production_time': [estimated_production_time],
            'product_type': [product_type],
            'customer_segment': [customer_segment],
            'machine_id': [machine_id],
            'status': [status]
        }
        input_df = pd.DataFrame(input_dict)
        try:
            processed_df = preprocess_order_data(input_df)
            pred = prioritize_orders(processed_df)[0]
            st.success(f"Predicted Priority Score: {pred:.3f}")
        except Exception as e:
            st.error(f"Error: {e}")
