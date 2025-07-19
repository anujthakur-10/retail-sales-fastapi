import streamlit as st
import requests

st.set_page_config(page_title="Sales Forecaster", layout="centered")

st.title("📈 Monthly Sales Forecasting App")
st.markdown("Enter the last 3 months' sales to predict the next month's sales.")

# Input fields
sales1 = st.number_input("Month 1 Sales", value=45000.0)
sales2 = st.number_input("Month 2 Sales", value=47000.0)
sales3 = st.number_input("Month 3 Sales", value=49000.0)

# Predict button
if st.button("🔮 Predict Next Month's Sales"):
    payload = {
        "sales_sequence": [sales1, sales2, sales3]
    }

    try:
        # Call FastAPI
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Predicted Sales: ₹{result['predicted_sales']:,.2f}")
        else:
            st.error("❌ Something went wrong with the prediction API.")

    except Exception as e:
        st.error(f"⚠️ Could not connect to API: {e}")
