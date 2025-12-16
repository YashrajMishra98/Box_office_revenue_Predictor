import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# 1. Load the saved model
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_rfm_model.cbm")
    return model

model = load_model()

# 2. App Title and Description
st.title("Customer Recommendation Engine")
st.write("""
This app predicts if we should recommend products to a customer based on their 
**Recency** (days since last purchase), **Frequency** (total purchases), and **Monetary** (total spend) value.
""")

# 3. Create Input Fields (Replacing your ipywidgets)
st.sidebar.header("Customer Details")

recency = st.sidebar.slider(
    "Recency (Days since last visit)", 
    min_value=0, max_value=365, value=30
)

frequency = st.sidebar.slider(
    "Frequency (Number of transactions)", 
    min_value=0, max_value=100, value=5
)

monetary = st.sidebar.number_input(
    "Monetary Value ($ Total Spend)", 
    min_value=0.0, max_value=10000.0, value=500.0, step=10.0
)

# 4. Make Prediction when button is clicked
if st.button("Generate Recommendation"):
    # Create a dataframe matching the training input
    input_data = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'Monetary': [monetary]
    })
    
    # Get prediction
    prediction = model.predict(input_data)[0]
    
    # Display result
    st.divider()
    if prediction == 1:
        st.success("## ✅ Recommendation: YES")
        st.write("This customer has a high likelihood of responding positively.")
    else:
        st.error("## ❌ Recommendation: NO")
        st.write("This customer falls below the target threshold.")