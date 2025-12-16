import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from catboost import CatBoostClassifier
import io

# Page Configuration
st.set_page_config(
    page_title="RFM Customer Intelligence",
    page_icon="🛒",
    layout="wide"
)

# --- 1. Model Loading ---
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    try:
        model.load_model("catboost_rfm_model.cbm")
        return model
    except Exception as e:
        st.error(f"Model file not found. Please ensure 'catboost_rfm_model.cbm' is in the same directory.")
        return None

model = load_model()

# --- 2. Helper Functions ---
def preprocess_raw_data(df):
    """
    Replicates the exact feature engineering logic from the training script
    to convert Raw Data (Invoices) -> Model Features (RFM).
    """
    try:
        # Standard cleaning
        df.dropna(inplace=True)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        df = df[df['Quantity'] > 0] # Remove returns

        # Set reference date (usually max date in data + 1 day)
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

        # Aggregation (The "Magic" Step)
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalPrice': 'sum'
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        return rfm
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

# --- 3. App Layout ---
st.title("🛒 RFM Customer Intelligence Dashboard")
st.markdown("""
This tool bridges the gap between raw transaction data and customer value prediction.
""")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["⚡ Instant Predictor", "📂 Batch Processor (Raw Data)", "ℹ️ Model Info"])

# --- TAB 1: INSTANT PREDICTOR (Manual Entry) ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Customer Profile")
        st.info("Adjust sliders to simulate a customer.")
        
        recency = st.slider("Recency (Days since last buy)", 0, 365, 30, help="Lower is better.")
        frequency = st.slider("Frequency (Total transactions)", 0, 100, 5, help="Higher is better.")
        monetary = st.number_input("Monetary (Total Spend $)", 0.0, 10000.0, 500.0, step=10.0, help="Higher is better.")
        
        predict_btn = st.button("Predict Customer Value", type="primary")

    with col2:
        if predict_btn and model:
            # Create DataFrame
            input_df = pd.DataFrame({
                'Recency': [recency],
                'Frequency': [frequency],
                'Monetary': [monetary]
            })
            
            # Predict
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] # Probability of being class 1
            
            st.subheader("Prediction Results")
            
            # Visual Gauge
            fig = px.bar(
                x=[prob], 
                y=["Score"], 
                orientation='h', 
                range_x=[0, 1],
                labels={'x': 'Probability', 'y': ''},
                title="Likelihood of Recommendation",
                color=[prob],
                color_continuous_scale=['red', 'yellow', 'green']
            )
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)

            if pred == 1:
                st.success(f"### ✅ Recommendation: TARGET")
                st.write(f"This customer is a high-value target with a **{prob:.1%}** probability score.")
            else:
                st.warning(f"### ⚠️ Recommendation: IGNORE")
                st.write(f"This customer falls below the threshold (Probability: **{prob:.1%}**).")

# --- TAB 2: BATCH PROCESSOR (Raw Data) ---
with tab2:
    st.subheader("Process Raw Transaction Data")
    st.markdown("Upload the raw `Online-Retail.xlsx` file here. The app will automatically convert the raw columns (`InvoiceDate`, `Quantity`, etc.) into the RFM features the model needs.")
    
    uploaded_file = st.file_uploader("Upload Excel or CSV", type=['xlsx', 'csv'])
    
    if uploaded_file and model:
        with st.spinner("Processing raw data... (This implies 'Feature Engineering')"):
            # Load Data
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            # Show Raw Snippet
            with st.expander("View Raw Data Snippet"):
                st.dataframe(raw_df.head())
            
            # Transform
            rfm_data = preprocess_raw_data(raw_df)
            
            if rfm_data is not None:
                st.success(f"Successfully processed **{len(rfm_data)}** unique customers!")
                
                # Predict on all
                predictions = model.predict(rfm_data)
                probabilities = model.predict_proba(rfm_data)[:, 1]
                
                # Add results to dataframe
                rfm_data['Recommended'] = ["Yes" if p == 1 else "No" for p in predictions]
                rfm_data['Score'] = probabilities
                
                # Display Results
                st.dataframe(rfm_data.style.highlight_max(axis=0, subset=['Score'], color='lightgreen'))
                
                # Download Button
                csv = rfm_data.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="rfm_predictions.csv",
                    mime="text/csv",
                )
                
                # Visual Summary of Batch
                st.subheader("Batch Analysis")
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    fig_pie = px.pie(rfm_data, names='Recommended', title="Recommendation Distribution")
                    st.plotly_chart(fig_pie, use_container_width=True)
                with chart_col2:
                    fig_scat = px.scatter(
                        rfm_data, 
                        x='Recency', 
                        y='Monetary', 
                        color='Recommended', 
                        title="Recency vs Monetary Segmentation",
                        log_y=True # Log scale for monetary helps visualization
                    )
                    st.plotly_chart(fig_scat, use_container_width=True)

# --- TAB 3: MODEL INFO ---
with tab3:
    st.markdown("""
    ### Why only 3 features?
    Although the raw data contains Country, Description, and StockCode, this specific model is an **RFM Classifier**.
    
    It intentionally aggregates complex transaction histories into three powerful metrics:
    1.  **Recency:** How "fresh" is the customer? (derived from *InvoiceDate*)
    2.  **Frequency:** How loyal are they? (derived from *InvoiceNo* count)
    3.  **Monetary:** How much are they worth? (derived from *Quantity * UnitPrice*)
    
    To use features like 'Country' or 'Product Type', the model training script (`.fit`) would need to be updated to include those columns before saving the model file.
    """)