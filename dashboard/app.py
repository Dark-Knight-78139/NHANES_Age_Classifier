import streamlit as st
import pandas as pd
import requests
import yaml
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import AgePredictor

st.set_page_config(page_title="NHANES Age intelligence", layout="wide", page_icon="🧬")

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 NHANES Age Intelligence")
st.markdown("### Accelerating Biological Age Research with ML")

# Load Config
try:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.error("Config file not found. Please ensure 'config/config.yaml' exists.")
    st.stop()

# Helper to check models
models_exist = os.path.exists("models/saved_model.pkl") and os.path.exists("models/preprocessor.joblib")

# Sidebar for navigation and pipeline control
with st.sidebar:
    st.image("https://www.cdc.gov/nchs/images/nhanes_logo.gif", width=200)
    st.header("Control Center")
    menu = ["👤 User Prediction", "📈 Dataset Insights", "⚙️ Model Dashboard"]
    choice = st.selectbox("Navigation", menu)
    
    st.divider()
    st.subheader("Data Pipeline")
    if st.button("🚀 Run Full Pipeline"):
        with st.spinner("Processing data and training model..."):
            result = subprocess.run(["python", "pipeline.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Pipeline completed successfully!")
                st.rerun()
            else:
                st.error(f"Pipeline failed: {result.stderr}")

if choice == "👤 User Prediction":
    st.header("Biological Age Predictor")
    
    if not models_exist:
        st.warning("⚠️ Model files not found. Please run the pipeline from the sidebar first.")
    
    with st.container():
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            bmx_wt = st.number_input("Weight (kg)", value=70.0, step=0.1)
            bmx_ht = st.number_input("Height (cm)", value=170.0, step=0.1)
            bmx_bmi = st.number_input("BMI", value=24.0, step=0.1)
            bmx_waist = st.number_input("Waist Circumference (cm)", value=85.0, step=0.1)
            
        with col2:
            lbx_stp = st.number_input("Total Protein (g/dL)", value=7.0, step=0.1)
            lbx_str = st.number_input("Triglycerides (mg/dL)", value=100.0, step=1.0)
            lbx_sch = st.number_input("Total Cholesterol (mg/dL)", value=180.0, step=1.0)
            gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x==1 else "Female")

        if st.button("Calculate Predicted Age"):
            if not models_exist:
                st.error("Cannot predict without a trained model.")
            else:
                payload_df = pd.DataFrame([{
                    "BMXWT": bmx_wt,
                    "BMXHT": bmx_ht,
                    "BMXBMI": bmx_bmi,
                    "BMXWAIST": bmx_waist,
                    "LBXSTP": lbx_stp,
                    "LBXSTR": lbx_str,
                    "LBXSCH": lbx_sch,
                    "RIAGENDR": gender
                }])
                
                try:
                    # Direct inference from local model
                    predictor = AgePredictor()
                    prediction = predictor.predict(payload_df)
                    predicted_age = round(float(prediction[0]), 2)
                    
                    st.metric("Estimated Biological Age", f"{predicted_age} yrs")
                    
                    if predicted_age > 50:
                        st.info("💡 High biomarker age detected. Consider consulting metabolic trends.")
                    else:
                        st.success("✅ Biomarkers indicate a youthful biological profile.")
                        
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

elif choice == "📈 Dataset Insights":
    st.header("Biomarker Distribution & Correlation")
    
    plot_dir = "data/processed"
    if os.path.exists(os.path.join(plot_dir, "missing_heatmap.png")):
        tab1, tab2, tab3 = st.tabs(["Data Health", "Correlations", "Feature Distributions"])
        
        with tab1:
            st.subheader("Data Completeness Heatmap")
            st.image(os.path.join(plot_dir, "missing_heatmap.png"), use_container_width=True)
            
        with tab2:
            st.subheader("Biomarker Correlation Matrix")
            st.image(os.path.join(plot_dir, "correlation_matrix.png"), use_container_width=True)
            
        with tab3:
            st.subheader("Feature Distributions")
            features = config['data']['features']
            feat_choice = st.selectbox("Select Feature", features)
            dist_path = os.path.join(plot_dir, f"dist_{feat_choice}.png")
            if os.path.exists(dist_path):
                st.image(dist_path, use_container_width=True)
            else:
                st.info("Distribution plot not found.")
    else:
        st.info("Run the pipeline to generate dataset insights and visualizations.")

elif choice == "⚙️ Model Dashboard":
    st.header("Model Parameters & Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hyperparameters")
        st.json(config['model'])
    
    with col2:
        st.subheader("ML Architecture")
        st.write(f"**Model Type:** {config['model']['type']}")
        st.write(f"**Target Column:** {config['data']['target_column']}")
        st.write(f"**Number of Features:** {len(config['data']['features'])}")

    st.divider()
    st.subheader("Feature Definitions")
    descriptions = {
        "BMXWT": "Body weight in kilograms",
        "BMXHT": "Standing height in centimeters",
        "BMXBMI": "Body Mass Index (kg/m^2)",
        "BMXWAIST": "Waist circumference in centimeters",
        "LBXSTP": "Serum total protein in g/dL",
        "LBXSTR": "Serum triglycerides in mg/dL",
        "LBXSCH": "Total cholesterol in mg/dL",
        "RIAGENDR": "Gender (1: Male, 2: Female)"
    }
    for feat, desc in descriptions.items():
        st.write(f"**{feat}**: {desc}")
