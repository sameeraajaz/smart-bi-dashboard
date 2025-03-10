import streamlit as st
import pandas as pd
import data_loader
import ui_config
import analytics
import visualizations
import kpi_metrics
from advanced_analysis import ensemble_forecast, advanced_anomaly_detection
import anomaly_dashboard  # ✅ NEW: import anomaly tab module
import os

# 📥 Load Default Sample if Nothing Provided
if not sheet_url and not uploaded_file:
    st.info("No file or link provided. Using sample data.")
    sample_path = os.path.join("sample_data", "enhanced_business_dataset.csv")
    df = pd.read_csv(sample_path)


# ✅ Apply UI Configuration
plot_theme = ui_config.apply_ui()

# ✅ Load Data (Google Sheets or CSV)
st.sidebar.header("Upload Data Source")
sheet_url = st.sidebar.text_input("Enter Google Sheet Public Link", "")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

df = pd.DataFrame()

if sheet_url:
    df = data_loader.load_google_sheet(sheet_url)
elif uploaded_file:
    df = data_loader.load_csv(uploaded_file)

if df.empty:
    st.error("⚠️ No data loaded. Please enter a valid Google Sheets link or upload a CSV file.")
    st.stop()

# ✅ Date Range Filter
st.sidebar.header("Filter Data")
start_date = pd.to_datetime(st.sidebar.date_input("Start Date", df["Date"].min()))
end_date = pd.to_datetime(st.sidebar.date_input("End Date", df["Date"].max()))
filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

# ✅ Forecasting Settings
st.sidebar.header("Forecasting Settings")
forecasting_model = st.sidebar.selectbox("Select Model", ["Prophet", "LSTM", "XGBoost"])
max_days = min(180, len(filtered_df) * 2)
forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=max_days, value=30)

# ✅ Advanced Intelligence Tools
st.sidebar.header("🧠 Advanced AI Options")
use_ensemble = st.sidebar.checkbox("Enable Ensemble Forecast")
use_isolation = st.sidebar.checkbox("Enable Advanced Anomaly Detection")

# ✅ Main Interface with Tabs
tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "🧠 Anomaly Dashboard"])

with tab1:
    # KPI Metrics
    kpi_metrics.show_kpi_metrics(filtered_df)

    # Charts
    visualizations.show_sales_trend(filtered_df, plot_theme)
    visualizations.show_profit_distribution(filtered_df, plot_theme)

    # Forecasting
    if forecasting_model == "Prophet":
        analytics.forecast_with_prophet(filtered_df, plot_theme, forecast_days)
    elif forecasting_model == "LSTM":
        analytics.forecast_with_lstm(filtered_df, plot_theme, forecast_days)
    elif forecasting_model == "XGBoost":
        analytics.forecast_with_xgboost(filtered_df, plot_theme, forecast_days)

    # Basic Anomaly Detection
    analytics.detect_anomalies(filtered_df, plot_theme)

    # Optional: Advanced Forecast + Anomaly
    if use_ensemble:
        ensemble_forecast(filtered_df, plot_theme, forecast_days)

    if use_isolation:
        advanced_anomaly_detection(filtered_df, plot_theme)

with tab2:
    anomaly_dashboard.display_anomaly_dashboard(filtered_df, plot_theme)

st.success("✅ Dashboard loaded with Analytics + Anomaly Intelligence View.")
