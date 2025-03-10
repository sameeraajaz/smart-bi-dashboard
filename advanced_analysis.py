import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ✅ Ensemble Forecasting: Prophet + LSTM
def ensemble_forecast(df, plot_theme, forecast_days=30):
    st.subheader("🔮 Advanced Forecasting (Prophet + LSTM Ensemble)")
    df = df.dropna(subset=["Sales"]).copy()
    df = df.sort_values("Date")

    prophet_df = df[["Date", "Sales"]].rename(columns={"Date": "ds", "Sales": "y"})
    m = Prophet()
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=forecast_days)
    prophet_forecast = m.predict(future)[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Prophet"})

    df = df.set_index("Date")
    values = df["Sales"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    lookback = 10
    X, y = [], []
    for i in range(len(values_scaled) - lookback):
        X.append(values_scaled[i:i+lookback])
        y.append(values_scaled[i+lookback])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=30, batch_size=8, verbose=0)

    input_seq = X[-1]
    preds = []
    for _ in range(forecast_days):
        pred = model.predict(input_seq.reshape(1, lookback, 1))[0, 0]
        preds.append(pred)
        input_seq = np.roll(input_seq, -1)
        input_seq[-1] = pred

    lstm_forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    lstm_df = pd.DataFrame({
        "Date": pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days),
        "LSTM": lstm_forecast
    })

    merged = prophet_forecast.merge(lstm_df, on="Date")
    merged["Ensemble"] = (merged["Prophet"] + merged["LSTM"]) / 2

    fig = px.line(merged, x="Date", y=["Prophet", "LSTM", "Ensemble"],
                  title="🔮 Ensemble Forecast", template=plot_theme)
    st.plotly_chart(fig, use_container_width=True)

# ✅ Isolation Forest Anomaly Detection + Root Cause Tracing
def advanced_anomaly_detection(df, plot_theme):
    st.subheader("🚨 AI-Powered Anomaly Detection (Isolation Forest + Feature Importance)")

    df = df.dropna(subset=["Sales", "Profit", "Costs"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])

    features = ["Sales", "Costs", "Profit"]
    model = IsolationForest(contamination=0.1, random_state=42)
    df["Anomaly Score"] = model.fit_predict(df[features])
    df["Anomaly Type"] = df["Anomaly Score"].map({-1: "Mixed Anomaly", 1: "Normal"})


    fig = px.scatter(df, x="Date", y="Sales", color="Anomaly Type",
                     hover_data=["Costs", "Profit"],
                     title="🚨 Multivariate Anomalies (Sales/Profit/Cost)",
                     template=plot_theme)
    st.plotly_chart(fig, use_container_width=True)

    # 🧠 Root Cause Tracing using Gradient Boosting
    st.markdown("### 🧠 Root Cause Tracing via Feature Importance")
    df_encoded = df[features + ["Anomaly Score"]].copy()
    df_encoded["Anomaly Score"] = df_encoded["Anomaly Score"].map({-1: 1, 1: 0})

    clf = GradientBoostingClassifier()
    clf.fit(df_encoded[features], df_encoded["Anomaly Score"])
    importance = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)

    for feature, score in importance.items():
        st.markdown(f"🔍 **{feature}** influenced anomalies with weight **{score:.2f}**")

    outliers = df[df["Anomaly Type"] == "Anomaly"]
    if not outliers.empty:
        st.markdown("### 📌 Most Suspicious Days")
        for _, row in outliers.sort_values(by="Sales", ascending=False).head(3).iterrows():
            st.markdown(f"📌 **{row['Date'].date()}** | Sales: {row['Sales']:.2f} | Profit: {row['Profit']:.2f}")
