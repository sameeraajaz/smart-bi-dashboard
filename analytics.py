import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from prophet import Prophet
from scipy.stats import zscore
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ✅ Prophet Forecasting
def forecast_with_prophet(df, plot_theme, forecast_days=30):
    st.subheader("📈 AI-Powered Sales Forecasting (Prophet)")
    st.write(f"Using Facebook Prophet to forecast the next **{forecast_days} days**.")

    df = df.dropna(subset=["Sales"])
    prophet_df = df[["Date", "Sales"]].rename(columns={"Date": "ds", "Sales": "y"})

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    fig = px.line(
        forecast, x="ds", y=["yhat", "yhat_lower", "yhat_upper"],
        title=f"📊 Prophet Forecast for the Next {forecast_days} Days",
        labels={"ds": "Date", "yhat": "Projected Sales"},
        template=plot_theme
    )
    st.plotly_chart(fig, use_container_width=True)

# ✅ LSTM Forecasting
def forecast_with_lstm(df, plot_theme, forecast_days=30):
    st.subheader("📈 AI-Powered Sales Forecasting (LSTM Neural Network)")
    st.write(f"Using LSTM to forecast the next **{forecast_days} days**.")

    df = df.dropna(subset=["Sales"])
    df = df.set_index("Date").sort_index()

    data = df["Sales"].values
    X, y = [], []
    lookback = 10

    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation="relu", return_sequences=True, input_shape=(lookback, 1)),
        LSTM(50, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)

    predictions = []
    current_input = X[-1]

    for _ in range(forecast_days):
        pred = model.predict(current_input.reshape((1, lookback, 1)))[0, 0]
        predictions.append(pred)
        current_input = np.roll(current_input, -1)
        current_input[-1] = pred

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    future_df = pd.DataFrame({"Date": future_dates, "Sales": predictions})

    fig = px.line(
        future_df, x="Date", y="Sales",
        title=f"📊 LSTM Forecast for the Next {forecast_days} Days",
        labels={"Date": "Date", "Sales": "Projected Sales"},
        template=plot_theme
    )
    st.plotly_chart(fig, use_container_width=True)

# ✅ XGBoost Forecasting
def forecast_with_xgboost(df, plot_theme, forecast_days=30):
    st.subheader("📈 AI-Powered Sales Forecasting (XGBoost)")
    st.write(f"Using XGBoost to forecast the next **{forecast_days} days**.")

    df = df.dropna(subset=["Sales"])
    df = df.set_index("Date").sort_index()

    df["Month"] = df.index.to_period("M").astype(str)
    df = pd.get_dummies(df, columns=["Month"], drop_first=True)

    df["Target"] = df["Sales"].shift(-1)
    df.dropna(inplace=True)

    X = df.drop(columns=["Target"])
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    st.write(f"📊 Mean Absolute Error (XGBoost): {error:.2f}")

    future_input = X.tail(forecast_days)
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_values = model.predict(future_input)

    future_df = pd.DataFrame({"Date": future_dates, "Sales": forecast_values})

    fig = px.line(
        future_df, x="Date", y="Sales",
        title=f"📊 XGBoost Forecast for the Next {forecast_days} Days",
        labels={"Date": "Date", "Sales": "Projected Sales"},
        template=plot_theme
    )
    st.plotly_chart(fig, use_container_width=True)

# ✅ Anomaly Detection
def detect_anomalies(df, plot_theme):
    st.subheader("⚠️ Sales Anomaly Detection")
    st.write("This chart highlights and explains abnormal sales behavior.")

    df["Z-Score"] = zscore(df["Sales"].fillna(0))

    def label_anomaly(z):
        if z <= -3:
            return "Critical Drop"
        elif z >= 3:
            return "Critical Spike"
        elif -3 < z <= -2:
            return "Minor Drop"
        elif 2 <= z < 3:
            return "Minor Spike"
        else:
            return "Normal"

    df["Anomaly Type"] = df["Z-Score"].apply(label_anomaly)
    df["Severity"] = df["Z-Score"].abs().round(2)

    fig = px.scatter(
        df, x="Date", y="Sales", color="Anomaly Type",
        size="Severity", hover_data=["Z-Score"],
        title="⚠️ Detected Sales Anomalies with Severity",
        labels={"Date": "Date", "Sales": "Sales Volume"},
        template=plot_theme
    )
    st.plotly_chart(fig, use_container_width=True)

    top_anomalies = df[df["Anomaly Type"] != "Normal"].sort_values(by="Severity", ascending=False).head(3)
    for _, row in top_anomalies.iterrows():
        st.markdown(
            f"📌 On **{row['Date'].strftime('%Y-%m-%d')}**, a **{row['Anomaly Type']}** was detected "
            f"with a Z-Score of **{row['Z-Score']:.2f}**, indicating a significant deviation."
        )
