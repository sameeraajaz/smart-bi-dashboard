import streamlit as st

# ✅ Function to Display KPI Metrics
def show_kpi_metrics(df):
    if df.empty:
        return  # Don't show KPIs if there's no data

    st.subheader("Key Business Metrics")

    # Compute KPIs
    total_revenue = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    average_profit_margin = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0

    df["Month"] = df["Date"].dt.to_period("M")
    monthly_sales = df.groupby("Month")["Sales"].sum()
    monthly_growth_rate = monthly_sales.pct_change().fillna(0).iloc[-1] * 100 if len(monthly_sales) > 1 else 0

    # ✅ Display Metrics with Correct Colors in Dark Mode
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(label="Total Revenue", value=f"${total_revenue:,.2f}")
    col2.metric(label="Total Profit", value=f"${total_profit:,.2f}")
    col3.metric(label="Avg. Profit Margin", value=f"{average_profit_margin:.2f}%")
    col4.metric(label="Monthly Growth Rate", value=f"{monthly_growth_rate:.2f}%")
