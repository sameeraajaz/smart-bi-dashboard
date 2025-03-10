import plotly.express as px
import streamlit as st

# ✅ Function for Sales Trend Visualization
def show_sales_trend(df, plot_theme):
    st.subheader("Sales Trend Over Time")
    fig = px.line(df, x="Date", y="Sales", title="Daily Sales Performance", template=plot_theme)
    st.plotly_chart(fig, use_container_width=True)

# ✅ Function for Profit Distribution
def show_profit_distribution(df, plot_theme):
    st.subheader("Profit Distribution")
    fig = px.histogram(df, x="Profit", nbins=20, title="Profit Distribution", template=plot_theme)
    st.plotly_chart(fig, use_container_width=True)
