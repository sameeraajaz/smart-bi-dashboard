import pandas as pd
import streamlit as st

# ✅ Function to Read Google Sheets (Public Mode)
@st.cache_data
def load_google_sheet(sheet_url):
    try:
        sheet_id = sheet_url.split("/d/")[1].split("/")[0]
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
        df = pd.read_csv(csv_url)
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading Google Sheet: {e}")
        return pd.DataFrame()

# ✅ Function to Read CSV Upload
@st.cache_data
def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading CSV: {e}")
        return pd.DataFrame()
