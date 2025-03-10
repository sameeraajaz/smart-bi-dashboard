import streamlit as st

# ✅ Function to Apply UI Configuration & Dark Mode
def apply_ui():
    st.set_page_config(page_title="Business Intelligence Dashboard", layout="wide")

    dark_mode = st.sidebar.toggle("Enable Dark Mode", value=False)

    if dark_mode:
        st.markdown(
            """
            <style>
            body, .stApp { background-color: #121212; color: #F9C349; }
            .css-18e3th9, .css-1d391kg { background-color: #121212; color: #F9C349; }
            .stSidebar, .css-1lcbmhc, .css-qri22k { background-color: #1E1E1E !important; color: #F9C349 !important; }
            .stButton>button { background-color: #4CAF50; color: white; }
            .stMetric { text-align: center; color: #F9C349 !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return "plotly_dark"
    
    return "plotly_white"
