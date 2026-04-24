import streamlit as st
import pickle
import pandas as pd
import json

st.set_page_config(page_title="Credit Risk AI", layout="wide")

# ✅ Initialize session state FIRST
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


# -------- LOGIN FUNCTION -------- #
def login():
    st.title("🔐 Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
            st.experimental_rerun()
        else:
            st.error("Invalid Credentials")


# -------- MAIN APP -------- #
if not st.session_state["logged_in"]:
    login()
else:
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.experimental_rerun()

    st.title("💳 Credit Default Prediction Dashboard")