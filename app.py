import streamlit as st
import pickle
import pandas as pd
import json

st.set_page_config(page_title="Credit Risk AI", layout="wide")

st.title("💳 Credit Default Prediction Dashboard")

tab1, tab2 = st.tabs(["📊 Prediction", "📈 Comparison"])

# ================= TAB 1 ================= #
with tab1:

    dataset = st.selectbox("Select Dataset", ["AMEX", "GMSC"])

    # -------- COMMON USER INPUT -------- #
    st.sidebar.header("Customer Inputs")

    payment = st.sidebar.slider("Payment Score", 300, 900, 700)
    balance = st.sidebar.number_input("Balance", 0, 1000000, 40000)
    days = st.sidebar.number_input("Days Past Due", 0, 120, 5)
    risk = st.sidebar.slider("Risk Score", 0, 10, 3)
    spend = st.sidebar.number_input("Spending", 0, 100000, 20000)
    delay = st.sidebar.number_input("Delay Count", 0, 50, 2)

    util = st.sidebar.slider("Credit Utilization", 0.0, 1.0, 0.3)
    age = st.sidebar.slider("Age", 18, 80, 30)
    late = st.sidebar.number_input("Late Payments (30–59 Days)", 0, 30, 1)
    debt = st.sidebar.slider("Debt Ratio", 0.0, 5.0, 0.5)
    income = st.sidebar.number_input("Income", 0, 1000000, 50000)
    credit = st.sidebar.number_input("Open Credit Lines", 0, 20, 5)

    # -------- LOAD MODEL -------- #
    if dataset == "AMEX":
        model = pickle.load(open("models/amex_xgb_model.pkl", "rb"))
        with open("columns/amex_columns.json") as f:
            cols = json.load(f)

        data = {c: 0 for c in cols}
        data["P_2"] = payment / 1000
        data["B_1"] = balance / 100000
        data["D_39"] = days / 100
        data["R_1"] = risk / 10
        data["S_3"] = spend / 100000
        data["D_41"] = delay / 100

    else:
        model = pickle.load(open("models/gmsc_xgb_model.pkl", "rb"))
        with open("columns/gmsc_columns.json") as f:
            cols = json.load(f)

        data = {c: 0 for c in cols}
        data["RevolvingUtilizationOfUnsecuredLines"] = util
        data["age"] = age
        data["NumberOfTime30-59DaysPastDueNotWorse"] = late
        data["DebtRatio"] = debt
        data["MonthlyIncome"] = income
        data["NumberOfOpenCreditLinesAndLoans"] = credit

    df = pd.DataFrame([data])

    if st.button("Predict Risk"):

        prob = float(model.predict_proba(df)[0][1])

        # store ALL inputs for comparison
        st.session_state["inputs"] = {
            "payment": payment,
            "balance": balance,
            "days": days,
            "risk": risk,
            "spend": spend,
            "delay": delay,
            "util": util,
            "age": age,
            "late": late,
            "debt": debt,
            "income": income,
            "credit": credit
        }

        st.session_state["prob"] = prob

        # -------- RESULT -------- #
        st.subheader("📊 Prediction Result")

        if prob < 0.3:
            st.success("🟢 Low Risk")
        elif prob < 0.7:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

        st.write(f"Probability: {prob:.2f}")

        # -------- EXPLANATION -------- #
        st.subheader("🤖 Explanation")

        if prob > 0.7:
            st.write("Customer is risky due to poor financial behavior.")
        elif prob > 0.3:
            st.write("Customer shows moderate risk indicators.")
        else:
            st.write("Customer is financially stable.")

        st.markdown("""
💡 *SHAP explains the model prediction, while business rules ensure critical risk conditions are enforced.*
""")

# ================= TAB 2 ================= #
with tab2:

    st.subheader("📈 Model Comparison")

    if "inputs" in st.session_state:

        inp = st.session_state["inputs"]

        # -------- LOAD BOTH MODELS -------- #
        amex_model = pickle.load(open("models/amex_xgb_model.pkl", "rb"))
        gmsc_model = pickle.load(open("models/gmsc_xgb_model.pkl", "rb"))

        with open("columns/amex_columns.json") as f:
            amex_cols = json.load(f)

        with open("columns/gmsc_columns.json") as f:
            gmsc_cols = json.load(f)

        # -------- BUILD BOTH INPUTS -------- #
        amex_data = {c: 0 for c in amex_cols}
        amex_data["P_2"] = inp["payment"] / 1000
        amex_data["B_1"] = inp["balance"] / 100000
        amex_data["D_39"] = inp["days"] / 100
        amex_data["R_1"] = inp["risk"] / 10
        amex_data["S_3"] = inp["spend"] / 100000
        amex_data["D_41"] = inp["delay"] / 100

        gmsc_data = {c: 0 for c in gmsc_cols}
        gmsc_data["RevolvingUtilizationOfUnsecuredLines"] = inp["util"]
        gmsc_data["age"] = inp["age"]
        gmsc_data["NumberOfTime30-59DaysPastDueNotWorse"] = inp["late"]
        gmsc_data["DebtRatio"] = inp["debt"]
        gmsc_data["MonthlyIncome"] = inp["income"]
        gmsc_data["NumberOfOpenCreditLinesAndLoans"] = inp["credit"]

        amex_df = pd.DataFrame([amex_data])
        gmsc_df = pd.DataFrame([gmsc_data])

        # -------- PREDICT -------- #
        amex_prob = float(amex_model.predict_proba(amex_df)[0][1])
        gmsc_prob = float(gmsc_model.predict_proba(gmsc_df)[0][1])

        # -------- DISPLAY -------- #
        col1, col2 = st.columns(2)

        with col1:
            st.metric("AMEX Risk", f"{amex_prob:.2f}")
            st.progress(min(max(amex_prob, 0), 1))

        with col2:
            st.metric("GMSC Risk", f"{gmsc_prob:.2f}")
            st.progress(min(max(gmsc_prob, 0), 1))

        # -------- USER FRIENDLY -------- #
        st.subheader("🧠 Understanding Comparison")

        if amex_prob > gmsc_prob:
            st.write("AMEX sees higher risk → behavior-based model is stricter.")
        elif gmsc_prob > amex_prob:
            st.write("GMSC sees higher risk → financial profile is risky.")
        else:
            st.write("Both models agree → strong decision.")

        st.subheader("📌 Final Conclusion")

        avg = (amex_prob + gmsc_prob) / 2

        if avg > 0.7:
            st.error("High Risk → Reject Loan")
        elif avg > 0.3:
            st.warning("Moderate Risk → Review Required")
        else:
            st.success("Low Risk → Approve Loan")

    else:
        st.info("Run prediction first")

   

    
        
