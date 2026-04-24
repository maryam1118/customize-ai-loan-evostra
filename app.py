import streamlit as st
import pickle
import pandas as pd
import json

st.set_page_config(page_title="Credit Risk AI", layout="wide")

# ---------------- UI STYLE ---------------- #
st.markdown("""
<style>
.main { background-color: #f5f7fa; }
h1, h2, h3 { color: #1f4e79; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ---------------- #
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

with open("users.json") as f:
    users = json.load(f)

def login():
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in users and users[u] == p:
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state["logged_in"]:
    login()
    st.stop()

if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

# ---------------- TITLE ---------------- #
st.title("💳 Credit Default Prediction Dashboard")

tab1, tab2 = st.tabs(["📊 Prediction", "📈 Comparison"])

# ================= TAB 1 ================= #
with tab1:

    dataset = st.selectbox("Select Dataset", ["AMEX", "GMSC"])

    try:
        if dataset == "AMEX":
            model = pickle.load(open("models/amex_xgb_model.pkl", "rb"))
            with open("columns/amex_columns.json") as f:
                all_columns = json.load(f)
        else:
            model = pickle.load(open("models/gmsc_xgb_model.pkl", "rb"))
            with open("columns/gmsc_columns.json") as f:
                all_columns = json.load(f)
    except:
        st.error("Model files not found ❌")
        st.stop()

    st.sidebar.header("Input Features")

    if dataset == "AMEX":
        payment = st.sidebar.slider("Payment Score", 300, 900, 700)
        balance = st.sidebar.number_input("Balance", 0, 1000000, 40000)
        days = st.sidebar.number_input("Days Due", 0, 120, 5)
        risk = st.sidebar.slider("Risk Score", 0, 10, 3)
        spend = st.sidebar.number_input("Spending", 0, 100000, 20000)
        delay = st.sidebar.number_input("Delay Count", 0, 50, 2)
    else:
        util = st.sidebar.slider("Credit Utilization", 0.0, 1.0, 0.3)
        age = st.sidebar.slider("Age", 18, 80, 30)
        late = st.sidebar.number_input("Late Payments (30–59 Days)", 0, 30, 1)
        debt = st.sidebar.slider("Debt Ratio", 0.0, 5.0, 0.5)
        income = st.sidebar.number_input("Income", 0, 1000000, 50000)
        credit = st.sidebar.number_input("Open Credit Lines", 0, 20, 5)

    full = {c: 0 for c in all_columns}

    if dataset == "AMEX":
        full["P_2"] = payment / 1000
        full["B_1"] = balance / 100000
        full["D_39"] = days / 100
        full["R_1"] = risk / 10
        full["S_3"] = spend / 100000
        full["D_41"] = delay / 100
    else:
        full["RevolvingUtilizationOfUnsecuredLines"] = util
        full["age"] = age
        full["NumberOfTime30-59DaysPastDueNotWorse"] = late
        full["DebtRatio"] = debt
        full["MonthlyIncome"] = income
        full["NumberOfOpenCreditLinesAndLoans"] = credit

    df = pd.DataFrame([full])

    if st.button("Predict Risk"):

        prob = float(model.predict_proba(df)[0][1])

        st.session_state["prob"] = prob
        st.session_state["dataset"] = dataset

        # -------- EXPLANATION FIRST -------- #
        st.markdown("## 🤖 Model Explanation")

        explanation = []

        if dataset == "AMEX":
            if payment < 500:
                explanation.append("Low payment score increases risk")
            if days > 30:
                explanation.append("High delay increases risk")
            if balance < 10000:
                explanation.append("Low balance indicates instability")

        else:
            if late > 10:
                explanation.append("Frequent late payments increase risk")
            if debt > 1:
                explanation.append("High debt ratio increases risk")
            if util > 0.8:
                explanation.append("High credit utilization increases risk")

        for e in explanation:
            st.write("•", e)

        # -------- RESULT -------- #
        st.markdown("## 📊 Prediction Result")
        st.write(f"Probability: {prob:.2f}")

        if prob < 0.3:
            st.success("🟢 Low Risk")
        elif prob < 0.7:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

# ================= TAB 2 ================= #
with tab2:

    st.markdown("## 📈 Model Comparison")

    if "prob" in st.session_state:

        prob = float(st.session_state["prob"])
        dataset = st.session_state["dataset"]

        amex = float(prob if dataset == "AMEX" else 0.5)
        gmsc = float(prob if dataset == "GMSC" else 0.5)

        # FIXED progress issue
        st.progress(min(max(amex, 0.0), 1.0), text=f"AMEX Risk: {amex:.2f}")
        st.progress(min(max(gmsc, 0.0), 1.0), text=f"GMSC Risk: {gmsc:.2f}")

        if amex > gmsc:
            st.error("AMEX predicts higher risk")
        elif gmsc > amex:
            st.warning("GMSC predicts higher risk")
        else:
            st.success("Both models show similar risk")

    else:
        st.info("Run prediction first")
                
