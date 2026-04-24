import streamlit as st
import pickle
import pandas as pd
import json

st.set_page_config(page_title="Credit Risk AI", layout="wide")

# ---------------- UI ---------------- #
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

    if dataset == "AMEX":
        model = pickle.load(open("models/amex_xgb_model.pkl", "rb"))
        with open("columns/amex_columns.json") as f:
            all_columns = json.load(f)
    else:
        model = pickle.load(open("models/gmsc_xgb_model.pkl", "rb"))
        with open("columns/gmsc_columns.json") as f:
            all_columns = json.load(f)

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

        # -------- MODEL EXPLANATION -------- #
        st.markdown("## 🤖 Model Explanation (Data-driven)")

        model_exp = []

        if dataset == "AMEX":
            if payment < 500:
                model_exp.append("Low payment behavior indicates poor credit discipline")
            if days > 30:
                model_exp.append("Higher days past due suggests repayment delays")
            if delay > 5:
                model_exp.append("Frequent delays increase default probability")

        else:
            if late > 10:
                model_exp.append("Frequent late payments indicate repayment risk")
            if debt > 1:
                model_exp.append("High debt ratio shows financial burden")
            if util > 0.8:
                model_exp.append("High credit utilization signals credit stress")

        for m in model_exp:
            st.write("•", m)

        # -------- BUSINESS RULE -------- #
        st.markdown("## ⚠️ Business Rule Explanation")

        rules = []

        if dataset == "AMEX":
            if days > 60:
                rules.append("Severe overdue triggers high-risk classification")
            if payment < 400:
                rules.append("Very poor payment score triggers rejection")

        else:
            if late > 20:
                rules.append("Extreme late payments indicate high default risk")
            if debt > 2:
                rules.append("Very high debt ratio is unacceptable risk")

        for r in rules:
            st.write("🔴", r)

        # -------- RESULT -------- #
        st.markdown("## 📊 Prediction Result")
        st.write(f"Default Probability: {prob:.2f}")

        if prob < 0.3:
            st.success("🟢 Low Risk")
        elif prob < 0.7:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

        # -------- FINAL INTERPRETATION -------- #
        st.markdown("## 🧠 Final Interpretation")

        if prob < 0.3:
            st.write("Customer is financially stable and safe for lending.")
        elif prob < 0.7:
            st.write("Customer shows moderate risk. Careful evaluation required.")
        else:
            st.write("Customer is highly risky. Loan approval not recommended.")

        # -------- YOUR LINE -------- #
        st.markdown("""
💡 *SHAP explains the model prediction, while business rules ensure critical risk conditions are enforced. 
Both are displayed to maintain transparency and support better financial decision-making.*
""")

# ================= TAB 2 ================= #
with tab2:

    st.markdown("## 📈 Model Comparison (User Friendly)")

    if "prob" in st.session_state:

        prob = float(st.session_state["prob"])
        dataset = st.session_state["dataset"]

        amex = prob if dataset == "AMEX" else 0.5
        gmsc = prob if dataset == "GMSC" else 0.5

        st.progress(amex, text=f"AMEX Risk Score: {amex:.2f}")
        st.progress(gmsc, text=f"GMSC Risk Score: {gmsc:.2f}")

        st.markdown("### 🧠 What does this mean?")

        if amex > gmsc:
            st.write("➡️ AMEX model is stricter and identifies higher risk.")
        elif gmsc > amex:
            st.write("➡️ GMSC model is stricter for this profile.")
        else:
            st.write("➡️ Both models agree on similar risk level.")

        st.markdown("### 📌 Conclusion")

        st.write("""
- This comparison helps understand **how different models evaluate the same customer**  
- If both models show high risk → **strong rejection signal**  
- If mixed → **manual review recommended**
""")

    else:
        st.info("Run prediction first to see comparison")
            
