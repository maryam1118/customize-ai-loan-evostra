import streamlit as st
import pickle
import pandas as pd
import json
import shap

# ---------------- UI STYLE ---------------- #
st.set_page_config(page_title="Credit Risk AI", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #1f4e79;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ---------------- #
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

with open("users.json") as f:
    users = json.load(f)

def login():
    st.title("🔐 Login Page")
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

# ---------------- TABS ---------------- #
tab1, tab2 = st.tabs(["📊 Prediction", "📈 Comparison"])

# ========================= TAB 1 ========================= #
with tab1:

    dataset = st.selectbox("Select Dataset", ["AMEX", "GMSC"])

    # -------- LOAD MODEL -------- #
    if dataset == "AMEX":
        model = pickle.load(open("models/amex_xgb_model.pkl", "rb"))
        with open("columns/amex_columns.json") as f:
            all_columns = json.load(f)
    else:
        model = pickle.load(open("models/gmsc_xgb_model.pkl", "rb"))
        with open("columns/gmsc_columns.json") as f:
            all_columns = json.load(f)

    # -------- INPUT -------- #
    st.sidebar.header("Input Features")

    if dataset == "AMEX":
        payment_score = st.sidebar.slider("Payment Behavior Score", 300, 900, 700)
        balance = st.sidebar.number_input("Account Balance (₹)", 0, 1000000, 40000)
        days_due = st.sidebar.number_input("Days Past Due", 0, 120, 5)
        risk_score = st.sidebar.slider("Risk Indicator Score", 0, 10, 3)
        spending = st.sidebar.number_input("Monthly Spending (₹)", 0, 100000, 20000)
        delay_count = st.sidebar.number_input("Recent Delay Count", 0, 50, 2)

    else:
        utilization = st.sidebar.slider("Credit Utilization", 0.0, 1.0, 0.3)
        age = st.sidebar.slider("Age", 18, 80, 30)

        past_due = st.sidebar.number_input(
            "Late Payments Count (30–59 Days)", 0, 30, 1
        )

        debt_ratio = st.sidebar.slider("Debt Ratio", 0.0, 5.0, 0.5)
        income = st.sidebar.number_input("Monthly Income (₹)", 0, 1000000, 50000)
        open_credit = st.sidebar.number_input("Open Credit Lines", 0, 20, 5)

    # -------- BUILD INPUT -------- #
    full_input = {col: 0 for col in all_columns}

    if dataset == "AMEX":
        full_input["P_2"] = payment_score / 1000
        full_input["B_1"] = balance / 100000
        full_input["D_39"] = days_due / 100
        full_input["R_1"] = risk_score / 10
        full_input["S_3"] = spending / 100000
        full_input["D_41"] = delay_count / 100
    else:
        full_input["RevolvingUtilizationOfUnsecuredLines"] = utilization
        full_input["age"] = age
        full_input["NumberOfTime30-59DaysPastDueNotWorse"] = past_due
        full_input["DebtRatio"] = debt_ratio
        full_input["MonthlyIncome"] = income
        full_input["NumberOfOpenCreditLinesAndLoans"] = open_credit

    df = pd.DataFrame([full_input])

    # -------- PREDICT -------- #
    if st.button("Predict Risk"):

        prob = model.predict_proba(df)[0][1]
        st.session_state["prob"] = prob
        st.session_state["dataset"] = dataset

        # -------- EXPLANATION FIRST -------- #
        st.markdown("## 🤖 Model Explanation")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            values = shap_values[0]

            top_idx = abs(values).argsort()[-3:][::-1]

            for i in top_idx:
                feature = df.columns[i]
                val = values[i]

                name_map = {
                    "P_2": "Payment Behavior",
                    "B_1": "Account Balance",
                    "D_39": "Days Past Due",
                    "R_1": "Risk Score",
                    "S_3": "Spending",
                    "D_41": "Delay Count"
                }

                fname = name_map.get(feature, feature)

                if val > 0:
                    st.write(f"🔺 {fname} increased risk")
                else:
                    st.write(f"🔻 {fname} reduced risk")

        except:
            st.info("Explanation not available")

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
            st.write("Customer is financially stable with low risk.")
        elif prob < 0.7:
            st.write("Customer shows moderate financial risk.")
        else:
            st.write("Customer shows high financial risk.")

        # -------- DECISION -------- #
        st.markdown("## 📌 Suggested Decision")

        if prob < 0.5:
            st.success("✔ Approve Loan")
        else:
            st.error("❌ Reject Loan")


# ========================= TAB 2 ========================= #
with tab2:

    st.markdown("## 📈 Model Comparison Dashboard")

    if "prob" in st.session_state:

        prob = st.session_state["prob"]
        dataset = st.session_state["dataset"]

        amex_prob = prob if dataset == "AMEX" else 0.5
        gmsc_prob = prob if dataset == "GMSC" else 0.5

        st.markdown("### Risk Comparison")

        st.progress(amex_prob, text=f"AMEX Risk: {amex_prob:.2f}")
        st.progress(gmsc_prob, text=f"GMSC Risk: {gmsc_prob:.2f}")

        st.markdown("### 🧠 Insight")

        if amex_prob > gmsc_prob:
            st.error("AMEX model predicts higher risk")
        elif gmsc_prob > amex_prob:
            st.warning("GMSC model predicts higher risk")
        else:
            st.success("Both models show similar risk")

    else:
        st.info("Run prediction first to see comparison")
            
