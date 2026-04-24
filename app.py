import streamlit as st
import pickle
import pandas as pd
import json
import shap

st.set_page_config(page_title="Credit Risk AI", layout="wide")

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

# ---------------- MAIN ---------------- #
st.title("💳 Credit Default Prediction Dashboard")

dataset = st.selectbox("Select Dataset", ["AMEX", "GMSC"])

# ---------------- LOAD MODEL ---------------- #
if dataset == "AMEX":
    model = pickle.load(open("models/amex_xgb_model.pkl", "rb"))
    with open("columns/amex_columns.json") as f:
        all_columns = json.load(f)
else:
    model = pickle.load(open("models/gmsc_xgb_model.pkl", "rb"))
    with open("columns/gmsc_columns.json") as f:
        all_columns = json.load(f)

# ---------------- INPUT ---------------- #
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
        "Late Payments Count (30–59 Days)",
        min_value=0,
        max_value=30,
        value=1
    )

    debt_ratio = st.sidebar.slider("Debt Ratio", 0.0, 5.0, 0.5)
    income = st.sidebar.number_input("Monthly Income (₹)", 0, 1000000, 50000)
    open_credit = st.sidebar.number_input("Open Credit Lines", 0, 20, 5)

# ---------------- BUILD INPUT ---------------- #
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

# ---------------- PREDICT ---------------- #
if st.button("Predict Risk"):

    prob = model.predict_proba(df)[0][1]

    # MODEL RISK
    if prob < 0.3:
        model_risk = "Low Risk"
    elif prob < 0.7:
        model_risk = "Medium Risk"
    else:
        model_risk = "High Risk"

    # ---------------- BUSINESS RULES ---------------- #
    high_flag = False
    medium_flag = False
    reasons = []

    if dataset == "AMEX":
        if days_due >= 60 or delay_count >= 15 or payment_score <= 400:
            high_flag = True
            reasons.append("severe payment issues")

        elif 15 <= days_due < 60 or 5 <= delay_count < 15:
            medium_flag = True
            reasons.append("moderate payment delays")

    else:
        if past_due >= 10 or utilization > 0.8 or debt_ratio > 1.5:
            high_flag = True
            reasons.append("frequent late payments and high financial stress")

        elif past_due >= 3 or utilization > 0.5:
            medium_flag = True
            reasons.append("moderate credit risk indicators")

    # FINAL RISK
    if high_flag:
        final_risk = "High Risk"
    elif medium_flag:
        final_risk = "Medium Risk"
    else:
        final_risk = model_risk

    # ---------------- DISPLAY ---------------- #
    st.markdown("## 📊 Prediction Result")
    st.write(f"Default Probability: {prob:.2f}")

    if final_risk == "High Risk":
        st.error("🔴 High Risk")
    elif final_risk == "Medium Risk":
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")

    # ---------------- MODEL EXPLANATION ---------------- #
    st.markdown("### 🤖 Model Explanation")

    explanation_points = []

    if dataset == "AMEX":
        if payment_score < 500:
            explanation_points.append("poor payment behavior")
        else:
            explanation_points.append("good payment history")

        if balance < 20000:
            explanation_points.append("low account balance")
        else:
            explanation_points.append("healthy account balance")

        if days_due > 30:
            explanation_points.append("high overdue days")

        if delay_count > 5:
            explanation_points.append("frequent payment delays")

    else:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df)
            values = shap_values[0] if isinstance(shap_values, list) else shap_values

            shap_df = pd.DataFrame({
                "feature": df.columns,
                "impact": values[0]
            })

            shap_df["abs"] = shap_df["impact"].abs()
            shap_df = shap_df.sort_values("abs", ascending=False).head(3)

            feature_map = {
                "RevolvingUtilizationOfUnsecuredLines": "credit utilization",
                "age": "age",
                "NumberOfTime30-59DaysPastDueNotWorse": "late payments",
                "DebtRatio": "debt ratio",
                "MonthlyIncome": "monthly income",
                "NumberOfOpenCreditLinesAndLoans": "credit accounts"
            }

            for _, row in shap_df.iterrows():
                name = feature_map.get(row["feature"], row["feature"])
                if row["impact"] > 0:
                    explanation_points.append(f"high {name}")
                else:
                    explanation_points.append(f"stable {name}")

        except:
            explanation_points.append("model insights unavailable")

    # ---------------- FINAL AI EXPLANATION ---------------- #
    st.markdown("### 🧠 AI Explanation")

    explanation_text = "Based on the analysis, "

    if final_risk == "High Risk":
        explanation_text += "the customer is classified as high risk due to "
    elif final_risk == "Medium Risk":
        explanation_text += "the customer shows moderate risk due to "
    else:
        explanation_text += "the customer is considered low risk because of "

    explanation_text += ", ".join(explanation_points)

    if reasons:
        explanation_text += f", along with {', '.join(reasons)}"

    explanation_text += "."

    st.info(explanation_text)

    # ---------------- DECISION ---------------- #
    st.markdown("### 📌 Suggested Decision")

    if final_risk == "Low Risk":
        st.success("✅ Approve Loan")
    elif final_risk == "Medium Risk":
        st.warning("⚠️ Review Manually")
    else:
        st.error("❌ Reject Loan")

    # ---------------- NOTE ---------------- #
    st.markdown("---")
    st.markdown(
        "💡 SHAP explains the model prediction, while business rules ensure critical risk conditions are enforced. I display both to maintain transparency."
    )
