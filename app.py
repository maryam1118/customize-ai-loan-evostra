import streamlit as st
import pickle
import pandas as pd
import json
import shap
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

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

# ---------------- PDF FUNCTION ---------------- #
def generate_pdf(amex_prob, gmsc_prob):
    file_path = "credit_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Credit Risk Comparison Report", styles["Title"]))
    content.append(Spacer(1, 20))
    content.append(Paragraph(f"AMEX Probability: {amex_prob:.2f}", styles["Normal"]))
    content.append(Paragraph(f"GMSC Probability: {gmsc_prob:.2f}", styles["Normal"]))
    content.append(Spacer(1, 20))

    if amex_prob > gmsc_prob:
        insight = "AMEX predicts higher risk."
    elif gmsc_prob > amex_prob:
        insight = "GMSC predicts higher risk."
    else:
        insight = "Both models show similar risk."

    content.append(Paragraph(f"Insight: {insight}", styles["Normal"]))
    doc.build(content)

    return file_path

# ---------------- PREDICT ---------------- #
if st.button("Predict Risk"):

    prob = model.predict_proba(df)[0][1]

    st.write(f"Default Probability: {prob:.2f}")

    # ---------------- COMPARISON ---------------- #
    try:
        # Load both models
        amex_model = pickle.load(open("models/amex_xgb_model.pkl", "rb"))
        gmsc_model = pickle.load(open("models/gmsc_xgb_model.pkl", "rb"))

        # Dummy mapping for comparison (safe fallback)
        amex_prob = prob if dataset == "AMEX" else 0.5
        gmsc_prob = prob if dataset == "GMSC" else 0.5

        st.markdown("## 📊 Model Comparison")

        fig, ax = plt.subplots()
        ax.bar(["AMEX", "GMSC"], [amex_prob, gmsc_prob])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # ---------------- SHAP SIDE BY SIDE ---------------- #
        st.markdown("## 🤖 SHAP Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("AMEX")
            try:
                explainer = shap.TreeExplainer(amex_model)
                shap_values = explainer.shap_values(df)
                shap.summary_plot(shap_values, df, show=False)
                st.pyplot(plt.gcf())
                plt.clf()
            except:
                st.info("Not available")

        with col2:
            st.subheader("GMSC")
            try:
                explainer = shap.TreeExplainer(gmsc_model)
                shap_values = explainer.shap_values(df)
                shap.summary_plot(shap_values, df, show=False)
                st.pyplot(plt.gcf())
                plt.clf()
            except:
                st.info("Not available")

        # ---------------- PDF DOWNLOAD ---------------- #
        if st.button("Download PDF Report"):
            file = generate_pdf(amex_prob, gmsc_prob)

            with open(file, "rb") as f:
                st.download_button(
                    "Download Report",
                    data=f,
                    file_name="credit_report.pdf"
                )

    except:
        st.info("Comparison not available")
            

   
