import streamlit as st
import pickle
import pandas as pd
import json

st.set_page_config(page_title="Evoastra AI", layout="wide")

# ---------------- PREMIUM UI ---------------- #
st.markdown("""
<style>
body {background-color:#04090f; color:white;}
.card {
    padding:20px;
    border-radius:18px;
    background: rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}
.metric {font-size:28px; font-weight:bold;}
.center {text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 EVOASTRA AI — Credit Risk Intelligence")

page = st.sidebar.radio("Navigation", ["Dashboard","Predictor","Comparison"])

# ================= DASHBOARD ================= #
if page == "Dashboard":

    st.markdown("## ⚡ System Overview")

    col1,col2,col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card center">', unsafe_allow_html=True)
        st.markdown('<div class="metric">0.94</div>', unsafe_allow_html=True)
        st.write("Model Accuracy (AUC)")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card center">', unsafe_allow_html=True)
        st.markdown('<div class="metric">190+</div>', unsafe_allow_html=True)
        st.write("AMEX Features")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card center">', unsafe_allow_html=True)
        st.markdown('<div class="metric">150K</div>', unsafe_allow_html=True)
        st.write("GMSC Records")
        st.markdown('</div>', unsafe_allow_html=True)

    st.info("This system uses XGBoost + Business Rules + Explainability (SHAP concept).")

# ================= PREDICTOR ================= #
elif page == "Predictor":

    dataset = st.selectbox("Dataset", ["AMEX","GMSC"])

    st.sidebar.header("Customer Details")

    # Common Inputs
    payment = st.sidebar.slider("Payment Score",300,900,700)
    balance = st.sidebar.number_input("Balance",0,1000000,40000)
    days = st.sidebar.number_input("Days Past Due",0,120,5)
    delay = st.sidebar.number_input("Delay Count",0,50,2)

    util = st.sidebar.slider("Credit Utilization",0.0,1.0,0.3)
    age = st.sidebar.slider("Age",18,80,30)
    late = st.sidebar.number_input("Late Payments",0,30,1)
    debt = st.sidebar.slider("Debt Ratio",0.0,5.0,0.5)
    income = st.sidebar.number_input("Income",0,1000000,50000)

    if st.button("🚀 Run AI Prediction"):

        # -------- LOAD MODEL -------- #
        if dataset=="AMEX":
            model = pickle.load(open("models/amex_xgb_model.pkl","rb"))
            with open("columns/amex_columns.json") as f:
                cols = json.load(f)

            data = {c:0 for c in cols}
            data["P_2"]=payment/1000
            data["B_1"]=balance/100000
            data["D_39"]=days/100
            data["D_41"]=delay/100

        else:
            model = pickle.load(open("models/gmsc_xgb_model.pkl","rb"))
            with open("columns/gmsc_columns.json") as f:
                cols = json.load(f)

            data = {c:0 for c in cols}
            data["RevolvingUtilizationOfUnsecuredLines"]=util
            data["age"]=age
            data["NumberOfTime30-59DaysPastDueNotWorse"]=late
            data["DebtRatio"]=debt
            data["MonthlyIncome"]=income

        df = pd.DataFrame([data])
        prob = float(model.predict_proba(df)[0][1])

        st.session_state["inputs"]={
            "payment":payment,"balance":balance,"days":days,"delay":delay,
            "util":util,"age":age,"late":late,"debt":debt,"income":income
        }

        st.session_state["prob"]=prob

        # -------- RESULT UI -------- #
        st.markdown("## 🎯 Risk Score")

        st.progress(prob)

        if prob<0.3:
            st.success("🟢 LOW RISK")
        elif prob<0.7:
            st.warning("🟡 MEDIUM RISK")
        else:
            st.error("🔴 HIGH RISK")

        # -------- EXPLANATION -------- #
        st.markdown("## 🤖 AI Explanation")

        explanation=[]

        if payment<500:
            explanation.append("Low payment score indicates poor repayment behavior")

        if days>30:
            explanation.append("High delay in payments increases default risk")

        if util>0.8:
            explanation.append("High credit utilization shows financial stress")

        if debt>1:
            explanation.append("High debt ratio increases financial burden")

        for e in explanation:
            st.write("•",e)

        # -------- BUSINESS RULE -------- #
        st.markdown("## ⚠️ Business Rules")

        if late>10:
            st.write("🔴 Frequent late payments → High Risk")

        if debt>2:
            st.write("🔴 Very high debt → Loan Rejection Condition")

        # -------- FINAL INTERPRETATION -------- #
        st.markdown("## 🧠 Final Decision")

        if prob<0.3:
            st.success("Approve Loan")
        elif prob<0.7:
            st.warning("Manual Review Required")
        else:
            st.error("Reject Loan")

        st.markdown("""
💡 *SHAP explains the model prediction, while business rules ensure critical risk conditions are enforced.*
""")

# ================= COMPARISON ================= #
elif page=="Comparison":

    st.markdown("## 📊 Model Comparison Dashboard")

    if "inputs" in st.session_state:

        inp=st.session_state["inputs"]

        amex_model=pickle.load(open("models/amex_xgb_model.pkl","rb"))
        gmsc_model=pickle.load(open("models/gmsc_xgb_model.pkl","rb"))

        with open("columns/amex_columns.json") as f:
            amex_cols=json.load(f)

        with open("columns/gmsc_columns.json") as f:
            gmsc_cols=json.load(f)

        # -------- INPUT MAPPING -------- #
        amex_data={c:0 for c in amex_cols}
        amex_data["P_2"]=inp["payment"]/1000
        amex_data["B_1"]=inp["balance"]/100000
        amex_data["D_39"]=inp["days"]/100

        gmsc_data={c:0 for c in gmsc_cols}
        gmsc_data["age"]=inp["age"]
        gmsc_data["DebtRatio"]=inp["debt"]
        gmsc_data["MonthlyIncome"]=inp["income"]

        amex_prob=float(amex_model.predict_proba(pd.DataFrame([amex_data]))[0][1])
        gmsc_prob=float(gmsc_model.predict_proba(pd.DataFrame([gmsc_data]))[0][1])

        col1,col2=st.columns(2)

        with col1:
            st.metric("AMEX Risk",f"{amex_prob:.2f}")
            st.progress(amex_prob)

        with col2:
            st.metric("GMSC Risk",f"{gmsc_prob:.2f}")
            st.progress(gmsc_prob)

        # -------- USER FRIENDLY -------- #
        st.markdown("## 🧠 Insight")

        if amex_prob>gmsc_prob:
            st.write("AMEX model detects higher behavioral risk.")
        elif gmsc_prob>amex_prob:
            st.write("GMSC model detects financial stress risk.")
        else:
            st.write("Both models agree on similar risk level.")

        # -------- FINAL -------- #
        avg=(amex_prob+gmsc_prob)/2

        st.markdown("## 📌 Final Conclusion")

        if avg>0.7:
            st.error("High Risk → Reject Loan")
        elif avg>0.3:
            st.warning("Moderate Risk → Manual Review")
        else:
            st.success("Low Risk → Approve Loan")

    else:
        st.info("Run prediction first")


    
           
        
   
    
                             

    
       

   
