# Claim Delay Prediction - Streamlit App

import pandas as pd
import numpy as np
import random
from faker import Faker
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Claim Delay Prediction", layout="wide")

# Faker setup
fake = Faker()

# Synthetic Data Generator
def generate_data(n=1000):
    data = []
    for _ in range(n):
        admit_date = fake.date_between(start_date='-2y', end_date='-1y')
        discharge_date = admit_date + datetime.timedelta(days=random.randint(1, 10))
        entry_date = discharge_date + datetime.timedelta(days=random.randint(1, 15))
        update_date = entry_date + datetime.timedelta(days=random.randint(1, 15))

        trans_amount = round(random.uniform(1000, 10000), 2)
        pay_amount = round(trans_amount * random.uniform(0.4, 0.9), 2)
        delay = (update_date - entry_date).days > 10
        notes = fake.sentence(nb_words=12) + (" missing documents" if delay else " all documents present")

        data.append({
            "AccountType": random.choice(["Inpatient", "Outpatient"]),
            "CurrentPrimaryInsurance": random.choice(["Insurer A", "Insurer B", "Insurer C"]),
            "CurrentFinancialClass": random.choice(["Private", "Govt", "Self-pay"]),
            "TransactionInsurance": random.choice(["Yes", "No"]),
            "BillType": random.choice(["Type A", "Type B", "Type C"]),
            "ClaimDeptResponsible": random.choice(["Cardiology", "Radiology", "Billing"]),
            "TransAmount": trans_amount,
            "PAYAmount": pay_amount,
            "REVAmount": round(trans_amount - pay_amount, 2),
            "Notes": notes,
            "IsDelayed": int(delay)
        })
    return pd.DataFrame(data)

# Load Data and Build Model
df = generate_data(1000)
features = [
    "AccountType", "CurrentPrimaryInsurance", "CurrentFinancialClass",
    "TransactionInsurance", "BillType", "ClaimDeptResponsible",
    "TransAmount", "PAYAmount", "REVAmount", "Notes"
]
X = df[features]
y = df["IsDelayed"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["TransAmount", "PAYAmount", "REVAmount"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), [
            "AccountType", "CurrentPrimaryInsurance", "CurrentFinancialClass",
            "TransactionInsurance", "BillType", "ClaimDeptResponsible"
        ]),
        ("txt", TfidfVectorizer(max_features=50), "Notes")
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Streamlit App
st.title("Claim Delay Prediction System")
st.markdown("""
### Predict whether a new claim is likely to be delayed
Enter the details below to simulate a new claim and get instant prediction results.
""")

with st.form("claim_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        account_type = st.selectbox("Account Type", ["Inpatient", "Outpatient"])
        financial_class = st.selectbox("Financial Class", ["Private", "Govt", "Self-pay"])
        bill_type = st.selectbox("Bill Type", ["Type A", "Type B", "Type C"])
    with col2:
        insurance = st.selectbox("Primary Insurance", ["Insurer A", "Insurer B", "Insurer C"])
        transaction_insurance = st.selectbox("Transaction Insurance", ["Yes", "No"])
        dept = st.selectbox("Department Responsible", ["Cardiology", "Radiology", "Billing"])
    with col3:
        trans_amt = st.number_input("Transaction Amount", min_value=0.0, value=5000.0)
        pay_amt = st.number_input("Payment Amount", min_value=0.0, value=3500.0)
        rev_amt = trans_amt - pay_amt

    notes = st.text_area("Notes", value="All documents are present and verified.")

    submitted = st.form_submit_button("Predict Delay")

    if submitted:
        input_data = pd.DataFrame([{
            "AccountType": account_type,
            "CurrentPrimaryInsurance": insurance,
            "CurrentFinancialClass": financial_class,
            "TransactionInsurance": transaction_insurance,
            "BillType": bill_type,
            "ClaimDeptResponsible": dept,
            "TransAmount": trans_amt,
            "PAYAmount": pay_amt,
            "REVAmount": rev_amt,
            "Notes": notes
        }])

        prediction = pipeline.predict(input_data)[0]
        proba = pipeline.predict_proba(input_data)[0][prediction]

        if prediction == 1:
            st.error(f"🚨 This claim is likely to be **DELAYED** (Confidence: {proba:.2f})")
        else:
            st.success(f"✅ This claim is likely to be **PROCESSED ON TIME** (Confidence: {proba:.2f})")

        st.markdown("---")
        st.markdown("#### Interpretation and Next Steps")
        st.markdown("""
        - **Delayed claims** are often linked with incomplete notes or specific insurers.
        - Ensure you have attached all required documents.
        - Verify fields like `Financial Class`, `Department`, and `Transaction Insurance`.
        - Use this tool as an **advisory system** to pre-screen claim forms before submission.
        """)
