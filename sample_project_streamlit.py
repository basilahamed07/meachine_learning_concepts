# Claim Delay Analysis - Streamlit Dashboard Version

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

# Set Streamlit page config
st.set_page_config(page_title="Claim Delay Analysis", layout="wide")

# Generate Synthetic Data
fake = Faker()
n_samples = 1000

def generate_data(n):
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

# Load and prepare data
df = generate_data(n_samples)
features = [
    "AccountType", "CurrentPrimaryInsurance", "CurrentFinancialClass",
    "TransactionInsurance", "BillType", "ClaimDeptResponsible",
    "TransAmount", "PAYAmount", "REVAmount", "Notes"
]
X = df[features]
y = df["IsDelayed"]

# Build Pipeline
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

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Streamlit Interface
st.title("Claim Delay Risk Dashboard")
st.markdown("---")

# Summary Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Claims", len(df))
col2.metric("Delayed Claims", df['IsDelayed'].sum())
col3.metric("Delay Rate (%)", f"{(df['IsDelayed'].mean() * 100):.2f}%")

# Confusion Matrix
st.subheader("Confusion Matrix")
conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Feature Importance (numeric approximation)
st.subheader("Feature Importance (Random Forest)")
model = pipeline.named_steps['classifier']
preprocessor_fitted = pipeline.named_steps['preprocessor']
importances = model.feature_importances_
feature_names = preprocessor_fitted.get_feature_names_out()
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(by="importance", ascending=False).head(10)
fig2, ax2 = plt.subplots()
sns.barplot(x="importance", y="feature", data=fi_df, ax=ax2)
ax2.set_title("Top 10 Features Contributing to Delay")
st.pyplot(fig2)

# Sample Recommendations
st.subheader("Sample Do's & Don'ts")
st.markdown("""
- ✅ **Do** attach all relevant lab reports when submitting to *Insurer A*.
- ❌ **Don't** submit claims to *Insurer C* without radiology notes – high delay correlation.
- ✅ **Do** prefer complete entries in `Notes` – claims with 'missing documents' flag are 2x more delayed.
""")
