import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from datetime import datetime
from faker import Faker
import random

# Initialize the Faker instance
fake = Faker()

# Function to generate random data for each column
def generate_data():
    return {
        "AccountNumber": fake.unique.random_number(digits=10),
        "VisitID": fake.unique.random_number(digits=8),
        "AdmitDate": fake.date_this_decade(),
        "DischargeDate": fake.date_this_decade(),
        "Name": fake.name(),
        "PatientID": fake.unique.random_number(digits=8),
        "LocationID": fake.random_number(digits=5),
        "AccountType": random.choice(['Inpatient', 'Outpatient']),
        "CurrentPrimaryInsurance": fake.company(),
        "CurrentFinancialClass": random.choice(['Private', 'Medicare', 'Medicaid']),
        "EntryBy": fake.name(),
        "EntryDate": fake.date_this_year(),
        "DS_YEAR": random.choice([2023, 2024, 2025]),
        "DS_MONTH": random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        "DS_DAY": random.randint(1, 31),
        "DS_WD": random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
        "Notes": fake.sentence(),
        "TransCount": random.randint(1, 10),
        "TransAmount": round(random.uniform(50, 5000), 2),
        "TransactionInsurance": fake.company(),
        "PSTAmount": round(random.uniform(10, 500), 2),
        "ADJAmount": round(random.uniform(0, 1000), 2),
        "PAYAmount": round(random.uniform(0, 1000), 2),
        "REVAmount": round(random.uniform(0, 1000), 2),
        "BillNo": fake.unique.random_number(digits=8),
        "BillStatus": random.choice(['Paid', 'Pending', 'Cancelled']),
        "BillType": random.choice(['Normal', 'Emergency']),
        "BillFromDate": fake.date_this_year(),
        "BillThroughDate": fake.date_this_year(),
        "Reference1": fake.word(),
        "Reference2": fake.word(),
        "ClaimID": fake.unique.random_number(digits=8),
        "RMCCode": fake.random_number(digits=4),
        "ClaimGroupCode": fake.random_number(digits=5),
        "ClaimDeptResponsible": fake.company(),
        "UpdateDateTime": fake.date_this_decade(),
    }

# Create a list of 1000 records
records = [generate_data() for _ in range(100000)]

# Convert to DataFrame
df = pd.DataFrame(records)

# Feature Engineering: Convert dates to datetime format
df['AdmitDate'] = pd.to_datetime(df['AdmitDate'])
df['DischargeDate'] = pd.to_datetime(df['DischargeDate'])
df['BillFromDate'] = pd.to_datetime(df['BillFromDate'])
df['BillThroughDate'] = pd.to_datetime(df['BillThroughDate'])

# Create a feature for claim processing time
df['ClaimProcessingTime'] = (df['BillThroughDate'] - df['DischargeDate']).dt.days

# Define the label as 'Fast' or 'Delayed' based on BillStatus and Processing Time
# A claim is considered 'fast' if it's paid within 30 days
df['ClaimStatus'] = df.apply(lambda row: 1 if row['BillStatus'] == 'Paid' and row['ClaimProcessingTime'] <= 30 else 0, axis=1)

# Prepare features (X) and target (y)
X = df[['ClaimProcessingTime', 'TransAmount', 'PSTAmount', 'ADJAmount', 'PAYAmount', 'REVAmount']]

# Encoding categorical variables (AccountType and CurrentFinancialClass)
le_account_type = LabelEncoder()
le_financial_class = LabelEncoder()

# Fit the encoders on the whole dataset before splitting the data
df['AccountType'] = le_account_type.fit_transform(df['AccountType'])
df['CurrentFinancialClass'] = le_financial_class.fit_transform(df['CurrentFinancialClass'])

# Add the encoded categorical variables to the feature set
X['AccountType'] = df['AccountType']
X['CurrentFinancialClass'] = df['CurrentFinancialClass']

# Define the target variable
y = df['ClaimStatus']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (RandomForestClassifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")

# Predicting a new claim (you will need to pass in similar features)
# Example: New claim data

new_claim_data = {
    'ClaimProcessingTime': 10,  # Example processing time in days
    'TransAmount': 300.0,  # Example transaction amount
    'PSTAmount': 10.0,  # Example PST amount
    'ADJAmount': 0.0,
    'PAYAmount': 250.0,
    'REVAmount': 40.0,
    'AccountType': le_account_type.transform(['Inpatient'])[0],  # Use the trained encoder
    'CurrentFinancialClass': le_financial_class.transform(['Private'])[0],  # Use the trained encoder
}

new_claim_df = pd.DataFrame([new_claim_data])

# Make prediction for new claim
prediction = model.predict(new_claim_df)

# Output prediction
if prediction == 1:
    print("The claim will be processed fast.")
else:
    print("The claim will be processed delayed.")
