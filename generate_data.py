"""
Step 1: Create fake (synthetic) Djezzy customers.

Idea: we invent 5 customer types. For each type we choose an average
value for each column (example: Data lovers use about 30 GB).
Then we add random variation so customers are not all identical.
"""

import numpy as np
import pandas as pd

np.random.seed(42)   # same random numbers every run, so results are repeatable
N = 5000             # total number of customers

# For each feature: (average, spread). Spread = how much customers differ.
personas = {
    "Data lovers": dict(
        share=0.25,
        data_gb=(30, 8), voice_min=(100, 40), sms_count=(20, 10),
        topup_count=(4, 1), avg_topup_da=(900, 200),
        tenure_months=(30, 15), night_data_ratio=(0.40, 0.10),
        roaming_prob=0.05),
    "Talkers": dict(
        share=0.20,
        data_gb=(4, 2), voice_min=(600, 150), sms_count=(60, 20),
        topup_count=(3, 1), avg_topup_da=(600, 150),
        tenure_months=(40, 15), night_data_ratio=(0.15, 0.05),
        roaming_prob=0.03),
    "Low spenders": dict(
        share=0.25,
        data_gb=(3, 1.5), voice_min=(80, 40), sms_count=(15, 10),
        topup_count=(1, 0.5), avg_topup_da=(200, 80),
        tenure_months=(20, 12), night_data_ratio=(0.20, 0.08),
        roaming_prob=0.01),
    "Premium": dict(
        share=0.10,
        data_gb=(45, 10), voice_min=(500, 120), sms_count=(50, 20),
        topup_count=(5, 1), avg_topup_da=(2500, 500),
        tenure_months=(50, 15), night_data_ratio=(0.25, 0.08),
        roaming_prob=0.35),
    "Students": dict(
        share=0.20,
        data_gb=(20, 6), voice_min=(120, 50), sms_count=(40, 15),
        topup_count=(3, 1), avg_topup_da=(400, 100),
        tenure_months=(12, 8), night_data_ratio=(0.55, 0.10),
        roaming_prob=0.02),
}

wilayas = ["Alger", "Blida", "Oran", "Constantine", "Annaba",
           "Setif", "Tizi Ouzou", "Batna", "Ouargla", "Tlemcen"]

parts = []
for name, p in personas.items():
    n = int(N * p["share"])
    df = pd.DataFrame({
        "true_persona": name,   # hidden answer, used later only to check our model
        "tenure_months": np.random.normal(*p["tenure_months"], n),
        "data_gb": np.random.normal(*p["data_gb"], n),
        "voice_min": np.random.normal(*p["voice_min"], n),
        "sms_count": np.random.normal(*p["sms_count"], n),
        "night_data_ratio": np.random.normal(*p["night_data_ratio"], n),
        "topup_count": np.random.normal(*p["topup_count"], n),
        "avg_topup_da": np.random.normal(*p["avg_topup_da"], n),
        "roaming_flag": np.random.binomial(1, p["roaming_prob"], n),
    })
    parts.append(df)

data = pd.concat(parts, ignore_index=True)

# Real numbers can't be negative, so we fix that
for col in ["tenure_months", "data_gb", "voice_min", "sms_count",
            "topup_count", "avg_topup_da"]:
    data[col] = data[col].clip(lower=0)
data["night_data_ratio"] = data["night_data_ratio"].clip(0, 1)

# Whole numbers where it makes sense
data["tenure_months"] = data["tenure_months"].round().astype(int)
data["voice_min"] = data["voice_min"].round().astype(int)
data["sms_count"] = data["sms_count"].round().astype(int)
data["topup_count"] = data["topup_count"].round().astype(int)
data["data_gb"] = data["data_gb"].round(1)
data["avg_topup_da"] = (data["avg_topup_da"] / 50).round() * 50
data["night_data_ratio"] = data["night_data_ratio"].round(2)

# Monthly spend = number of top-ups x average amount (plus small noise)
data["monthly_spend_da"] = (
    data["topup_count"] * data["avg_topup_da"] * np.random.uniform(0.9, 1.1, len(data))
).round(-1)

# Extra columns
data["wilaya"] = np.random.choice(wilayas, len(data))
data["plan_type"] = np.random.choice(["Prepaid", "Postpaid"], len(data), p=[0.85, 0.15])

# Make the data a bit messy on purpose (real data is never clean)
# 1) about 3% missing values in two columns
for col in ["data_gb", "voice_min"]:
    missing_rows = data.sample(frac=0.03, random_state=1 if col == "data_gb" else 2).index
    data.loc[missing_rows, col] = np.nan
# 2) about 0.5% extreme outliers in data usage
outlier_rows = data.sample(frac=0.005, random_state=3).index
data.loc[outlier_rows, "data_gb"] = data.loc[outlier_rows, "data_gb"] * 5

# Shuffle rows and add anonymous customer IDs
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.insert(0, "customer_id", [f"C{i:05d}" for i in range(1, len(data) + 1)])

# Put columns in a nice order
cols = ["customer_id", "wilaya", "plan_type", "tenure_months", "data_gb",
        "voice_min", "sms_count", "night_data_ratio", "topup_count",
        "avg_topup_da", "monthly_spend_da", "roaming_flag", "true_persona"]
data = data[cols]

data.to_csv("customers.csv", index=False)
print("Saved customers.csv with", len(data), "customers")
print(data.head())
print("\nMissing values per column:")
print(data.isna().sum()[data.isna().sum() > 0])
