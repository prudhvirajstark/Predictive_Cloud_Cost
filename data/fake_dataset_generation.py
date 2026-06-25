import pandas as pd
import numpy as np

# 1. Setup dimensions
np.random.seed(42)
date_range = pd.date_range(start="2026-03-01", end="2026-05-30", freq="h")
services = ["EC2", "S3", "RDS", "Lambda"]
regions = ["us-east-1", "eu-west-1"]

# 2. Build base dataframe structure
records = []
for dt in date_range:
    for service in services:
        for region in regions:
            records.append({"Timestamp": dt, "Service": service, "Region": region})

df = pd.DataFrame(records)

# 3. Inject realistic baseline usage and costs (with daily seasonality)
df["Hour"] = df["Timestamp"].dt.hour
# Higher usage during business hours (9 AM to 5 PM)
seasonality = np.sin((df["Hour"] - 6) * np.pi / 12) + 1.5

df["Usage_Amount"] = np.random.gamma(shape=2, scale=10, size=len(df)) * seasonality
df["Cost"] = df["Usage_Amount"] * np.random.uniform(0.05, 0.20, size=len(df))

# 4. Inject explicit anomalies for the ML pipeline to detect
# Anomaly 1: A massive spike in Lambda costs due to an infinite loop on May 10th
df.loc[(df["Timestamp"].dt.date == pd.to_datetime("2026-05-10").date()) & (df["Service"] == "Lambda"), "Cost"] *= 15

# Anomaly 2: Unused, runaway EC2 instances starting May 20th
df.loc[(df["Timestamp"] >= "2026-05-20") & (df["Service"] == "EC2"), "Cost"] *= 3

# Save to CSV for your Streamlit dashboard and Scikit-Learn pipeline
df.to_csv("cloud_cost_poc_data.csv", index=False)
print(f"Generated PoC dataset with {len(df)} rows. Saved to 'cloud_cost_poc_data.csv'.")
