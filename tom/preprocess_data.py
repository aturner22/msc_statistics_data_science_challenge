import pandas as pd
import numpy as np

from datetime import datetime as dt
import os
print(os.getcwd())

df = pd.read_csv("data/transactions_training_sept_oct_2023.csv", sep=";", decimal=",", parse_dates=["DATETIME_GMT"])

columns_to_keep = [
    'ID_TRX', 'ID_CARD', 'DATETIME_GMT', 'AMOUNT', 'Anomaly_amount_6',
    'FLAG_BEHAVIOUR_Anomaly_1', 'FLAG_BEHAVIOUR_Anomaly_2', 'FLAG_BEHAVIOUR_Anomaly_3',
    'FLAG_BEHAVIOUR_Anomaly_4', 'FLAG_BEHAVIOUR_Anomaly_5', 'FLAG_BEHAVIOUR_Anomaly_6',
    'FLAG_BEHAVIOUR_Anomaly7', 'FLAG_BEHAVIOUR_Anomaly_8',
    'Population_Anomaly_1', 'Population_Anomaly_2', 'Population_Anomaly_3',
    'Population_Anomaly_4', 'Population_Anomaly_5', 'Population_Anomaly_6',
    'FLAG_FRAUD'
]
# Subset the dataframe
df = df[columns_to_keep]

# Drop rows with missing values
df = df.dropna()

# Define your feature lists
numerical_cols = [col for col in df.columns if (
    col.startswith("Anomaly_amount") or
    col.startswith("Population_Anomaly") or
    col == "AMOUNT"
)]

categorical_cols = [col for col in df.columns if "FLAG_BEHAVIOUR_Anomaly" in col]

# Log-transform numerical columns, with +1 to handle zeros safely
for col in numerical_cols:
    log_col = f"log_{col}"
    df[log_col] = np.log1p(df[col])  # log1p(x) = log(1 + x), safer for zero or small values

# Drop original numerical columns
df.drop(columns=numerical_cols, inplace=True)

df["DAY_OF_WEEK"] = df["DATETIME_GMT"].dt.dayofweek
df["HOUR_OF_DAY"] = df["DATETIME_GMT"].dt.hour

print(df)
df.to_csv("processed_initial.csv")