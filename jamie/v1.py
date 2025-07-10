import pandas as pd
from sklearn.preprocessing import OrdinalEncoder



# a. Load dataset
df = pd.read_csv("data/transactions_training_sept_oct_2023.csv", sep=';')



# b. Parse datetime
df['DATETIME_GMT'] = pd.to_datetime(df['DATETIME_GMT'])

# c. Feature engineering
df['hour'] = df['DATETIME_GMT'].dt.hour
df['day_of_week'] = df['DATETIME_GMT'].dt.dayofweek

print(df)

# # d. Encode categorical anomalies (e.g., one-hot or ordinal)

# categorical_cols = [col for col in df.columns if "FLAG_BEHAVIOUR_Anomaly" in col]
# encoder = OrdinalEncoder()
# df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

# # e. Drop unused columns
# drop_cols = ['ID_TRX', 'ID_CARD_BEN', 'DATETIME_GMT']
# X = df.drop(columns=drop_cols + ['FLAG_FRAUD'])
# y = df['FLAG_FRAUD']
