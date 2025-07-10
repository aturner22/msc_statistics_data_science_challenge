import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix

# --- Load Data ---
df = pd.read_csv("data/old_transactions_training_sept_oct_2023.csv", sep=';', decimal=',')
df['DATETIME_GMT'] = pd.to_datetime(df['DATETIME_GMT'])
df['hour'] = df['DATETIME_GMT'].dt.hour

# --- Feature Engineering ---
df['time_since_last_transaction'] = df.groupby('ID_CARD')['DATETIME_GMT'].diff().dt.total_seconds()
df['diff_to_last_transaction'] = df.groupby('ID_CARD')['AMOUNT'].diff()
df['trx_count_card'] = df.groupby('ID_CARD').cumcount() + 1
df = df.sort_values(['ID_CARD', 'DATETIME_GMT'])
df['rolling_mean_amt_card5'] = (
    df.groupby('ID_CARD')['AMOUNT']
      .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
)
pop_anoms = [col for col in df.columns if 'Population_Anomaly_' in col]
df['pop_anom_sum'] = df[pop_anoms].sum(axis=1)
df['day_of_week'] = df['DATETIME_GMT'].dt.dayofweek.astype('category')
df['hour_cat'] = df['hour'].astype('category')
ordinal = OrdinalEncoder()
df[['day_of_week', 'hour_cat']] = ordinal.fit_transform(df[['day_of_week', 'hour_cat']])

# --- Best Feature Combo ---
base_features = [
    'ID_CARD', 'time_since_last_transaction', 'diff_to_last_transaction', 'AMOUNT',
    'Anomaly_amount_1', 'Anomaly_amount_2', 'Anomaly_amount_3', 'Anomaly_amount_4',
    'Anomaly_amount_5', 'Anomaly_amount_6', 'Anomaly_amount_7', 'Anomaly_amount_8',
    'FLAG_BEHAVIOUR_Anomaly_1', 'FLAG_BEHAVIOUR_Anomaly_2', 'FLAG_BEHAVIOUR_Anomaly_3',
    'FLAG_BEHAVIOUR_Anomaly_4', 'FLAG_BEHAVIOUR_Anomaly_5', 'FLAG_BEHAVIOUR_Anomaly_6',
    'FLAG_BEHAVIOUR_Anomaly7', 'FLAG_BEHAVIOUR_Anomaly_8', 'Anomaly_amount_9',
    'Population_Anomaly_1', 'Population_Anomaly_2', 'Population_Anomaly_3', 'Population_Anomaly_4',
    'Population_Anomaly_5', 'Population_Anomaly_6', 'Population_Anomaly_7', 'Population_Anomaly_8',
    'hour_cat', 'day_of_week'
]
best_feats = base_features + ['trx_count_card', 'rolling_mean_amt_card5', 'pop_anom_sum']

X = df[best_feats].copy()
y = df['FLAG_FRAUD']
scaler = StandardScaler()
X[X.columns.difference(['ID_CARD'])] = scaler.fit_transform(X[X.columns.difference(['ID_CARD'])])

# --- Train Model on All Data ---
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# --- Predict on Afternoon Subset Only (without refitting) ---
afternoon_mask = (df['hour'] >= 12) & (df['hour'] < 18)
X_afternoon = X[afternoon_mask]
y_afternoon = y[afternoon_mask]
y_pred_proba_afternoon = model.predict_proba(X_afternoon)[:, 1]
THRESHOLD = 0.65
y_pred_afternoon = (y_pred_proba_afternoon > THRESHOLD).astype(int)

print(f"\nEvaluation on AFTERNOON transactions (12 <= hour < 18) with threshold {THRESHOLD}:")
print(classification_report(y_afternoon, y_pred_afternoon, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_afternoon, y_pred_afternoon)) 


df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)



df['amount_zscore_card'] = (
    df.groupby('ID_CARD')['AMOUNT']
      .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
)