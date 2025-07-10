import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import f1_score

# --- Load Data ---
df = pd.read_csv("data/transactions_training_sept_oct_2023.csv", sep=';', decimal=',')
df['DATETIME_GMT'] = pd.to_datetime(df['DATETIME_GMT'])
df['hour'] = df['DATETIME_GMT'].dt.hour
df['day_of_week'] = df['DATETIME_GMT'].dt.dayofweek

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
df['day_of_week'] = df['day_of_week'].astype('category')
df['hour'] = df['hour'].astype('category')
ordinal = OrdinalEncoder()
df[['day_of_week', 'hour']] = ordinal.fit_transform(df[['day_of_week', 'hour']])

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
    'hour', 'day_of_week'
]
best_feats = base_features + ['trx_count_card', 'rolling_mean_amt_card5', 'pop_anom_sum']

X = df[best_feats].copy()
y = df['FLAG_FRAUD']
scaler = StandardScaler()
X[X.columns.difference(['ID_CARD'])] = scaler.fit_transform(X[X.columns.difference(['ID_CARD'])])

# --- 5-Fold CV F1 for reporting ---
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1s = []
for train_idx, val_idx in kf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    f1 = f1_score(y_val, y_pred)
    f1s.append(f1)
print(f"Best Model 5-Fold F1 Scores: {np.round(f1s, 4)}")
print(f"Mean F1: {np.mean(f1s):.4f}, Std: {np.std(f1s):.4f}\n")

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

# --- Predict (preserve order) ---
y_pred_proba = model.predict_proba(X)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)
df_out = df.copy()
df_out['PREDICTED_FRAUD'] = y_pred
df_out = df_out.sort_index()  # Restore original order

# Save predictions
out_path = 'jamie/fraud_predictions.csv'
df_out[['ID_TRX', 'PREDICTED_FRAUD']].to_csv(out_path, index=False)
print(f"Predictions saved to {out_path}") 