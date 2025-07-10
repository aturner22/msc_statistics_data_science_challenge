import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading & Preprocessing ---
df = pd.read_csv("data/old_transactions_training_sept_oct_2023.csv", sep=';', decimal=',')
df['DATETIME_GMT'] = pd.to_datetime(df['DATETIME_GMT'])
df['hour'] = df['DATETIME_GMT'].dt.hour
df['day_of_week'] = df['DATETIME_GMT'].dt.dayofweek
df = df.sort_values(by=['ID_CARD', 'DATETIME_GMT'])
df['time_since_last_transaction'] = df.groupby('ID_CARD')['DATETIME_GMT'].diff().dt.total_seconds()
df['diff_to_last_transaction'] = df.groupby('ID_CARD')['AMOUNT'].diff()
num_cols = ['AMOUNT', 'time_since_last_transaction', 'diff_to_last_transaction',
            'Anomaly_amount_1', 'Anomaly_amount_2', 'Anomaly_amount_3', 'Anomaly_amount_4', 'Anomaly_amount_5',
            'Anomaly_amount_6', 'Anomaly_amount_7', 'Anomaly_amount_8', 'Anomaly_amount_9',
            'Population_Anomaly_1', 'Population_Anomaly_2', 'Population_Anomaly_3', 'Population_Anomaly_4',
            'Population_Anomaly_5', 'Population_Anomaly_6', 'Population_Anomaly_7', 'Population_Anomaly_8']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['day_of_week'] = df['day_of_week'].astype('category')
df['hour'] = df['hour'].astype('category')
ordinal = OrdinalEncoder()
df[['day_of_week', 'hour']] = ordinal.fit_transform(df[['day_of_week', 'hour']])
selected_cols = [
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
X = df[selected_cols]
y = df['FLAG_FRAUD']
scaler = StandardScaler()
X[X.columns.difference(['ID_CARD'])] = scaler.fit_transform(X[X.columns.difference(['ID_CARD'])])

# --- Helper: Train/Eval Model ---
def train_eval(X_train, y_train, X_val, y_val):
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
    best_f1, best_threshold = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_pred_proba > t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return best_f1, best_threshold

# --- 1. F1 vs. Test Size (Random Split) ---
test_sizes = [0.1, 0.2, 0.3, 0.4]
f1_random = []
for ts in test_sizes:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=ts, stratify=y, random_state=42)
    f1, _ = train_eval(X_train, y_train, X_val, y_val)
    f1_random.append(f1)

plt.figure(figsize=(6,4))
plt.plot([int(ts*100) for ts in test_sizes], f1_random, marker='o', label='Random Split')
plt.xlabel('Test Size (%)')
plt.ylabel('Best F1 Score')
plt.title('F1 vs. Test Size (Random Split)')
plt.grid(True)
plt.tight_layout()
plt.savefig('jamie/f1_vs_testsize_random.png')
plt.show()

# --- 2. Random vs. Time Split (20% test) ---
# Random split
X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
f1_random20, _ = train_eval(X_train_r, y_train_r, X_val_r, y_val_r)
# Time split
df_sorted = df.sort_values('DATETIME_GMT')
split_idx = int(len(df_sorted) * 0.8)
train_df = df_sorted.iloc[:split_idx]
test_df = df_sorted.iloc[split_idx:]
X_train_t = train_df[selected_cols]
y_train_t = train_df['FLAG_FRAUD']
X_val_t = test_df[selected_cols]
y_val_t = test_df['FLAG_FRAUD']
scaler2 = StandardScaler()
X_train_t[X_train_t.columns.difference(['ID_CARD'])] = scaler2.fit_transform(X_train_t[X_train_t.columns.difference(['ID_CARD'])])
X_val_t[X_val_t.columns.difference(['ID_CARD'])] = scaler2.transform(X_val_t[X_val_t.columns.difference(['ID_CARD'])])
f1_time20, _ = train_eval(X_train_t, y_train_t, X_val_t, y_val_t)

plt.figure(figsize=(5,4))
plt.bar(['Random Split', 'Time Split'], [f1_random20, f1_time20], color=['#4c72b0', '#dd8452'])
plt.ylabel('Best F1 Score')
plt.title('Random vs. Time Split (20% test)')
plt.tight_layout()
plt.savefig('jamie/f1_random_vs_time.png')
plt.show()

# --- 3. K-Fold Cross-Validation ---
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1s = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    f1, _ = train_eval(X_train, y_train, X_val, y_val)
    f1s.append(f1)
    print(f"Fold {fold+1}: F1 = {f1:.4f}")
print(f"Mean F1: {np.mean(f1s):.4f}, Std: {np.std(f1s):.4f}")

plt.figure(figsize=(7,4))
sns.barplot(x=[f'Fold {i+1}' for i in range(5)], y=f1s)
plt.axhline(np.mean(f1s), color='red', linestyle='--', label='Mean F1')
plt.ylabel('Best F1 Score')
plt.title('K-Fold Cross-Validation F1 Scores')
plt.legend()
plt.tight_layout()
plt.savefig('jamie/f1_kfold.png')
plt.show()

# --- Summary Table ---
print("\nSummary:")
print("F1 vs. Test Size (Random):", dict(zip([int(ts*100) for ts in test_sizes], np.round(f1_random, 4))))
print(f"Random Split (20%): F1 = {f1_random20:.4f}")
print(f"Time Split (20%): F1 = {f1_time20:.4f}")
print(f"K-Fold Mean F1: {np.mean(f1s):.4f} (Std: {np.std(f1s):.4f})")

# --- Feature Engineering: New Features ---
# 1. Transaction count per card (historical)
df['trx_count_card'] = df.groupby('ID_CARD').cumcount() + 1

# 2. Rolling mean of amount per card (last 5 transactions)
df = df.sort_values(['ID_CARD', 'DATETIME_GMT'])
df['rolling_mean_amt_card5'] = (
    df.groupby('ID_CARD')['AMOUNT']
      .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
)

# 3. Z-score of amount per card
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)
df['zscore_amt_card'] = df.groupby('ID_CARD')['AMOUNT'].transform(zscore)

# 4. Sum of anomaly flags
anomaly_flags = [col for col in df.columns if 'FLAG_BEHAVIOUR_Anomaly' in col]
df['anomaly_flag_sum'] = df[anomaly_flags].sum(axis=1)

# 5. Sum of population anomaly scores
pop_anoms = [col for col in df.columns if 'Population_Anomaly_' in col]
df['pop_anom_sum'] = df[pop_anoms].sum(axis=1)

# --- Feature Sets to Test ---
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
new_feats = ['trx_count_card', 'rolling_mean_amt_card5', 'zscore_amt_card', 'anomaly_flag_sum', 'pop_anom_sum']

from itertools import combinations

# All single, 2-way, and 3-way combinations of new features
to_test = []
for k in range(1, 4):
    to_test.extend(list(combinations(new_feats, k)))

results = []

for combo in to_test:
    feats = base_features + list(combo)
    X = df[feats].copy()
    scaler = StandardScaler()
    X[X.columns.difference(['ID_CARD'])] = scaler.fit_transform(X[X.columns.difference(['ID_CARD'])])
    y = df['FLAG_FRAUD']
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        f1, _ = train_eval(X_train, y_train, X_val, y_val)
        f1s.append(f1)
    results.append({
        'features': combo,
        'mean_f1': np.mean(f1s),
        'std_f1': np.std(f1s)
    })
    print(f"Features: {combo}, Mean F1: {np.mean(f1s):.4f}, Std: {np.std(f1s):.4f}")

# Plot results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('mean_f1', ascending=False)
plt.figure(figsize=(10,6))
plt.barh(
    [', '.join(r) for r in results_df['features']],
    results_df['mean_f1'],
    xerr=results_df['std_f1'],
    color='skyblue'
)
plt.xlabel('Mean F1 (5-fold CV)')
plt.title('Feature Combinations: K-Fold F1 Scores')
plt.tight_layout()
plt.savefig('jamie/feature_combo_kfold_f1.png')
plt.show()

print("\nTop Feature Combinations by Mean F1:")
print(results_df[['features', 'mean_f1', 'std_f1']].head(10)) 