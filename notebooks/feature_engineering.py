#Cell 1 — Imports & Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../data/upi_transactions.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head(3)

#Cell 2 — Time-Based Features
# Peak hours: 9-11 AM and 7-10 PM (highest UPI traffic)
df['is_peak_hour'] = df['hour_of_day'].apply(
    lambda x: 1 if (9 <= x <= 11) or (19 <= x <= 22) else 0
)

# Weekend flag
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# High stress day — salary OR festival
df['is_high_stress_day'] = ((df['is_salary_day'] == 1) | (df['is_festival_day'] == 1)).astype(int)

print("Peak hour distribution:")
print(df['is_peak_hour'].value_counts())
print(f"\nPeak hour failure rate:     {df[df['is_peak_hour']==1]['is_failed'].mean()*100:.2f}%")
print(f"Non-peak hour failure rate: {df[df['is_peak_hour']==0]['is_failed'].mean()*100:.2f}%")
print(f"\nHigh stress day failure rate: {df[df['is_high_stress_day']==1]['is_failed'].mean()*100:.2f}%")
print(f"Normal day failure rate:      {df[df['is_high_stress_day']==0]['is_failed'].mean()*100:.2f}%")

#Cell 3 — Amount Features
# Log transform — reduces right skew of amount
df['log_amount'] = np.log1p(df['amount'])

# High value transaction flag
df['is_high_value'] = (df['amount'] > 10000).astype(int)

print("High value transaction distribution:")
print(df['is_high_value'].value_counts())
print(f"\nHigh value failure rate:  {df[df['is_high_value']==1]['is_failed'].mean()*100:.2f}%")
print(f"Normal value failure rate: {df[df['is_high_value']==0]['is_failed'].mean()*100:.2f}%")

# Verify log transform reduced skew
print(f"\nAmount skew (original): {df['amount'].skew():.3f}")
print(f"Amount skew (log):      {df['log_amount'].skew():.3f}")

#Cell 4 — Bank Health Interaction Features
# Difference in bank health — if sender is weak but receiver is strong, still risky
df['bank_health_diff'] = df['sender_bank_health'] - df['receiver_bank_health']

# Combined fail rate — average of both sides
df['combined_fail_rate'] = (df['sender_recent_fail_rate'] + df['receiver_recent_fail_rate']) / 2

# Weakest link — minimum health on either side
df['min_bank_health'] = df[['sender_bank_health', 'receiver_bank_health']].min(axis=1)

# Health × fail rate interaction — most powerful feature
df['health_x_failrate'] = df['sender_bank_health'] * df['sender_recent_fail_rate']

print("New interaction features created:")
print(df[['bank_health_diff', 'combined_fail_rate', 'min_bank_health', 'health_x_failrate']].describe().round(3))

#Cell 5 — Sender Risk Score
# Network type risk mapping (from EDA — 2G/3G had higher failure)
network_risk = {'2G': 1.0, '3G': 0.6, '4G': 0.2, 'WiFi': 0.0}
df['network_risk'] = df['network_type'].map(network_risk)

# Sender risk score — weighted combination
# Higher bank fail rate + lower health + risky network = high risk
df['sender_risk_score'] = (
    0.4 * df['sender_recent_fail_rate'] +
    0.3 * (1 - df['sender_bank_health']) +   # invert health: low health = high risk
    0.2 * df['network_risk'] +
    0.1 * df['is_high_value']
)

print("Sender risk score stats:")
print(df['sender_risk_score'].describe().round(3))
print(f"\nHigh risk (score > 0.5) failure rate: {df[df['sender_risk_score'] > 0.5]['is_failed'].mean()*100:.2f}%")
print(f"Low risk  (score < 0.2) failure rate: {df[df['sender_risk_score'] < 0.2]['is_failed'].mean()*100:.2f}%")

#Cell 6 — Encode Categorical Features
# One-hot encode network_type, device_type, amount_bucket
df = pd.get_dummies(df, columns=['network_type', 'device_type', 'amount_bucket'], drop_first=False)

print("Shape after encoding:", df.shape)
print("\nNew columns added:")
new_cols = [c for c in df.columns if any(x in c for x in ['network_type_', 'device_type_', 'amount_bucket_'])]
print(new_cols)

#Cell 7 — Drop Unnecessary Columns
drop_cols = [
    'transaction_id',   # just an ID
    'timestamp',        # already extracted hour, day
    'sender_vpa',       # high cardinality, no signal
    'receiver_vpa',     # high cardinality, no signal
    'sender_bank',      # already have sender_bank_encoded
    'receiver_bank',    # already have receiver_bank_encoded
    'hour_of_day',      # already have is_peak_hour
    'failure_reason_code'  # TARGET LEAKAGE — only exists when is_failed=1
]

df_model = df.drop(columns=drop_cols)

print(f"Shape before drop: {df.shape}")
print(f"Shape after drop:  {df_model.shape}")
print(f"\nFinal features ({df_model.shape[1]-1} features + 1 target):")
feature_cols = [c for c in df_model.columns if c != 'is_failed']
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2}. {col}")

#Cell 8 — Verify No Target Leakage
# Quick check — make sure no column is perfectly correlated with target
correlations = df_model.corr()['is_failed'].abs().sort_values(ascending=False)

print("Top 10 correlations with is_failed:")
print(correlations.head(11).round(3))  # 11 because is_failed itself is #1

# Red flag: if any feature has correlation > 0.95, it's likely leakage
leakage_check = correlations[correlations > 0.95].drop('is_failed')
if len(leakage_check) > 0:
    print(f"\n⚠️  POTENTIAL LEAKAGE DETECTED: {list(leakage_check.index)}")
else:
    print("\n✅ No target leakage detected")

#Cell 9 — Feature Distribution Plot (Top 8 Features)
top_features = correlations.drop('is_failed').head(8).index.tolist()

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()

for i, col in enumerate(top_features):
    failed = df_model[df_model['is_failed'] == 1][col].astype(float)
    success = df_model[df_model['is_failed'] == 0][col].astype(float)
    axes[i].hist(success.sample(min(5000, len(success)), random_state=42),
                 bins=30, alpha=0.6, color='#2ecc71', label='Success', density=True)
    axes[i].hist(failed.sample(min(5000, len(failed)), random_state=42),
                 bins=30, alpha=0.6, color='#e74c3c', label='Failure', density=True)
    axes[i].set_title(col, fontsize=9, fontweight='bold')
    axes[i].legend(fontsize=7)

plt.suptitle('Top 8 Features — Success vs Failure Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../notebooks/plots/09_top_features_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: 09_top_features_distribution.png")

#Cell 10 — Save Engineered Dataset
import os
os.makedirs('../models', exist_ok=True)

# Save engineered dataset
df_model.to_csv('../data/upi_features.csv', index=False)

print(f"✅ Saved: ../data/upi_features.csv")
print(f"   Shape: {df_model.shape}")
print(f"   Features: {df_model.shape[1] - 1}")
print(f"   Target: is_failed")
print(f"   Failure rate: {df_model['is_failed'].mean()*100:.2f}%")
print("\n🎯 Day 3 Complete! Ready for Day 4: XGBoost + SMOTE + SHAP")