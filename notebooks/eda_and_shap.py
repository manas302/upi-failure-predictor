#Cell 1 — Imports & Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
sns.set_palette('husl')

df = pd.read_csv('../data/upi_transactions.csv')
print(f"Shape: {df.shape}")
print(f"Failure rate: {df['is_failed'].mean()*100:.2f}%")
df.head(3)

#Cell 2 — Sanity Check
print("NULL VALUES:")
print(df.isnull().sum())

print(f"\nDUPLICATES: {df.duplicated().sum()}")

print("\nDTYPES:")
print(df.dtypes)

#Cell 3 — Parse Timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
print("Date range:", df['timestamp'].min(), "→", df['timestamp'].max())

print("\nTarget distribution:")
print(df['is_failed'].value_counts())
print(f"\nFailure rate: {df['is_failed'].mean()*100:.2f}%")

#Cell 4 — Class Imbalance + Hour + Day Charts
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Class balance
counts = df['is_failed'].value_counts()
axes[0].bar(['Success', 'Failure'], counts.values, color=['#2ecc71', '#e74c3c'], width=0.5)
axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 2000, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=10)
axes[0].set_ylabel('Count')

# Failure rate by hour
hourly = df.groupby('hour_of_day')['is_failed'].mean() * 100
axes[1].plot(hourly.index, hourly.values, marker='o', color='#e74c3c', linewidth=2)
axes[1].fill_between(hourly.index, hourly.values, alpha=0.15, color='#e74c3c')
axes[1].set_title('Failure Rate by Hour of Day', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Failure Rate (%)')
axes[1].set_xticks(range(0, 24, 2))

# Failure rate by day of week
day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
daily = df.groupby('day_of_week')['is_failed'].mean() * 100
daily.index = [day_map.get(d, d) for d in daily.index]
axes[2].bar(daily.index, daily.values, color='#3498db')
axes[2].set_title('Failure Rate by Day of Week', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Day')
axes[2].set_ylabel('Failure Rate (%)')

plt.tight_layout()
plt.savefig('../notebooks/plots/01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: 01_target_distribution.png")

#Cell 5 — Special Days Analysis
salary_fail = df[df['is_salary_day'] == 1]['is_failed'].mean() * 100
festival_fail = df[df['is_festival_day'] == 1]['is_failed'].mean() * 100
normal_fail = df[(df['is_salary_day'] == 0) & (df['is_festival_day'] == 0)]['is_failed'].mean() * 100

print(f"Failure rate on SALARY DAY:   {salary_fail:.2f}%")
print(f"Failure rate on FESTIVAL DAY: {festival_fail:.2f}%")
print(f"Failure rate on NORMAL DAY:   {normal_fail:.2f}%")

#Cell 6 — Numerical Features: Distribution + Boxplot
num_cols = ['amount', 'sender_bank_health', 'receiver_bank_health',
            'sender_recent_fail_rate', 'receiver_recent_fail_rate']

fig, axes = plt.subplots(2, len(num_cols), figsize=(22, 8))

for i, col in enumerate(num_cols):
    failed = df[df['is_failed'] == 1][col]
    success = df[df['is_failed'] == 0][col]

    # Top row: histogram
    axes[0][i].hist(success.sample(min(10000, len(success)), random_state=42),
                    bins=40, alpha=0.6, color='#2ecc71', label='Success', density=True)
    axes[0][i].hist(failed.sample(min(10000, len(failed)), random_state=42),
                    bins=40, alpha=0.6, color='#e74c3c', label='Failure', density=True)
    axes[0][i].set_title(col, fontsize=10, fontweight='bold')
    axes[0][i].legend(fontsize=8)

    # Bottom row: boxplot
    bp_data = [success.sample(min(5000, len(success)), random_state=42),
               failed.sample(min(5000, len(failed)), random_state=42)]
    bp = axes[1][i].boxplot(bp_data, patch_artist=True, labels=['Success', 'Failure'])
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for box in bp['boxes']:
        box.set_alpha(0.7)

plt.suptitle('Numerical Features: Success vs Failure', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../notebooks/plots/02_numerical_features.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: 02_numerical_features.png")

#Cell 7 — Amount Bucket Analysis
bucket_fail = df.groupby('amount_bucket')['is_failed'].mean().sort_values(ascending=False) * 100

plt.figure(figsize=(10, 4))
bucket_fail.plot(kind='bar', color='#e67e22', edgecolor='white')
plt.title('Failure Rate by Amount Bucket', fontsize=13, fontweight='bold')
plt.xlabel('Amount Bucket')
plt.ylabel('Failure Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../notebooks/plots/03_amount_bucket.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: 03_amount_bucket.png")

#Cell 8 — Categorical Features: Bank, Network, Device
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Sender bank
bank_fail = df.groupby('sender_bank')['is_failed'].mean().sort_values(ascending=False) * 100
bank_fail.plot(kind='bar', ax=axes[0], color='#9b59b6', edgecolor='white')
axes[0].set_title('Failure Rate by Sender Bank', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Failure Rate (%)')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
axes[0].axhline(df['is_failed'].mean() * 100, color='red', linestyle='--', alpha=0.6, label='Avg')
axes[0].legend()

# Network type
net_fail = df.groupby('network_type')['is_failed'].mean().sort_values(ascending=False) * 100
net_fail.plot(kind='bar', ax=axes[1], color='#1abc9c', edgecolor='white')
axes[1].set_title('Failure Rate by Network Type', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Failure Rate (%)')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
axes[1].axhline(df['is_failed'].mean() * 100, color='red', linestyle='--', alpha=0.6)

# Device type
dev_fail = df.groupby('device_type')['is_failed'].mean().sort_values(ascending=False) * 100
dev_fail.plot(kind='bar', ax=axes[2], color='#3498db', edgecolor='white')
axes[2].set_title('Failure Rate by Device Type', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Failure Rate (%)')
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')
axes[2].axhline(df['is_failed'].mean() * 100, color='red', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('../notebooks/plots/04_categorical_features.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: 04_categorical_features.png")

#Cell 9 — Failure Reason Code Distribution
failed_df = df[df['is_failed'] == 1]
reason_counts = failed_df['failure_reason_code'].value_counts()

plt.figure(figsize=(10, 4))
reason_counts.plot(kind='bar', color='#e74c3c', edgecolor='white')
plt.title('Failure Reason Code Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Failure Reason Code')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../notebooks/plots/05_failure_reason_codes.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTop failure reasons (%):")
print((reason_counts / len(failed_df) * 100).round(2))

#Cell 10 — Bank × Hour Heatmap
df['hour_bucket'] = pd.cut(
    df['hour_of_day'],
    bins=[0, 6, 12, 18, 24],
    labels=['Night(0-6)', 'Morning(6-12)', 'Afternoon(12-18)', 'Evening(18-24)'],
    right=False
)

heatmap_data = df.groupby(['sender_bank', 'hour_bucket'])['is_failed'].mean() * 100
heatmap_data = heatmap_data.unstack()

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'label': 'Failure Rate (%)'})
plt.title('Failure Rate (%) by Bank × Time of Day', fontsize=13, fontweight='bold')
plt.xlabel('Time of Day')
plt.ylabel('Sender Bank')
plt.tight_layout()
plt.savefig('../notebooks/plots/06_bank_hour_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: 06_bank_hour_heatmap.png")

#Cell 11 - Correlation Matrix
corr_cols = ['amount', 'hour_of_day', 'day_of_week',
             'is_salary_day', 'is_festival_day',
             'sender_bank_health', 'receiver_bank_health',
             'sender_recent_fail_rate', 'receiver_recent_fail_rate',
             'sender_bank_encoded', 'receiver_bank_encoded',
             'is_failed']

corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../notebooks/plots/07_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: 07_correlation_matrix.png")

#Cell 12 — Top Features Correlated with Target
target_corr = corr_matrix['is_failed'].drop('is_failed').abs().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
target_corr.plot(kind='barh', color='#2c3e50')
plt.title('Feature Correlation with is_failed (absolute)', fontsize=13, fontweight='bold')
plt.xlabel('|Correlation|')
plt.axvline(0.1, color='red', linestyle='--', alpha=0.5, label='0.1 threshold')
plt.legend()
plt.tight_layout()
plt.savefig('../notebooks/plots/08_target_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTop predictive features:")
print(target_corr.head(8).round(3))

#Cell 13 - EDA Summary
print("""
╔══════════════════════════════════════════════════════════════╗
║           EDA SUMMARY — KEY INSIGHTS FOR INTERVIEWS         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. CLASS IMBALANCE: ~10% failure rate → SMOTE needed        ║
║     before training, else model predicts all success         ║
║                                                              ║
║  2. TEMPORAL PATTERN: Failure spikes at peak hours           ║
║     (salary day / festival day = higher stress on banks)     ║
║                                                              ║
║  3. BANK HEALTH: sender/receiver_bank_health strongly        ║
║     correlates with failure — top engineered feature         ║
║                                                              ║
║  4. RECENT FAIL RATE: sender/receiver_recent_fail_rate       ║
║     captures rolling behavioral signal — key for model       ║
║                                                              ║
║  5. AMOUNT BUCKET: High-value transactions fail more         ║
║     — banks throttle large transfers under load              ║
║                                                              ║
║  6. NETWORK TYPE: 2G/3G users see significantly higher       ║
║     failure rates than 4G/WiFi users                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
print("✅ EDA Complete! Ready for Day 3: Feature Engineering")