#cell 1
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import uuid

#cell 2
fake = Faker('en_IN')
np.random.seed(42)
random.seed(42)

NUM_TRANSACTIONS = 500000

BANKS = {
    'SBI':  0.18,
    'HDFC': 0.08,
    'ICICI': 0.09,
    'Axis': 0.11,
    'Kotak': 0.07,
    'BOB': 0.16,
    'PNB': 0.19,
    'YesBank': 0.14
}

BANK_NAMES = list(BANKS.keys())
BANK_FAILURE_RATES = list(BANKS.values())

DEVICE_TYPES = ['android', 'ios', 'feature_phone']
DEVICE_WEIGHTS = [0.70, 0.20, 0.10]

NETWORK_TYPES = ['4G', '3G', '2G', 'wifi']
NETWORK_WEIGHTS = [0.50, 0.20, 0.10, 0.20]

NETWORK_FAILURE_RATES = {
    'wifi': 0.08,
    '4G':   0.10,
    '3G':   0.15,
    '2G':   0.24
}

HOUR_FAILURE_RATES = {
    0: 0.08, 1: 0.08, 2: 0.08, 3: 0.08, 4: 0.08, 5: 0.08,
    6: 0.10, 7: 0.10, 8: 0.10,
    9: 0.09, 10: 0.09, 11: 0.09,
    12: 0.09, 13: 0.09, 14: 0.09,
    15: 0.11, 16: 0.11, 17: 0.11,
    18: 0.14, 19: 0.14, 20: 0.14,
    21: 0.18, 22: 0.18, 23: 0.13
}

FAILURE_REASON_DISTRIBUTION = {
    'RB':  0.35,
    'Z9':  0.25,
    'ZM':  0.10,
    'Z6':  0.10,
    'U30': 0.07,
    'U69': 0.05,
    'XD':  0.04,
    'B1':  0.02,
    'U16': 0.01,
    'RP':  0.01
}

#cell 3
def get_amount():
    amount = np.random.lognormal(mean=6.5, sigma=1.2)
    amount = round(min(max(amount, 1), 100000), 2)
    return amount

def get_amount_bucket(amount):
    if amount < 100:
        return 'micro'
    elif amount < 1000:
        return 'small'
    elif amount < 10000:
        return 'medium'
    elif amount < 100000:
        return 'large'
    else:
        return 'very_large'

def get_amount_failure_boost(amount_bucket):
    boosts = {
        'micro':      -0.03,
        'small':       0.00,
        'medium':      0.02,
        'large':       0.06,
        'very_large':  0.12
    }
    return boosts[amount_bucket]

def get_bank_health(bank, hour, is_salary_day):
    base_failure = BANKS[bank]

    # Add time-based stress
    if 19 <= hour <= 22:
        base_failure += 0.12
    elif 9 <= hour <= 11:
        base_failure += 0.06
    if is_salary_day:
        base_failure += 0.08

    # Add realistic random variation — this is the key change
    base_failure += np.random.normal(0, 0.05)

    health = 1.0 - base_failure
    health = round(float(np.clip(health, 0.20, 0.95)), 2)
    return health

def get_failure_reason():
    codes = list(FAILURE_REASON_DISTRIBUTION.keys())
    weights = list(FAILURE_REASON_DISTRIBUTION.values())
    return random.choices(codes, weights=weights, k=1)[0]

def get_upi_id(bank):
    phone = ''.join([str(random.randint(0, 9)) for _ in range(10)])
    suffixes = {
        'SBI':      'oksbi',
        'HDFC':     'okhdfc',
        'ICICI':    'okicici',
        'Axis':     'okaxis',
        'Kotak':    'kotak',
        'BOB':      'okbob',
        'PNB':      'okpnb',
        'YesBank':  'yesbank'
    }
    return f"{phone}@{suffixes[bank]}"

#cell 4
def generate_transaction():
    
    # --- timestamp ---
    days_ago = random.randint(0, 180)
    hour = random.choices(
        list(HOUR_FAILURE_RATES.keys()),
        weights=[1.5 if h in range(18, 24) else 1.0 
                 for h in HOUR_FAILURE_RATES.keys()],
        k=1
    )[0]
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    timestamp = datetime.now() - timedelta(days=days_ago)
    timestamp = timestamp.replace(hour=hour, minute=minute, second=second)

    # --- date features ---
    day_of_week = timestamp.weekday()
    day_of_month = timestamp.day
    is_salary_day = 1 if day_of_month in [1, 2, 30, 31] else 0
    is_festival_day = 1 if (timestamp.month == 11 and 
                             timestamp.day in range(1, 6)) else 0

    # --- banks ---
    sender_bank = random.choices(BANK_NAMES, 
                                  weights=BANK_FAILURE_RATES, k=1)[0]
    receiver_bank = random.choices(BANK_NAMES, k=1)[0]

    # --- device and network ---
    device_type = random.choices(DEVICE_TYPES, 
                                  weights=DEVICE_WEIGHTS, k=1)[0]
    
    if device_type == 'feature_phone':
        network_type = random.choices(
            ['2G', '3G'], weights=[0.7, 0.3], k=1)[0]
    else:
        network_type = random.choices(
            NETWORK_TYPES, weights=NETWORK_WEIGHTS, k=1)[0]

    # --- amount ---
    amount = get_amount()
    amount_bucket = get_amount_bucket(amount)

    # --- bank health ---
    sender_bank_health = get_bank_health(
        sender_bank, hour, is_salary_day)
    receiver_bank_health = get_bank_health(
        receiver_bank, hour, is_salary_day)
    
    # sender recent fail rate — independent signal, higher for weak banks
    sender_base_fail = BANKS[sender_bank]
    sender_recent_fail_rate = round(
        np.clip(np.random.beta(
            a=max(sender_base_fail * 10, 0.5),
            b=max((1 - sender_base_fail) * 10, 0.5)
        ) + random.uniform(-0.03, 0.03), 0, 1), 2
    )

    # receiver recent fail rate — same logic
    receiver_base_fail = BANKS[receiver_bank]
    receiver_recent_fail_rate = round(
        np.clip(np.random.beta(
            a=max(receiver_base_fail * 10, 0.5),
            b=max((1 - receiver_base_fail) * 10, 0.5)
        ) + random.uniform(-0.03, 0.03), 0, 1), 2
    )

    # --- failure probability (stronger signal version) ---
    failure_prob = 0.04  # base rate

    # Bank health — strongest driver (range: 0 to 0.50)
    failure_prob += (1 - sender_bank_health) * 0.50

    # Recent fail rate — second strongest (range: 0 to 0.35)
    failure_prob += sender_recent_fail_rate * 0.35

    # Network type — clear hierarchy
    network_boost = {'2G': 0.18, '3G': 0.10, '4G': 0.02, 'wifi': 0.00}
    failure_prob += network_boost[network_type]

    # High stress days
    if is_festival_day == 1:
        failure_prob += 0.15
    if is_salary_day == 1:
        failure_prob += 0.10

    # Peak hours
    if 9 <= hour <= 11 or 19 <= hour <= 22:
        failure_prob += 0.08

    # High value transactions
    if amount > 10000:
        failure_prob += 0.07

    # Feature phone
    if device_type == 'feature_phone':
        failure_prob += 0.05

    failure_prob = min(max(failure_prob, 0), 0.95)

    # --- determine outcome ---
    is_failed = 1 if random.random() < failure_prob else 0
    failure_reason_code = get_failure_reason() if is_failed else None

    # --- build transaction ---
    return {
        'transaction_id':           str(uuid.uuid4()),
        'timestamp':                timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'sender_vpa':               get_upi_id(sender_bank),
        'receiver_vpa':             get_upi_id(receiver_bank),
        'sender_bank':              sender_bank,
        'receiver_bank':            receiver_bank,
        'sender_bank_encoded':      BANK_NAMES.index(sender_bank),
        'receiver_bank_encoded':    BANK_NAMES.index(receiver_bank),
        'hour_of_day':              hour,
        'day_of_week':              day_of_week,
        'is_salary_day':            is_salary_day,
        'is_festival_day':          is_festival_day,
        'amount':                   amount,
        'amount_bucket':            amount_bucket,
        'device_type':              device_type,
        'network_type':             network_type,
        'sender_bank_health':       sender_bank_health,
        'receiver_bank_health':     receiver_bank_health,
        'sender_recent_fail_rate':  sender_recent_fail_rate,
        'receiver_recent_fail_rate':receiver_recent_fail_rate,
        'failure_reason_code':      failure_reason_code,
        'is_failed':                is_failed
    }

#cell 5
def generate_dataset(num_transactions=NUM_TRANSACTIONS):
    
    print(f"Generating {num_transactions:,} transactions...")
    
    transactions = []
    
    for i in range(num_transactions):
        transactions.append(generate_transaction())
        
        if (i + 1) % 50000 == 0:
            print(f"  {i + 1:,} transactions generated...")
    
    print("Creating DataFrame...")
    df = pd.DataFrame(transactions)
    
    print("Saving to CSV...")
    df.to_csv('upi_transactions.csv', index=False)
    
    print(f"\nDone! Dataset saved as 'upi_transactions.csv'")
    return df

#cell 6
df = generate_dataset()

print("\n--- Dataset Summary ---")
print(f"Total transactions:  {len(df):,}")
print(f"Total columns:       {len(df.columns)}")
print(f"\nFailure rate:        {df['is_failed'].mean()*100:.1f}%")
print(f"Total failures:      {df['is_failed'].sum():,}")
print(f"Total successes:     {(df['is_failed']==0).sum():,}")

print("\n--- Failure rate by bank ---")
bank_stats = df.groupby('sender_bank')['is_failed'].mean()*100
print(bank_stats.round(1).sort_values(ascending=False))

print("\n--- Failure rate by network ---")
network_stats = df.groupby('network_type')['is_failed'].mean()*100
print(network_stats.round(1).sort_values(ascending=False))

print("\n--- Failure rate by hour (peak hours) ---")
hour_stats = df.groupby('hour_of_day')['is_failed'].mean()*100
print(hour_stats.round(1).sort_values(ascending=False).head(5))

print("\n--- Amount distribution ---")
print(df['amount'].describe().round(2))

print("\n--- Failure reason distribution ---")
print(df[df['is_failed']==1]['failure_reason_code'].value_counts())

print("\n--- First 3 rows ---")
df.head(3)
    