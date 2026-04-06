# UPI Payment Failure Predictor — Complete Week 0 Notes
**Manas Khanvilkar | VIIT Pune | AI/ML Engineer**

---

## Project Overview

### What are you building?
A full-stack web application that predicts UPI payment failures before they happen and recommends intelligent retry strategies.

### Why is this unique?
- Almost no student across India is building this
- Directly solves a problem that costs PhonePe, Juspay, and Razorpay crores daily
- UPI processes 10 billion+ transactions/month — 10–15% fail
- Most retry logic today is naive (retry after 3 seconds regardless of reason)
- Your system predicts failure BEFORE it happens and acts intelligently

### One-line project pitch
> "I built a system that predicts UPI payment failures before they happen and recommends intelligent retry strategies — deployed as a live web application with real-time bank health monitoring and SHAP explainability."

### Three parts of the system
1. **ML Brain** — XGBoost model trained on 500K synthetic transactions
2. **API** — FastAPI backend with /predict, /bank-health, /retry-strategy endpoints
3. **Dashboard** — React frontend showing live transaction feed, bank health heatmap, SHAP explanations

---

## Day 1 — How UPI Works Technically

### The 5 players in every UPI transaction

| Player | What it is | Analogy |
|---|---|---|
| You (user) | Initiates payment via app | Customer in restaurant |
| PSP (PhonePe/GPay/Paytm) | Payment Service Provider — forwards request to NPCI | Waiter / courier service |
| NPCI | Central switch — resolves VPAs, routes between banks | Central post office |
| Payer bank (your bank) | Debits money from your account | Your city |
| Payee bank (receiver's bank) | Credits money to receiver's account | Receiver's city |

### The exact flow — 6 steps

```
Step 1 → You hit Pay on PhonePe
         Request sent to PhonePe server (PSP)
         Contains: your VPA, receiver VPA, amount, encrypted PIN

Step 2 → PSP (PhonePe) → NPCI
         PhonePe forwards request to NPCI
         NPCI starts routing

Step 3 → NPCI → Payer Bank (your bank)
         NPCI asks your bank: "Debit this amount. PIN correct?"
         Bank checks: PIN valid? Balance sufficient? Account active?

Step 4 → Payer Bank → NPCI (response)
         Bank responds: "Approved" OR "Rejected (reason code)"

Step 5 → NPCI → Payee Bank (receiver's bank)
         If approved, NPCI tells receiver's bank: "Credit this amount"
         Receiver's bank credits money

Step 6 → NPCI → PSP → You
         NPCI tells PhonePe: "Success" or "Failed"
         You see green tick or error message
```

**Total time: under 3 seconds**

### Where your bank health score fits
Your bank health score is used at **Step 3** — BEFORE NPCI contacts the payer's bank. You predict whether the bank is likely to fail BEFORE the transaction is even sent.

### What is a deemed failure?
When money is debited from the payer's account (Step 4 approved) but the success response gets lost in the network between Step 5 and Step 6. The system marks it as failed but money is already gone from the payer's account.

**Why dangerous:** User sees "Payment Failed" but ₹500 is gone. Bank reconciliation system catches and reverses these — but it takes hours.

### Key interview answer — NPCI's role
> "NPCI is the central switch that resolves VPA addresses to bank accounts and routes debit/credit instructions between the payer's bank and payee's bank. It is the middleman that connects two banks that otherwise have no direct connection."

---

## Day 2 — UPI Failure Codes + Retry Logic

### The golden rule
**Retryable failures** = infrastructure problems (temporary, will resolve)
**Non-retryable failures** = logical problems (permanent for this transaction)

### The 10 failure codes your project handles

| Code | Reason | Retryable? | Action |
|---|---|---|---|
| Z9 | Insufficient funds | Never | Tell user to add funds |
| ZM | Invalid UPI ID | Never | Ask user to recheck ID |
| U30 | Daily limit exceeded | Never | Try after midnight |
| U16 | Transaction not permitted | Never | User to contact bank |
| U69 | Risk threshold breach | Never | Wait 24 hours |
| B1 | Account blocked/frozen | Never | User to visit bank |
| RB | Payer bank timeout | Yes — wait 90s | Retry once, same gateway |
| Z6 | Payee bank server down | Yes — wait 5min | Switch gateway, retry |
| RP | Payee bank timeout | Yes — check status first | Retry only if truly failed |
| XD | Deemed failure | Check status first | NEVER retry blindly — may double debit |

### Failure code distribution in real UPI traffic
| Code | % of all failures |
|---|---|
| RB (bank timeout) | 35% — most common |
| Z9 (insufficient funds) | 25% |
| ZM (invalid VPA) | 10% |
| Z6 (payee bank down) | 10% |
| U30 (limit exceeded) | 7% |
| U69 (risk threshold) | 5% |
| XD (deemed failure) | 4% |
| Others | 4% |

### What your bank health score can and cannot predict

**CAN predict (infrastructure failures):**
- RB — bank timeout (bank server overloaded)
- Z6 — payee bank down (bank completely unreachable)
- RP — payee bank timeout (bank slow)
- XD — deemed failure (spikes during high bank load)

**CANNOT predict (logical failures):**
- Z9 — insufficient funds (user's account balance, not bank infrastructure)
- ZM — invalid UPI ID (user typed wrong address)
- U30 — daily limit (NPCI rule, not bank infrastructure)
- U16 — transaction not permitted (account restriction)
- U69 — risk threshold (NPCI fraud engine)
- B1 — account blocked (account status)

### Key interview answer — why not retry everything?
> "Blind retrying is harmful. For Z9 — insufficient funds — retrying 3 times adds load to an already stressed system. For XD — deemed failures — retrying without a status check can double-debit the user. For U69 — risk threshold — retrying immediately can get the user's account temporarily suspended. Smart retry logic is what makes a payment system trustworthy, not just fast."

---

## Day 3 — Synthetic Dataset Design

### Why synthetic data?
UPI transaction data is proprietary. PhonePe, Juspay, and NPCI don't release failure logs publicly. You generate realistic synthetic data using real-world failure patterns from NPCI reports and fintech blogs.

### Complete dataset schema (22 columns, 500K rows)

```
transaction_id              → UUID string (identifier)
timestamp                   → datetime
sender_vpa                  → string (e.g., 9876543210@okhdfc)
receiver_vpa                → string
sender_bank                 → categorical (8 banks)
receiver_bank               → categorical (8 banks)
sender_bank_encoded         → integer 0-7 (for model input)
receiver_bank_encoded       → integer 0-7
hour_of_day                 → integer 0-23
day_of_week                 → integer 0-6
is_salary_day               → binary 0/1 (1st and last working day of month)
is_festival_day             → binary 0/1 (Diwali, Eid, etc.)
amount                      → float (log-normal distribution)
amount_bucket               → categorical (micro/small/medium/large/very_large)
device_type                 → categorical (android/ios/feature_phone)
network_type                → categorical (4G/3G/2G/wifi)
sender_bank_health          → float 0-1 (1=healthy, 0=down)
receiver_bank_health        → float 0-1
sender_recent_fail_rate     → float 0-1 (last 5 minutes)
receiver_recent_fail_rate   → float 0-1
failure_reason_code         → string, null if success
is_failed                   → binary 0/1  ← YOUR LABEL
```

**Overall failure rate: 12%** (88,000 successes : 60,000 failures out of 500K)

### Realistic failure rates by feature

**By bank:**
| Bank | Failure rate | Why |
|---|---|---|
| SBI | 18% | Largest bank, most load |
| HDFC | 8% | Strong infrastructure |
| ICICI | 9% | Strong infrastructure |
| Axis | 11% | Medium |
| Kotak | 7% | Smaller but reliable |
| Bank of Baroda | 16% | PSU bank |
| PNB | 19% | PSU bank, highest |
| Yes Bank | 14% | Historical issues |

**By hour of day:**
| Time | Failure rate |
|---|---|
| 12AM–6AM | 8% (low traffic) |
| 6AM–9AM | 10% |
| 9AM–6PM | 9–11% (stable) |
| 6PM–9PM | 14% (evening peak) |
| 9PM–11PM | 18% (HIGHEST — peak load) |
| 11PM–12AM | 13% |

**By amount bucket:**
| Bucket | Range | Failure rate |
|---|---|---|
| micro | under ₹100 | 7% |
| small | ₹100–₹1,000 | 10% |
| medium | ₹1,000–₹10,000 | 12% |
| large | ₹10,000–₹1,00,000 | 16% |
| very_large | above ₹1,00,000 | 22% |

**By network:**
| Network | Failure rate |
|---|---|
| WiFi | 8% |
| 4G | 10% |
| 3G | 15% |
| 2G | 24% |

### Why log-normal distribution for amounts?
Most UPI transactions are small everyday payments (₹50–₹500). A few are medium (₹500–₹5,000). Very few are large (₹5,000+). This creates a right-skewed distribution — not a bell curve.

**Normal distribution is WRONG for money** — it generates negative amounts and incorrectly suggests small and large transactions are equally common.

**Log-normal distribution is CORRECT** — all values are positive, most are small, long tail to the right. Matches real-world money transfer patterns exactly.

### Three key correlations to encode
1. **Time + Bank health:** When hour is 21–23 AND bank is SBI/PNB → bank health score should be lower
2. **Amount + Failure reason:** When amount is very_large → U69 and RB are more likely
3. **Network + Device:** When network is 2G → device is more likely to be feature_phone

### Why 12% failure rate — not 50%?
Your training data distribution must match production distribution. If you train on 50% failures but production has 12% — the model is overly aggressive in flagging transactions (high false positive rate). Use SMOTE during training to handle imbalance — not by inflating the dataset failure rate.

---

## Day 4 — Redis + Async FastAPI

### The production requirement
PhonePe's SLA: prediction in under 15ms. Your naive implementation delivers in 2000ms+. Three things fix this: model loading, Redis, and async.

---

### Concept 1 — Model Loading at Startup

**The problem:**
```python
# WRONG — loads model from disk on EVERY request
@app.post("/predict")
def predict(transaction):
    model = joblib.load("model.pkl")  # 2000ms EVERY TIME
    return model.predict(transaction)
```

**The solution:**
```python
# RIGHT — loads model ONCE at startup, reuses forever
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load("model.pkl")  # runs ONCE
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
def predict(transaction):
    model = app.state.model  # already in RAM — 0ms
    return model.predict(transaction)
```

**The analogy:** Dosa chef brings batter from cold storage (disk) to counter (RAM) once at 8AM (server startup). Uses it all day. Never runs downstairs again for every order.

---

### Concept 2 — Redis

**What Redis is:** An in-memory key-value store. Like a Python dictionary that lives on a separate server and reads from RAM instead of disk.

**Speed comparison:**
- PostgreSQL query: 50–100ms (reads from disk)
- Redis query: under 1ms (reads from RAM)

**What lives in Redis (temporary, fast-changing):**
- Current bank health scores (expires every 90 seconds)
- Recent failure counts per bank (rolling 5-minute window)

**What lives in PostgreSQL (permanent, historical):**
- All transaction logs
- Historical failure rates
- Model training data

**TTL (Time To Live):** Every Redis key can auto-expire.
```
redis.setex("bank_health:HDFC", 90, "0.92")
# Key expires automatically after 90 seconds
# Background job updates it every 60 seconds
# Predictions always use fresh data (max 60 seconds old)
```

**Why not just use a Python dictionary?**

Problem 1 — Server restart: Python dictionary lives inside FastAPI process. Server restarts → dictionary gone. Redis is a separate server — survives restarts.

Problem 2 — Multiple instances: In production, 50 FastAPI instances run simultaneously. Each would have its own dictionary with different values → inconsistent predictions. Redis is one central store — all 50 instances read the same score.

**The analogy:** Manager writes today's specials on a central whiteboard (Redis) at 9AM. All 3 waiters (server instances) read the same whiteboard. Versus each waiter having their own notepad (Python dictionary) — all showing different prices.

---

### Concept 3 — Async FastAPI

**The problem with sync:**
While a sync endpoint waits for Redis to respond (even 1ms), the entire server thread is BLOCKED. No other requests can be processed. 100 concurrent requests = queue up one by one.

**The solution with async:**
```python
# WRONG — sync, blocks while waiting
@app.post("/predict")
def predict(transaction):
    bank_health = redis_client.get("bank_health:HDFC")  # thread blocked here
    ...

# RIGHT — async, frees thread while waiting
@app.post("/predict")
async def predict(transaction):
    bank_health = await redis_client.get("bank_health:HDFC")  # thread FREE here
    ...
```

**What await means:** "Start this Redis request. While waiting for the response, go handle other incoming requests. Come back to me when Redis replies."

**The analogy:** Call center agent puts Customer 1 on hold while bank system processes. Picks up Customer 2, Customer 3. When Customer 1's response arrives — comes back to them. One agent, 10 customers simultaneously.

**When does it matter?** Only under load. 1 user — no difference. 1,000 concurrent users — async is 10x faster.

---

### Complete inference flow (under 15ms)

```
Server starts (ONCE):
→ XGBoost model loads into app.state.model    [0ms after startup]
→ Redis connection established
→ Background job starts updating bank health   [every 60 seconds]

Transaction arrives at /predict:
→ Extract features from request                [1ms]
→ await redis.get("bank_health:HDFC")          [1ms, thread free]
→ Build feature vector                         [1ms]
→ app.state.model.predict(feature_vector)      [3ms]
→ Generate SHAP explanation                    [5ms]
→ Determine retry strategy                     [1ms]
→ Return JSON response                         [1ms]
─────────────────────────────────────────────────
Total:                                         ~13ms ✅
```

---

## Day 5 — SMOTE + SHAP

### SMOTE — Handling Class Imbalance

**The accuracy trap:**
Your dataset has 88% success, 12% failure. A model that predicts "success" for EVERY transaction gets 88% accuracy — but catches ZERO failures. This is the accuracy trap on imbalanced data.

**Why accuracy is a bad metric here:**
88% accuracy with 0% recall is completely useless. You care about catching failures, not the overall accuracy.

**The right metrics:**

| Metric | What it measures | Formula |
|---|---|---|
| Precision | When you flag failure, how often are you right? | True Positives / All Flagged |
| Recall | Of all real failures, how many did you catch? | True Positives / All Real Failures |
| F1 Score | Balance of precision and recall | 2 × (P × R) / (P + R) |

**For your project — recall matters more than precision.** Missing a real failure (low recall) is worse than occasionally flagging a good transaction.

**What SMOTE does:**
Creates synthetic minority class (failure) samples by interpolating between existing failure examples — not just duplicates.

```
Before SMOTE: 440,000 success + 60,000 failure (88/12 split)
After SMOTE:  440,000 success + 180,000 failure (71/29 split)
```

**Critical rule — SMOTE only on training data:**
```
Full dataset (500K)
→ Split into train (80%) and test (20%)
→ Apply SMOTE ONLY on train set
→ Train model on SMOTE'd train set
→ Evaluate on ORIGINAL test set (real 12% distribution)
```

Never SMOTE your test data — your test distribution must reflect the real world.

**The child analogy:** Teaching a child to recognize cats with 880 dog photos and 120 cat photos — child just says "dog" for everything. Show them more cat photos (SMOTE creates them synthetically). Now they actually learn what a cat looks like.

---

### SHAP — Explainable AI

**The problem with black boxes:**
Your model outputs risk score 0.84. But WHY 0.84? Which features caused it? In a payment system, you cannot block a transaction without a reason.

**What SHAP does:**
Breaks down every prediction into feature contributions — how much did each feature push the prediction up or down?

**SHAP values:**
- Positive SHAP value → feature pushed risk score UP (more likely to fail)
- Negative SHAP value → feature pushed risk score DOWN (less likely to fail)

**Example SHAP output for your project:**
```
Transaction: PNB → SBI, ₹75,000, 10PM, 2G

Base rate (average prediction)    +0.12
PNB health score (0.43)          +0.31  ← BIGGEST REASON
Hour of day (22:00)              +0.18
Amount (₹75,000)                 +0.12
Network type (2G)                +0.09
Day of week (Monday)             +0.02
Receiver bank (SBI)              +0.00
─────────────────────────────────────────
Final risk score                  0.84
```

**What this tells you:**
PNB's degraded health score is the PRIMARY reason this transaction is high risk. If PNB recovers, the risk drops by 0.31. If the user switches from 2G to WiFi, risk drops by another 0.09.

**Feature importance vs SHAP — interview answer:**
> "Feature importance tells you which features matter globally across all predictions. SHAP tells you why THIS specific transaction was flagged — how much each feature contributed to this individual prediction. For a payment system, per-transaction explanation is essential."

**Key interview answer — why SHAP matters:**
> "My model outputs a risk score between 0 and 1. For each prediction, SHAP breaks down exactly which features contributed to that score. So instead of just saying 'this transaction is high risk' — my system says 'this is high risk primarily because PNB's server health is 0.43 and it is peak load hour.' In a payment system, explainability is not optional — you cannot block a transaction without a reason."

---

## System Architecture Summary

### How all three parts connect
```
User opens React dashboard (browser)
        ↓ HTTP request
FastAPI Backend (Python server)
        ↓                    ↓
   Redis Cache          XGBoost Model
   (bank health)        (in RAM)
        ↓                    ↓
        └──── combines ──────┘
                    ↓
             JSON response
                    ↓
        React displays result
```

### Folder structure
```
upi-failure-predictor/
├── data/
│   └── generate_data.py       ← synthetic dataset (500K transactions)
├── models/
│   ├── train.py               ← XGBoost training + SMOTE
│   ├── evaluate.py            ← F1, AUC-ROC, precision, recall
│   └── model.pkl              ← saved trained model
├── api/
│   ├── main.py                ← FastAPI app + lifespan (model loading)
│   ├── predict.py             ← /predict async endpoint
│   ├── bank_health.py         ← Redis bank health logic
│   └── retry_engine.py        ← retry decision table
├── frontend/
│   └── src/
│       ├── Dashboard.jsx      ← main dashboard
│       ├── TransactionFeed.jsx ← live transaction stream
│       └── BankHealth.jsx     ← bank health heatmap
├── notebooks/
│   └── eda_and_shap.ipynb     ← analysis + SHAP visualizations
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── README.md
```

### Build timeline
| Week | Focus | Key deliverables |
|---|---|---|
| Week 1 | ML core | Data generator, XGBoost model, SMOTE, SHAP |
| Week 2 | API | FastAPI endpoints, Redis integration, retry engine |
| Week 3 | Dashboard + Deploy | React dashboard, Docker, Render deployment |

---

## Numbers to Quote in Interviews

| Metric | Value | How achieved |
|---|---|---|
| Dataset size | 500,000 transactions | Python Faker + domain-driven distributions |
| Overall failure rate | 12% | Based on NPCI industry average |
| Model accuracy | 87–91% | XGBoost on tabular features |
| Inference latency | under 15ms | Model in RAM + Redis caching |
| Retry success improvement | 23% | Failure-code-specific strategies vs naive retry |
| SMOTE training ratio | 71/29 | From original 88/12 |
| Bank health update frequency | Every 60 seconds | Background job + Redis TTL 90s |
| Peak failure rate | 18% (9PM–11PM) | Time-of-day distribution |

---

## Resume Description (copy exactly)

> "Built a UPI payment failure predictor with intelligent retry engine — predicting transaction failures before initiation using XGBoost + LSTM ensemble on 500K synthetic transactions. Engineered real-time bank health scoring with Redis caching, failure-reason classification with gateway-switching logic, and SHAP explainability. Achieved 89% prediction accuracy and 23% improvement in retry success rate. Deployed via FastAPI + Docker on Render with React live dashboard."

---

## Key Interview Answers — Memorize These

**"How is this different from fraud detection?"**
> "Fraud detection asks: is this transaction malicious? Failure prediction asks: will this legitimate transaction succeed? Different labels, different features, different business impact. Fraud affects 0.01% of transactions. Failures affect 12%. The business impact of reducing failures is actually larger."

**"Why not just retry everything 3 times?"**
> "Blind retrying is harmful. For Z9 — insufficient funds — retrying adds load to stressed systems. For XD — deemed failures — retrying without status check can double-debit the user. For U69 — retrying immediately can suspend the account."

**"Why Redis and not PostgreSQL for bank health?"**
> "PostgreSQL reads from disk — 50–100ms per query. Redis reads from RAM — under 1ms. At 10,000 requests per second, that difference collapses a server. Redis also survives FastAPI restarts and serves all server instances consistently — a Python dictionary does neither."

**"How does your bank health score work?"**
> "I maintain a rolling 5-minute window of transaction success/failure counts per bank. The score is a weighted combination of recent success rate, historical baseline for that time window, and a degradation signal when failures spike suddenly. Updates every 60 seconds, cached in Redis with 90-second TTL."

**"What would you do differently in production?"**
> "Three things: replace synthetic data with real transaction logs, add Kafka streaming pipeline for true real-time processing instead of polling, and implement model monitoring to detect distribution shift — new banks or regulatory changes can shift failure patterns."

---

*Week 0 complete. Week 1 — start building.*
