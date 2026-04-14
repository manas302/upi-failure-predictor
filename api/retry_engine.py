# api/retry_engine.py
"""
Predicts the likely NPCI failure code for a transaction
and returns an intelligent retry strategy.

NPCI failure codes reference:
- U30: Request timeout
- U09: Remitter bank timeout
- U16: Risk threshold exceeded
- U68: Transaction not permitted
- BT:  Bank server busy
- XB:  Payer account blocked
- Z9:  Insufficient funds
"""

# ── NPCI Failure Code Definitions ────────────────────────────────
FAILURE_CODES = {
    "U30": {
        "description": "Request timeout — transaction did not complete in time",
        "cause": "High network latency or bank server slowness",
        "retry_allowed": True,
    },
    "U09": {
        "description": "Remitter bank timeout — sender bank did not respond",
        "cause": "Sender bank server under heavy load",
        "retry_allowed": True,
    },
    "BT": {
        "description": "Bank server busy — bank temporarily overloaded",
        "cause": "Peak traffic or salary day load",
        "retry_allowed": True,
    },
    "U16": {
        "description": "Risk threshold exceeded — transaction flagged as risky",
        "cause": "High transaction amount, new account, or unusual pattern",
        "retry_allowed": False,
    },
    "U68": {
        "description": "Transaction not permitted by bank",
        "cause": "Bank-level restriction on transaction type or amount",
        "retry_allowed": False,
    },
    "Z9": {
        "description": "Insufficient funds in sender account",
        "cause": "Low balance",
        "retry_allowed": False,
    },
    "XB": {
        "description": "Payer account blocked or frozen",
        "cause": "Account suspended due to suspicious activity",
        "retry_allowed": False,
    },
}

# ── Retry Strategies ──────────────────────────────────────────────
RETRY_STRATEGIES = {
    "U30": {
        "should_retry": True,
        "max_attempts": 3,
        "wait_seconds": [5, 15, 30],        # exponential-style backoff
        "suggestion": "Retry after 5 seconds. If it fails again, wait 15 seconds before the next attempt.",
        "user_message": "Transaction timed out. Please wait a moment and try again.",
    },
    "U09": {
        "should_retry": True,
        "max_attempts": 2,
        "wait_seconds": [30, 120],
        "suggestion": "Sender bank is under load. Retry after 30 seconds or try during off-peak hours.",
        "user_message": "Your bank is experiencing high traffic. Please try again in 30 seconds.",
    },
    "BT": {
        "should_retry": True,
        "max_attempts": 3,
        "wait_seconds": [10, 30, 60],
        "suggestion": "Bank server busy. Retry with short delays. Avoid peak hours (8-11 AM, 7-10 PM).",
        "user_message": "Bank server is busy. Retrying automatically in 10 seconds.",
    },
    "U16": {
        "should_retry": False,
        "max_attempts": 0,
        "wait_seconds": [],
        "suggestion": "Transaction flagged as high risk. Do not retry — contact your bank.",
        "user_message": "Transaction declined for security reasons. Please contact your bank.",
    },
    "U68": {
        "should_retry": False,
        "max_attempts": 0,
        "wait_seconds": [],
        "suggestion": "Transaction not permitted. Check UPI daily limit or bank restrictions.",
        "user_message": "This transaction is not permitted. Please check your UPI limits.",
    },
    "Z9": {
        "should_retry": False,
        "max_attempts": 0,
        "wait_seconds": [],
        "suggestion": "Insufficient funds. Add money to your account before retrying.",
        "user_message": "Insufficient balance. Please add funds and try again.",
    },
    "XB": {
        "should_retry": False,
        "max_attempts": 0,
        "wait_seconds": [],
        "suggestion": "Account blocked. Contact your bank immediately.",
        "user_message": "Your account has been blocked. Please contact your bank.",
    },
}


# ── Failure Code Prediction ───────────────────────────────────────

def predict_likely_failure_code(risk_score: float, features: dict) -> str:
    """
    Predicts the most likely NPCI failure code based on
    risk score and transaction features.

    Logic mirrors real NPCI failure patterns:
    - High latency → timeout codes (U30, U09)
    - High risk score + high amount → risk block (U16)
    - Peak hour + salary day → bank busy (BT)
    - Very high sender fail rate → account issues (XB)
    """
    network_risk = features.get("network_risk", 0.2)
    sender_health = features.get("sender_bank_health", 0.8)
    receiver_health = features.get("receiver_bank_health", 0.8)
    is_high_value = features.get("is_high_value", 0)
    is_peak_hour = features.get("is_peak_hour", 0)
    is_salary_day = features.get("is_salary_day", 0)
    sender_fail_rate = features.get("sender_recent_fail_rate", 0.1)

    # Rule 1: Very high sender failure rate → account likely blocked
    if sender_fail_rate > 0.7:
        return "XB"

    # Rule 2: High risk + high value → risk threshold exceeded
    if risk_score >= 0.75 and is_high_value:
        return "U16"

    # Rule 3: High network risk → request timeout
    if network_risk >= 0.6:
        return "U30"

    # Rule 4: Sender bank unhealthy → remitter bank timeout
    if sender_health < 0.65:
        return "U09"

    # Rule 5: Peak hour + salary day → bank busy
    if is_peak_hour and is_salary_day:
        return "BT"

    # Rule 6: Receiver bank unhealthy → general timeout
    if receiver_health < 0.65:
        return "U30"

    # Rule 7: Moderate risk during peak → bank busy
    if risk_score >= 0.5 and is_peak_hour:
        return "BT"

    # Default for elevated risk
    return "U30"


def get_retry_strategy(failure_code: str) -> dict:
    """
    Returns the retry strategy for a given NPCI failure code.

    Args:
        failure_code: NPCI failure code string (e.g. "U30")

    Returns:
        dict with retry strategy details
    """
    strategy = RETRY_STRATEGIES.get(failure_code)
    if not strategy:
        return {
            "should_retry": True,
            "max_attempts": 1,
            "wait_seconds": [30],
            "suggestion": "Unknown failure. Retry once after 30 seconds.",
            "user_message": "Transaction failed. Please try again.",
        }

    code_info = FAILURE_CODES.get(failure_code, {})
    return {
        **strategy,
        "failure_code": failure_code,
        "description": code_info.get("description", ""),
        "cause": code_info.get("cause", ""),
    }