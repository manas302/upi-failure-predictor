# api/bank_health.py
"""
Computes a real-time health score (0.0 to 1.0) for a given bank.
Higher score = healthier bank = lower failure probability.

Factors considered:
- Base reliability of the bank
- Hour of day (peak hours = more stress = lower health)
- Salary day stress (1st, 2nd, 30th, 31st of month)
- Known high-stress banks during peak load
"""

import json

# ── Base reliability scores per bank ─────────────────────────────
# Based on RBI/NPCI public failure rate data patterns
BANK_BASE_HEALTH = {
    "HDFC":     0.92,
    "ICICI":    0.90,
    "Axis":     0.88,
    "Kotak":    0.87,
    "SBI":      0.78,   # High volume = more failures during peak
    "PNB":      0.74,
    "BOB":      0.72,
    "YesBank":  0.68,   # Historically more unstable
}

DEFAULT_HEALTH = 0.75   # For unknown banks

# ── Peak hour penalty ─────────────────────────────────────────────
# These hours see highest UPI traffic → more timeouts
PEAK_HOUR_PENALTIES = {
    9:  0.08,
    10: 0.10,
    11: 0.10,
    12: 0.06,
    13: 0.05,
    19: 0.08,
    20: 0.12,   # Highest traffic hour
    21: 0.10,
    22: 0.07,
}

# ── Bank-specific peak sensitivity ───────────────────────────────
# PSU banks (SBI, PNB, BOB) degrade more under peak load
PEAK_SENSITIVITY = {
    "SBI":      1.5,
    "PNB":      1.4,
    "BOB":      1.4,
    "YesBank":  1.3,
    "HDFC":     0.8,
    "ICICI":    0.8,
    "Axis":     0.9,
    "Kotak":    0.9,
}

DEFAULT_SENSITIVITY = 1.0


def compute_bank_health(bank_name: str, hour: int, is_salary_day: bool) -> float:
    """
    Returns a health score between 0.0 and 1.0 for a bank.

    Args:
        bank_name:     Bank name (must match BANK_BASE_HEALTH keys)
        hour:          Current hour (0-23)
        is_salary_day: True if today is 1st, 2nd, 30th, or 31st of month

    Returns:
        float: Health score between 0.0 and 1.0
    """
    base = BANK_BASE_HEALTH.get(bank_name, DEFAULT_HEALTH)
    sensitivity = PEAK_SENSITIVITY.get(bank_name, DEFAULT_SENSITIVITY)

    # Apply peak hour penalty
    peak_penalty = PEAK_HOUR_PENALTIES.get(hour, 0.0) * sensitivity

    # Salary day penalty — all banks see higher load
    salary_penalty = 0.06 if is_salary_day else 0.0

    # PSU banks hit harder on salary days
    if is_salary_day and bank_name in ("SBI", "PNB", "BOB"):
        salary_penalty = 0.10

    health = base - peak_penalty - salary_penalty

    # Clamp between 0.1 and 1.0
    return round(max(0.1, min(1.0, health)), 4)


def get_all_bank_health(hour: int, is_salary_day: bool) -> dict:
    """
    Returns health scores for all known banks.
    Useful for the dashboard bank health overview panel.
    """
    return {
        bank: compute_bank_health(bank, hour, is_salary_day)
        for bank in BANK_BASE_HEALTH
    }


def classify_bank_health(score: float) -> str:
    """Converts a health score to a human-readable status."""
    if score >= 0.85:
        return "Excellent"
    elif score >= 0.75:
        return "Good"
    elif score >= 0.60:
        return "Degraded"
    else:
        return "Critical"

def get_all_bank_health_cached(redis_client, hour: int, is_salary_day: bool) -> dict:
    """
    Same as get_all_bank_health() but caches result in Redis for 5 minutes.
    Falls back to direct compute if Redis is unavailable.
    """
    if redis_client is None:
        return get_all_bank_health(hour, is_salary_day)

    cache_key = f"bank_health:{hour}:{int(is_salary_day)}"

    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    scores = get_all_bank_health(hour, is_salary_day)

    try:
        redis_client.setex(cache_key, 300, json.dumps(scores))
    except Exception:
        pass

    return scores        