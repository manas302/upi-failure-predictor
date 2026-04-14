# api/predict.py

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from datetime import datetime
import time
import math
import json
import hashlib

import os

# Load optimal threshold
_threshold_path = os.path.join(os.path.dirname(__file__), "../models/threshold_config.json")
try:
    with open(_threshold_path) as f:
        PREDICTION_THRESHOLD = json.load(f)["optimal_threshold"]
except Exception:
    PREDICTION_THRESHOLD = 0.2782

print(f"✅ Threshold loaded: {PREDICTION_THRESHOLD}")

from bank_health import compute_bank_health
from retry_engine import get_retry_strategy, predict_likely_failure_code

router = APIRouter()

# ── Schemas ───────────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    sender_bank: str
    receiver_bank: str
    amount: float
    hour_of_day: Optional[int] = None
    network_type: str = "4G"
    device_type: str = "android"
    is_salary_day: Optional[int] = None
    is_festival_day: int = 0
    model_type: str = "xgboost"  # "xgboost" or "lightgbm"

class PredictionResponse(BaseModel):
    risk_score: float
    risk_level: str
    predicted_failure_code: Optional[str]
    retry_strategy: Optional[dict]
    feature_contributions: dict
    prediction_time_ms: float
    timestamp: str

# ── Constants ─────────────────────────────────────────────────────
BANK_NAMES     = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "BOB", "PNB", "YesBank"]
NETWORK_TYPES  = ["4G", "3G", "2G", "wifi"]
DEVICE_TYPES   = ["android", "ios", "feature_phone"]
AMOUNT_BUCKETS = ["micro", "small", "medium", "large", "very_large"]

# ── Helpers ───────────────────────────────────────────────────────
def get_amount_bucket(amount: float) -> str:
    if amount < 100:      return "micro"
    elif amount < 1000:   return "small"
    elif amount < 10000:  return "medium"
    elif amount < 100000: return "large"
    else:                 return "very_large"

def get_risk_level(score: float) -> str:
    if score < 0.3:    return "LOW"
    elif score < 0.55: return "MEDIUM"
    else:              return "HIGH"

def build_feature_vector(req: TransactionRequest, feature_names: list) -> pd.DataFrame:
    now          = datetime.now()
    hour         = req.hour_of_day if req.hour_of_day is not None else now.hour
    day_of_week  = now.weekday()
    day_of_month = now.day
    is_salary_day = req.is_salary_day if req.is_salary_day is not None else \
                    (1 if day_of_month in [1, 2, 30, 31] else 0)

    sender_health   = compute_bank_health(req.sender_bank,   hour, bool(is_salary_day))
    receiver_health = compute_bank_health(req.receiver_bank, hour, bool(is_salary_day))
    sender_fail_rate   = round(1 - sender_health + 0.02, 2)
    receiver_fail_rate = round(1 - receiver_health + 0.02, 2)

    sender_enc   = BANK_NAMES.index(req.sender_bank)   if req.sender_bank   in BANK_NAMES else 0
    receiver_enc = BANK_NAMES.index(req.receiver_bank) if req.receiver_bank in BANK_NAMES else 0

    amount_bucket      = get_amount_bucket(req.amount)
    log_amount         = math.log1p(req.amount)
    is_high_value      = 1 if req.amount > 10000 else 0
    is_peak_hour       = 1 if (9 <= hour <= 11 or 19 <= hour <= 22) else 0
    is_weekend         = 1 if day_of_week >= 5 else 0
    is_high_stress_day = 1 if (is_salary_day or req.is_festival_day) else 0
    bank_health_diff   = sender_health - receiver_health
    combined_fail_rate = (sender_fail_rate + receiver_fail_rate) / 2
    min_bank_health    = min(sender_health, receiver_health)
    health_x_failrate  = sender_health * sender_fail_rate
    network_risk_map   = {"2G": 1.0, "3G": 0.6, "4G": 0.2, "wifi": 0.0}
    network_risk       = network_risk_map.get(req.network_type, 0.2)
    sender_risk_score  = (
        0.4 * sender_fail_rate +
        0.3 * (1 - sender_health) +
        0.2 * network_risk +
        0.1 * is_high_value
    )

    features = {
        "sender_bank_encoded":       sender_enc,
        "receiver_bank_encoded":     receiver_enc,
        "day_of_week":               day_of_week,
        "is_salary_day":             is_salary_day,
        "is_festival_day":           req.is_festival_day,
        "amount":                    req.amount,
        "sender_bank_health":        sender_health,
        "receiver_bank_health":      receiver_health,
        "sender_recent_fail_rate":   sender_fail_rate,
        "receiver_recent_fail_rate": receiver_fail_rate,
        "is_peak_hour":              is_peak_hour,
        "is_weekend":                is_weekend,
        "is_high_stress_day":        is_high_stress_day,
        "log_amount":                log_amount,
        "is_high_value":             is_high_value,
        "bank_health_diff":          bank_health_diff,
        "combined_fail_rate":        combined_fail_rate,
        "min_bank_health":           min_bank_health,
        "health_x_failrate":         health_x_failrate,
        "network_risk":              network_risk,
        "sender_risk_score":         sender_risk_score,
    }
    for nt in NETWORK_TYPES:
        features[f"network_type_{nt}"] = 1 if req.network_type == nt else 0
    for dt in DEVICE_TYPES:
        features[f"device_type_{dt}"] = 1 if req.device_type == dt else 0
    for ab in AMOUNT_BUCKETS:
        features[f"amount_bucket_{ab}"] = 1 if amount_bucket == ab else 0

    df = pd.DataFrame([features])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


# ── Core logic (also used by /predict/batch in main.py) ──────────
async def run_prediction(req: TransactionRequest, state) -> dict:
    start = time.time()

    # ── Redis cache check ─────────────────────────────────────────
    cache_key = "predict:" + hashlib.md5(
        f"{req.sender_bank}:{req.receiver_bank}:{req.amount}:{req.hour_of_day}:{req.network_type}:{req.device_type}:{req.is_salary_day}:{req.is_festival_day}".encode()
    ).hexdigest()

    redis_client = getattr(state, "redis", None)

    if redis_client:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                result = json.loads(cached_result)
                result["cached"] = True
                return result
        except Exception:
            pass

    # ── Validation ────────────────────────────────────────────────
    if req.sender_bank not in BANK_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown sender_bank: {req.sender_bank}")
    if req.receiver_bank not in BANK_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown receiver_bank: {req.receiver_bank}")

    # ── Prediction ──────────────────────────────────────────────── 
    feature_df   = build_feature_vector(req, state.feature_names)
    risk_score   = float(state.model.predict_proba(feature_df)[0][1])

    # Yeh dono daalo:
    feature_df = build_feature_vector(req, state.feature_names)
    if req.model_type == "lightgbm" and getattr(state, "lgbm_model", None) is not None:
        risk_score = float(state.lgbm_model.predict_proba(feature_df)[0][1])
        threshold  = state.lgbm_threshold
    else:
        risk_score = float(state.model.predict_proba(feature_df)[0][1])
        threshold  = PREDICTION_THRESHOLD
        
    risk_level   = get_risk_level(risk_score)

    features_dict = feature_df.iloc[0].to_dict()
    failure_code  = predict_likely_failure_code(risk_score, features_dict) if risk_score >= 0.3 else None
    retry         = get_retry_strategy(failure_code) if failure_code else None

    active_model = state.lgbm_model if (req.model_type == 'lightgbm' and getattr(state, 'lgbm_model', None) is not None) else state.model
    importances = active_model.feature_importances_
    top_contrib = {
        k: float(v) for k, v in sorted(
            zip(state.feature_names, importances),
            key=lambda x: x[1], reverse=True
        )[:8]
    }

    result = {
        "transaction_id":         "N/A",
        "failure_probability":    round(risk_score, 4),
        "prediction":             "LIKELY_FAIL" if risk_score >= threshold  else "LIKELY_SUCCESS",
        "risk_level":             risk_level,
        "top_risk_factors": [
            {"feature": k, "shap_value": v}
            for k, v in top_contrib.items()
        ],
        "cached":                 False,
        "retry_recommended":      risk_score >= 0.35,
        "retry_suggestion":       retry.get("suggestion") if retry else None,
        "predicted_failure_code": failure_code,
        "prediction_time_ms":     round((time.time() - start) * 1000, 2),
        "timestamp":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Store in Redis (TTL: 5 min) ───────────────────────────────
    if redis_client:
        try:
            redis_client.setex(cache_key, 300, json.dumps(result))
        except Exception:
            pass

    return result


# ── /predict endpoint ─────────────────────────────────────────────
@router.post("/predict")
async def predict(req: TransactionRequest, request: Request):
    return await run_prediction(req, request.app.state)