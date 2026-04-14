# api/main.py
"""
Entry point for the UPI Payment Failure Predictor API.
Handles:
- App startup (loads model, SHAP explainer, Redis)
- Registers router from predict.py
- /health endpoint
- /predict/batch endpoint
"""

import os
import json
import pickle
import redis
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from predict import router as predict_router, TransactionRequest, run_prediction

from bank_health import get_all_bank_health_cached, classify_bank_health
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────
MODEL_PATH    = os.getenv("MODEL_PATH",    "../models/xgb_model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "../models/feature_names.pkl")
REDIS_URL     = os.getenv("REDIS_URL",     "redis://localhost:6379")
CACHE_TTL     = int(os.getenv("CACHE_TTL", 300))


# ── Lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load XGBoost model
    with open(MODEL_PATH, "rb") as f:
        app.state.model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        app.state.feature_names = pickle.load(f)

    # Load LightGBM model
    lgbm_path = os.getenv("LGBM_MODEL_PATH", "../models/lgbm_model.pkl")
    lgbm_threshold_path = os.getenv("LGBM_THRESHOLD_PATH", "../models/lgbm_threshold_config.json")
    try:
        with open(lgbm_path, "rb") as f:
            app.state.lgbm_model = pickle.load(f)
        with open(lgbm_threshold_path) as f:
            app.state.lgbm_threshold = json.load(f)["optimal_threshold"]
        print(f"✅ LightGBM loaded, threshold: {app.state.lgbm_threshold:.4f}")
    except Exception as e:
        app.state.lgbm_model = None
        app.state.lgbm_threshold = 0.4455
        print(f"⚠️ LightGBM unavailable: {e}")

    # SHAP explainer
    app.state.explainer = shap.TreeExplainer(app.state.model)

    # Redis (graceful degradation)
    try:
        client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        app.state.redis = client
        print("✅ Redis connected")
    except Exception:
        app.state.redis = None
        print("⚠️  Redis unavailable — caching disabled")

    print(f"✅ Model loaded: {MODEL_PATH}")
    yield


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="UPI Payment Failure Predictor",
    version="2.0.0",
    description="Predicts UPI transaction failure with NPCI failure codes and retry strategies.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register predict router (handles /predict)
app.include_router(predict_router)


# ── /health ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
        "redis_connected": getattr(app.state, "redis", None) is not None,
        "version": "2.0.0",
    }

@app.get("/bank-health")
def bank_health():
    now = datetime.now()
    hour = now.hour
    day = now.day
    is_salary = day in [1, 2, 30, 31]
    scores = get_all_bank_health_cached(app.state.redis, hour, is_salary)
    return {
        "banks": {
            bank: {
                "score": round(score * 100, 1),
                "status": classify_bank_health(score),
                "is_peak_hour": (9 <= hour <= 11 or 19 <= hour <= 22),
                "is_salary_day": is_salary,
            }
            for bank, score in scores.items()
        },
        "fetched_at": now.strftime("%H:%M:%S"),
        "hour": hour,
    }


# ── /predict/batch ────────────────────────────────────────────────
@app.post("/predict/batch")
async def predict_batch(transactions: list[TransactionRequest]):
    """Batch endpoint — up to 50 transactions."""
    if len(transactions) > 50:
        raise HTTPException(status_code=400, detail="Batch limit is 50 transactions.")

    results = []
    for txn in transactions:
        result = await run_prediction(txn, app.state)
        results.append(result)

    return {"results": results, "count": len(results)}