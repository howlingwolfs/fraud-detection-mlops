# src/api.py
# ------------------------------------------------------------
# src/api.py – FastAPI service for XGB fraud prediction
# ------------------------------------------------------------
"""
Prerequisites

    pip install fastapi uvicorn joblib pandas numpy xgboost

Run the app

    uvicorn src.api:app --host 0.0.0.0 --port 8000

Test the endpoint (JSON body example)

    POST http://localhost:8000/predict
    {
        "Time": 0.0,
        "V1": -1.3598071336738,
        ...
        "V28": -0.0265233482418,
        "Amount": 100
    }

Response example

    {
        "prediction": 0,
        "probability": 0.0023
    }
"""

import json
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# Resolve the paths relative to this file (src/api.py)
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILE = MODEL_DIR / "XGBClassifier.joblib"

# Feature names – keep in sync with the training data
FEATURE_NAMES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]

# ------------------------------------------------------------------
# Pydantic model – one transaction record
# ------------------------------------------------------------------
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# ------------------------------------------------------------------
# Lifespan context manager – loads the model once on startup
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the pre‑trained XGB model into a global variable.
    """
    global model  # noqa: PLW0602 – intentional global
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    yield
    # No teardown logic required


# ------------------------------------------------------------------
# FastAPI instance
# ------------------------------------------------------------------
app = FastAPI(
    title="Credit‑Card Fraud Prediction API",
    description="Returns the probability of a transaction being fraudulent",
    version="1.0.0",
    lifespan=lifespan,
)

# ------------------------------------------------------------------
# Health‑check endpoint
# ------------------------------------------------------------------
@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """Simple health‑check – returns 200 OK if the server is running."""
    return {"status": "ok"}

# ------------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------------
@app.post("/predict", tags=["prediction"])
async def predict(transaction: Transaction) -> dict:
    """
    Predict whether the given transaction is fraudulent.
    """
    try:
        df = pd.DataFrame([transaction.model_dump()])[FEATURE_NAMES]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        pred = model.predict(df)[0]               # class label (0 or 1)
        prob = model.predict_proba(df)[0, 1]      # probability of class 1
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    return {"prediction": int(pred), "probability": float(prob)}

# ------------------------------------------------------------------
# Batch prediction endpoint
# ------------------------------------------------------------------
@app.post("/predict_batch", tags=["prediction"])
async def predict_batch(transactions: List[Transaction]) -> List[dict]:
    """
    Batch prediction – accepts a list of transactions and returns a list
    of predictions in the same order.
    """
    df = pd.DataFrame([t.model_dump() for t in transactions])[FEATURE_NAMES]
    preds = model.predict(df).astype(int).tolist()
    probs = model.predict_proba(df)[:, 1].tolist()
    return [{"prediction": int(p), "probability": float(q)} for p, q in zip(preds, probs)]

# ------------------------------------------------------------------
# Main entry‑point for local testing
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
