"""
Azure ML / Azure Function scoring script.

This follows the standard Azure ML `init()` / `run()` scoring contract, so it
can be registered as an Azure ML online endpoint, or wrapped in an HTTP-triggered
Azure Function (see function_app.py) for a lighter-weight deployment.
"""
import json
from pathlib import Path
import joblib
import pandas as pd

model = None


def init():
    """Called once when the endpoint/function cold-starts."""
    global model
    model_path = Path(__file__).resolve().parent / "model.pkl"
    model = joblib.load(model_path)


def run(raw_data):
    """
    Expects JSON: {"data": [{"tenure_months": 5, "monthly_charge": 80.0,
                              "contract_type": "month-to-month", "num_support_calls": 3,
                              "has_autopay": 0, "num_products": 1, "satisfaction_score": 2}]}
    Returns: {"churn_probability": [0.71]}
    """
    body = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    df = pd.DataFrame(body["data"])
    proba = model.predict_proba(df)[:, 1]
    return {"churn_probability": [round(float(p), 4) for p in proba]}
