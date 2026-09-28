import time
import joblib
import pandas as pd
from clearml import Model, Task
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize ClearML Monitoring Task
task = Task.init(
    project_name="Credit Card Fraud Detection",
    task_name="Production Inference Endpoint",
    task_type=Task.TaskTypes.inference,
    reuse_last_task_id=False,
)
logger = task.get_logger()

# Fetching published model from ClearML Registry
model_info = Model(model_id="626fa416cc304203af12935da4703143")
model_path = model_info.get_local_copy()
model = joblib.load(model_path)

# Extract expected feature count
expected_features = getattr(model, "n_features_in_", 30)
feature_names = getattr(
    model, "feature_names_in_", [f"V{i}" for i in range(1, expected_features + 1)]
)

app = FastAPI(title="Credit Card Fraud Detection API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_count = 0


class TransactionPayload(BaseModel):
    features: list[float]  # Expects 30 features


@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_id": "626fa416cc304203af12935da4703143",
        "expected_features": expected_features,
    }


@app.post("/predict")
def predict(payload: TransactionPayload):
    global prediction_count

    if len(payload.features) != expected_features:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid payload shape. Expected {expected_features} features, got {len(payload.features)}.",
        )

    prediction_count += 1
    start_time = time.time()

    # Pass as DataFrame with expected feature names to suppress sklearn warnings
    df = pd.DataFrame([payload.features], columns=feature_names)

    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    latency_ms = (time.time() - start_time) * 1000

    # Log telemetry directly to ClearML
    logger.report_scalar(
        title="Production Telemetry",
        series="Fraud Probability",
        value=prob,
        iteration=prediction_count,
    )
    logger.report_scalar(
        title="Production Telemetry",
        series="Prediction Class",
        value=pred,
        iteration=prediction_count,
    )
    logger.report_scalar(
        title="Performance",
        series="Latency (ms)",
        value=latency_ms,
        iteration=prediction_count,
    )

    return {"is_fraud": pred, "fraud_probability": round(prob, 4)}