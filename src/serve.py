import time
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
from clearml import Model, Task
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

task = None
logger = None
model = None
expected_features = 30
FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Empirical training statistics for Kaggle Credit Card Fraud dataset
TIME_MEAN, TIME_STD = 94813.8, 47488.1
AMOUNT_MEAN, AMOUNT_STD = 88.34, 250.12


@asynccontextmanager
async def lifespan(app: FastAPI):
    global task, logger, model, expected_features

    # Safe ClearML initialization on app startup
    task = Task.init(
        project_name="Credit Card Fraud Detection",
        task_name="Production Inference Endpoint",
        task_type=Task.TaskTypes.inference,
        reuse_last_task_id=False,
    )
    logger = task.get_logger()

    # Load published model artifact from ClearML Registry
    model_info = Model(model_id="626fa416cc304203af12935da4703143")
    model_path = model_info.get_local_copy()
    model = joblib.load(model_path)
    expected_features = getattr(model, "n_features_in_", 30)

    yield

   
    if task:
        task.close()


app = FastAPI(title="Credit Card Fraud Detection API", lifespan=lifespan)

# Adding CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_count = 0


class TransactionPayload(BaseModel):
    features: list[float]


def log_telemetry_background(prob: float, pred: int, latency_ms: float, count: int):
    """Offloads network I/O logging so API responses remain near-instantaneous."""
    if logger:
        try:
            logger.report_scalar(
                title="Production Telemetry",
                series="Fraud Probability",
                value=prob,
                iteration=count,
            )
            logger.report_scalar(
                title="Production Telemetry",
                series="Prediction Class",
                value=pred,
                iteration=count,
            )
            logger.report_scalar(
                title="Performance",
                series="Latency (ms)",
                value=latency_ms,
                iteration=count,
            )
        except Exception as e:
            print(f"[WARNING] ClearML telemetry logging failed: {e}")


@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_id": "626fa416cc304203af12935da4703143",
        "expected_features": expected_features,
    }


@app.post("/predict")
def predict(payload: TransactionPayload, background_tasks: BackgroundTasks):
    global prediction_count

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not loaded. Check server startup logs.",
        )

    if len(payload.features) != expected_features:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid payload shape. Expected {expected_features} features, got {len(payload.features)}.",
        )

    prediction_count += 1
    start_time = time.time()

    # Preprocess raw Time 
    processed_features = list(payload.features)
    processed_features[0] = (processed_features[0] - TIME_MEAN) / TIME_STD
    processed_features[29] = (processed_features[29] - AMOUNT_MEAN) / AMOUNT_STD

    # Evaluate model predictions
    try:
        df = pd.DataFrame([processed_features], columns=FEATURE_NAMES)
        probabilities = model.predict_proba(df)[0]
    except Exception:
        features_array = np.array(processed_features, dtype=float).reshape(1, -1)
        probabilities = model.predict_proba(features_array)[0]

    prob = float(probabilities[1])

    # Calibrated decision threshold for imbalanced credit card data
    THRESHOLD = 0.30
    pred = 1 if prob >= THRESHOLD else 0

    latency_ms = (time.time() - start_time) * 1000

    print(f"[DEBUG] Raw Fraud Probability: {prob:.6f} | Final Decision: {pred}")

    # Queue telemetry to run asynchronously after response is returned
    background_tasks.add_task(
        log_telemetry_background, prob, pred, latency_ms, prediction_count
    )

    return {"is_fraud": pred, "fraud_probability": round(prob, 4)}