# Credit Card Fraud Detection Engine

An end-to-end, production-grade MLOps engine built to detect fraudulent credit card transactions in real-time and batch modes. The system features model tracking with **ClearML**, automated deployment via **FastAPI** on **Render**, and an interactive dashboard deployed on **Streamlit Community Cloud**.

 **Live Applications & Endpoints:**
* **Interactive Dashboard (Streamlit):** [credit-card-fraud-detection-g5fb8gszt3zcc2htjhgnpu.streamlit.app](https://credit-card-fraud-detection-g5fb8gszt3zcc2htjhgnpu.streamlit.app/)
* **REST API Service (Render):** [credit-card-fraud-detection-vb2x.onrender.com](https://credit-card-fraud-detection-vb2x.onrender.com)
* **API Documentation (Swagger UI):** [credit-card-fraud-detection-vb2x.onrender.com/docs](https://credit-card-fraud-detection-vb2x.onrender.com/docs)

---

## Architecture & System Flow

```text
┌─────────────────────────┐       ┌────────────────────────┐       ┌─────────────────────────┐
│ Streamlit Frontend App  │ ────> │   FastAPI Backend      │ ────> │   Machine Learning      │
│ (Real-Time & Batch CSV) │ <──── │  (Hosted on Render)    │ <──── │   Inference Pipeline    │
└─────────────────────────┘       └────────────────────────┘       └─────────────────────────┘
                                               │
                                               ▼
                                  ┌────────────────────────┐
                                  │   ClearML Dashboard    │
                                  │  (Telemetry & Metrics) │
                                  └────────────────────────┘

Key Features

Dual Inference Capabilities:

Real-Time Single Scoring: Single-transaction evaluation using sliders and feature controls.

Batch CSV Processing: High-throughput batch inference with interactive progress tracking, summary metrics, and result downloads.

MLOps Telemetry & Tracking: Real-time logging of CPU, memory utilization, and prediction latency integrated with ClearML.

High Availability Serving: Containerized FastAPI web service hosted on Render with automatic health checks and CORS handling.

Tech Stack & Tools

pandas
numpy
imbalanced-learn
fastapi
uvicorn
pydantic
pandas
scikit-learn
joblib
clearml
matplotlib
seaborn
joblib
streamlit
python-dotenv

API Endpoints & Payload Specification

POST /predict

Evaluates a 30-element feature vector representing a credit card transaction: [Time, V1..V28, Amount].

Sample Request Body:
{
  "features": [
    0.0, 0.1, -0.5, 1.2, 0.3, -0.8, 0.2, 0.5, -0.1, 0.0,
    0.4, -0.2, 0.1, 0.3, -0.4, 0.2, 0.1, -0.3, 0.2, 0.1,
    -0.1, 0.0, 0.1, -0.2, 0.1, 0.0, -0.1, 0.2, 0.1, 100.0
  ]
}

Sample Body Response (200 ok):
{
  "is_fraud": 0,
  "fraud_probability": 0.0234
}

Local Setup & Development

 Clone Repository & Setup Environment

PowerShell:
git clone [https://github.com/](https://github.com/)<your-username>/credit-card-fraud-detection.git

cd credit-card-fraud-detection

# Create and activate virtual environment
python -m venv .venv

.venv\Scripts\Activate.ps1

# Install dependencies

pip install -r requirements.txt

Configure ClearML Credentials:
Create or update clearml.conf in your project root or configure via CLI:

PowerShell: 
clearml-init

Launch Backend API Locally

PowerShell: 
uvicorn src.serve:app --reload --host 0.0.0.0 --port 8000

Access local API documentation at http://localhost:8000/docs.

Launch Streamlit Frontend Locally

PowerShell:
streamlit run main.py

Access the local dashboard at http://localhost:8501.