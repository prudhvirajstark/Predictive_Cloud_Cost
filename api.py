# /// script
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "pandas",
#   "scikit-learn",
#   "mlflow",
# ]
# ///
import os
import pickle
import uvicorn
import pandas as pd
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn # type: ignore

# 1. Define global memory storage for live API assets
ml_assets = {}

# 2. Updated Lifespan Event Handler querying the local MLflow Registry
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Connect to your absolute SQLite tracking workspace
        db_path = os.path.abspath("mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        print("\n[INIT] Connecting to local MLflow backend database...")

        # 1. Fetch the active model directly using the modern model alias syntax
        model_uri = "models:/Cloud_Cost_Optimizer_Model@Production" # Note the @Production alias syntax
        ml_assets["model"] = mlflow.sklearn.load_model(model_uri)
        print("[SUCCESS] Active Production Model loaded from MLflow.")

        # 2. Query the registry client using modern aliases instead of deprecated stages
        client = mlflow.tracking.MlflowClient()

        # Extract the model version that owns the active "Production" alias
        model_version_details = client.get_model_version_by_alias("Cloud_Cost_Optimizer_Model", "Production")
        run_id = model_version_details.run_id

        print(f"[INFO] Production Model points to Run ID: {run_id} (Version {model_version_details.version})")

        # 3. FIX: download_artifacts uses 'path' to specify the artifact file name
        scaler_local_path = client.download_artifacts(run_id=run_id, path="cloud_cost_scaler.pkl")
        print(f"[INFO] Coupled StandardScaler artifact downloaded to: {scaler_local_path}")
        with open(scaler_local_path, "rb") as f:
            ml_assets["scaler"] = pickle.load(f)

        print(f"[SUCCESS] Coupled StandardScaler loaded from: {scaler_local_path}\n")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to fetch active production assets from local MLflow: {str(e)}")
        print("💡 Hint: Ensure you re-ran your notebook training cell and it successfully saved 'cloud_cost_scaler.pkl'.\n")
        raise SystemExit("Missing deployment catalog assets. Shutting down server.")

    yield
    ml_assets.clear()
    print("\n[SHUTDOWN] Microservice cache flushed smoothly.\n")

# 3. Initialize FastAPI application
app = FastAPI(
    title="Cloud Cost Anomaly Detection Engine (MLflow Powered)",
    description="Production-grade API fetching dynamic models natively from a local MLflow version registry.",
    version="2.0.0",
    lifespan=lifespan
)

# 4. Strict data typing and validation guardrails using Pydantic
class CloudLogInput(BaseModel):
    Usage_Amount: float
    Cost: float
    Service: Literal["EC2", "S3", "RDS", "Lambda"]
    Region: Literal["us-east-1", "eu-west-1"]

# 5. Live Anomaly Evaluation Endpoint
@app.post("/predict", summary="Evaluate live metrics against the active MLflow Production Model")
def predict_anomaly(data: CloudLogInput):
    model = ml_assets.get("model")
    scaler = ml_assets.get("scaler")

    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Machine learning engine components are uninitialized.")

    try:
        # Format input log exactly matching the original training feature dimensions
        input_dict = {
            "Usage_Amount": data.Usage_Amount,
            "Cost": data.Cost,
            "Service_EC2": 1 if data.Service == "EC2" else 0,
            "Service_Lambda": 1 if data.Service == "Lambda" else 0,
            "Service_RDS": 1 if data.Service == "RDS" else 0,
            "Service_S3": 1 if data.Service == "S3" else 0,
            "Region_eu-west-1": 1 if data.Region == "eu-west-1" else 0,
            "Region_us-east-1": 1 if data.Region == "us-east-1" else 0,
        }

        df_input = pd.DataFrame([input_dict])
        scaled_input = scaler.transform(df_input)

        # Process predictions through the active MLflow asset assembly
        prediction = model.predict(scaled_input)
        score = model.decision_function(scaled_input)
        is_anomaly = bool(prediction == -1)

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": float(score[0]), # Extracted safely out of the 1D NumPy array shape
            "meta": {
                "service_evaluated": data.Service,
                "region_evaluated": data.Region,
                "cost_received": data.Cost,
                "usage_received": data.Usage_Amount
            },
            "incident_response": "ALERT: Triggering DevOps on-call infrastructure review" if is_anomaly else "Nominal operation"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failure processing features: {str(e)}")

# 6. Admin Hot-Swap / Reload Endpoint
@app.post("/reload", summary="Hot-swap server memory to a newly promoted MLflow model version without downtime")
def hot_swap_production_model():
    try:
        db_path = os.path.abspath("mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        model_uri = "models:/Cloud_Cost_Optimizer_Model@Production"
        new_model = mlflow.sklearn.load_model(model_uri)

        client = mlflow.tracking.MlflowClient()
        model_version_details = client.get_model_version_by_alias("Cloud_Cost_Optimizer_Model", "Production")

        scaler_local_path = client.download_artifacts(run_id=model_version_details.run_id, path="cloud_cost_scaler.pkl")
        with open(scaler_local_path, "rb") as f:
            new_scaler = pickle.load(f)

        ml_assets["model"] = new_model
        ml_assets["scaler"] = new_scaler

        return {
            "status": "success",
            "message": f"API hot-swapped seamlessly to MLflow Run ID: {model_version_details.run_id} (Version {model_version_details.version})"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hot-swap transaction aborted: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
