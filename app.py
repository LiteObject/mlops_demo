"""
FastAPI app file for serving predictions.
"""

from contextlib import asynccontextmanager
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.ml_project.steps.predict import Predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Global variables to hold model/predictor
metric_model = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Load model on startup.
    """
    logger.info("Loading model...")
    try:
        # Initialize predictor
        predictor = Predictor()
        # Find latest model URI
        model_uri = predictor.get_latest_model_uri()
        if not model_uri:
            logger.warning("No model found! API will not work correctly.")
        else:
            logger.info("loading model from %s", model_uri)
            # We can pre-load the model here if we extend Predictor,
            # but Predictor.predict loads it.
            # For simplicity in this demo, we store the predictor and URI.
            metric_model["predictor"] = predictor
            metric_model["model_uri"] = model_uri

            # Warm up / Validate load
            # from mlflow import sklearn as mlflow_sklearn
            # metric_model["loaded_model"] = mlflow_sklearn.load_model(model_uri)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to load model: %s", e)

    yield

    # Cleanup if needed
    metric_model.clear()


app = FastAPI(title="Heart Disease Prediction API", lifespan=lifespan)


class HeartDiseaseInput(BaseModel):
    """Input parameters for heart disease prediction"""

    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/")
def read_root():
    """Health check"""
    return {"status": "ok", "message": "Heart Disease Prediction API is running"}


@app.post("/predict")
def predict(input_data: HeartDiseaseInput):
    """
    Generate predictions for a single sample.
    """
    if "predictor" not in metric_model or "model_uri" not in metric_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        data = pd.DataFrame([input_data.model_dump()])

        # Predict
        predictor = metric_model["predictor"]
        uri = metric_model["model_uri"]

        # Note: In a real high-throughput app, we would cache the loaded model object
        # instead of loading from MLflow/disk on every request.
        # But this uses the existing steps/predict.py logic.
        predictions = predictor.predict(data, model_uri=uri)

        return {"prediction": int(predictions[0])}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
