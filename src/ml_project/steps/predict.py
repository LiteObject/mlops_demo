"""
Prediction Step.

This module handles the inference using the trained model.
The process involves:
1. Loading the latest trained model from MLflow.
2. Preprocessing input data (if necessary, though usually done before).
3. Generating predictions.

Example:
    Run this step directly:
    $ python src/ml_project/steps/predict.py
"""

import logging
from typing import Optional, cast, Any
import mlflow
from mlflow.exceptions import MlflowException
from mlflow import sklearn as mlflow_sklearn
import pandas as pd
import numpy as np
from src.ml_project.utils import load_config
from src.ml_project.steps.ingest import ingest_data
from src.ml_project.steps.clean import clean_data


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Predictor:
    """
    Class to load model and make predictions.
    """

    def __init__(self, config_path: str = "config.yml"):
        """
        Initialize Predictor.

        Args:
            config_path (str): Path to configuration YAML file.
        """
        self.config = load_config(config_path)
        self.mlflow_config = self.config.get("mlflow", {})
        self.experiment_name = self.mlflow_config.get("experiment_name", "default_experiment")
        self.tracking_uri = self.mlflow_config.get("tracking_uri", "mlruns")

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

    def get_latest_model_uri(self) -> Optional[str]:
        """
        Get the URI of the latest model from the configured experiment.

        Returns:
            Optional[str]: Model URI if found, else None.
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                logger.error("Experiment '%s' not found.", self.experiment_name)
                return None

            # Search for runs, order by start_time desc
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
            runs = cast(pd.DataFrame, runs)

            if runs.empty:
                logger.warning("No runs found for experiment '%s'.", self.experiment_name)
                return None

            latest_run_id = runs.iloc[0]["run_id"]
            # Assuming model is logged as "model" artifact
            model_uri = f"runs:/{latest_run_id}/model"
            logger.info("Found latest model URI: %s", model_uri)
            return model_uri

        except MlflowException as e:
            logger.error("Error finding latest model: %s", e)
            raise e
        except Exception as e:
            logger.error("Unexpected error finding latest model: %s", e)
            raise e

    def predict(self, data: pd.DataFrame, model_uri: Optional[str] = None) -> np.ndarray:
        """
        Generate predictions using the loaded model.

        Args:
            data (pd.DataFrame): Input features.
            model_uri (Optional[str]): Specific model URI. If None, fetches latest.

        Returns:
            np.ndarray: Predictions.
        """
        try:
            if model_uri is None:
                model_uri = self.get_latest_model_uri()
                if model_uri is None:
                    raise ValueError("Could not determine model URI.")

            logger.info("Loading model from %s...", model_uri)
            model: Any = mlflow_sklearn.load_model(model_uri)

            logger.info("Generating predictions for %d samples...", len(data))
            predictions = model.predict(data)
            return predictions

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error during prediction: %s", e)
            raise e


def make_predictions(data: pd.DataFrame, config_path: str = "config.yml") -> np.ndarray:
    """
    Convenience function to run prediction pipeline.

    Args:
        data (pd.DataFrame): Features to predict on.
        config_path (str): Path to config.

    Returns:
        np.ndarray: Predictions.
    """
    predictor = Predictor(config_path)
    return predictor.predict(data)


if __name__ == "__main__":
    try:
        # Load some test data (using ingestion & clean for consistency)
        _, raw_test = ingest_data()

        # We need to clean it to get X features.
        # Note: clean_data returns (x_train, y_train, x_test, y_test)
        # We can pass raw_test as both args just to get the x_test part transformed if clean_data is rigid,
        # but clean_data(raw_train, raw_test) is the signature.
        # Let's just process the test set part.

        # Simulating the pipeline flow to get valid features
        # In a real scenario, we might have a dedicated transform method,
        # but here we rely on the clean step's output.
        raw_train_dummy, raw_test_real = ingest_data()
        _, _, x_test_processed, y_test_real = clean_data(raw_train_dummy, raw_test_real)

        # Predict
        preds = make_predictions(x_test_processed)

        print(f"Predictions (first 5): {preds[:5]}")
        print(f"Actuals (first 5): {y_test_real.values[:5]}")

    except (FileNotFoundError, ValueError, RuntimeError) as err:
        logger.error("Prediction script failed: %s", err)
