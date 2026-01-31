"""
Model Training Step.

This module handles the training and evaluation of the machine learning model.
The process involves:
1. Training a Random Forest Classifier.
2. Logging parameters, metrics, and the model to MLflow.
3. Saving requirements for reproducibility.

Example:
    Run this step directly:
    $ python src/ml_project/steps/train.py
"""

import logging
import os
from typing import Any
import yaml
import mlflow
from mlflow import sklearn as mlflow_sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from src.ml_project.steps.ingest import ingest_data
    from src.ml_project.steps.clean import clean_data
except ImportError:
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    from src.ml_project.steps.ingest import ingest_data
    from src.ml_project.steps.clean import clean_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Class to filter and train model.
    """

    def __init__(self, config_path: str = "config.yml"):
        """
        Initialize ModelTrainer with configuration.

        Args:
            config_path (str): Path to config file.
        """
        self.config = self._load_config(config_path)
        self.mlflow_config = self.config.get("mlflow", {})
        self.model_config = self.config.get("model", {})
        self.params = self.model_config.get("params", {})

    def _load_config(self, path: str) -> dict:
        """
        Load YAML configuration.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Error loading config: %s", e)
            raise e

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train the model.

        Args:
            x_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.

        Returns:
            Any: Trained model.
        """
        algo = self.model_config.get("algorithm", "random_forest")
        logger.info("Training model using algorithm: %s", algo)

        if algo == "random_forest":
            model = RandomForestClassifier(**self.params)
        else:
            # Fallback
            logger.warning("Algorithm %s not supported. Defaulting to Random Forest.", algo)
            model = RandomForestClassifier(**self.params)

        model.fit(x_train, y_train)
        logger.info("Model training completed.")
        return model

    def evaluate(self, model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate the model.

        Args:
            model (Any): Trained model.
            x_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test labels.

        Returns:
            dict: Evaluation metrics.
        """
        y_pred = model.predict(x_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="binary"),
            "recall": recall_score(y_test, y_pred, average="binary"),
            "f1": f1_score(y_test, y_pred, average="binary"),
        }
        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    def log_to_mlflow(self, model: Any, metrics: dict):
        """
        Log parameters, metrics, and model to MLflow.

        Args:
            model (Any): Trained model.
            metrics (dict): Evaluation metrics.
        """
        experiment_name = self.mlflow_config.get("experiment_name", "default_experiment")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log params
            mlflow.log_params(self.params)
            mlflow.log_param("algorithm", self.model_config.get("algorithm"))

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow_sklearn.log_model(model, "model")
            logger.info("Model and metrics logged to MLflow experiment: %s", experiment_name)


def train_model(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, config_path: str = "config.yml"
) -> Any:
    """
    Convenience function to run training pipeline.

    Returns:
        Any: Trained model.
    """
    trainer = ModelTrainer(config_path)

    # Train
    model = trainer.train(x_train, y_train)

    # Evaluate
    metrics = trainer.evaluate(model, x_test, y_test)

    # Log
    trainer.log_to_mlflow(model, metrics)

    return model


if __name__ == "__main__":
    try:
        # End-to-end test
        raw_train, raw_test = ingest_data()
        x_tr, y_tr, x_te, y_te = clean_data(raw_train, raw_test)
        trained_model = train_model(x_tr, y_tr, x_te, y_te)
    except (FileNotFoundError, ValueError, KeyError) as err:
        logger.error("Training failed: %s", err)
    except (IOError, OSError, RuntimeError) as err:
        logger.error("Unexpected error during training: %s", err)
