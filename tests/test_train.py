"""
Unit tests for model training step
"""

# pylint: disable=redefined-outer-name
from unittest.mock import patch
import pandas as pd
import pytest
from src.ml_project.steps.train import ModelTrainer


@pytest.fixture
def sample_data():
    """Create sample training data"""
    features = pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8]})
    y = pd.Series([0, 1, 0, 1], name="target")
    return features, y


@pytest.fixture
def mock_config():
    """Mock configuration"""
    return {
        "model": {"algorithm": "random_forest", "params": {"n_estimators": 10, "random_state": 42}},
        "mlflow": {"experiment_name": "test", "tracking_uri": "mlruns"},
    }


class TestModelTrainer:
    """Tests for ModelTrainer class"""

    @patch("src.ml_project.steps.train.load_config")
    def test_train_returns_model(self, mock_load_config, mock_config, sample_data):
        """Test that training returns a fitted model"""
        mock_load_config.return_value = mock_config
        features, y = sample_data

        trainer = ModelTrainer("config.yml")
        model = trainer.train(features, y)

        assert hasattr(model, "predict")
        predictions = model.predict(features)
        assert len(predictions) == len(y)

    @patch("src.ml_project.steps.train.load_config")
    def test_evaluate_returns_metrics(self, mock_load_config, mock_config, sample_data):
        """Test evaluation returns expected metrics"""
        mock_load_config.return_value = mock_config
        features, y = sample_data

        trainer = ModelTrainer("config.yml")
        model = trainer.train(features, y)
        metrics = trainer.evaluate(model, features, y)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
