"""
Unit tests for the predict step
"""

# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import numpy as np
from src.ml_project.steps.predict import Predictor, make_predictions


@pytest.fixture
def mock_config():
    """Mock configuration dictionary"""
    return {
        "data": {"raw_path": "data/raw", "train_file": "train.csv", "test_file": "test.csv"},
        "model": {"algorithm": "random_forest", "params": {}},
        "mlflow": {"experiment_name": "test_experiment", "tracking_uri": "mlruns"},
    }


class TestPredict:
    """Tests for prediction functionality"""

    @pytest.fixture
    def predictor(self, mock_config):
        """Create a Predictor instance with mocked config"""
        with patch("builtins.open"), patch("yaml.safe_load", return_value=mock_config), patch("mlflow.set_tracking_uri"):
            return Predictor("config.yml")

    @patch("mlflow.search_runs")
    @patch("mlflow.get_experiment_by_name")
    def test_get_latest_model_uri_success(self, mock_get_exp, mock_search, predictor):
        """Test successfully finding the latest model"""
        # Mock experiment
        mock_exp = MagicMock()
        mock_exp.experiment_id = "123"
        mock_get_exp.return_value = mock_exp

        # Mock runs
        mock_runs = pd.DataFrame({"run_id": ["run_abc"]})
        mock_search.return_value = mock_runs

        uri = predictor.get_latest_model_uri()
        assert uri == "runs:/run_abc/model"

        mock_get_exp.assert_called_with("test_experiment")
        mock_search.assert_called()

    @patch("mlflow.search_runs")
    @patch("mlflow.get_experiment_by_name")
    def test_get_latest_model_uri_no_experiment(self, mock_get_exp, _, predictor):
        """Test behavior when experiment doesn't exist"""
        mock_get_exp.return_value = None
        uri = predictor.get_latest_model_uri()
        assert uri is None

    @patch("mlflow.sklearn.load_model")
    def test_predict_success(self, mock_load_model, predictor):
        """Test making predictions"""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1])
        mock_load_model.return_value = mock_model

        data = pd.DataFrame({"col1": [1, 2]})

        # Test with explicit URI
        preds = predictor.predict(data, model_uri="runs:/fake/model")

        assert len(preds) == 2
        mock_load_model.assert_called_with("runs:/fake/model")
        mock_model.predict.assert_called_with(data)

    @patch("src.ml_project.steps.predict.Predictor")
    def test_make_predictions_wrapper(self, mock_cls):
        """Test convenience wrapper"""
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        data = pd.DataFrame()
        make_predictions(data)

        mock_cls.assert_called_once()
        mock_instance.predict.assert_called_with(data)
