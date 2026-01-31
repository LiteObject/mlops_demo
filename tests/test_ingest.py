"""
Unit tests for data ingestion step
"""

# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from src.ml_project.steps.ingest import DataIngestion, ingest_data


@pytest.fixture
def mock_config():
    """Mock configuration dictionary"""
    return {"data": {"raw_path": "data/raw", "train_file": "train.csv", "test_file": "test.csv"}}


class TestDataIngestion:
    """Test cases for DataIngestion class"""

    def test_init(self):
        """Test initialization loads config"""
        with patch("builtins.open"), patch("yaml.safe_load", return_value={"test": "config"}):
            ingestor = DataIngestion("fake_path.yml")
            assert ingestor.config == {"test": "config"}

    @patch("src.ml_project.steps.ingest.pd.read_csv")
    @patch("src.ml_project.steps.ingest.os.path.exists")
    def test_get_data_success(self, mock_exists, mock_read_csv, mock_config):
        """Test successful data retrieval"""
        # Setup mocks
        mock_exists.return_value = True
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.return_value = mock_df

        # Initialize with mocked config
        with patch("builtins.open"), patch("yaml.safe_load", return_value=mock_config):
            ingestor = DataIngestion("fake_path.yml")
            train_df, test_df = ingestor.get_data()

        # Assertions
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert train_df.shape == (2, 2)
        assert mock_read_csv.call_count == 2  # Called for train and test

    @patch("src.ml_project.steps.ingest.os.path.exists")
    def test_get_data_file_not_found(self, mock_exists, mock_config):
        """Test error when file is missing"""
        mock_exists.return_value = False

        with patch("builtins.open"), patch("yaml.safe_load", return_value=mock_config):
            ingestor = DataIngestion("fake_path.yml")

            with pytest.raises(FileNotFoundError):
                ingestor.get_data()

    @patch("src.ml_project.steps.ingest.DataIngestion")
    def test_ingest_data_convenience_function(self, mock_cls):
        """Test the convenience wrapper function"""
        # Setup mock instance
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.get_data.return_value = (pd.DataFrame(), pd.DataFrame())

        # Call function
        ingest_data("config.yml")

        # Verify calls
        mock_cls.assert_called_once_with("config.yml")
        mock_instance.get_data.assert_called_once()
