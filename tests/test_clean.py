"""
Unit tests for data cleaning step
"""

import pandas as pd
import numpy as np
import pytest
from src.ml_project.steps.clean import DataCleaning, clean_data as clean_data_fn


class TestDataCleaning:
    """Test cases for DataCleaning class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing"""
        df = pd.DataFrame(
            {"age": [50, 60, np.nan, 45], "chol": [200, 240, 300, np.nan], "target": [0, 1, 3, 0]}  # 3 should become 1
        )
        return df

    def test_handle_missing_values(self, sample_data):
        """Test missing value imputation"""
        cleaner = DataCleaning()
        cleaned = cleaner.handle_missing_values(sample_data.copy())

        assert cleaned.isnull().sum().sum() == 0
        # Median of [50, 60, 45] is 50
        assert cleaned.loc[2, "age"] == 50.0

    def test_prepare_target(self, sample_data):
        """Test target binarization"""
        cleaner = DataCleaning(target_col="target")
        processed = cleaner.prepare_target(sample_data.copy())

        # Check values
        unique_targets = processed["target"].unique()
        assert set(unique_targets).issubset({0, 1})
        assert processed.loc[2, "target"] == 1  # 3 -> 1

    def test_split_features_target(self, sample_data):
        """Test splitting features and target"""
        cleaner = DataCleaning(target_col="target")
        x_df, y = cleaner.split_features_target(sample_data)

        assert "target" not in x_df.columns
        assert "target" == y.name
        assert x_df.shape == (4, 2)
        assert y.shape == (4,)

    def test_clean_data_convenience_function(self, sample_data):
        """Test full pipeline wrapper"""
        # Create identical train and test sets for simplicity
        train_df = sample_data.copy()
        test_df = sample_data.copy()

        x_tr, y_tr, _, _ = clean_data_fn(train_df, test_df, target_col="target")

        assert isinstance(x_tr, pd.DataFrame)
        assert isinstance(y_tr, pd.Series)
        assert x_tr.isnull().sum().sum() == 0
