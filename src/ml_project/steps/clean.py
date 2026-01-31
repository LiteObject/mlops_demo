"""
Data Cleaning Step.

This module handles the cleaning and preprocessing of data for the machine learning pipeline.

The process involves:
1. Handling missing values (imputation).
2. Data transformation and target binarization.
3. Returning processed pandas DataFrames ready for training.

Example:
    Run this step directly:
    $ python src/ml_project/steps/clean.py
"""

import logging
from typing import Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataCleaning:
    """
    Class to handle data cleaning and preprocessing.
    """

    def __init__(self, target_col: str = "target"):
        """
        Initialize DataCleaning.

        Args:
            target_col (str): Name of the target column.
        """
        self.target_col = target_col

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        Strategy: Fill numeric columns with median.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with valid values.
        """
        try:
            # Check for nulls
            null_counts = df.isnull().sum().sum()
            if null_counts > 0:
                logger.info("Found %s missing values. Filling with median.", null_counts)
                # Numeric columns only for median
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

            return df
        except Exception as e:
            logger.error("Error handling missing values: %s", e)
            raise e

    def prepare_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the target variable.
        For Heart Disease dataset: 0 = No Disease, 1-4 = Disease.
        We convert this to Binary Classification (0 vs 1).

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with binary target.
        """
        try:
            if self.target_col in df.columns:
                # If values > 0, treat as positive class (1)
                df[self.target_col] = df[self.target_col].apply(lambda x: 1 if x > 0 else 0)
                logger.info("Converted target column '%s' to binary classes.", self.target_col)
            return df
        except Exception as e:
            logger.error("Error preparing target: %s", e)
            raise e

    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into Features (X) and Target (y).

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
             Tuple[pd.DataFrame, pd.Series]: X, y
        """
        try:
            x_features = df.drop(columns=[self.target_col])
            y_target = df[self.target_col]
            return x_features, y_target
        except Exception as e:
            logger.error("Error splitting features and target: %s", e)
            raise e


def clean_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "target"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Convenience function to run data cleaning pipeline.

    Args:
        train_df (pd.DataFrame): Raw training data.
        test_df (pd.DataFrame): Raw test data.
        target_col (str): Name of target column.

    Returns:
        Tuple: x_train, y_train, x_test, y_test
    """
    cleaner = DataCleaning(target_col)

    # Process Training Data
    logger.info("Cleaning Training Data...")
    train_df = cleaner.handle_missing_values(train_df)
    train_df = cleaner.prepare_target(train_df)
    x_train_processed, y_train_processed = cleaner.split_features_target(train_df)

    # Process Test Data
    logger.info("Cleaning Test Data...")
    test_df = cleaner.handle_missing_values(test_df)
    test_df = cleaner.prepare_target(test_df)
    x_test_processed, y_test_processed = cleaner.split_features_target(test_df)

    logger.info("Data cleaning completed.")
    return x_train_processed, y_train_processed, x_test_processed, y_test_processed


if __name__ == "__main__":
    # Test the script works by importing ingest dynamically
    import sys
    import os

    # Add project root to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

    # pylint: disable=import-outside-toplevel
    from src.ml_project.steps.ingest import ingest_data

    try:
        # Load data using ingest step
        raw_train, raw_test = ingest_data()

        # Run clean
        x_tr, y_tr, x_te, y_te = clean_data(raw_train, raw_test)

        print("\nCleaned Data Stats:")
        print(f"X_train shape: {x_tr.shape}")
        print(f"y_train distribution:\n{y_tr.value_counts()}")

    except Exception as err:  # pylint: disable=broad-exception-caught
        logger.error("Failed to execute data cleaning: %s", err)
