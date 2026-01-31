"""
Data Ingestion Step.

This module handles the ingestion of data for the machine learning pipeline.

The process involves:
1. Loading the configuration from a YAML file.
2. Reading raw training and testing data from CSV files specified in the config.
3. Returning pandas DataFrames for downstream processing.

Example:
    Run this step directly:
    $ python src/ml_project/steps/ingest.py
"""

import logging
import os
from typing import Tuple, Dict, Any
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Class to handle data ingestion from CSV files.
    """

    def __init__(self, config_path: str = "config.yml"):
        """
        Initialize DataIngestion with configuration path.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            dict: Configuration dictionary.
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded from %s", self.config_path)
            return config
        except Exception as e:
            logger.error("Error loading config: %s", e)
            raise e

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ingest data from raw data paths defined in config.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing (train_df, test_df).
        """
        try:
            data_config = self.config["data"]
            raw_path = data_config["raw_path"]

            train_path = os.path.join(raw_path, data_config["train_file"])
            test_path = os.path.join(raw_path, data_config["test_file"])

            logger.info("Reading training data from %s", train_path)
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Training file not found at {train_path}")
            df_train = pd.read_csv(train_path)

            logger.info("Reading test data from %s", test_path)
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test file not found at {test_path}")
            df_test = pd.read_csv(test_path)

            logger.info("Data ingestion completed. Train shape: %s, Test shape: %s", df_train.shape, df_test.shape)
            return df_train, df_test

        except Exception as e:
            logger.error("Error in data ingestion: %s", e)
            raise e


def ingest_data(config_path: str = "config.yml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to run data ingestion.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing (train_df, test_df).
    """
    ingestor = DataIngestion(config_path)
    return ingestor.get_data()


if __name__ == "__main__":
    # Allow running this step as a script
    try:
        train_df, test_df = ingest_data()
        print(f"Ingested training data:\n{train_df.head()}")
    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        logger.error("Failed to execute data ingestion: %s", e)
