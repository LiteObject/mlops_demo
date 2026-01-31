"""
Main PIPELINE Execution Script.

This script orchestrates the entire machine learning pipeline:
1. Ingestion
2. Cleaning
3. Training & Evaluation
"""

import logging
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.abspath("src"))

# pylint: disable=wrong-import-position
from src.ml_project.steps.ingest import ingest_data
from src.ml_project.steps.clean import clean_data
from src.ml_project.steps.train import train_model

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    Main pipeline execution.
    """
    try:
        logger.info("Starting pipeline execution...")

        # Step 1: Ingest
        logger.info("Step 1: Ingesting data...")
        raw_train, raw_test = ingest_data()

        # Step 2: Clean
        logger.info("Step 2: Cleaning data...")
        x_train, y_train, x_test, y_test = clean_data(raw_train, raw_test)

        # Step 3: Train
        logger.info("Step 3: Training model...")
        train_model(x_train, y_train, x_test, y_test)

        logger.info("Pipeline execution completed successfully.")

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        raise e


if __name__ == "__main__":
    main()
