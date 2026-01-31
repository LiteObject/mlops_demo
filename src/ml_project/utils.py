"""
Utility functions for the ML project.
"""

import logging
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_config(config: dict) -> None:
    """
    Validate configuration has required keys.

    Args:
        config: Configuration dictionary

    Raises:
        KeyError: If required keys are missing
    """
    required_keys = {
        "data": ["raw_path", "train_file", "test_file"],
        "model": ["algorithm", "params"],
        "mlflow": ["experiment_name", "tracking_uri"],
    }

    for section, keys in required_keys.items():
        if section not in config:
            raise KeyError(f"Missing config section: {section}")
        for key in keys:
            if key not in config[section]:
                raise KeyError(f"Missing config key: {section}.{key}")


def load_config(path: str) -> dict:
    """
    Load YAML configuration.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        validate_config(config)
        return config
    except Exception as e:
        logger.error("Error loading config: %s", e)
        raise e
