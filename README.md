# MLOps Demo Project (WIP)

This is a demo project showcasing MLOps best practices, including modular pipeline structure, experiment tracking, and containerization.

## Project Structure

The project follows a modular structure to separate concerns between data, source code, configuration, and environment definition.

```text
mlops_demo/
├── .github/                 # CI/CD workflows and actions
├── data/                    # Data directory (managed by DVC)
│   ├── raw/                 # Immutable original data
│   └── processed/           # Cleaned data ready for training
├── docs/                    # Documentation
├── mlruns/                  # MLflow experiment tracking logs
├── models/                  # Serialized models
├── notebooks/               # Jupyter notebooks for EDA and experiments
├── src/                     # Source code package
│   └── ml_project/
│       ├── pipelines/       # Pipeline orchestration logic
│       ├── steps/           # Individual pipeline steps (ingest, clean, train)
│       └── dataset.py       # Data generation and loading utilities
├── tests/                   # Unit and integration tests
├── app.py                   # FastAPI application for model serving
├── config.yml               # Configuration parameters
├── data.dvc                 # DVC configuration for data versioning
├── Dockerfile               # Container definition for reproducibility
├── main.py                  # Entry point for running the training pipeline
├── Makefile                 # Utility commands for build, test, and run
└── requirements.txt         # Python dependencies
```

## Key Components

- **src/ml_project/**: Contains the core logic of the machine learning pipeline, structured as a Python package.
- **steps/**: Modular functions for each stage of the pipeline (Ingest -> Clean -> Train).
- **notebooks/**: A sandbox for data exploration that is kept separate from production code.
- **Dockerfile**: Ensures the project runs in a consistent environment across different machines.
- **mlruns/**: Stores experiment metrics and artifacts locally (can be configured for remote storage).
