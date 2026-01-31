# MLOps Demo Project (WIP)

This is a demo project showcasing MLOps best practices, including modular pipeline structure, experiment tracking, and containerization.

## What is MLOps?

**MLOps (Machine Learning Operations)** brings together Machine Learning, DevOps, and Data Engineering to help teams build and maintain ML systems that actually work in production.

### The Problem MLOps Solves

Training a model in a notebook is easy. Keeping it running reliably in production is hard. MLOps addresses challenges like:

- **"It worked on my machine"** → Containerization and environment management ensure consistency
- **"Which model version is in production?"** → Experiment tracking and model registries provide visibility
- **"The model's predictions are getting worse"** → Monitoring and automated retraining catch model drift
- **"I can't reproduce last month's results"** → Version control for data, code, and models enables reproducibility

### Core MLOps Practices

| Practice | Purpose | Tools in This Project |
|----------|---------|----------------------|
| **Experiment Tracking** | Log parameters, metrics, and artifacts | MLflow |
| **Data Versioning** | Track changes to datasets | DVC |
| **CI/CD Pipelines** | Automate testing and deployment | GitHub Actions |
| **Containerization** | Package code with dependencies | Docker |
| **Model Serving** | Expose models via APIs | FastAPI |

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
│       ├── steps/           # Individual pipeline steps (ingest, clean, train, predict)
│       └── dataset.py       # Data generation and loading utilities
├── tests/                   # Unit and integration tests
├── app.py                   # FastAPI application for model serving
├── config.yml               # Configuration parameters (paths, hyperparameters)
├── data.dvc                 # DVC configuration for data versioning
├── Dockerfile               # Container definition for reproducibility
├── main.py                  # Entry point for running the training pipeline
├── Makefile                 # Utility commands for build, test, and run
├── pyproject.toml           # Python packaging and tool configuration
├── requirements.txt         # Python dependencies
└── .pre-commit-config.yaml  # Configuration for git hooks (linting/formatting)
```

## Quick Start

```bash
# Install dependencies
make install

# Setup pre-commit hooks (runs checks before commit)
pre-commit install

# Install in development mode (editable)
pip install -e .

# Run training pipeline
make train

# Run tests
make test

# Start FastAPI server
make serve

# Build Docker image
make docker-build
```

## API Usage

The project includes a FastAPI model server.

1. **Start the server**:
   ```bash
   make serve
   ```

2. **Access Swagger UI**:
   Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to test the `predict` endpoint interactively.

3. **Sample Request**:
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
   ```

## Development & Quality Assurance

### Pre-commit Hooks
This project uses `pre-commit` to ensure code quality. Before every commit, the following checks run automatically:
- **Black**: Formats code to standard style.
- **Pylint**: Checks for static errors (scoped to `src/`, `tests/`, `app.py`, `main.py`).
- **Standard checks**: Trailing whitespace, end-of-file fixers, YAML validation.

If a hook fails (e.g., Black reformats a file), simply stage the changes (`git add .`) and commit again.

### Data Version Control (DVC)
Data is managed via DVC. The actual CSV files are not in Git.
- **Pull data**: `dvc pull` (uses local storage in `dvc_storage/` by default).
- **Track new data**: `dvc add data/` then `git add data.dvc`.

## CI/CD

This project includes a GitHub Actions workflow ([.github/workflows/ci.yml](.github/workflows/ci.yml)) that runs on every push and pull request to `main`:

- **Linting**: Checks code quality with Pylint
- **Testing**: Runs pytest to verify pipeline steps and API functionality

## Key Components

- **src/ml_project/**: Contains the core logic of the machine learning pipeline, structured as a Python package.
- **steps/**: Modular functions for each stage of the pipeline (Ingest -> Clean -> Train -> Predict).
- **notebooks/**: A sandbox for data exploration that is kept separate from production code.
- **Dockerfile**: Ensures the project runs in a consistent environment across different machines.
- **mlruns/**: Stores experiment metrics and artifacts locally (can be configured for remote storage).
