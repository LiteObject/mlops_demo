.PHONY: install train test lint docker-build docker-run clean serve

# Install dependencies
install:
	pip install -r requirements.txt

# Install package in development mode
dev-install:
	pip install -e .

# Run the training pipeline
train:
	python main.py

# Run unit tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src/ml_project --cov-report=html

# Lint the code
lint:
	flake8 src/ tests/

# Format code
format:
	black src/ tests/

# Build Docker image
docker-build:
	docker build -t mlops_demo .

# Run Docker container
docker-run:
	docker run -p 8000:8000 mlops_demo

# Start the FastAPI server
serve:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Pull data with DVC
dvc-pull:
	dvc pull

# Push data with DVC
dvc-push:
	dvc push

# Clean up artifacts
clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
