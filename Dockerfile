FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY app.py .
COPY main.py .
COPY config.yml .

# Set python path to include src
ENV PYTHONPATH="/app/src"

EXPOSE 8000

CMD ["python", "main.py"]
