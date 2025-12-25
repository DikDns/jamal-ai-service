# JAMAL AI Service Dockerfile for Hugging Face Spaces
# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.11-slim

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install system dependencies (as root temporarily)
USER root
RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy requirements first for better caching
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY --chown=user . /app

# Create model directory with correct permissions
RUN mkdir -p /app/model

# Expose port 7860 (required by HF Spaces)
EXPOSE 7860

# Run the application on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
