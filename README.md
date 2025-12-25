# JAMAL AI Service

FastAPI microservice for Similarity Metric Learning.

## Quick Start

```bash
# Build
docker build -t jamal-ai -f Dockerfile.ai .

# Run
docker run -p 8000:8000 jamal-ai
```

## Endpoints

- `POST /similarity` - Check similarity between 2 texts
- `POST /group` - Group multiple ideas by similarity
- `GET /health` - Health check

## Environment Variables

| Variable             | Default                            | Description                    |
| -------------------- | ---------------------------------- | ------------------------------ |
| `MODEL_PATH`         | `./model/jamal_metric_learning.h5` | Path to model file             |
| `TOKENIZER_PATH`     | `./model/tokenizer.pkl`            | Path to tokenizer              |
| `GROUPING_THRESHOLD` | `0.3`                              | Default threshold for grouping |
