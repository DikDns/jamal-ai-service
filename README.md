---
title: JAMAL AI Service
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# JAMAL AI Service

FastAPI microservice for Similarity Metric Learning - semantic similarity detection for brainstorming idea grouping.

## API Endpoints

| Endpoint      | Method | Description                        |
| ------------- | ------ | ---------------------------------- |
| `/health`     | GET    | Health check                       |
| `/similarity` | POST   | Check similarity between 2 texts   |
| `/group`      | POST   | Group multiple ideas by similarity |
| `/docs`       | GET    | Swagger UI documentation           |

## Usage

**Check Similarity:**

```bash
curl -X POST https://dikdns-jamal-ai-service.hf.space/similarity \
  -H "Content-Type: application/json" \
  -d '{"text1": "belajar coding", "text2": "belajar programming"}'
```

**Group Ideas:**

```bash
curl -X POST https://dikdns-jamal-ai-service.hf.space/group \
  -H "Content-Type: application/json" \
  -d '{"ideas": ["belajar python", "belajar javascript", "main game"]}'
```
