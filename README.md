# JAMAL AI Service

FastAPI microservice for Similarity Metric Learning.

## 🚀 Quick Start

### Prerequisites

1. **Model Files** - Pastikan file model sudah ada di folder `model/`:
   - `jamal_metric_learning.h5` - Model TensorFlow
   - `tokenizer.pkl` - Tokenizer

### Option 1: Run dengan Docker Compose (Recommended)

```bash
# Build dan run
docker-compose up -d

# Lihat logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 2: Run dengan Docker Manual

```bash
# Build image
docker build -t jamal-ai-service .

# Run container
docker run -d \
  --name jamal-ai-service \
  -p 8000:8000 \
  -v $(pwd)/model:/app/model:ro \
  jamal-ai-service
```

### Option 3: Run Langsung dengan Python (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run aplikasi
python main.py

# Atau dengan uvicorn (hot reload)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

| Endpoint      | Method | Deskripsi                                      |
| ------------- | ------ | ---------------------------------------------- |
| `/health`     | GET    | Health check status                            |
| `/similarity` | POST   | Cek similarity antara 2 teks                   |
| `/group`      | POST   | Grouping multiple ideas berdasarkan similarity |
| `/docs`       | GET    | Swagger UI documentation                       |

### Contoh Request

**Check Similarity:**

```bash
curl -X POST http://localhost:8000/similarity \
  -H "Content-Type: application/json" \
  -d '{"text1": "belajar coding", "text2": "belajar programming", "threshold": 0.5}'
```

**Group Ideas:**

```bash
curl -X POST http://localhost:8000/group \
  -H "Content-Type: application/json" \
  -d '{"ideas": ["belajar python", "belajar javascript", "main game", "coding tutorial"]}'
```

## ⚙️ Environment Variables

| Variable             | Default                            | Description                    |
| -------------------- | ---------------------------------- | ------------------------------ |
| `MODEL_PATH`         | `./model/jamal_metric_learning.h5` | Path to TensorFlow model       |
| `TOKENIZER_PATH`     | `./model/tokenizer.pkl`            | Path to tokenizer pickle       |
| `GROUPING_THRESHOLD` | `0.3`                              | Default threshold for grouping |
| `MAX_LEN`            | `50`                               | Max sequence length            |

## 🔧 CI/CD Deployment

Repo ini sudah dikonfigurasi untuk auto-deploy ke VPS via GitHub Actions.

### Required GitHub Secrets

| Secret               | Deskripsi                          |
| -------------------- | ---------------------------------- |
| `DOCKERHUB_USERNAME` | Docker Hub username                |
| `DOCKERHUB_TOKEN`    | Docker Hub access token            |
| `VPS_HOST`           | VPS IP address atau domain         |
| `VPS_USERNAME`       | SSH username (biasanya `deployer`) |
| `VPS_SSH_KEY`        | Private SSH key untuk akses VPS    |

### VPS Setup (One-time)

Di VPS, buat folder dan upload model files:

```bash
# Buat folder
mkdir -p /home/deployer/jamal-ai-service/model

# Upload model files (dari local)
scp model/jamal_metric_learning.h5 deployer@YOUR_VPS:/home/deployer/jamal-ai-service/model/
scp model/tokenizer.pkl deployer@YOUR_VPS:/home/deployer/jamal-ai-service/model/
```

## 📁 Project Structure

```
jamal-ai-service/
├── .github/
│   └── workflows/
│       └── deploy.yml      # GitHub Actions CI/CD
├── model/
│   ├── jamal_metric_learning.h5  # Model file (upload after training)
│   └── tokenizer.pkl             # Tokenizer (upload after training)
├── main.py                 # FastAPI application
├── Dockerfile              # Docker image config
├── docker-compose.yml      # Docker Compose config
├── requirements.txt        # Python dependencies
└── export_model.py         # Script untuk export model dari Kaggle
```

## 🤖 Training Model

Model ditraining menggunakan Kaggle notebook. Setelah training:

1. Jalankan `export_for_deployment()` di notebook
2. Download file dari folder `export/`
3. Upload ke folder `model/` di repo ini
