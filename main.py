"""
JAMAL AI Service - Similarity Metric Learning API
FastAPI microservice for semantic similarity detection.
"""

import os
import pickle
import numpy as np
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K

# --- Configuration ---
MODEL_PATH = os.getenv("MODEL_PATH", "./model/jamal_metric_learning.h5")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "./model/tokenizer.pkl")
MAX_LEN = int(os.getenv("MAX_LEN", "50"))
DEFAULT_THRESHOLD = float(os.getenv("GROUPING_THRESHOLD", "0.3"))
MARGIN = 1.0

# --- Custom Functions (needed for model loading) ---


def contrastive_loss(y_true, y_pred):
    """Contrastive Loss for Metric Learning."""
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(MARGIN - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def l2_norm(x):
    """L2 Normalization for embedding vectors."""
    return K.l2_normalize(x, axis=1)


# --- Global variables ---
model = None
tokenizer = None
model_loading = False

# --- Lazy Loading Functions ---


def ensure_model_loaded():
    """Lazy load model on first request."""
    global model, model_loading

    if model is not None:
        return True

    if model_loading:
        return False  # Loading in progress

    model_loading = True
    print("🔄 Lazy loading model...")

    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={
                    'contrastive_loss': contrastive_loss,
                    'l2_norm': l2_norm,
                    'K': K
                },
                compile=False  # Skip compilation for faster loading
            )
            print(f"✅ Model loaded from {MODEL_PATH}")
            return True
        else:
            print(f"⚠️ Model not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_loading = False
        return False


def ensure_tokenizer_loaded():
    """Lazy load tokenizer on first request."""
    global tokenizer

    if tokenizer is not None:
        return True

    print("🔄 Lazy loading tokenizer...")

    try:
        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH, 'rb') as f:
                tokenizer = pickle.load(f)
            print(f"✅ Tokenizer loaded from {TOKENIZER_PATH}")
            return True
        else:
            print(f"⚠️ Tokenizer not found at {TOKENIZER_PATH}")
            return False
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return False


# --- Lifespan (startup/shutdown) ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fast startup - no model loading here
    print("🚀 Server starting (lazy loading enabled)...")
    print(f"📁 Model path: {MODEL_PATH}")
    print(f"📁 Tokenizer path: {TOKENIZER_PATH}")

    yield

    print("👋 Shutting down...")

# --- FastAPI App ---
app = FastAPI(
    title="JAMAL AI Service",
    description="Similarity Metric Learning API for brainstorming idea grouping",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response Models ---


class SimilarityRequest(BaseModel):
    text1: str
    text2: str
    threshold: Optional[float] = 0.5


class SimilarityResponse(BaseModel):
    distance: float
    is_similar: bool
    threshold_used: float


class GroupRequest(BaseModel):
    ideas: List[str]
    threshold: Optional[float] = None


class GroupItem(BaseModel):
    index: int
    text: str


class GroupResponse(BaseModel):
    groups: dict
    n_groups: int
    threshold_used: float
    distance_matrix: List[List[float]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    tokenizer_loaded: bool

# --- Helper Functions ---


def preprocess_text(texts: List[str]) -> np.ndarray:
    """Convert texts to padded sequences."""
    # Trigger lazy loading
    if not ensure_tokenizer_loaded():
        raise HTTPException(
            status_code=503, detail="Tokenizer still loading, please retry")

    if tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer not loaded")

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
    return padded


def compute_distance(text1: str, text2: str) -> float:
    """Compute distance between two texts."""
    # Trigger lazy loading
    if not ensure_model_loaded():
        raise HTTPException(
            status_code=503, detail="Model still loading, please retry in a few minutes")

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    pad1 = preprocess_text([text1])
    pad2 = preprocess_text([text2])

    distance = model.predict([pad1, pad2], verbose=0)[0][0]
    return float(distance)


def compute_pairwise_distances(ideas: List[str]) -> np.ndarray:
    """Compute pairwise distance matrix."""
    # Trigger lazy loading
    if not ensure_model_loaded():
        raise HTTPException(
            status_code=503, detail="Model still loading, please retry in a few minutes")

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    n = len(ideas)
    padded = preprocess_text(ideas)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = model.predict(
                [padded[i:i+1], padded[j:j+1]], verbose=0
            )[0][0]
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def find_groups(ideas: List[str], distance_matrix: np.ndarray, threshold: float) -> dict:
    """Find connected components based on distance threshold."""
    n = len(ideas)
    adjacency = distance_matrix < threshold
    np.fill_diagonal(adjacency, False)

    visited = [False] * n
    groups = []

    def dfs(node, group):
        visited[node] = True
        group.append(node)
        for neighbor in range(n):
            if not visited[neighbor] and adjacency[node, neighbor]:
                dfs(neighbor, group)

    for i in range(n):
        if not visited[i]:
            group = []
            dfs(i, group)
            groups.append(group)

    # Format output
    result = {}
    group_counter = 0
    ungrouped = []

    for group in groups:
        if len(group) > 1:
            group_counter += 1
            result[f"group_{group_counter}"] = [
                {"index": i, "text": ideas[i]} for i in group
            ]
        else:
            ungrouped.extend(group)

    if ungrouped:
        result["ungrouped"] = [
            {"index": i, "text": ideas[i]} for i in ungrouped
        ]

    return result, group_counter

# --- API Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        tokenizer_loaded=tokenizer is not None
    )


@app.post("/similarity", response_model=SimilarityResponse)
async def check_similarity(request: SimilarityRequest):
    """Check similarity between two texts."""
    distance = compute_distance(request.text1, request.text2)

    return SimilarityResponse(
        distance=distance,
        is_similar=distance < request.threshold,
        threshold_used=request.threshold
    )


@app.post("/group", response_model=GroupResponse)
async def group_ideas(request: GroupRequest):
    """Group ideas by semantic similarity."""
    if len(request.ideas) == 0:
        return GroupResponse(
            groups={},
            n_groups=0,
            threshold_used=0,
            distance_matrix=[]
        )

    # Compute distances
    distance_matrix = compute_pairwise_distances(request.ideas)

    # Determine threshold
    all_distances = distance_matrix[np.triu_indices(len(request.ideas), k=1)]

    if request.threshold is not None:
        threshold = request.threshold
    else:
        # Adaptive threshold based on distribution
        if len(all_distances) > 0:
            percentile_25 = np.percentile(all_distances, 25)
            threshold = min(percentile_25, DEFAULT_THRESHOLD)
        else:
            threshold = DEFAULT_THRESHOLD

    # Find groups
    groups, n_groups = find_groups(request.ideas, distance_matrix, threshold)

    return GroupResponse(
        groups=groups,
        n_groups=n_groups,
        threshold_used=threshold,
        distance_matrix=distance_matrix.tolist()
    )

# --- Run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
