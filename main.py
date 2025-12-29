"""
JAMAL AI Service - Similarity Metric Learning API
FastAPI microservice for semantic similarity detection.
"""

# CRITICAL: Set environment variables BEFORE importing TensorFlow
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import sys
import numpy as np
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K

# --- Configuration ---
# SavedModel format uses folder path, not .h5 file
MODEL_PATH = os.getenv("MODEL_PATH", "./model/jamal_model")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "./model/tokenizer.pkl")
MAX_LEN = int(os.getenv("MAX_LEN", "50"))
DEFAULT_THRESHOLD = float(os.getenv("GROUPING_THRESHOLD", "0.3"))
MARGIN = 1.0

# --- Custom Functions (for backward compatibility) ---
def contrastive_loss(y_true, y_pred):
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(MARGIN - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# Global variables
model = None
tokenizer = None
model_loading = False
model_load_error = None

def load_tokenizer_sync():
    global tokenizer
    print("Loading tokenizer...", flush=True)
    try:
        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH, 'rb') as f:
                tokenizer = pickle.load(f)
            print(f"Tokenizer loaded from {TOKENIZER_PATH}", flush=True)
            return True
        else:
            print(f"Tokenizer not found at {TOKENIZER_PATH}", flush=True)
            return False
    except Exception as e:
        print(f"Error loading tokenizer: {e}", flush=True)
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_loading, model_load_error
    print("=== JAMAL AI Service Starting ===", flush=True)
    print(f"Python: {sys.version}", flush=True)
    print(f"TensorFlow: {tf.__version__}", flush=True)
    print(f"Model path: {MODEL_PATH}", flush=True)
    print(f"Model exists: {os.path.exists(MODEL_PATH)}", flush=True)
    
    # List model directory contents
    if os.path.exists(MODEL_PATH):
        print(f"Model contents: {os.listdir(MODEL_PATH)}", flush=True)
    
    load_tokenizer_sync()
    
    print("Loading SavedModel...", flush=True)
    model_loading = True
    try:
        if os.path.exists(MODEL_PATH):
            # SavedModel format - no custom_objects needed!
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print(f"Model loaded successfully!", flush=True)
            print(f"Model parameters: {model.count_params():,}", flush=True)
        else:
            model_load_error = f"Model not found at {MODEL_PATH}"
            print(f"ERROR: {model_load_error}", flush=True)
    except Exception as e:
        model_load_error = str(e)
        print(f"ERROR loading model: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        model_loading = False
    
    if model:
        print("=== Ready to serve requests ===", flush=True)
    
    yield
    print("Shutting down...", flush=True)

app = FastAPI(
    title="JAMAL AI Service",
    description="Similarity Metric Learning API for brainstorming idea grouping",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Pydantic Models ---
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

class GroupResponse(BaseModel):
    groups: dict
    n_groups: int
    threshold_used: float
    distance_matrix: List[List[float]]

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_loading: bool
    tokenizer_loaded: bool
    error: Optional[str] = None

# --- Helper Functions ---
def preprocess_text(texts: List[str]) -> np.ndarray:
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

def compute_distance(text1: str, text2: str) -> float:
    if model_loading:
        raise HTTPException(status_code=503, detail="Model still loading, please retry")
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {model_load_error}")
    pad1 = preprocess_text([text1])
    pad2 = preprocess_text([text2])
    return float(model.predict([pad1, pad2], verbose=0)[0][0])

def compute_pairwise_distances(ideas: List[str]) -> np.ndarray:
    if model_loading:
        raise HTTPException(status_code=503, detail="Model still loading, please retry")
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {model_load_error}")
    n = len(ideas)
    padded = preprocess_text(ideas)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = model.predict([padded[i:i+1], padded[j:j+1]], verbose=0)[0][0]
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix

def find_groups(ideas: List[str], distance_matrix: np.ndarray, threshold: float):
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
    
    result = {}
    group_counter = 0
    ungrouped = []
    
    for group in groups:
        if len(group) > 1:
            group_counter += 1
            result[f"group_{group_counter}"] = [{"index": i, "text": ideas[i]} for i in group]
        else:
            ungrouped.extend(group)
    
    if ungrouped:
        result["ungrouped"] = [{"index": i, "text": ideas[i]} for i in ungrouped]
    
    return result, group_counter

# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok" if model and not model_load_error else "error",
        model_loaded=model is not None,
        model_loading=model_loading,
        tokenizer_loaded=tokenizer is not None,
        error=model_load_error
    )

@app.get("/debug")
async def debug_info():
    model_dir = os.path.dirname(MODEL_PATH) or "./model"
    return {
        "cwd": os.getcwd(),
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_contents": os.listdir(MODEL_PATH) if os.path.exists(MODEL_PATH) else [],
        "tokenizer_exists": os.path.exists(TOKENIZER_PATH),
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "error": model_load_error,
        "tf_version": tf.__version__,
    }

@app.post("/similarity", response_model=SimilarityResponse)
async def check_similarity(request: SimilarityRequest):
    distance = compute_distance(request.text1, request.text2)
    return SimilarityResponse(
        distance=distance,
        is_similar=distance < request.threshold,
        threshold_used=request.threshold
    )

@app.post("/group", response_model=GroupResponse)
async def group_ideas(request: GroupRequest):
    if len(request.ideas) == 0:
        return GroupResponse(groups={}, n_groups=0, threshold_used=0, distance_matrix=[])
    
    distance_matrix = compute_pairwise_distances(request.ideas)
    all_distances = distance_matrix[np.triu_indices(len(request.ideas), k=1)]
    
    if request.threshold is not None:
        threshold = request.threshold
    elif len(all_distances) > 0:
        threshold = min(np.percentile(all_distances, 25), DEFAULT_THRESHOLD)
    else:
        threshold = DEFAULT_THRESHOLD
    
    groups, n_groups = find_groups(request.ideas, distance_matrix, threshold)
    
    return GroupResponse(
        groups=groups,
        n_groups=n_groups,
        threshold_used=threshold,
        distance_matrix=distance_matrix.tolist()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
