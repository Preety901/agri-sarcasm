import hashlib, os, threading
import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "emb_cache")

_model = None
_lock = threading.Lock()

def _get_model():
    global _model
    with _lock:                      # Streamlit reruns run in threads: load only once
        if _model is None:
            from sentence_transformers import SentenceTransformer
            try:
                _model = SentenceTransformer(MODEL_NAME, device="cpu",
                                             model_kwargs={"low_cpu_mem_usage": False})
            except NotImplementedError:   # "meta tensor" error -> plain load
                _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model

def embed(texts, model_name=MODEL_NAME):
    texts = list(texts)
    key = hashlib.md5(("|".join(texts) + model_name).encode()).hexdigest()
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, key + ".npy")
    if os.path.exists(path):
        return np.load(path)
    X = _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    np.save(path, X)
    return X
