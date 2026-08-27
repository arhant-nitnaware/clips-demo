import os
import torch
import gc
from utils.device import DEVICE
from models.clip_model import load_clip
from models.clip4clip_model import load_clip4clip
from models.clap_model import load_clap
from models.tinyclip_model import load_tinyclip

# Global cache for loaded model instances
_models = {
    "clip": None,      # (model, processor)
    "clip4clip": None, # (processor, model)
    "clap": None,      # (model, tokenizer, extractor)
    "tinyclip": None,  # (pipe, model, processor)
}

def get_hf_key():
    """Helper to retrieve Hugging Face API key/token from Streamlit secrets or environment."""
    try:
        import streamlit as st
        if "HF_KEY" in st.secrets and st.secrets["HF_KEY"]:
            return st.secrets["HF_KEY"].strip() or None
    except Exception:
        pass
    key = os.environ.get("HF_KEY", "").strip()
    return key if key else None


def initialize_models():
    """Load all required models into cache. Typically called on startup."""
    token = get_hf_key()
    get_clip(token)
    get_clip4clip(token)
    get_clap(token)
    get_tinyclip(token)

def get_clip(token: str = None):
    """Retrieve CLIP model and processor, loading them if not cached."""
    global _models
    if _models["clip"] is None:
        if token is None:
            token = get_hf_key()
        _models["clip"] = load_clip(token)
    return _models["clip"]

def get_clip4clip(token: str = None):
    """Retrieve CLIP4Clip model and processor, loading them if not cached."""
    global _models
    if _models["clip4clip"] is None:
        if token is None:
            token = get_hf_key()
        _models["clip4clip"] = load_clip4clip(token)
    return _models["clip4clip"]

def get_clap(token: str = None):
    """Retrieve CLAP model, tokenizer, and extractor, loading them if not cached."""
    global _models
    if _models["clap"] is None:
        if token is None:
            token = get_hf_key()
        _models["clap"] = load_clap(token)
    return _models["clap"]

def get_tinyclip(token: str = None):
    """Retrieve TinyCLIP pipe, model, and processor, loading them if not cached."""
    global _models
    if _models["tinyclip"] is None:
        if token is None:
            token = get_hf_key()
        _models["tinyclip"] = load_tinyclip(token)
    return _models["tinyclip"]

def get_av_models(token: str = None):
    """Retrieve CLIP4Clip and CLAP models which are needed for A+V retrieval."""
    return get_clip4clip(token), get_clap(token)

def unload_model_instance(name: str):
    """Remove model from cache and trigger garbage collection/CUDA cache clearing."""
    global _models
    if name in _models:
        _models[name] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
