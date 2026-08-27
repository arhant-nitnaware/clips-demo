import os

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoFeatureExtractor
)
from utils.device import DEVICE

MODEL_ID = "laion/clap-htsat-unfused"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DIR = os.path.join(
    BASE_DIR,
    "models_local",
    "clap"
)



def load_clap(token):

    # ======================================
    # LOCAL PROJECT COPY
    # ======================================

    if os.path.exists(
        LOCAL_DIR
    ):

        print(
            "[INFO] loading CLAP from local directory..."
        )

        model = AutoModel.from_pretrained(
            LOCAL_DIR
        )

        model = model.to(DEVICE)
        
        tokenizer = (
            AutoTokenizer.from_pretrained(
                LOCAL_DIR
            )
        )

        extractor = (
            AutoFeatureExtractor
            .from_pretrained(
                LOCAL_DIR
            )
        )

        model.eval()

        print(
            "[INFO] CLAP loaded from local directory"
        )

        return (
            model,
            tokenizer,
            extractor
        )

    # ======================================
    # HF CACHE
    # ======================================

    try:

        print(
            "[INFO] loading CLAP from HF cache..."
        )

        model = AutoModel.from_pretrained(
            MODEL_ID,
            local_files_only=True
        )

        model = model.to(DEVICE)

        tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_ID,
                local_files_only=True
            )
        )

        extractor = (
            AutoFeatureExtractor
            .from_pretrained(
                MODEL_ID,
                local_files_only=True
            )
        )

        print(
            "[INFO] CLAP loaded from cache"
        )

        model.eval()

        return (
            model,
            tokenizer,
            extractor
        )

    except Exception:

        print(
            "[INFO] cache miss"
        )

    # ======================================
    # ONLINE DOWNLOAD
    # ======================================

    try:

        print(
            "[INFO] downloading CLAP..."
        )

        model = AutoModel.from_pretrained(
            MODEL_ID,
            token=token
        )

        model = model.to(DEVICE)

        tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_ID,
                token=token
            )
        )

        extractor = (
            AutoFeatureExtractor
            .from_pretrained(
                MODEL_ID,
                token=token
            )
        )

        os.makedirs(
            LOCAL_DIR,
            exist_ok=True
        )

        model.save_pretrained(
            LOCAL_DIR
        )

        tokenizer.save_pretrained(
            LOCAL_DIR
        )

        extractor.save_pretrained(
            LOCAL_DIR
        )

        print(
            "[INFO] CLAP downloaded and saved locally"
        )

        model.eval()

        return (
            model,
            tokenizer,
            extractor
        )

    except Exception:

        print(
            f"[ERROR] unable to load '{MODEL_ID}'"
        )

        raise