import os

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoFeatureExtractor
)

MODEL_ID = "laion/clap-htsat-unfused"

LOCAL_DIR = (
    "models_local/clap"
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