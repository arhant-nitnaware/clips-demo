import os

from transformers import (
    AutoProcessor,
    AutoModel
)

MODEL_ID = (
    "Searchium-ai/clip4clip-webvid150k"
)

LOCAL_DIR = os.path.join(
    "models_local",
    "clip4clip"
)


def load_clip4clip(token):

    # ======================================
    # LOCAL PROJECT COPY
    # ======================================

    if os.path.exists(
        LOCAL_DIR
    ):

        print(
            "[INFO] loading Clip4Clip from local directory..."
        )

        processor = (
            AutoProcessor
            .from_pretrained(
                LOCAL_DIR
            )
        )

        model = (
            AutoModel
            .from_pretrained(
                LOCAL_DIR
            )
        )

        model.eval()

        print(
            "[INFO] Clip4Clip loaded from local directory"
        )

        return processor, model

    # ======================================
    # HF CACHE
    # ======================================

    try:

        print(
            "[INFO] loading Clip4Clip from HF cache..."
        )

        processor = (
            AutoProcessor
            .from_pretrained(
                MODEL_ID,
                local_files_only=True
            )
        )

        model = (
            AutoModel
            .from_pretrained(
                MODEL_ID,
                local_files_only=True
            )
        )

        print(
            "[INFO] Clip4Clip loaded from cache"
        )

        model.eval()

        return processor, model

    except Exception:

        print(
            "[INFO] cache miss"
        )

    # ======================================
    # ONLINE DOWNLOAD
    # ======================================

    try:

        print(
            "[INFO] downloading Clip4Clip..."
        )

        processor = (
            AutoProcessor
            .from_pretrained(
                MODEL_ID,
                token=token
            )
        )

        model = (
            AutoModel
            .from_pretrained(
                MODEL_ID,
                token=token
            )
        )

        os.makedirs(
            LOCAL_DIR,
            exist_ok=True
        )

        processor.save_pretrained(
            LOCAL_DIR
        )

        model.save_pretrained(
            LOCAL_DIR
        )

        print(
            "[INFO] Clip4Clip downloaded and saved locally"
        )

        model.eval()

        return processor, model

    except Exception:

        print(
            f"[ERROR] unable to load '{MODEL_ID}'"
        )

        raise