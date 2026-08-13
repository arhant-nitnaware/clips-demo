import os

from transformers import (
    CLIPModel,
    CLIPProcessor
)
from utils.device import DEVICE

MODEL_ID = (
    "openai/clip-vit-base-patch32"
)

LOCAL_DIR = os.path.join(
    "/mnt/E/UG/cdac/implementation(s)/inference/decoupled/models_local",
    "clip"
)




def load_clip(token):

    # ======================================
    # LOCAL PROJECT COPY
    # ======================================

    if os.path.exists(
        LOCAL_DIR
    ):

        print(
            "[INFO] loading CLIP from local directory..."
        )

        processor = (
            CLIPProcessor.from_pretrained(
                LOCAL_DIR
            )
        )

        model = (
            CLIPModel.from_pretrained(
                LOCAL_DIR
            )
        )

        model = model.to(DEVICE)

        model.eval()

        print(
            "[INFO] CLIP loaded from local directory"
        )

        return model, processor

    # ======================================
    # HF CACHE
    # ======================================

    try:

        print(
            "[INFO] loading CLIP from HF cache..."
        )

        processor = (
            CLIPProcessor.from_pretrained(
                MODEL_ID,
                local_files_only=True
            )
        )

        model = (
            CLIPModel.from_pretrained(
                MODEL_ID,
                local_files_only=True
            )
        )

        model = model.to(DEVICE)

        print(
            "[INFO] CLIP loaded from cache"
        )

        model.eval()

        return model, processor

    except Exception:

        print(
            "[INFO] cache miss"
        )

    # ======================================
    # ONLINE DOWNLOAD
    # ======================================

    try:

        print(
            "[INFO] downloading CLIP..."
        )

        processor = (
            CLIPProcessor.from_pretrained(
                MODEL_ID,
                token=token
            )
        )

        model = (
            CLIPModel.from_pretrained(
                MODEL_ID,
                token=token
            )
        )

        model = model.to(DEVICE)

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
            "[INFO] CLIP downloaded and saved locally"
        )

        model.eval()

        return model, processor

    except Exception:

        print(
            f"[ERROR] unable to load '{MODEL_ID}'"
        )

        raise