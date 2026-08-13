import os

from transformers import (
    AutoModel,
    AutoProcessor,
    pipeline
)

from utils.device import DEVICE

MODEL_ID = (
    "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
)

LOCAL_DIR = os.path.join(
    "/mnt/E/UG/cdac/implementation(s)/inference/decoupled/models_local",
    "tinyclip"
)



def load_tinyclip(token):

    # ======================================
    # LOCAL PROJECT COPY
    # ======================================

    if os.path.exists(
        LOCAL_DIR
    ):

        print(
            "[INFO] loading TinyCLIP from local directory..."
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

        model = model.to(DEVICE)

        pipe = pipeline(
            task="zero-shot-image-classification",
            model=model,
            image_processor=processor,
            tokenizer=processor.tokenizer
        )

        model.eval()

        print(
            "[INFO] TinyCLIP loaded from local directory"
        )

        return (
            pipe,
            model,
            processor
        )

    # ======================================
    # HF CACHE
    # ======================================

    try:

        print(
            "[INFO] loading TinyCLIP from HF cache..."
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

        model = model.to(DEVICE)

        pipe = pipeline(
            task="zero-shot-image-classification",
            model=model,
            image_processor=processor,
            tokenizer=processor.tokenizer
        )

        model.eval()

        print(
            "[INFO] TinyCLIP loaded from cache"
        )

        return (
            pipe,
            model,
            processor
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
            "[INFO] downloading TinyCLIP..."
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

        pipe = pipeline(
            task="zero-shot-image-classification",
            model=model,
            image_processor=processor,
            tokenizer=processor.tokenizer
        )

        model.eval()

        print(
            "[INFO] TinyCLIP downloaded and saved locally"
        )

        return (
            pipe,
            model,
            processor
        )

    except Exception:

        print(
            f"[ERROR] unable to load '{MODEL_ID}'"
        )

        raise