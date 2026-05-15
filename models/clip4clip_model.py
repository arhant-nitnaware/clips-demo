from transformers import (
    AutoProcessor,
    AutoModel
)

MODEL_ID = (
    "Searchium-ai/clip4clip-webvid150k"
)


def load_clip4clip(token):

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

    model.eval()

    return processor, model