from transformers import (
    CLIPModel,
    CLIPProcessor
)

MODEL_ID = "openai/clip-vit-base-patch32"


def load_clip(token):

    processor = (
        CLIPProcessor
        .from_pretrained(
            MODEL_ID,
            token=token
        )
    )

    model = (
        CLIPModel
        .from_pretrained(
            MODEL_ID,
            token=token
        )
    )

    model.eval()

    return model, processor