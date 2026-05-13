from transformers import (
    AutoProcessor,
    CLIPModel
)

MODEL_ID = "Searchium-ai/clip4clip-webvid150k"

def load_clip4clip(token):

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        token=token
    )

    model = CLIPModel.from_pretrained(
        MODEL_ID,
        token=token
    )

    model.eval()

    return processor, model