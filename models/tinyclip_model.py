from transformers import pipeline

MODEL_ID = (
    "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
)

def load_tinyclip(token):

    pipe = pipeline(
        "zero-shot-image-classification",
        model=MODEL_ID,
        device=-1,
        token=token
    )

    return pipe