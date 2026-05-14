from transformers import (
    AutoModel,
    AutoProcessor,
    pipeline
)

MODEL_ID = (
    "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
)


def load_tinyclip(token):

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

    pipe = pipeline(
        task="zero-shot-image-classification",
        model=model,
        image_processor=processor,
        tokenizer=processor.tokenizer
    )

    model.eval()

    return (
        pipe,
        model,
        processor
    )