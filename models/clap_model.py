from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoFeatureExtractor
)

MODEL_ID = "laion/clap-htsat-unfused"

def load_clap(token):

    model = AutoModel.from_pretrained(
        MODEL_ID,
        token=token
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=token
    )

    extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_ID,
        token=token
    )

    model.eval()

    return model, tokenizer, extractor