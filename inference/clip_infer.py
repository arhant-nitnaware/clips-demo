import torch
import torch.nn.functional as F

from utils.timers import Timer


def extract_embedding(output):

    if isinstance(
        output,
        torch.Tensor
    ):

        return output

    if hasattr(
        output,
        "image_embeds"
    ):

        return output.image_embeds

    if hasattr(
        output,
        "text_embeds"
    ):

        return output.text_embeds

    if hasattr(
        output,
        "pooler_output"
    ):

        return output.pooler_output

    if hasattr(
        output,
        "last_hidden_state"
    ):

        return (
            output
            .last_hidden_state
            .mean(dim=1)
        )

    raise ValueError(
        "Could not extract embeddings"
    )


def run_clip(
    model,
    processor,
    image,
    labels
):

    with Timer() as timer:

        inputs = processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():

            image_output = (
                model.get_image_features(
                    pixel_values=
                    inputs["pixel_values"]
                )
            )

            text_output = (
                model.get_text_features(
                    input_ids=
                    inputs["input_ids"],

                    attention_mask=
                    inputs[
                        "attention_mask"
                    ]
                )
            )

        image_features = (
            extract_embedding(
                image_output
            )
        )

        text_features = (
            extract_embedding(
                text_output
            )
        )

        image_features = F.normalize(
            image_features,
            dim=-1
        )

        text_features = F.normalize(
            text_features,
            dim=-1
        )

        similarities = torch.matmul(
            image_features,
            text_features.T
        ).squeeze(0)

        scores = (
            similarities
            .cpu()
            .numpy()
            .tolist()
        )

    results = sorted(
        zip(labels, scores),
        key=lambda x: -x[1]
    )

    return {
        "time_taken": timer.elapsed,

        "results": results,

        "input_details": {
            "image_size":
                image.size,

            "num_labels":
                len(labels),

            "embedding_model":
                "CLIP ViT-B/32"
        }
    }