import torch
import torch.nn.functional as F

from PIL import Image

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


def retrieve_tinyclip_images(
    model,
    processor,
    query,
    uploaded_images
):

    with Timer() as timer:

        # ==============================
        # TEXT EMBEDDING
        # ==============================

        text_inputs = processor(
            text=[query],
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():

            text_output = (
                model.get_text_features(
                    **text_inputs
                )
            )

        text_features = (
            extract_embedding(
                text_output
            )
        )

        text_features = F.normalize(
            text_features,
            dim=-1
        )

        # ==============================
        # IMAGE EMBEDDINGS
        # ==============================

        pil_images = []

        image_names = []

        for uploaded in uploaded_images:

            image = Image.open(
                uploaded
            ).convert("RGB")

            pil_images.append(
                image
            )

            image_names.append(
                uploaded.name
            )

        image_inputs = processor(
            images=pil_images,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():

            image_output = (
                model.get_image_features(
                    pixel_values=
                    image_inputs[
                        "pixel_values"
                    ]
                )
            )

        image_features = (
            extract_embedding(
                image_output
            )
        )

        image_features = F.normalize(
            image_features,
            dim=-1
        )

        # ==============================
        # SIMILARITY
        # ==============================

        similarities = torch.matmul(
            image_features,
            text_features.T
        ).squeeze(-1)

        scores = (
            similarities
            .cpu()
            .numpy()
        )

        results = []

        for idx, score in enumerate(
            scores
        ):

            results.append(
                (
                    image_names[idx],
                    pil_images[idx],
                    float(score)
                )
            )

    results.sort(
        key=lambda x: -x[2]
    )

    return {
        "query": query,
        "time_taken": timer.elapsed,
        "results": results
    }