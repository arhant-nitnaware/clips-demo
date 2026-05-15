import torch
import torch.nn.functional as F
import numpy as np

from utils.timers import Timer


def encode_frames(
    processor,
    model,
    frames
):

    inputs = processor(
        images=frames,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        image_features = (
            model.get_image_features(
                **inputs
            )
        )

    # compatibility handling
    if hasattr(image_features, "pooler_output"):

        image_features = (
            image_features.pooler_output
        )

    image_features = F.normalize(
        image_features,
        dim=-1
    )

    return image_features


def encode_text(
    processor,
    model,
    texts
):

    text_inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        text_features = (
            model.get_text_features(
                **text_inputs
            )
        )

    if hasattr(text_features, "pooler_output"):

        text_features = (
            text_features.pooler_output
        )

    text_features = F.normalize(
        text_features,
        dim=-1
    )

    return text_features


# =====================================================
# VIDEO LABELING
# =====================================================

def classify_video(
    processor,
    model,
    frames,
    labels
):

    with Timer() as timer:

        frame_features = encode_frames(
            processor,
            model,
            frames
        )

        # =====================================
        # TEMPORAL POSITIONAL WEIGHTING
        # =====================================

        num_frames = (
            frame_features.shape[0]
        )

        temporal_weights = torch.linspace(
            0.5,
            1.5,
            num_frames
        ).unsqueeze(-1)

        weighted_features = (
            frame_features
            * temporal_weights
        )

        video_embedding = (
            weighted_features.sum(
                dim=0,
                keepdim=True
            )
        )

        video_embedding = F.normalize(
            video_embedding,
            dim=-1
        )

        text_features = encode_text(
            processor,
            model,
            labels
        )

        similarities = torch.matmul(
            text_features,
            video_embedding.T
        ).squeeze()

        scores = similarities.cpu().tolist()

    results = sorted(
        zip(labels, scores),
        key=lambda x: -x[1]
    )

    return {
        "time_taken": timer.elapsed,

        "results": results,

        "input_details": {
            "num_frames_used":
                len(frames),

            "frame_shape":
                np.array(frames[0]).shape,

            "num_labels":
                len(labels),

            "embedding_strategy":
                "mean pooled temporal embedding"
        }
    }


# =====================================================
# FRAME RETRIEVAL
# =====================================================

def query_video(
    processor,
    model,
    frames,
    query
):

    with Timer() as timer:

        frame_features = encode_frames(
            processor,
            model,
            frames
        )

        text_features = encode_text(
            processor,
            model,
            [query]
        )

        similarities = torch.matmul(
            frame_features,
            text_features.T
        ).squeeze()

        scores = similarities.cpu().tolist()

    ranked_frames = sorted(
        list(enumerate(scores)),
        key=lambda x: -x[1]
    )

    return {
        "time_taken": timer.elapsed,

        "query": query,

        "frame_scores":
            ranked_frames,

        "input_details": {
            "num_frames_used":
                len(frames),

            "frame_shape":
                np.array(frames[0]).shape,

            "retrieval_strategy":
                "framewise cosine similarity"
        }
    }