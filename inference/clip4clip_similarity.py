import torch
import torch.nn.functional as F

from utils.timers import Timer

from inference.clip4clip_infer import (
    encode_frames,
    encode_text
)


def compute_video_similarity(
    processor,
    model,
    frames,
    prompt
):

    with Timer() as timer:

        frame_features = encode_frames(
            processor,
            model,
            frames
        )

        # temporal aggregation
        video_embedding = (
            frame_features.mean(
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
            [prompt]
        )

        similarity = torch.matmul(
            video_embedding,
            text_features.T
        ).squeeze()

        score = similarity.item()

    return {
        "prompt": prompt,
        "similarity": score,
        "time_taken": timer.elapsed,
        "num_frames": len(frames)
    }