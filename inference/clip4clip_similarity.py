from inference.clip4clip_infer import (
    classify_video
)


def compute_video_similarity(
    processor,
    model,
    frames,
    prompts
):

    result = classify_video(
        processor,
        model,
        frames,
        prompts
    )

    return result["results"]