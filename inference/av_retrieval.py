import torch
import torch.nn.functional as F

from utils.timers import Timer

from inference.clip4clip_infer import (
    encode_frames,
    encode_text
)

from inference.clap_infer import (
    preprocess_audio
)


def run_av_retrieval(

    clip_processor,
    clip_model,

    clap_model,
    clap_tokenizer,
    clap_extractor,

    segments,

    query,

    visual_weight,
    audio_weight
):

    with Timer() as timer:

        # ==================================
        # TEXT EMBEDDINGS
        # ==================================

        visual_text_features = (
            encode_text(
                clip_processor,
                clip_model,
                [query]
            )
        )

        text_inputs = clap_tokenizer(
            [query],
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():

            audio_text_output = (
                clap_model
                .get_text_features(
                    **text_inputs
                )
            )

        if hasattr(
            audio_text_output,
            "pooler_output"
        ):

            audio_text_features = (
                audio_text_output
                .pooler_output
            )

        else:

            audio_text_features = (
                audio_text_output
            )

        audio_text_features = (
            F.normalize(
                audio_text_features,
                dim=-1
            )
        )

        # ==================================
        # SEGMENT RETRIEVAL
        # ==================================

        results = []

        for segment in segments:

            start_t = segment[
                "start_time"
            ]

            end_t = segment[
                "end_time"
            ]

            frames = segment[
                "frames"
            ]

            waveform = segment[
                "audio"
            ]

            sample_rate = segment[
                "sample_rate"
            ]

            waveform, sample_rate = (
                preprocess_audio(
                    waveform,
                    sample_rate
                )
            )

            # ==========================
            # VISUAL
            # ==========================

            frame_features = (
                encode_frames(
                    clip_processor,
                    clip_model,
                    frames
                )
            )

            num_frames = (
                frame_features
                .shape[0]
            )

            temporal_weights = (
                torch.linspace(
                    0.5,
                    1.5,
                    num_frames
                ).unsqueeze(-1)
            )

            weighted_features = (
                frame_features
                * temporal_weights
            )

            video_embedding = (
                weighted_features
                .sum(
                    dim=0,
                    keepdim=True
                )
            )

            video_embedding = (
                F.normalize(
                    video_embedding,
                    dim=-1
                )
            )

            visual_similarity = (
                torch.matmul(
                    video_embedding,
                    visual_text_features.T
                )
                .squeeze()
                .item()
            )

            # ==========================
            # AUDIO
            # ==========================

            audio_inputs = (
                clap_extractor(
                    raw_speech=waveform,
                    sampling_rate=
                    sample_rate,
                    return_tensors="pt"
                )
            )

            with torch.no_grad():

                audio_output = (
                    clap_model
                    .get_audio_features(
                        **audio_inputs
                    )
                )

            if hasattr(
                audio_output,
                "pooler_output"
            ):

                audio_features = (
                    audio_output
                    .pooler_output
                )

            else:

                audio_features = (
                    audio_output
                )

            audio_features = (
                F.normalize(
                    audio_features,
                    dim=-1
                )
            )

            audio_similarity = (
                torch.matmul(
                    audio_features,
                    audio_text_features.T
                )
                .squeeze()
                .item()
            )

            # ==========================
            # FUSION
            # ==========================

            fused_similarity = (
                (
                    visual_weight
                    * visual_similarity
                )
                +
                (
                    audio_weight
                    * audio_similarity
                )
            )

            results.append({

                "start_time":
                    start_t,

                "end_time":
                    end_t,

                "visual_score":
                    visual_similarity,

                "audio_score":
                    audio_similarity,

                "fused_score":
                    fused_similarity,

                "frames":
                    frames
            })

    results = sorted(
        results,
        key=lambda x:
            -x["fused_score"]
    )

    return {

        "query": query,

        "results": results,

        "time_taken":
            timer.elapsed
    }