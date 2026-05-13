import numpy as np
import torch
import torch.nn.functional as F

from utils.timers import Timer

from inference.clap_infer import (
    preprocess_audio
)

BATCH_SIZE = 4


def extract_embedding(output):

    if isinstance(
        output,
        torch.Tensor
    ):

        return output

    if hasattr(
        output,
        "audio_embeds"
    ):

        return output.audio_embeds

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
        "Could not extract embedding tensor"
    )


def retrieve_audio_segments(
    model,
    tokenizer,
    extractor,
    waveform,
    sample_rate,
    query,
    segments
):

    waveform, sample_rate = (
        preprocess_audio(
            waveform,
            sample_rate
        )
    )

    with Timer() as timer:

        # ==================================
        # TEXT EMBEDDING
        # ==================================

        text_inputs = tokenizer(
            [query],
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():

            text_output = (
                model.get_text_features(
                    **text_inputs
                )
            )

        text_features = extract_embedding(
            text_output
        )

        text_features = F.normalize(
            text_features,
            dim=-1
        )

        # ==================================
        # AUDIO RETRIEVAL
        # ==================================

        results = []

        for batch_start in range(
            0,
            len(segments),
            BATCH_SIZE
        ):

            batch_segments = segments[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            batch_waveforms = []

            for _, _, segment in (
                batch_segments
            ):

                segment = np.asarray(
                    segment,
                    dtype=np.float32
                )

                if segment.ndim > 1:

                    segment = segment.mean(
                        axis=1
                    )

                batch_waveforms.append(
                    segment
                )

            audio_inputs = extractor(
                raw_speech=batch_waveforms,

                sampling_rate=sample_rate,

                return_tensors="pt",

                truncation="rand_trunc",

                padding="max_length",

                max_length_s=10
            )

            with torch.no_grad():

                audio_output = (
                    model.get_audio_features(
                        **audio_inputs
                    )
                )

            audio_features = (
                extract_embedding(
                    audio_output
                )
            )

            audio_features = F.normalize(
                audio_features,
                dim=-1
            )

            similarities = torch.matmul(
                audio_features,
                text_features.T
            ).squeeze(-1)

            similarities = (
                similarities
                .cpu()
                .numpy()
            )

            if np.isscalar(
                similarities
            ):

                similarities = np.array(
                    [similarities]
                )

            for idx, (
                start_t,
                end_t,
                _
            ) in enumerate(batch_segments):

                results.append(
                    (
                        start_t,
                        end_t,
                        float(
                            similarities[idx]
                        )
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