import numpy as np

def split_audio_segments(
    waveform,
    sample_rate,
    segment_seconds=2
):

    segment_size = int(
        sample_rate * segment_seconds
    )

    segments = []

    for start in range(
        0,
        len(waveform),
        segment_size
    ):

        end = start + segment_size

        segment = waveform[start:end]

        if len(segment) < segment_size:
            continue

        segments.append(
            (
                start / sample_rate,
                end / sample_rate,
                segment
            )
        )

    return segments