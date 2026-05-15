import tempfile
import subprocess

import soundfile as sf
import numpy as np


# ==========================================
# EXTRACT AUDIO FROM VIDEO
# ==========================================

def extract_audio_from_video(
    video_path
):

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as tmp:

        audio_path = tmp.name

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "48000",
        "-ac",
        "1",
        audio_path
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    waveform, sample_rate = sf.read(
        audio_path,
        always_2d=False
    )

    return waveform, sample_rate


# ==========================================
# SPLIT AUDIO SEGMENTS
# ==========================================

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