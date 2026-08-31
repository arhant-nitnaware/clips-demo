import soundfile as sf
import numpy as np
import av
import librosa

# ==========================================
# EXTRACT AUDIO FROM VIDEO
# ==========================================
TARGET_SR = 48000


def extract_audio_from_video(video_path):
    container = av.open(video_path)
    try:
        audio_stream = next(
            s for s in container.streams
            if s.type == "audio"
        )
    except StopIteration:
        container.close()
        raise ValueError("No audio stream found in the video file.")

    samples = []
    try:
        for frame in container.decode(audio_stream):
            arr = frame.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            samples.append(arr)
    finally:
        container.close()

    if not samples:
        raise ValueError("Audio stream is present but contains no audio frames.")

    waveform = np.concatenate(
        samples
    ).astype(np.float32)

    original_sr = audio_stream.rate

    if original_sr != TARGET_SR:
        waveform = librosa.resample(
            waveform,
            orig_sr=original_sr,
            target_sr=TARGET_SR
        )

    return waveform, TARGET_SR


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
    if segment_size <= 0:
        segment_size = len(waveform)

    segments = []

    for start in range(
        0,
        len(waveform),
        segment_size
    ):

        end = min(start + segment_size, len(waveform))

        segment = waveform[start:end]

        if len(segment) < segment_size and len(segments) > 0:
            continue

        segments.append(
            (
                start / sample_rate,
                end / sample_rate,
                segment
            )
        )

    if not segments and len(waveform) > 0:
        segments.append(
            (
                0.0,
                len(waveform) / sample_rate,
                waveform
            )
        )

    return segments