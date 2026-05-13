import numpy as np
import torch
import librosa

from utils.timers import Timer

TARGET_SR = 48000

def preprocess_audio(
    waveform,
    sample_rate
):

    if waveform.ndim > 1:

        waveform = waveform.mean(axis=1)

    waveform = waveform.astype(np.float32)

    if sample_rate != TARGET_SR:

        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=TARGET_SR
        )

        sample_rate = TARGET_SR

    return waveform, sample_rate


def extract_embedding(output):

    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "audio_embeds"):
        return output.audio_embeds

    if hasattr(output, "text_embeds"):
        return output.text_embeds

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.mean(dim=1)

    raise ValueError(
        "Could not extract embedding tensor"
    )


def run_clap(
    model,
    tokenizer,
    extractor,
    waveform,
    sample_rate,
    texts
):

    waveform, sample_rate = preprocess_audio(
        waveform,
        sample_rate
    )

    with Timer() as timer:

        audio_inputs = extractor(
            raw_speech=waveform,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        text_inputs = tokenizer(
            texts,
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():

            audio_output = (
                model.get_audio_features(
                    **audio_inputs
                )
            )

            text_output = (
                model.get_text_features(
                    **text_inputs
                )
            )

        audio_feat = extract_embedding(
            audio_output
        )

        text_feat = extract_embedding(
            text_output
        )

        audio_norm = (
            audio_feat /
            audio_feat.norm(
                dim=-1,
                keepdim=True
            )
        )

        text_norm = (
            text_feat /
            text_feat.norm(
                dim=-1,
                keepdim=True
            )
        )

        sims = (
            audio_norm @ text_norm.T
        ).squeeze(0)

        probs = sims.softmax(
            dim=-1
        ).tolist()

    ranked = sorted(
        zip(texts, probs),
        key=lambda x: -x[1]
    )

    return {
        "time_taken": timer.elapsed,
        "results": ranked,
        "input_details": {
            "original_sample_rate":
                sample_rate,

            "processed_sample_rate":
                TARGET_SR,

            "duration_seconds":
                len(waveform) / TARGET_SR,

            "num_texts":
                len(texts)
        }
    }