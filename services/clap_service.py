import os
import soundfile as sf
from models import model_manager
from inference.clap_retrieval import retrieve_audio_segments
from inference.clap_infer import run_clap
from utils.audio_utils import extract_audio_from_video, split_audio_segments

def run_clap_retrieval(query: str, video_path: str, segment_seconds: float = 5.0, top_k: int = 4) -> dict:
    """Run CLAP audio retrieval on video or audio file and format results to be JSON-serializable."""
    model, tokenizer, extractor = model_manager.get_clap()
    
    # Load audio waveform
    if video_path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
        waveform, sample_rate = sf.read(video_path, always_2d=False)
    else:
        waveform, sample_rate = extract_audio_from_video(video_path)
        
    segments = split_audio_segments(waveform, sample_rate, segment_seconds)
    if not segments:
        raise ValueError("Could not extract any audio segments from the input file.")
        
    result = retrieve_audio_segments(
        model,
        tokenizer,
        extractor,
        waveform,
        sample_rate,
        query,
        segments
    )
    
    import io
    import base64
    import numpy as np
 
    def audio_to_b64(segment_waveform, sr) -> str:
        segment_audio = np.asarray(segment_waveform, dtype=np.float32)
        if segment_audio.ndim > 1:
            segment_audio = segment_audio.mean(axis=1)
        max_val = np.abs(segment_audio).max()
        if max_val > 1.0:
            segment_audio = segment_audio / max_val
        
        buffered = io.BytesIO()
        sf.write(buffered, segment_audio, sr, format="WAV")
        audio_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:audio/wav;base64,{audio_str}"
    
    # Format results, adding base64 audio only to the top_k
    formatted_results = []
    for idx, r in enumerate(result["results"]):
        start_time = float(r[0])
        end_time = float(r[1])
        score = float(r[2])
        
        b64_audio = ""
        if idx < top_k:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_waveform = waveform[start_sample:end_sample]
            b64_audio = audio_to_b64(segment_waveform, sample_rate)
            
        formatted_results.append({
            "start_time": start_time,
            "end_time": end_time,
            "score": score,
            "audio": b64_audio
        })
        
    return {
        "query": result["query"],
        "time_taken": float(result["time_taken"]),
        "results": formatted_results
    }

def run_clap_labeling(video_path: str, labels: list[str]) -> dict:
    """Run CLAP audio labeling classification on video/audio files."""
    model, tokenizer, extractor = model_manager.get_clap()
    
    if video_path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
        waveform, sample_rate = sf.read(video_path, always_2d=False)
    else:
        waveform, sample_rate = extract_audio_from_video(video_path)
        
    result = run_clap(
        model=model,
        tokenizer=tokenizer,
        extractor=extractor,
        waveform=waveform,
        sample_rate=sample_rate,
        texts=labels
    )
    
    formatted_results = [
        {
            "label": str(r[0]),
            "score": float(r[1])
        }
        for r in result["results"]
    ]
    return {
        "time_taken": float(result["time_taken"]),
        "results": formatted_results
    }
