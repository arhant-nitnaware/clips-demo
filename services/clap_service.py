import os
import soundfile as sf
from models import model_manager
from inference.clap_retrieval import retrieve_audio_segments
from inference.clap_infer import run_clap
from utils.audio_utils import extract_audio_from_video, split_audio_segments

def run_clap_retrieval(query: str, video_path: str, segment_seconds: float = 5.0) -> dict:
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
    
    formatted_results = [
        {
            "start_time": float(r[0]),
            "end_time": float(r[1]),
            "score": float(r[2])
        }
        for r in result["results"]
    ]
    
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
