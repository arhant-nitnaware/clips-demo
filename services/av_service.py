import os
from models import model_manager
from inference.av_retrieval import run_av_retrieval
from utils.video_utils import extract_segment_frames, get_video_duration
from utils.audio_utils import extract_audio_from_video

def run_av_retrieval_service(
    query: str,
    video_path: str,
    visual_weight: float = 0.5,
    audio_weight: float = 0.5,
    segment_seconds: float = 3.0,
    max_frames: int = 8
) -> dict:
    """Run fused Audio+Video retrieval on a video file and return JSON-serializable results."""
    # Retrieve cached models from model manager
    (clip_processor, clip_model), (clap_model, clap_tokenizer, clap_extractor) = model_manager.get_av_models()
    
    # Extract audio waveform and sample rate
    waveform, sample_rate = extract_audio_from_video(video_path)
    duration = get_video_duration(video_path)
    
    # Segment video and audio synchronously
    segments = []
    current_t = 0.0
    while current_t < duration:
        end_t = min(current_t + segment_seconds, duration)
        frames = extract_segment_frames(
            video_path,
            current_t,
            end_t,
            max_frames=max_frames
        )
        start_sample = int(current_t * sample_rate)
        end_sample = int(end_t * sample_rate)
        audio_segment = waveform[start_sample:end_sample]
        
        if len(frames) == 0:
            current_t += segment_seconds
            continue
            
        segments.append({
            "start_time": current_t,
            "end_time": end_t,
            "frames": frames,
            "audio": audio_segment,
            "sample_rate": sample_rate
        })
        current_t += segment_seconds

    if not segments:
        raise ValueError("Could not extract any segments from the video file.")
        
    # Run AV retrieval
    result = run_av_retrieval(
        clip_processor,
        clip_model,
        clap_model,
        clap_tokenizer,
        clap_extractor,
        segments,
        query,
        visual_weight,
        audio_weight
    )
    
    # Format results to be JSON-serializable (strip the frame images)
    formatted_results = [
        {
            "start_time": float(r["start_time"]),
            "end_time": float(r["end_time"]),
            "visual_score": float(r["visual_score"]),
            "audio_score": float(r["audio_score"]),
            "fused_score": float(r["fused_score"])
        }
        for r in result["results"]
    ]
    
    return {
        "query": result["query"],
        "time_taken": float(result["time_taken"]),
        "results": formatted_results
    }
