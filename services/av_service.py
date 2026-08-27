import os
from models import model_manager
from inference.av_retrieval import run_av_retrieval
from utils.video_utils import extract_segment_frames, get_video_duration
from utils.audio_utils import extract_audio_from_video
from utils.device import get_device_name

def run_av_retrieval_service(
    query: str,
    video_path: str,
    visual_weight: float = 0.5,
    audio_weight: float = 0.5,
    segment_seconds: float = 5.0,
    max_frames: int = 8,
    top_k: int = 4
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
            "video_name": os.path.basename(video_path),
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
    
    # Format results to be JSON-serializable (strip frame arrays, encode representative frame to base64, keep top_k)
    import io
    import base64
    from PIL import Image

    def frame_to_b64(frame_np) -> str:
        pil_img = Image.fromarray(frame_np)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    formatted_results = []
    for idx, r in enumerate(result["results"]):
        b64_image = ""
        if idx < top_k and r.get("frames") and len(r["frames"]) > 0:
            b64_image = frame_to_b64(r["frames"][0])
            
        formatted_results.append({
            "video_name": r.get("video_name", ""),
            "start_time": float(r["start_time"]),
            "end_time": float(r["end_time"]),
            "visual_score": float(r["visual_score"]),
            "audio_score": float(r["audio_score"]),
            "fused_score": float(r["fused_score"]),
            "image": b64_image
        })
    
    return {
        "query": result["query"],
        "device": get_device_name(),
        "time_taken": float(result["time_taken"]),
        "results": formatted_results
    }


def run_av_batch_retrieval_service(
    query: str,
    videos: list,  # list of tuples (video_name, video_path)
    visual_weight: float = 0.5,
    audio_weight: float = 0.5,
    segment_seconds: float = 5.0,
    max_frames: int = 8,
    top_k: int = 4
) -> dict:
    """Run fused Audio+Video retrieval on multiple video files and return global top_k JSON-serializable results."""
    # Retrieve cached models from model manager
    (clip_processor, clip_model), (clap_model, clap_tokenizer, clap_extractor) = model_manager.get_av_models()
    
    segments = []
    
    for video_name, video_path in videos:
        # Extract audio waveform and sample rate
        try:
            waveform, sample_rate = extract_audio_from_video(video_path)
            duration = get_video_duration(video_path)
        except Exception as e:
            print(f"[WARNING] Skipping video {video_name} due to extraction error: {e}")
            continue
        
        # Segment video and audio synchronously
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
                "video_name": video_name,
                "start_time": current_t,
                "end_time": end_t,
                "frames": frames,
                "audio": audio_segment,
                "sample_rate": sample_rate
            })
            current_t += segment_seconds

    if not segments:
        raise ValueError("Could not extract any segments from the uploaded video files.")
        
    # Run AV retrieval on all segments combined
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
    
    # Format results to be JSON-serializable
    import io
    import base64
    from PIL import Image

    def frame_to_b64(frame_np) -> str:
        pil_img = Image.fromarray(frame_np)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    formatted_results = []
    for idx, r in enumerate(result["results"]):
        b64_image = ""
        if idx < top_k and r.get("frames") and len(r["frames"]) > 0:
            b64_image = frame_to_b64(r["frames"][0])
            
        formatted_results.append({
            "video_name": r.get("video_name", ""),
            "start_time": float(r["start_time"]),
            "end_time": float(r["end_time"]),
            "visual_score": float(r["visual_score"]),
            "audio_score": float(r["audio_score"]),
            "fused_score": float(r["fused_score"]),
            "image": b64_image
        })
    
    return {
        "query": result["query"],
        "device": get_device_name(),
        "time_taken": float(result["time_taken"]),
        "results": formatted_results
    }
