import os
from PIL import Image
from models import model_manager
from inference.clip_retrieval import retrieve_images
from inference.clip_infer import run_clip
from inference.clip4clip_infer import query_video, classify_video
from inference.clip4clip_similarity import compute_video_similarity
from inference.tinyclip_retrieval import retrieve_tinyclip_images
from inference.tinyclip_infer import run_tinyclip
from utils.video_utils import extract_frames

class PathWrapper(str):
    @property
    def name(self):
        return os.path.basename(self)

# ==========================================
# CLIP SERVICES
# ==========================================

def run_clip_retrieval(query: str, image_paths: list[str]) -> dict:
    """Run CLIP image retrieval and format the output to be JSON-serializable."""
    model, processor = model_manager.get_clip()
    wrapped_images = [PathWrapper(p) for p in image_paths]
    result = retrieve_images(model, processor, query, wrapped_images)
    
    formatted_results = [
        {
            "image_path": str(r[0]),
            "score": float(r[2])
        }
        for r in result["results"]
    ]
    return {
        "query": result["query"],
        "time_taken": float(result["time_taken"]),
        "results": formatted_results
    }

def run_clip_labeling(image_path: str, labels: list[str]) -> dict:
    """Run CLIP image zero-shot labeling classification."""
    model, processor = model_manager.get_clip()
    image = Image.open(image_path).convert("RGB")
    result = run_clip(model, processor, image, labels)
    
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

# ==========================================
# CLIP4CLIP SERVICES
# ==========================================

def run_clip4clip_retrieval(query: str, video_path: str, max_frames: int = 12, top_k: int = 4) -> dict:
    """Run CLIP4Clip video frame retrieval, returning base64 images for the top_k matching frames."""
    import io
    import base64
    from PIL import Image

    def frame_to_b64(frame_np) -> str:
        pil_img = Image.fromarray(frame_np)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    processor, model = model_manager.get_clip4clip()
    frames = extract_frames(video_path, max_frames=max_frames)
    if not frames:
        raise ValueError(f"Could not extract frames from video at {video_path}")
    result = query_video(processor, model, frames, query)
    
    # Return top_k results with base64 image frames
    formatted_results = []
    for r in result["frame_scores"][:top_k]:
        frame_idx = int(r[0])
        score = float(r[1])
        b64_image = frame_to_b64(frames[frame_idx])
        formatted_results.append({
            "frame_index": frame_idx,
            "score": score,
            "image": b64_image
        })
        
    return {
        "query": result["query"],
        "time_taken": float(result["time_taken"]),
        "results": formatted_results,
        "all_scores": [
            {"frame_index": int(r[0]), "score": float(r[1])}
            for r in result["frame_scores"]
        ]
    }

def run_clip4clip_labeling(video_path: str, labels: list[str], max_frames: int = 12) -> dict:
    """Run CLIP4Clip zero-shot video labeling classification."""
    processor, model = model_manager.get_clip4clip()
    frames = extract_frames(video_path, max_frames=max_frames)
    if not frames:
        raise ValueError(f"Could not extract frames from video at {video_path}")
    result = classify_video(processor, model, frames, labels)
    
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

def run_clip4clip_similarity(video_path: str, prompts: list[str], max_frames: int = 12) -> dict:
    """Run CLIP4Clip temporal video similarity comparisons."""
    processor, model = model_manager.get_clip4clip()
    frames = extract_frames(video_path, max_frames=max_frames)
    if not frames:
        raise ValueError(f"Could not extract frames from video at {video_path}")
    result = compute_video_similarity(processor, model, frames, prompts)
    
    formatted_results = [
        {
            "prompt": str(r[0]),
            "score": float(r[1])
        }
        for r in result
    ]
    return {
        "results": formatted_results
    }

# ==========================================
# TINYCLIP SERVICES
# ==========================================

def run_tinyclip_retrieval(query: str, image_paths: list[str]) -> dict:
    """Run TinyCLIP image retrieval and format the output to be JSON-serializable."""
    _, model, processor = model_manager.get_tinyclip()
    wrapped_images = [PathWrapper(p) for p in image_paths]
    result = retrieve_tinyclip_images(model, processor, query, wrapped_images)
    
    formatted_results = [
        {
            "image_path": str(r[0]),
            "score": float(r[2])
        }
        for r in result["results"]
    ]
    return {
        "query": result["query"],
        "time_taken": float(result["time_taken"]),
        "results": formatted_results
    }

def run_tinyclip_labeling(image_path: str, labels: list[str]) -> dict:
    """Run TinyCLIP zero-shot image classification."""
    pipe, _, _ = model_manager.get_tinyclip()
    image = Image.open(image_path).convert("RGB")
    result = run_tinyclip(pipe, image, labels)
    
    return {
        "time_taken": float(result["time_taken"]),
        "results": result["results"]
    }
