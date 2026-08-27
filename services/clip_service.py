import os
from PIL import Image
from models import model_manager
from inference.clip_retrieval import retrieve_images
from inference.clip_infer import run_clip
from inference.clip4clip_infer import query_video, classify_video
from inference.tinyclip_retrieval import retrieve_tinyclip_images
from inference.tinyclip_infer import run_tinyclip
from utils.video_utils import extract_frames
from utils.device import get_device_name

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
        "device": get_device_name(),
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
        "device": get_device_name(),
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
        "device": get_device_name(),
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
        "device": get_device_name(),
        "time_taken": float(result["time_taken"]),
        "results": formatted_results
    }


def run_clip4clip_batch_retrieval(
    query: str,
    videos: list,  # list of tuples (video_name, video_path)
    max_frames: int = 12,
    top_k: int = 4
) -> dict:
    """Run CLIP4Clip retrieval across multiple video files, returning globally ranked top_k results."""
    import io
    import base64
    import cv2
    import os
    import torch
    import numpy as np
    from PIL import Image
    from inference.clip4clip_infer import encode_frames, encode_text
    from utils.timers import Timer
    from utils.device import DEVICE

    def frame_to_b64(frame_np) -> str:
        pil_img = Image.fromarray(frame_np)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    processor, model = model_manager.get_clip4clip()
    
    # We will extract frames from each video and keep track of video name and timestamp
    pooled_frames = []
    
    for video_name, video_path in videos:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARNING] Skipping video {video_name} because it cannot be opened.")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0 or fps <= 0:
            cap.release()
            continue
            
        step = max(total_frames // max_frames, 1)
        
        extracted_frames_count = 0
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if idx % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = idx / fps
                
                pooled_frames.append({
                    "video_name": video_name,
                    "frame": frame_rgb,
                    "frame_index": idx,
                    "timestamp": timestamp
                })
                extracted_frames_count += 1
                if extracted_frames_count >= max_frames:
                    break
            idx += 1
        cap.release()

    if not pooled_frames:
        raise ValueError("Could not extract any frames from the uploaded video files.")

    # Now run CLIP4Clip matching on all pooled frames
    raw_frames = [item["frame"] for item in pooled_frames]
    
    with Timer() as timer:
        # Encode frames
        frame_features = encode_frames(processor, model, raw_frames)
        # Encode query text
        text_features = encode_text(processor, model, [query])
        
        # Compute similarities
        similarities = torch.matmul(frame_features, text_features.T).squeeze()
        
        if len(raw_frames) == 1:
            scores = [similarities.item()]
        else:
            scores = similarities.cpu().tolist()

    # Create ranked results
    ranked_results = []
    for idx, score in enumerate(scores):
        meta = pooled_frames[idx]
        ranked_results.append({
            "video_name": meta["video_name"],
            "frame_index": meta["frame_index"],
            "timestamp": meta["timestamp"],
            "score": float(score),
            "frame_np": meta["frame"] # Keep temporarily to encode to b64 only for top_k
        })
        
    ranked_results = sorted(ranked_results, key=lambda x: -x["score"])
    
    # Format and keep top_k
    formatted_results = []
    for idx, r in enumerate(ranked_results):
        b64_image = ""
        # Only encode top_k images to base64
        if idx < top_k:
            b64_image = frame_to_b64(r["frame_np"])
            
        formatted_results.append({
            "video_name": r["video_name"],
            "frame_index": r["frame_index"],
            "timestamp": r["timestamp"],
            "score": r["score"],
            "image": b64_image
        })

    return {
        "query": query,
        "device": get_device_name(),
        "time_taken": float(timer.elapsed),
        "results": formatted_results,
        "all_scores": [
            {
                "video_name": r["video_name"],
                "frame_index": r["frame_index"],
                "timestamp": r["timestamp"],
                "score": r["score"]
            }
            for r in ranked_results
        ]
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
        "device": get_device_name(),
        "time_taken": float(result["time_taken"]),
        "results": formatted_results
    }

def run_tinyclip_labeling(image_path: str, labels: list[str]) -> dict:
    """Run TinyCLIP zero-shot image classification."""
    pipe, _, _ = model_manager.get_tinyclip()
    image = Image.open(image_path).convert("RGB")
    result = run_tinyclip(pipe, image, labels)
    
    return {
        "device": get_device_name(),
        "time_taken": float(result["time_taken"]),
        "results": result["results"]
    }
