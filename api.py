import os
import shutil
import tempfile
from typing import List, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form

from models.model_manager import initialize_models, _models
from services.clip_service import (
    run_clip_retrieval,
    run_clip_labeling,
    run_clip4clip_retrieval,
    run_clip4clip_labeling,
    run_clip4clip_similarity,
    run_tinyclip_retrieval,
    run_tinyclip_labeling
)
from services.clap_service import run_clap_retrieval, run_clap_labeling
from services.av_service import run_av_retrieval_service

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Multimodal ML Inference API",
    description="REST API backend exposing all multimodal AI model inference capabilities, supporting both file uploads and local file paths.",
    version="1.1.0"
)

# Enable CORS for cross-origin requests (e.g., from React dev servers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# FILE UPLOAD HELPERS
# ==========================================

import uuid

def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save an UploadFile object to a unique sub-directory, preserving its original filename."""
    unique_id = uuid.uuid4().hex
    temp_dir = os.path.join(os.getcwd(), "tmp_uploads", unique_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Safe filename extraction
    original_filename = os.path.basename(upload_file.filename)
    dest_path = os.path.join(temp_dir, original_filename)
    
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
        
    return dest_path

def cleanup_file(path: Optional[str]):
    """Safely delete temporary uploaded files and their unique parent directories."""
    if path and os.path.exists(path) and "tmp_uploads" in path:
        try:
            os.unlink(path)
            parent_dir = os.path.dirname(path)
            if os.path.basename(parent_dir) != "tmp_uploads":
                os.rmdir(parent_dir)
        except Exception as e:
            print(f"[WARNING] Failed to clean up temp file/dir {path}: {e}")

def parse_labels(labels_str: Optional[str], labels_list: Optional[List[str]]) -> List[str]:
    """Parse candidate labels/prompts from either multiple Form inputs or a single string (comma/newline separated)."""
    if labels_list:
        # FastAPI might receive labels_list as a list of strings
        # Check if the list contains only a single string that has delimiters
        if len(labels_list) == 1 and ("," in labels_list[0] or "\n" in labels_list[0]):
            labels_str = labels_list[0]
        else:
            return [l.strip() for l in labels_list if l.strip()]
            
    if labels_str:
        if "\n" in labels_str:
            return [l.strip() for l in labels_str.splitlines() if l.strip()]
        return [l.strip() for l in labels_str.split(",") if l.strip()]
        
    return []

# ==========================================
# STARTUP EVENT
# ==========================================

@app.on_event("startup")
async def startup():
    print("[INFO] Initializing and caching all models on startup...")
    initialize_models()
    print("[INFO] All models loaded successfully.")

# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/health")
def health():
    """Health check endpoint to verify server status."""
    return {"status": "healthy"}

@app.get("/models")
def get_models():
    """Retrieve status of loaded/cached models."""
    return {
        name: ("Loaded" if val is not None else "Not Loaded")
        for name, val in _models.items()
    }

# ---------- CLIP (Image) ----------

@app.post("/clip/search")
async def clip_search(
    files: Optional[List[UploadFile]] = File(None),
    image_paths: Optional[List[str]] = Form(None),
    query: str = Form(...)
):
    """Run text-to-image similarity search using OpenAI CLIP (supports multiple uploads)."""
    paths = []
    temp_paths = []
    try:
        if files and any(f.filename for f in files):
            for f in files:
                tmp_p = save_uploaded_file(f)
                paths.append(tmp_p)
                temp_paths.append(tmp_p)
        elif image_paths:
            paths = [p.strip() for p in image_paths if p.strip()]
        else:
            raise HTTPException(
                status_code=400,
                detail="Request must include either uploaded 'files' or a list of 'image_paths'."
            )
            
        for path in paths:
            if not os.path.exists(path):
                raise HTTPException(status_code=400, detail=f"Image file not found: {path}")
                
        return run_clip_retrieval(query, paths)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in temp_paths:
            cleanup_file(p)

@app.post("/clip/label")
async def clip_label(
    file: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    labels: Optional[List[str]] = Form(None),
    labels_str: Optional[str] = Form(None)
):
    """Classify an image using OpenAI CLIP (zero-shot classification)."""
    temp_path = None
    try:
        parsed_labels = parse_labels(labels_str, labels)
        if not parsed_labels:
            raise HTTPException(status_code=400, detail="Must provide candidate 'labels' or 'labels_str'.")
            
        if file and file.filename:
            temp_path = save_uploaded_file(file)
            path_to_use = temp_path
        elif image_path:
            path_to_use = image_path
        else:
            raise HTTPException(status_code=400, detail="Must provide uploaded 'file' or 'image_path'.")
            
        if not os.path.exists(path_to_use):
            raise HTTPException(status_code=400, detail=f"Image file not found: {path_to_use}")
            
        return run_clip_labeling(path_to_use, parsed_labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(temp_path)

# ---------- TinyCLIP (Image) ----------

@app.post("/tinyclip/search")
async def tinyclip_search(
    files: Optional[List[UploadFile]] = File(None),
    image_paths: Optional[List[str]] = Form(None),
    query: str = Form(...)
):
    """Run text-to-image similarity search using TinyCLIP (supports multiple uploads)."""
    paths = []
    temp_paths = []
    try:
        if files and any(f.filename for f in files):
            for f in files:
                tmp_p = save_uploaded_file(f)
                paths.append(tmp_p)
                temp_paths.append(tmp_p)
        elif image_paths:
            paths = [p.strip() for p in image_paths if p.strip()]
        else:
            raise HTTPException(
                status_code=400,
                detail="Request must include either uploaded 'files' or a list of 'image_paths'."
            )
            
        for path in paths:
            if not os.path.exists(path):
                raise HTTPException(status_code=400, detail=f"Image file not found: {path}")
                
        return run_tinyclip_retrieval(query, paths)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in temp_paths:
            cleanup_file(p)

@app.post("/tinyclip/label")
async def tinyclip_label(
    file: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    labels: Optional[List[str]] = Form(None),
    labels_str: Optional[str] = Form(None)
):
    """Classify an image using TinyCLIP (zero-shot classification)."""
    temp_path = None
    try:
        parsed_labels = parse_labels(labels_str, labels)
        if not parsed_labels:
            raise HTTPException(status_code=400, detail="Must provide candidate 'labels' or 'labels_str'.")
            
        if file and file.filename:
            temp_path = save_uploaded_file(file)
            path_to_use = temp_path
        elif image_path:
            path_to_use = image_path
        else:
            raise HTTPException(status_code=400, detail="Must provide uploaded 'file' or 'image_path'.")
            
        if not os.path.exists(path_to_use):
            raise HTTPException(status_code=400, detail=f"Image file not found: {path_to_use}")
            
        return run_tinyclip_labeling(path_to_use, parsed_labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(temp_path)

# ---------- CLIP4Clip (Video) ----------

@app.post("/clip4clip/search")
async def clip4clip_search(
    file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None),
    query: str = Form(...),
    max_frames: int = Form(12)
):
    """Run text-to-video search on extracted frames using CLIP4Clip."""
    temp_path = None
    try:
        if file and file.filename:
            temp_path = save_uploaded_file(file)
            path_to_use = temp_path
        elif video_path:
            path_to_use = video_path
        else:
            raise HTTPException(status_code=400, detail="Must provide uploaded 'file' or 'video_path'.")
            
        if not os.path.exists(path_to_use):
            raise HTTPException(status_code=400, detail=f"Video file not found: {path_to_use}")
            
        return run_clip4clip_retrieval(query, path_to_use, max_frames)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(temp_path)

@app.post("/clip4clip/label")
async def clip4clip_label(
    file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None),
    labels: Optional[List[str]] = Form(None),
    labels_str: Optional[str] = Form(None),
    max_frames: int = Form(12)
):
    """Classify a video using CLIP4Clip (zero-shot classification)."""
    temp_path = None
    try:
        parsed_labels = parse_labels(labels_str, labels)
        if not parsed_labels:
            raise HTTPException(status_code=400, detail="Must provide candidate 'labels' or 'labels_str'.")
            
        if file and file.filename:
            temp_path = save_uploaded_file(file)
            path_to_use = temp_path
        elif video_path:
            path_to_use = video_path
        else:
            raise HTTPException(status_code=400, detail="Must provide uploaded 'file' or 'video_path'.")
            
        if not os.path.exists(path_to_use):
            raise HTTPException(status_code=400, detail=f"Video file not found: {path_to_use}")
            
        return run_clip4clip_labeling(path_to_use, parsed_labels, max_frames)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(temp_path)

@app.post("/clip4clip/similarity")
async def clip4clip_similarity(
    file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None),
    prompts: Optional[List[str]] = Form(None),
    prompts_str: Optional[str] = Form(None),
    max_frames: int = Form(12)
):
    """Compute temporal similarity scores between a video and multiple text prompts."""
    temp_path = None
    try:
        parsed_prompts = parse_labels(prompts_str, prompts)
        if not parsed_prompts:
            raise HTTPException(status_code=400, detail="Must provide 'prompts' or 'prompts_str'.")
            
        if file and file.filename:
            temp_path = save_uploaded_file(file)
            path_to_use = temp_path
        elif video_path:
            path_to_use = video_path
        else:
            raise HTTPException(status_code=400, detail="Must provide uploaded 'file' or 'video_path'.")
            
        if not os.path.exists(path_to_use):
            raise HTTPException(status_code=400, detail=f"Video file not found: {path_to_use}")
            
        return run_clip4clip_similarity(path_to_use, parsed_prompts, max_frames)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(temp_path)

# ---------- CLAP (Audio) ----------

@app.post("/clap/search")
async def clap_search(
    file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None),
    query: str = Form(...),
    segment_seconds: float = Form(5.0)
):
    """Run text-to-audio search on segments of an audio or video file using CLAP."""
    temp_path = None
    try:
        if file and file.filename:
            temp_path = save_uploaded_file(file)
            path_to_use = temp_path
        elif video_path:
            path_to_use = video_path
        else:
            raise HTTPException(status_code=400, detail="Must provide uploaded 'file' or 'video_path'.")
            
        if not os.path.exists(path_to_use):
            raise HTTPException(status_code=400, detail=f"Audio/video file not found: {path_to_use}")
            
        return run_clap_retrieval(query, path_to_use, segment_seconds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(temp_path)

@app.post("/clap/label")
async def clap_label(
    file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None),
    labels: Optional[List[str]] = Form(None),
    labels_str: Optional[str] = Form(None)
):
    """Classify audio content of a file using CLAP (zero-shot classification)."""
    temp_path = None
    try:
        parsed_labels = parse_labels(labels_str, labels)
        if not parsed_labels:
            raise HTTPException(status_code=400, detail="Must provide candidate 'labels' or 'labels_str'.")
            
        if file and file.filename:
            temp_path = save_uploaded_file(file)
            path_to_use = temp_path
        elif video_path:
            path_to_use = video_path
        else:
            raise HTTPException(status_code=400, detail="Must provide uploaded 'file' or 'video_path'.")
            
        if not os.path.exists(path_to_use):
            raise HTTPException(status_code=400, detail=f"Audio/video file not found: {path_to_use}")
            
        return run_clap_labeling(path_to_use, parsed_labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(temp_path)

# ---------- Audio + Video Fusion ----------

@app.post("/av/search")
async def av_search(
    file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None),
    query: str = Form(...),
    visual_weight: float = Form(0.5),
    audio_weight: float = Form(0.5),
    segment_seconds: float = Form(5.0)
):
    """Run fused Audio+Video search on segments of a video file using CLIP4Clip and CLAP."""
    temp_path = None
    try:
        if file and file.filename:
            temp_path = save_uploaded_file(file)
            path_to_use = temp_path
        elif video_path:
            path_to_use = video_path
        else:
            raise HTTPException(status_code=400, detail="Must provide uploaded 'file' or 'video_path'.")
            
        if not os.path.exists(path_to_use):
            raise HTTPException(status_code=400, detail=f"Video file not found: {path_to_use}")
            
        return run_av_retrieval_service(
            query=query,
            video_path=path_to_use,
            visual_weight=visual_weight,
            audio_weight=audio_weight,
            segment_seconds=segment_seconds
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
