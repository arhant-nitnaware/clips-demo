# Multimodal AI Inference API

A streamlined REST API backend exposing multimodal AI models (**CLIP**, **TinyCLIP**, **CLIP4Clip**, and **CLAP**) with a built-in **Streamlit** client application.

---

## 1. Setup & Installation

### A. Clone Repository
```bash
git clone <repository-url>
cd clips-demo
```

### B. Create Virtual Environment
> [!NOTE]
> Python 3.13.5 was used while developing this codebase, therefore Python 3.13.x is recommended.

#### Linux / macOS
```bash
python3 -m venv <env-name>
source <env-name>/bin/activate
```

#### Windows
```bash
python -m venv <env-name>
<env-name>\Scripts\activate
```

#### Conda
```bash
conda create -n <env_name> python=3.13
conda activate <env-name>
```

### C. Install Dependencies
```bash
pip install -r requirements.txt
```

#### PyTorch Installation Options
* **CPU Torch:**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
* **GPU Torch:**
  ```bash
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  ```
  *Note: Check the [PyTorch Website](https://pytorch.org/get-started/locally/) to select the appropriate `cu` version based on your GPU driver and CUDA version (run `nvidia-smi` to verify). `cu126` was used in development.*

---

## 2. Model Setup

Large model weights are not stored inside the repository. You can download all required model files using the setup script:

### Linux / macOS
```bash
python3 utils/setup.py
```

### Windows
```bash
python utils/setup.py
```

This setup script downloads the `.safetensors` weight files for:
* **CLIP**
* **CLAP**
* **TinyCLIP**
* **CLIP4Clip**

The models are extracted into their respective directories inside `models_local/`. Each directory must contain:
```text
model.safetensors
```

---

## 3. Running the Application

### A. Run FastAPI Backend
```bash
python api.py
```

### B. Run Streamlit Client
```bash
streamlit run app.py
```
Open the URL shown in the terminal output, typically:
```text
http://localhost:8501
```
If port `8501` is unavailable, Streamlit will automatically select another port and display it in the terminal.

---

## 4. Demo Videos

Explore the capabilities of the system by watching the following demo videos:
* **Audio Search:** [https://youtu.be/D4UeC6aDbd0](https://youtu.be/D4UeC6aDbd0)
* **Video Search:** [https://youtu.be/6ssBksFQoHo](https://youtu.be/6ssBksFQoHo)
* **Audio-Video (AV) Search:** [https://youtu.be/lBf_itk5blk](https://youtu.be/lBf_itk5blk)

---

## 5. API Reference

All inference endpoints expect inputs as `multipart/form-data`. You can supply either an uploaded binary file via `file` / `files` or a local server path via `video_path` / `image_path` / `video_paths` / `image_paths`.

---

### A. Media Metadata

#### `POST /media/info`
Extracts duration, framerate, resolution, audio sample rate, and size from a media file.

**Parameters:**
* `file` (*UploadFile, optional*): Binary audio/video file.
* `video_path` (*string, optional*): Local path on the server to the media file.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/media/info" \
  -F "file=@dataset/audio/birds-5.mp3"
```

**Response Schema:**
```json
{
  "duration": 12.34,
  "fps": 29.97,
  "frame_count": 370,
  "resolution": "1920x1080",
  "sample_rate": 48000,
  "file_size_mb": 4.12
}
```

---

### B. OpenAI CLIP (Image Encoders)

#### `POST /clip/search`
Compares a text query against multiple images and ranks them by similarity.

**Parameters:**
* `files` (*UploadFile[], optional*): One or more binary image files.
* `image_paths` (*string[], optional*): Local paths on the server to image files.
* `query` (*string, required*): Search query text.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/clip/search" \
  -F "files=@dataset/images/car1.jpg" \
  -F "files=@dataset/images/car2.jpg" \
  -F "query=a red sports car"
```

**Response Schema:**
```json
{
  "query": "a red sports car",
  "device": "cuda:0",
  "time_taken": 0.1245,
  "results": [
    { "image_path": "car1.jpg", "score": 0.2842 },
    { "image_path": "car2.jpg", "score": 0.1503 }
  ]
}
```

#### `POST /clip/label`
Runs zero-shot classification on a single image against candidate labels (scores sum to 1.0).

**Parameters:**
* `file` (*UploadFile, optional*): Binary image file.
* `image_path` (*string, optional*): Local server path to the image file.
* `labels_str` (*string, optional*): Comma- or newline-separated candidate labels.
* `labels` (*string[], optional*): Repeated form field for candidate labels.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/clip/label" \
  -F "file=@dataset/images/car1.jpg" \
  -F "labels_str=nature,car,city,dog"
```

**Response Schema:**
```json
{
  "device": "cuda:0",
  "time_taken": 0.0812,
  "results": [
    { "label": "car", "score": 0.8125 },
    { "label": "nature", "score": 0.1204 },
    { "label": "city", "score": 0.0451 },
    { "label": "dog", "score": 0.0220 }
  ]
}
```

---

### C. TinyCLIP (Resource-Efficient Image Encoders)

#### `POST /tinyclip/search`
Resource-efficient version of standard CLIP image similarity search.

**Parameters:**
* `files` (*UploadFile[], optional*): One or more binary image files.
* `image_paths` (*string[], optional*): Local paths on the server to image files.
* `query` (*string, required*): Search query text.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/tinyclip/search" \
  -F "files=@dataset/images/car1.jpg" \
  -F "query=a red sports car"
```

**Response Schema:**
```json
{
  "query": "a red sports car",
  "device": "cuda:0",
  "time_taken": 0.0654,
  "results": [
    { "image_path": "car1.jpg", "score": 0.2791 }
  ]
}
```

#### `POST /tinyclip/label`
Resource-efficient zero-shot image classification.

**Parameters:**
* `file` (*UploadFile, optional*): Binary image file.
* `image_path` (*string, optional*): Local server path to the image file.
* `labels_str` (*string, optional*): Comma- or newline-separated candidate labels.
* `labels` (*string[], optional*): Repeated form field for candidate labels.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/tinyclip/label" \
  -F "file=@dataset/images/car1.jpg" \
  -F "labels_str=nature,car,city,dog"
```

**Response Schema:**
```json
{
  "device": "cuda:0",
  "time_taken": 0.0412,
  "results": [
    { "label": "car", "score": 0.8350 },
    { "label": "nature", "score": 0.1021 },
    { "label": "city", "score": 0.0415 },
    { "label": "dog", "score": 0.0214 }
  ]
}
```

---

### D. CLIP4Clip (Video Frame Encoders)

#### `POST /clip4clip/search`
Compares extracted video frames against a text query and returns top matching frames as base64 images.

**Parameters:**
* `file` (*UploadFile, optional*): Binary video file.
* `video_path` (*string, optional*): Local server path to the video file.
* `query` (*string, required*): Search query text.
* `max_frames` (*int, default: 12*): Maximum frames sampled across the video.
* `top_k` (*int, default: 4*): Number of top frame results returned.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/clip4clip/search" \
  -F "file=@demo_videos/sample.mp4" \
  -F "query=a yellow race car" \
  -F "max_frames=12" \
  -F "top_k=4"
```

**Response Schema:**
```json
{
  "query": "a yellow race car",
  "device": "cuda:0",
  "time_taken": 0.6543,
  "results": [
    {
      "frame_index": 8,
      "score": 0.2981,
      "image": "data:image/jpeg;base64,..."
    }
  ],
  "all_scores": [
    { "frame_index": 8, "score": 0.2981 },
    { "frame_index": 4, "score": 0.1872 }
  ]
}
```

#### `POST /clip4clip/label`
Runs zero-shot video classification based on visual temporal features across extracted frames.

**Parameters:**
* `file` (*UploadFile, optional*): Binary video file.
* `video_path` (*string, optional*): Local server path to the video file.
* `labels_str` (*string, optional*): Comma- or newline-separated candidate labels.
* `labels` (*string[], optional*): Repeated form field for candidate labels.
* `max_frames` (*int, default: 12*): Maximum frames sampled for classification.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/clip4clip/label" \
  -F "file=@demo_videos/sample.mp4" \
  -F "labels_str=sports,cooking,news" \
  -F "max_frames=12"
```

**Response Schema:**
```json
{
  "device": "cuda:0",
  "time_taken": 0.4812,
  "results": [
    { "label": "sports", "score": 0.7621 },
    { "label": "news", "score": 0.1542 },
    { "label": "cooking", "score": 0.0837 }
  ]
}
```

#### `POST /clip4clip/batch_search`
Compares video frames across multiple video files against a text query and returns globally ranked top matching frames.

**Parameters:**
* `files` (*UploadFile[], optional*): Multiple binary video files.
* `video_paths` (*string, optional*): JSON array or comma-separated server video paths.
* `query` (*string, required*): Search query text.
* `max_frames` (*int, default: 12*): Maximum frames sampled per video.
* `top_k` (*int, default: 4*): Total top results returned across all videos.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/clip4clip/batch_search" \
  -F "files=@demo_videos/sample1.mp4" \
  -F "files=@demo_videos/sample2.mp4" \
  -F "query=a yellow race car" \
  -F "max_frames=12" \
  -F "top_k=4"
```

**Response Schema:**
```json
{
  "query": "a yellow race car",
  "device": "cuda:0",
  "time_taken": 1.1245,
  "results": [
    {
      "video_name": "sample1.mp4",
      "frame_index": 8,
      "timestamp": 2.4,
      "score": 0.3125,
      "image": "data:image/jpeg;base64,..."
    }
  ],
  "all_scores": [
    {
      "video_name": "sample1.mp4",
      "frame_index": 8,
      "timestamp": 2.4,
      "score": 0.3125
    }
  ]
}
```

---

### E. CLAP (Contrastive Language-Audio Pretraining)

#### `POST /clap/search`
Segments the audio track of an audio or video file and ranks segments based on query similarity.

**Parameters:**
* `file` (*UploadFile, optional*): Binary audio/video file.
* `video_path` (*string, optional*): Local server path to the audio/video file.
* `query` (*string, required*): Audio search query text.
* `segment_seconds` (*float, default: 5.0*): Duration in seconds of each audio segment.
* `top_k` (*int, default: 4*): Number of top audio segment results returned.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/clap/search" \
  -F "file=@dataset/audio/birds-5.mp3" \
  -F "query=birds chirping" \
  -F "segment_seconds=3.0" \
  -F "top_k=4"
```

**Response Schema:**
```json
{
  "query": "birds chirping",
  "device": "cuda:0",
  "time_taken": 0.3512,
  "results": [
    {
      "start_time": 6.0,
      "end_time": 9.0,
      "score": 0.7241,
      "audio": "data:audio/wav;base64,..."
    }
  ]
}
```

#### `POST /clap/label`
Classifies the complete audio track zero-shot against candidate labels.

**Parameters:**
* `file` (*UploadFile, optional*): Binary audio/video file.
* `video_path` (*string, optional*): Local server path to the audio/video file.
* `labels_str` (*string, optional*): Comma- or newline-separated candidate labels.
* `labels` (*string[], optional*): Repeated form field for candidate labels.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/clap/label" \
  -F "file=@dataset/audio/birds-5.mp3" \
  -F "labels_str=birds,applause,coughing"
```

**Response Schema:**
```json
{
  "device": "cuda:0",
  "time_taken": 0.1845,
  "results": [
    { "label": "birds", "score": 0.8912 },
    { "label": "applause", "score": 0.0815 },
    { "label": "coughing", "score": 0.0273 }
  ]
}
```

---

### F. Audio+Video Fusion Encoders

#### `POST /av/search`
Combines visual (CLIP4Clip) and audio (CLAP) representations synchronously (`fused_score = visual_weight * visual + audio_weight * audio`) to find matching temporal segments within a single video.

**Parameters:**
* `file` (*UploadFile, optional*): Binary video file.
* `video_path` (*string, optional*): Local server path to the video file.
* `query` (*string, required*): Search query text.
* `visual_weight` (*float, default: 0.5*): Relative weight for visual modality.
* `audio_weight` (*float, default: 0.5*): Relative weight for audio modality.
* `segment_seconds` (*float, default: 5.0*): Segment length in seconds.
* `max_frames` (*int, default: 8*): Number of frames sampled per segment.
* `top_k` (*int, default: 4*): Number of top fused segment results returned.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/av/search" \
  -F "file=@demo_videos/sample.mp4" \
  -F "query=helicopter blades roaring" \
  -F "visual_weight=0.5" \
  -F "audio_weight=0.5" \
  -F "segment_seconds=5.0" \
  -F "max_frames=8" \
  -F "top_k=4"
```

**Response Schema:**
```json
{
  "query": "helicopter blades roaring",
  "device": "cuda:0",
  "time_taken": 1.4589,
  "results": [
    {
      "video_name": "sample.mp4",
      "start_time": 5.0,
      "end_time": 10.0,
      "visual_score": 0.4512,
      "audio_score": 0.8124,
      "fused_score": 0.6318,
      "image": "data:image/jpeg;base64,..."
    }
  ]
}
```

#### `POST /av/batch_search`
Combines visual and audio representations across multiple video files to find matching segments globally ranked across all input videos.

**Parameters:**
* `files` (*UploadFile[], optional*): Multiple binary video files.
* `video_paths` (*string, optional*): JSON array or comma-separated server video paths.
* `query` (*string, required*): Search query text.
* `visual_weight` (*float, default: 0.5*): Relative weight for visual modality.
* `audio_weight` (*float, default: 0.5*): Relative weight for audio modality.
* `segment_seconds` (*float, default: 5.0*): Segment length in seconds.
* `max_frames` (*int, default: 8*): Number of frames sampled per segment.
* `top_k` (*int, default: 4*): Total top fused results returned across all videos.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/av/batch_search" \
  -F "files=@demo_videos/sample1.mp4" \
  -F "files=@demo_videos/sample2.mp4" \
  -F "query=helicopter blades roaring" \
  -F "visual_weight=0.5" \
  -F "audio_weight=0.5" \
  -F "segment_seconds=5.0" \
  -F "max_frames=8" \
  -F "top_k=4"
```

**Response Schema:**
```json
{
  "query": "helicopter blades roaring",
  "device": "cuda:0",
  "time_taken": 2.8150,
  "results": [
    {
      "video_name": "sample1.mp4",
      "start_time": 5.0,
      "end_time": 10.0,
      "visual_score": 0.4512,
      "audio_score": 0.8124,
      "fused_score": 0.6318,
      "image": "data:image/jpeg;base64,..."
    }
  ]
}
```

---

### G. System & Model Management

#### `GET /health`
Health check endpoint to verify backend status.
```json
{ "status": "healthy" }
```

#### `GET /gpu`
Retrieves CUDA availability, device name, and GPU VRAM allocations.
```json
{
  "cuda_available": true,
  "allocated_gb": 3.42,
  "reserved_gb": 4.10,
  "device": "cuda:0"
}
```

#### `GET /models`
Retrieves the loaded status of all inference models (`clip`, `clip4clip`, `clap`, `tinyclip`).
```json
{
  "clip": "Loaded",
  "clip4clip": "Loaded",
  "clap": "Loaded",
  "tinyclip": "Loaded"
}
```

#### `POST /models/load/{model_name}`
Explicitly loads and caches a specific model into memory.
* `model_name`: `clip` | `clip4clip` | `clap` | `tinyclip`

#### `POST /models/unload/{model_name}`
Unloads a model from memory and clears GPU cache.
* `model_name`: `clip` | `clip4clip` | `clap` | `tinyclip`

