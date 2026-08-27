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

All endpoints expect inputs as `multipart/form-data`. You can supply either an uploaded binary file via `file` / `files` or a local server path via `video_path` / `image_path` / `video_paths`.

### A. Media Metadata

#### `POST /media/info`
Extracts duration, framerate, resolution, audio sample rate, and size from a media file.

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
Compares a query against a batch of images and ranks them by similarity.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/clip/search" \
  -F "files=@dataset/images/car1.jpg" \
  -F "query=a red sports car"
```

**Response Schema:**
```json
{
  "query": "a red sports car",
  "time_taken": 0.1245,
  "results": [
    { "image_path": "car1.jpg", "score": 0.2842 }
  ]
}
```

#### `POST /clip/label`
Runs zero-shot classification on a single image.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/clip/label" \
  -F "file=@dataset/images/car1.jpg" \
  -F "labels_str=nature,car,city,dog"
```

**Response Schema:**
```json
{
  "time_taken": 0.0812,
  "results": [
    { "label": "car", "score": 0.8125 },
    { "label": "nature", "score": 0.1204 }
  ]
}
```

---

### C. TinyCLIP (Resource-Efficient Image Encoders)

#### `POST /tinyclip/search`
Optimized version of standard CLIP image search.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/tinyclip/search" \
  -F "files=@dataset/images/car1.jpg" \
  -F "query=a red sports car"
```

#### `POST /tinyclip/label`
Optimized version of standard CLIP zero-shot classification.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/tinyclip/label" \
  -F "file=@dataset/images/car1.jpg" \
  -F "labels_str=nature,car,city,dog"
```

---

### D. CLIP4Clip (Video Frame Encoders)

#### `POST /clip4clip/search`
Compares video frames to a text query and returns top matching frames.

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
  "time_taken": 0.6543,
  "results": [
    {
      "frame_index": 8,
      "score": 0.2981,
      "image": "data:image/jpeg;base64,..."
    }
  ]
}
```

#### `POST /clip4clip/label`
Runs zero-shot video classification based on visual temporal features.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/clip4clip/label" \
  -F "file=@demo_videos/sample.mp4" \
  -F "labels_str=sports,cooking,news" \
  -F "max_frames=12"
```

#### `POST /clip4clip/batch_search`
Compares video frames across multiple video files to a text query.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/clip4clip/batch_search" \
  -F "files=@demo_videos/sample1.mp4" \
  -F "files=@demo_videos/sample2.mp4" \
  -F "query=a yellow race car" \
  -F "max_frames=12" \
  -F "top_k=4"
```

---

### E. CLAP (Contrastive Language-Audio Pretraining)

#### `POST /clap/search`
Segments the audio track of a file and ranks audio segments based on query similarity.

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
Classifies the full audio track zero-shot.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/clap/label" \
  -F "file=@dataset/audio/birds-5.mp3" \
  -F "labels_str=birds,applause,coughing"
```

---

### F. Audio+Video Fusion Encoders

#### `POST /av/search`
Combines visual (CLIP4Clip) and audio (CLAP) features synchronously to find sections of a video matching a query.

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
  "time_taken": 1.4589,
  "results": [
    {
      "video_name": "chopper.mp4",
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
Combines visual and audio features across multiple video files to find matching sections globally.

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
