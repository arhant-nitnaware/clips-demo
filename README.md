# Multimodal AI Inference Backend & Streamlit Application

This project exposes the existing Multimodal Machine Learning (ML) inference pipeline through a **FastAPI backend** while preserving the **Streamlit frontend application**. Both components share a centralized **Model Manager** to load and cache models efficiently without duplication.

---

## 1. Installation

Install all required Python packages:

```bash
pip install -r requirements.txt
```

---

## 2. How to Run

### Streamlit Application

To start the interactive web user interface:

```bash
streamlit run app.py
```

### FastAPI Backend

To start the FastAPI REST API server (runs on `http://localhost:8000` by default):

```bash
python api.py
```

Or run via Uvicorn directly:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

*Note: All models are loaded and cached into GPU/CPU memory on FastAPI startup.*

---

## 3. Exposing Features via FastAPI (How to Upload Files or Use Local Paths)

Every endpoint supports **both** uploading files via standard multipart form data (`file`/`files`) **and** specifying a path to a file already on the server (`video_path`/`image_paths`).

### 1. Health & Models Info

* **Check API Health (`GET /health`):**
  ```bash
  curl -X GET http://localhost:8000/health
  ```
* **Retrieve Cache Status (`GET /models`):**
  ```bash
  curl -X GET http://localhost:8000/models
  ```

---

### 2. OpenAI CLIP Endpoints (Image)

#### Image Retrieval (`POST /clip/search`)
Search over a set of images using a query.
* **Option A: Uploading Files**
  ```bash
  curl -X POST http://localhost:8000/clip/search \
    -F "files=@dataset/dogs/black-dog.jpg" \
    -F "files=@dataset/dogs/white-dog.jpg" \
    -F "query=a white dog"
  ```
* **Option B: Using Paths**
  ```bash
  curl -X POST http://localhost:8000/clip/search \
    -F "image_paths=dataset/dogs/black-dog.jpg" \
    -F "image_paths=dataset/dogs/white-dog.jpg" \
    -F "query=a white dog"
  ```

#### Image Labeling (`POST /clip/label`)
Run zero-shot classification on an image.
* **Option A: Uploading File**
  ```bash
  curl -X POST http://localhost:8000/clip/label \
    -F "file=@dataset/dogs/brown-dog.jpg" \
    -F "labels_str=a puppy,a cat,a bird,nature"
  ```
* **Option B: Using Path**
  ```bash
  curl -X POST http://localhost:8000/clip/label \
    -F "image_path=dataset/dogs/brown-dog.jpg" \
    -F "labels_str=a puppy,a cat,a bird,nature"
  ```

---

### 3. TinyCLIP Endpoints (Lightweight Image)

#### Image Retrieval (`POST /tinyclip/search`)
* **Example (Using Paths):**
  ```bash
  curl -X POST http://localhost:8000/tinyclip/search \
    -F "image_paths=dataset/dogs/brown-dog.jpg" \
    -F "image_paths=dataset/dogs/golden-dog.jpg" \
    -F "query=golden retriever puppy"
  ```

#### Image Labeling (`POST /tinyclip/label`)
* **Example (Uploading File):**
  ```bash
  curl -X POST http://localhost:8000/tinyclip/label \
    -F "file=@dataset/dogs/golden-dog.jpg" \
    -F "labels_str=puppy,cat,car"
  ```

---

### 4. CLIP4Clip Endpoints (Video)

#### Frame Retrieval (`POST /clip4clip/search`)
Find which video frames best match the query.
* **Option A: Uploading File**
  ```bash
  curl -X POST http://localhost:8000/clip4clip/search \
    -F "file=@demo_videos/video-search.mp4" \
    -F "query=a car driving by" \
    -F "max_frames=12"
  ```
* **Option B: Using Path**
  ```bash
  curl -X POST http://localhost:8000/clip4clip/search \
    -F "video_path=demo_videos/video-search.mp4" \
    -F "query=a car driving by" \
    -F "max_frames=12"
  ```

#### Video Labeling (`POST /clip4clip/label`)
Classify the video content zero-shot.
* **Example (Uploading File):**
  ```bash
  curl -X POST http://localhost:8000/clip4clip/label \
    -F "file=@demo_videos/video-search.mp4" \
    -F "labels_str=driving in city,sports event,nature scene"
  ```

#### Video Similarity (`POST /clip4clip/similarity`)
Rank multiple prompt options against the video content.
* **Example (Uploading File):**
  ```bash
  curl -X POST http://localhost:8000/clip4clip/similarity \
    -F "file=@demo_videos/video-search.mp4" \
    -F "prompts_str=a racing vehicle,a parked car,a nature view"
  ```

---

### 5. CLAP Endpoints (Audio)

#### Audio Segment Retrieval (`POST /clap/search`)
Split audio into segments and rank them by query similarity.
* **Option A: Uploading File**
  ```bash
  curl -X POST http://localhost:8000/clap/search \
    -F "file=@demo_videos/audio-search.mp4" \
    -F "query=roaring engines" \
    -F "segment_seconds=2.0"
  ```
* **Option B: Using Path**
  ```bash
  curl -X POST http://localhost:8000/clap/search \
    -F "video_path=demo_videos/audio-search.mp4" \
    -F "query=roaring engines" \
    -F "segment_seconds=2.0"
  ```

#### Audio Labeling (`POST /clap/label`)
Run zero-shot classification on the full audio track.
* **Example (Uploading File):**
  ```bash
  curl -X POST http://localhost:8000/clap/label \
    -F "file=@demo_videos/audio-search.mp4" \
    -F "labels_str=car engine,dog barking,people talking,applause"
  ```

---

### 6. Audio+Video Fusion Endpoint (`POST /av/search`)

Combines both visual (CLIP4Clip) and audio (CLAP) encoders to rank video segments by temporal visual+audio similarity to the query.
* **Option A: Uploading File**
  ```bash
  curl -X POST http://localhost:8000/av/search \
    -F "file=@demo_videos/av-search.mp4" \
    -F "query=helicopter blades roaring" \
    -F "visual_weight=0.5" \
    -F "audio_weight=0.5" \
    -F "segment_seconds=3.0"
  ```
* **Option B: Using Path**
  ```bash
  curl -X POST http://localhost:8000/av/search \
    -F "video_path=demo_videos/av-search.mp4" \
    -F "query=helicopter blades roaring" \
    -F "visual_weight=0.5" \
    -F "audio_weight=0.5" \
    -F "segment_seconds=3.0"
  ```
