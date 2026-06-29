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

#### Check API Health (`GET /health`)
* **Request:**
  ```bash
  curl -X GET http://localhost:8000/health
  ```
* **Response:**
  ```json
  {
    "status": "healthy"
  }
  ```

#### Retrieve Cache Status (`GET /models`)
* **Request:**
  ```bash
  curl -X GET http://localhost:8000/models
  ```
* **Response:**
  ```json
  {
    "clip": "Loaded",
    "clip4clip": "Loaded",
    "clap": "Loaded",
    "tinyclip": "Loaded"
  }
  ```

---

### 2. OpenAI CLIP Endpoints (Image)

#### Image Retrieval (`POST /clip/search`)
Search over a set of images using a query.
* **Request (Uploading Files):**
  ```bash
  curl -X POST http://localhost:8000/clip/search \
    -F "files=@dataset/dogs/black-dog.jpg" \
    -F "files=@dataset/dogs/white-dog.jpg" \
    -F "query=a white dog"
  ```
* **Request (Using Paths):**
  ```bash
  curl -X POST http://localhost:8000/clip/search \
    -F "image_paths=dataset/dogs/black-dog.jpg" \
    -F "image_paths=dataset/dogs/white-dog.jpg" \
    -F "query=a white dog"
  ```
* **Response Example:**
  ```json
  {
    "query": "a white dog",
    "time_taken": 0.1423,
    "results": [
      {
        "image_path": "white-dog.jpg",
        "score": 0.2854
      },
      {
        "image_path": "black-dog.jpg",
        "score": 0.1102
      }
    ]
  }
  ```

#### Image Labeling (`POST /clip/label`)
Run zero-shot classification on an image.
* **Request (Uploading File):**
  ```bash
  curl -X POST http://localhost:8000/clip/label \
    -F "file=@dataset/dogs/brown-dog.jpg" \
    -F "labels_str=a puppy,a cat,a bird,nature"
  ```
* **Response Example:**
  ```json
  {
    "time_taken": 0.0852,
    "results": [
      {
        "label": "a puppy",
        "score": 0.8241
      },
      {
        "label": "a cat",
        "score": 0.1032
      },
      {
        "label": "nature",
        "score": 0.0521
      },
      {
        "label": "a bird",
        "score": 0.0206
      }
    ]
  }
  ```

---

### 3. TinyCLIP Endpoints (Lightweight Image)

#### Image Retrieval (`POST /tinyclip/search`)
* **Request (Using Paths):**
  ```bash
  curl -X POST http://localhost:8000/tinyclip/search \
    -F "image_paths=dataset/dogs/brown-dog.jpg" \
    -F "image_paths=dataset/dogs/golden-dog.jpg" \
    -F "query=golden retriever puppy"
  ```
* **Response Example:**
  ```json
  {
    "query": "golden retriever puppy",
    "time_taken": 0.0912,
    "results": [
      {
        "image_path": "golden-dog.jpg",
        "score": 0.2914
      },
      {
        "image_path": "brown-dog.jpg",
        "score": 0.1245
      }
    ]
  }
  ```

#### Image Labeling (`POST /tinyclip/label`)
* **Request (Uploading File):**
  ```bash
  curl -X POST http://localhost:8000/tinyclip/label \
    -F "file=@dataset/dogs/golden-dog.jpg" \
    -F "labels_str=puppy,cat,car"
  ```
* **Response Example:**
  ```json
  {
    "time_taken": 0.0654,
    "results": [
      {
        "label": "puppy",
        "score": 0.7421
      },
      {
        "label": "cat",
        "score": 0.1852
      },
      {
        "label": "car",
        "score": 0.0727
      }
    ]
  }
  ```

---

### 4. CLIP4Clip Endpoints (Video)

#### Frame Retrieval (`POST /clip4clip/search`)
Find which video frames best match the query.
* **Request (Uploading File):**
  ```bash
  curl -X POST http://localhost:8000/clip4clip/search \
    -F "file=@demo_videos/video-search.mp4" \
    -F "query=a car driving by" \
    -F "max_frames=12"
  ```
* **Response Example:**
  ```json
  {
    "query": "a car driving by",
    "time_taken": 0.8521,
    "results": [
      {
        "frame_index": 5,
        "score": 0.3125,
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
      },
      {
        "frame_index": 12,
        "score": 0.1843,
        "image": "data:image/jpeg;base64,/9j/4QBYRXhpZg..."
      }
    ]
  }
  ```

#### Video Labeling (`POST /clip4clip/label`)
Classify the video content zero-shot.
* **Request (Uploading File):**
  ```bash
  curl -X POST http://localhost:8000/clip4clip/label \
    -F "file=@demo_videos/video-search.mp4" \
    -F "labels_str=driving in city,sports event,nature scene"
  ```
* **Response Example:**
  ```json
  {
    "time_taken": 0.7241,
    "results": [
      {
        "label": "driving in city",
        "score": 0.6542
      },
      {
        "label": "nature scene",
        "score": 0.2415
      },
      {
        "label": "sports event",
        "score": 0.1043
      }
    ]
  }
  ```

#### Video Similarity (`POST /clip4clip/similarity`)
Rank multiple prompt options against the video content.
* **Request (Uploading File):**
  ```bash
  curl -X POST http://localhost:8000/clip4clip/similarity \
    -F "file=@demo_videos/video-search.mp4" \
    -F "prompts_str=a racing vehicle,a parked car,a nature view"
  ```
* **Response Example:**
  ```json
  {
    "results": [
      {
        "prompt": "a racing vehicle",
        "score": 0.4125
      },
      {
        "prompt": "a parked car",
        "score": 0.1842
      },
      {
        "prompt": "a nature view",
        "score": 0.0763
      }
    ]
  }
  ```

---

### 5. CLAP Endpoints (Audio)

#### Audio Segment Retrieval (`POST /clap/search`)
Split audio into segments and rank them by query similarity.
* **Request (Uploading File):**
  ```bash
  curl -X POST http://localhost:8000/clap/search \
    -F "file=@demo_videos/audio-search.mp4" \
    -F "query=roaring engines" \
    -F "segment_seconds=2.0"
  ```
* **Response Example:**
  ```json
  {
    "query": "roaring engines",
    "time_taken": 0.4287,
    "results": [
      {
        "start_time": 2.0,
        "end_time": 4.0,
        "score": 0.6521
      },
      {
        "start_time": 0.0,
        "end_time": 2.0,
        "score": 0.1245
      }
    ]
  }
  ```

#### Audio Labeling (`POST /clap/label`)
Run zero-shot classification on the full audio track.
* **Request (Uploading File):**
  ```bash
  curl -X POST http://localhost:8000/clap/label \
    -F "file=@demo_videos/audio-search.mp4" \
    -F "labels_str=car engine,dog barking,people talking,applause"
  ```
* **Response Example:**
  ```json
  {
    "time_taken": 0.3842,
    "results": [
      {
        "label": "car engine",
        "score": 0.7842
      },
      {
        "label": "dog barking",
        "score": 0.0954
      },
      {
        "label": "people talking",
        "score": 0.0821
      },
      {
        "label": "applause",
        "score": 0.0383
      }
    ]
  }
  ```

---

### 6. Audio+Video Fusion Endpoint (`POST /av/search`)

Combines both visual (CLIP4Clip) and audio (CLAP) encoders to rank video segments by temporal visual+audio similarity to the query.
* **Request (Uploading File):**
  ```bash
  curl -X POST http://localhost:8000/av/search \
    -F "file=@demo_videos/av-search.mp4" \
    -F "query=helicopter blades roaring" \
    -F "visual_weight=0.5" \
    -F "audio_weight=0.5" \
    -F "segment_seconds=3.0"
  ```
* **Response Example:**
  ```json
  {
    "query": "helicopter blades roaring",
    "time_taken": 1.2541,
    "results": [
      {
        "start_time": 3.0,
        "end_time": 6.0,
        "visual_score": 0.2851,
        "audio_score": 0.7423,
        "fused_score": 0.4680,
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
      },
      {
        "start_time": 0.0,
        "end_time": 3.0,
        "visual_score": 0.3214,
        "audio_score": 0.2105,
        "fused_score": 0.2770,
        "image": "data:image/jpeg;base64,/9j/4QBYRXhpZg..."
      }
    ]
  }
  ```
