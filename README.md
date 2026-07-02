# Multimodal AI Inference API Developer Guide

This repository contains the **Multimodal AI REST API** backend and **Streamlit** client application. The backend exposes state-of-the-art multimodal models—**CLIP**, **TinyCLIP**, **CLIP4Clip**, and **CLAP**—over REST endpoints. Both applications share a centralized **Model Manager** to cache weights in CPU/GPU memory.

---

## 1. Quick Start

### Installation
Install all dependencies inside a virtual environment:
```bash
pip install -r requirements.txt
```

### Run FastAPI Backend
Start the REST API server on `http://localhost:8000`:
```bash
python api.py
```
Or start manually via Uvicorn:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Run Streamlit Client
Start the web demonstration client:
```bash
streamlit run app.py
```

---

## 2. Frontend Integration Guide

All inference APIs expect inputs sent as **Multipart Form Data** (`multipart/form-data`). This allows uploading raw binary files directly from the browser's `<input type="file">` element.

### Uploading Media via JavaScript `fetch`
To query the API from a frontend application (e.g., React, Vue, or Vanilla JS):

```javascript
async function searchVideoFrames(fileBlob, queryText, topK = 6) {
  const formData = new FormData();
  // Upload the binary file
  formData.append("file", fileBlob, "user-video.mp4");
  // Set parameters
  formData.append("query", queryText);
  formData.append("top_k", topK);
  formData.append("max_frames", 12);

  try {
    const response = await fetch("http://localhost:8000/clip4clip/search", {
      method: "POST",
      body: formData // Browser automatically sets Content-Type to multipart/form-data
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "API request failed");
    }

    const data = await response.json();
    console.log("Query Time:", data.time_taken);
    return data.results; // Array of top matching frames
  } catch (error) {
    console.error("Inference Error:", error.message);
  }
}
```

### Rendering Base64 Frame Images in React / HTML
The video frame retrieval (`/clip4clip/search`) and AV temporal fusion (`/av/search`) endpoints return matching frames encoded as **Base64 Data URIs** under the `"image"` field. 

You can bind these directly to the `src` attribute of a standard HTML `<img>` tag:

```jsx
// React Component to render matching frame results
function FrameResult({ frameIndex, score, base64Image }) {
  return (
    <div className="frame-card">
      <h4>Frame {frameIndex} (Score: {score.toFixed(4)})</h4>
      {base64Image ? (
        <img 
          src={base64Image} 
          alt={`Matching frame ${frameIndex}`} 
          style={{ width: '100%', borderRadius: '8px' }} 
        />
      ) : (
        <p>No preview frame available</p>
      )}
    </div>
  );
}
```

### Playing Base64 Audio Segments in HTML5
The CLAP audio segment retrieval (`/clap/search`) endpoint returns base64-encoded WAV files under the `"audio"` field. 

You can bind these directly to the `<audio>` player's `src` attribute:

```javascript
// Playing retrieved base64 audio dynamically in JavaScript
function playAudioSegment(base64AudioString) {
  const audioPlayer = new Audio(base64AudioString); // base64AudioString looks like "data:audio/wav;base64,..."
  audioPlayer.play();
}
```

---

## 3. Complete API Specifications

### Common Response Headers
All endpoints support **CORS** (Cross-Origin Resource Sharing) with the following default headers:
- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Methods: *`
- `Access-Control-Allow-Headers: *`

### General Error Response Schema
Whenever an error occurs (such as a missing audio track in video processing, invalid files, or runtime exceptions), the API returns a `500` or `400` status code with a descriptive detail payload:
```json
{
  "detail": "No audio stream found in the video file."
}
```

---

### A. Media Metadata Endpoint

#### `POST /media/info`
Extracts duration, framerate, resolution, audio sample rate, and size from a media file.
* **Request Params (`multipart/form-data`):**
  - `file`: `UploadFile` (Optional binary file upload)
  - `video_path`: `str` (Optional path to a local server file if not uploading)
* **Response:**
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
Compares a query string against a batch of images and ranks them by similarity.
* **Request Params (`multipart/form-data`):**
  - `files`: `List[UploadFile]` (Optional list of image file uploads)
  - `image_paths`: `List[str]` (Optional list of local server image paths if not uploading)
  - `query`: `str` (Search description query, e.g., `"a red sports car"`)
* **Response:**
  ```json
  {
    "query": "a red sports car",
    "time_taken": 0.1245,
    "results": [
      { "image_path": "car1.jpg", "score": 0.2842 },
      { "image_path": "car2.jpg", "score": 0.1105 }
    ]
  }
  ```

#### `POST /clip/label`
Run zero-shot classification on a single image.
* **Request Params (`multipart/form-data`):**
  - `file`: `UploadFile` (Optional image file upload)
  - `image_path`: `str` (Optional local image path)
  - `labels_str`: `str` (Comma-separated candidate labels, e.g., `"nature,car,city,dog"`)
* **Response:**
  ```json
  {
    "time_taken": 0.0812,
    "results": [
      { "label": "nature", "score": 0.8125 },
      { "label": "city", "score": 0.1204 }
    ]
  }
  ```

---

### C. TinyCLIP (Lightweight Image Encoders)
Provides the same image search and zero-shot labeling capabilities as standard CLIP, but uses a highly optimized, resource-efficient architecture.

#### `POST /tinyclip/search`
* **Request Params (`multipart/form-data`):**
  - `files`: `List[UploadFile]`
  - `image_paths`: `List[str]`
  - `query`: `str`
* **Response:** same format as `/clip/search`.

#### `POST /tinyclip/label`
* **Request Params (`multipart/form-data`):**
  - `file`: `UploadFile`
  - `image_path`: `str`
  - `labels_str`: `str`
* **Response:** same format as `/clip/label`.

---

### D. CLIP4Clip (Video Frame Encoders)

#### `POST /clip4clip/search`
Compares video frames to a text query, and returns the top matching frames.
* **Request Params (`multipart/form-data`):**
  - `file`: `UploadFile` (Optional video file upload)
  - `video_path`: `str` (Optional local video path)
  - `query`: `str` (Visual target query)
  - `max_frames`: `int` (Default: `12`. Total frames to sample from the video)
  - `top_k`: `int` (Default: `4`. Number of top matching frames to return with Base64 representations)
* **Response:**
  ```json
  {
    "query": "a yellow race car",
    "time_taken": 0.6543,
    "results": [
      {
        "frame_index": 8,
        "score": 0.2981,
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
      }
    ],
    "all_scores": [
      { "frame_index": 0, "score": 0.0821 },
      { "frame_index": 8, "score": 0.2981 }
    ]
  }
  ```

#### `POST /clip4clip/label`
Run zero-shot video classification based on visual temporal features.
* **Request Params (`multipart/form-data`):**
  - `file`: `UploadFile`
  - `video_path`: `str`
  - `labels_str`: `str` (Comma-separated candidate labels)
  - `max_frames`: `int` (Total frames to sample from the video)
* **Response:** same format as `/clip/label`.

#### `POST /clip4clip/batch_search`
Compares video frames across multiple video files to a text query, and returns the top matching frames globally.
* **Request Params (`multipart/form-data`):**
  - `files`: `List[UploadFile]` (Optional list of video file uploads)
  - `video_paths`: `str` (Optional JSON string or comma-separated list of local video paths)
  - `query`: `str` (Visual target query)
  - `max_frames`: `int` (Default: `12`. Total frames to sample from each video)
  - `top_k`: `int` (Default: `4`. Number of top matching frames to return globally with Base64 representations)
* **Response:**
  ```json
  {
    "query": "a yellow race car",
    "time_taken": 1.2543,
    "results": [
      {
        "video_name": "video1.mp4",
        "frame_index": 8,
        "timestamp": 4.25,
        "score": 0.2981,
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
      }
    ],
    "all_scores": [
      { "video_name": "video1.mp4", "frame_index": 0, "timestamp": 0.0, "score": 0.0821 },
      { "video_name": "video1.mp4", "frame_index": 8, "timestamp": 4.25, "score": 0.2981 }
    ]
  }
  ```

---

### E. CLAP (Contrastive Language-Audio Pretraining)
*Note: Audio operations will return a `500` error with `No audio stream found in the video file.` if the input file does not contain a valid audio track.*

#### `POST /clap/search`
Segments the audio track of a file and ranks audio segments based on query similarity.
* **Request Params (`multipart/form-data`):**
  - `file`: `UploadFile` (Optional audio or video file upload)
  - `video_path`: `str` (Optional local audio/video path)
  - `query`: `str` (Audio descriptor query, e.g. `"dog barking"`)
  - `segment_seconds`: `float` (Duration of each audio segment window in seconds)
  - `top_k`: `int` (Default: `4`. Number of matching audio segments to return with Base64 audio arrays)
* **Response:**
  ```json
  {
    "query": "dog barking",
    "time_taken": 0.3512,
    "results": [
      {
        "start_time": 6.0,
        "end_time": 9.0,
        "score": 0.7241,
        "audio": "data:audio/wav;base64,UklGRtS5AgBXQV..."
      }
    ]
  }
  ```

#### `POST /clap/label`
Classify the full audio track zero-shot.
* **Request Params (`multipart/form-data`):**
  - `file`: `UploadFile`
  - `video_path`: `str`
  - `labels_str`: `str` (Comma-separated candidate audio events)
* **Response:** same format as `/clip/label`.

---

### F. Audio+Video Fusion Encoders

#### `POST /av/search`
Combines visual (CLIP4Clip) and audio (CLAP) features synchronously to find sections of a video that match a multimodal query.
* **Request Params (`multipart/form-data`):**
  - `file`: `UploadFile` (Optional video file upload)
  - `video_path`: `str` (Optional local video path)
  - `query`: `str` (Target query description)
  - `visual_weight`: `float` (Default: `0.5`. Weight given to the visual model similarity score)
  - `audio_weight`: `float` (Default: `0.5`. Weight given to the audio model similarity score)
  - `segment_seconds`: `float` (Default: `5.0`. Duration of each segment)
  - `max_frames`: `int` (Default: `8`. Max frames to sample from each segment)
  - `top_k`: `int` (Default: `4`. Number of results to return with representative frames)
* **Response:**
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
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
      }
    ]
  }
  ```

#### `POST /av/batch_search`
Combines visual (CLIP4Clip) and audio (CLAP) features synchronously across multiple video files to find sections that match a query, returning globally ranked matches.
* **Request Params (`multipart/form-data`):**
  - `files`: `List[UploadFile]` (Optional list of video file uploads)
  - `video_paths`: `str` (Optional JSON string or comma-separated list of local video paths)
  - `query`: `str` (Target query description)
  - `visual_weight`: `float` (Default: `0.5`. Weight for visual model similarity)
  - `audio_weight`: `float` (Default: `0.5`. Weight for audio model similarity)
  - `segment_seconds`: `float` (Default: `5.0`. Duration of segments in seconds)
  - `max_frames`: `int` (Default: `8`. Max frames to sample from each segment)
  - `top_k`: `int` (Default: `4`. Number of global results to return with representative frames)
* **Response:**
  ```json
  {
    "query": "helicopter blades roaring",
    "time_taken": 2.4589,
    "results": [
      {
        "video_name": "chopper.mp4",
        "start_time": 5.0,
        "end_time": 10.0,
        "visual_score": 0.4512,
        "audio_score": 0.8124,
        "fused_score": 0.6318,
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
      }
    ]
  }
  ```
