# Multimodal AI Demo

## Requirements

* Python 3.13
* Virtual Environment



## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd main-demo-3
```

### 2. Create Virtual Environment

#### Linux

```bash
python3 -m venv env
source env/bin/activate
```

#### Windows

```cmd
python -m venv env
env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```



## Model Setup

Large model weights are not stored inside the repository.

Download all required model files using:

### Linux

```bash
python3 utils/setup.py
```

### Windows

```cmd
python utils/setup.py
```

This downloads the `.safetensors` weight files for:

* CLIP
* CLAP
* TinyCLIP
* CLIP4Clip

into:

```text
models_local/
├── clap
├── clip
├── clip4clip
└── tinyclip
```

Each directory should contain:

```text
model.safetensors
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the URL shown in the terminal output, typically:

```text
http://localhost:8501
```

If port `8501` is unavailable, Streamlit will automatically select another port and display it in the terminal.



## GPU Support

The application automatically uses CUDA when a compatible GPU is available.

If CUDA is unavailable, execution automatically falls back to CPU.

Current device information and GPU memory usage are displayed in the application sidebar.
