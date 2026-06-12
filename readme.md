# Multimodal AI Demo

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd main-demo-3
```

### 2. Create Virtual Environment

#### Python 3.13.5 was used while developing this codebase, therefore python 3.13.x is recommended.

#### Linux

```bash
python3 -m venv env
source env/bin/activate
```

#### Windows

```bash
python -m venv env
env\Scripts\activate
```

#### Conda

```bash
conda create --name <env-name>
conda activate <env-name>
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Above command will not install torch torchvision

### CPU torch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### GPU torch

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Check torch website: https://pytorch.org/get-started/locally/ to check which ```cu``` version you should install based on your driver version and CUDA version, use ```nvidia-smi``` to check. cu130 was used in development.

## Model Setup

Large model weights are not stored inside the repository.

Download all required model files using:

### Linux

```bash
python3 utils/setup.py
```

### Windows

```bash
python utils/setup.py
```

This downloads the `.safetensors` weight files for:

* CLIP
* CLAP
* TinyCLIP
* CLIP4Clip

into respective model folder inside models_local/

Each directory should contain:

```bash
model.safetensors
```

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

## Take a look at demo videos at: 

Audio Search : https://youtu.be/D4UeC6aDbd0

Video Search : https://youtu.be/6ssBksFQoHo

AV Search : https://youtu.be/lBf_itk5blk