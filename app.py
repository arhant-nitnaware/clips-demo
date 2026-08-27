import streamlit as st
import requests
from utils.config import get_api_url

API_URL = get_api_url()

def get_backend_models():
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {
        "clip": "Not Loaded",
        "clip4clip": "Not Loaded",
        "clap": "Not Loaded",
        "tinyclip": "Not Loaded"
    }

def load_backend_model(model_name: str):
    try:
        requests.post(f"{API_URL}/models/load/{model_name}")
    except Exception:
        pass

def unload_backend_model(model_name: str):
    try:
        requests.post(f"{API_URL}/models/unload/{model_name}")
    except Exception:
        pass

from ui.clip4clip_tab import (
    render_clip4clip_tab
)

from ui.clap_tab import (
    render_clap_tab
)

from ui.tinyclip_tab import (
    render_tinyclip_tab
)

from ui.clip_tab import (
    render_clip_tab
)

from ui.av_tab import (
    render_av_tab
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Multimodal AI",
    layout="wide"
)

st.title(
    "Multimodal AI Demo"
)

# ==========================================
# SESSION STATE
# ==========================================

# ---------- CLIP4Clip ----------

if (
    "clip4clip_model"
    not in st.session_state
):

    st.session_state.clip4clip_model = None

if (
    "clip4clip_processor"
    not in st.session_state
):

    st.session_state.clip4clip_processor = None

if (
    "clip4clip_result"
    not in st.session_state
):

    st.session_state.clip4clip_result = None

if (
    "clip4clip_frames"
    not in st.session_state
):

    st.session_state.clip4clip_frames = None

# ---------- CLAP ----------

if (
    "clap_model"
    not in st.session_state
):

    st.session_state.clap_model = None

if (
    "clap_tokenizer"
    not in st.session_state
):

    st.session_state.clap_tokenizer = None

if (
    "clap_extractor"
    not in st.session_state
):

    st.session_state.clap_extractor = None

if (
    "clap_result"
    not in st.session_state
):

    st.session_state.clap_result = None

# ---------- TinyCLIP ----------

if (
    "tinyclip_pipe"
    not in st.session_state
):

    st.session_state.tinyclip_pipe = None

if (
    "tinyclip_model"
    not in st.session_state
):

    st.session_state.tinyclip_model = None

if (
    "tinyclip_processor"
    not in st.session_state
):

    st.session_state.tinyclip_processor = None

if (
    "tinyclip_result"
    not in st.session_state
):

    st.session_state.tinyclip_result = None

if (
    "tinyclip_retrieval_result"
    not in st.session_state
):

    st.session_state[
        "tinyclip_retrieval_result"
    ] = None

if (
    "tinyclip_top_k"
    not in st.session_state
):


    st.session_state[
        "tinyclip_top_k"
    ] = 4

if "av_result" not in st.session_state:

    st.session_state.av_result = None

# ---------- Original CLIP ----------

if (
    "clip_model"
    not in st.session_state
):

    st.session_state.clip_model = None

if (
    "clip_processor"
    not in st.session_state
):

    st.session_state.clip_processor = None

if (
    "clip_label_result"
    not in st.session_state
):

    st.session_state.clip_label_result = None

if (
    "clip_retrieval_result"
    not in st.session_state
):

    st.session_state.clip_retrieval_result = None

# - - - -- - - - for blocking buttons during loading 
if (
    "clip4clip_loading"
    not in st.session_state
):

    st.session_state[
        "clip4clip_loading"
    ] = False

if (
    "clap_loading"
    not in st.session_state
):

    st.session_state[
        "clap_loading"
    ] = False

if (
    "tinyclip_loading"
    not in st.session_state
):

    st.session_state[
        "tinyclip_loading"
    ] = False

if (
    "clip_loading"
    not in st.session_state
):

    st.session_state[
        "clip_loading"
    ] = False

# Query backend models status
backend_models = get_backend_models()

clip4clip_loaded = backend_models.get("clip4clip") == "Loaded"
clap_loaded = backend_models.get("clap") == "Loaded"
tinyclip_loaded = backend_models.get("tinyclip") == "Loaded"
clip_loaded = backend_models.get("clip") == "Loaded"

clip4clip_loading = st.session_state.get("clip4clip_loading", False)
clap_loading = st.session_state.get("clap_loading", False)
tinyclip_loading = st.session_state.get("tinyclip_loading", False)
clip_loading = st.session_state.get("clip_loading", False)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("Model Manager")

    st.subheader("CLIP4Clip")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Load",
            key="load_clip4clip",
            use_container_width=True,
            disabled=(
                clip4clip_loaded
                or
                clip4clip_loading
            )
        ):

            st.session_state[
                "clip4clip_loading"
            ] = True

            st.rerun()

    with col2:

        if st.button(
            "Unload",
            key="unload_clip4clip",
            use_container_width=True,
            disabled=(
                not clip4clip_loaded
                or
                clip4clip_loading
            )
        ):

            unload_backend_model("clip4clip")

            st.rerun()

    if clip4clip_loading:

        st.info(
            "Loading..."
        )

    elif clip4clip_loaded:

        st.success(
            "Loaded CLIP4Clip"
        )

    else:

        st.error(
            "Not Loaded"
        )

    st.markdown("---")
    
    st.subheader("CLAP")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Load",
            key="load_clap",
            use_container_width=True,
            disabled=(
                clap_loaded
                or
                clap_loading
            )
        ):

            st.session_state[
                "clap_loading"
            ] = True

            st.rerun()

    with col2:

        if st.button(
            "Unload",
            key="unload_clap",
            use_container_width=True,
            disabled=(
                not clap_loaded
                or
                clap_loading
            )
        ):

            unload_backend_model("clap")

            st.rerun()

    if clap_loading:

        st.info(
            "Loading..."
        )

    elif clap_loaded:

        st.success(
            "Loaded CLAP"
        )

    else:

        st.error(
            "Not Loaded"
        )

    st.markdown("---")
    
    st.subheader("TinyCLIP")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Load",
            key="load_tinyclip",
            use_container_width=True,
            disabled=(
                tinyclip_loaded
                or
                tinyclip_loading
            )
        ):

            st.session_state[
                "tinyclip_loading"
            ] = True

            st.rerun()

    with col2:

        if st.button(
            "Unload",
            key="unload_tinyclip",
            use_container_width=True,
            disabled=(
                not tinyclip_loaded
                or
                tinyclip_loading
            )
        ):

            unload_backend_model("tinyclip")

            st.rerun()

    if tinyclip_loading:

        st.info(
            "Loading..."
        )

    elif tinyclip_loaded:

        st.success(
            "Loaded TinyCLIP"
        )

    else:

        st.error(
            "Not Loaded"
        )

    st.markdown("---")

    st.subheader("Original CLIP")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Load",
            key="load_clip",
            use_container_width=True,
            disabled=(
                clip_loaded
                or
                clip_loading
            )
        ):

            st.session_state[
                "clip_loading"
            ] = True

            st.rerun()

    with col2:

        if st.button(
            "Unload",
            key="unload_clip",
            use_container_width=True,
            disabled=(
                not clip_loaded
                or
                clip_loading
            )
        ):

            unload_backend_model("clip")

            st.rerun()

    if clip_loading:

        st.info(
            "Loading..."
        )

    elif clip_loaded:

        st.success(
            "Loaded Original CLIP"
        )

    else:

        st.error(
            "Not Loaded"
        )

    st.markdown("---")

    try:
        gpu_resp = requests.get(f"{API_URL}/gpu")
        if gpu_resp.status_code == 200:
            gpu_data = gpu_resp.json()
            if gpu_data.get("cuda_available"):
                st.write(f"GPU Allocated: {gpu_data['allocated_gb']:.2f} GB")
                st.write(f"GPU Reserved: {gpu_data['reserved_gb']:.2f} GB")
            else:
                st.write(f"Running on {gpu_data.get('device', 'CPU')}")
        else:
            st.write("FastAPI server offline/unreachable")
    except Exception:
        st.write("FastAPI server offline/unreachable")

if st.session_state.get("clip4clip_loading", False):

    with st.spinner(
        "Loading CLIP4Clip on backend..."
    ):

        load_backend_model("clip4clip")

        st.session_state[
            "clip4clip_loading"
        ] = False

    st.rerun()

if st.session_state.get("clap_loading", False):

    with st.spinner(
        "Loading CLAP on backend..."
    ):

        load_backend_model("clap")

        st.session_state[
            "clap_loading"
        ] = False

    st.rerun()

if st.session_state.get("tinyclip_loading", False):

    with st.spinner(
        "Loading TinyCLIP on backend..."
    ):

        load_backend_model("tinyclip")

        st.session_state[
            "tinyclip_loading"
        ] = False

    st.rerun()

if st.session_state.get("clip_loading", False):

    with st.spinner(
        "Loading Original CLIP on backend..."
    ):

        load_backend_model("clip")

        st.session_state[
            "clip_loading"
        ] = False

    st.rerun()

# ==========================================
# TABS
# ==========================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "CLIP4Clip",
        "CLAP",
        "TinyCLIP",
        "Original CLIP",
        "Audio+Video"
    ]
)

# ==========================================
# CLIP4Clip TAB
# ==========================================

with tab1:

    if clip4clip_loaded:

        render_clip4clip_tab()

    else:

        st.warning(
            "Load CLIP4Clip first"
        )

# ==========================================
# CLAP TAB
# ==========================================

with tab2:

    if clap_loaded:

        render_clap_tab()

    else:

        st.warning(
            "Load CLAP first"
        )

# ==========================================
# TinyCLIP TAB
# ==========================================

with tab3:

    if tinyclip_loaded:

        render_tinyclip_tab()

    else:

        st.warning(
            "Load TinyCLIP first"
        )

# ==========================================
# ORIGINAL CLIP TAB
# ==========================================

with tab4:

    if clip_loaded:

        render_clip_tab()

    else:

        st.warning(
            "Load Original CLIP first"
        )

# ==========================================
# AV CLIP TAB
# ==========================================
with tab5:

    if clip4clip_loaded and clap_loaded:

        render_av_tab()

    else:

        st.warning(
            "Load CLIP4Clip and CLAP first"
        )
