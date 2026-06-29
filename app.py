import streamlit as st
import torch
from utils.device import DEVICE

try:
    HF_KEY = st.secrets["HF_KEY"]
except Exception:
    HF_KEY = None

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

from utils.model_man import (
    unload_model
)

from models.model_manager import (
    get_clip,
    get_clip4clip,
    get_clap,
    get_tinyclip
)

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

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("Model Manager")

    st.subheader("CLIP4Clip")

    col1, col2 = st.columns(2)

    clip4clip_loaded = (
        st.session_state.clip4clip_model
        is not None
    )

    clip4clip_loading = (
        st.session_state.clip4clip_loading
    )

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

            unload_model(
                "clip4clip_model",
                "clip4clip_processor"
            )

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

    clap_loaded = (
        st.session_state.clap_model
        is not None
    )

    clap_loading = (
        st.session_state.clap_loading
    )

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

            unload_model(
                "clap_model",
                "clap_tokenizer",
                "clap_extractor"
            )

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

    tinyclip_loaded = (
        st.session_state.tinyclip_model
        is not None
    )

    tinyclip_loading = (
        st.session_state.tinyclip_loading
    )

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

            unload_model(
                "tinyclip_pipe",
                "tinyclip_model",
                "tinyclip_processor"
            )

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

    clip_loaded = (
        st.session_state.clip_model
        is not None
    )

    clip_loading = (
        st.session_state.clip_loading
    )

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

            unload_model(
                "clip_model",
                "clip_processor"
            )

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

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        st.write(
            f"GPU Allocated: "
            f"{allocated:.2f} GB"
        )

        st.write(
            f"GPU Reserved: "
            f"{reserved:.2f} GB"
        )

    else:

        st.write(
            f"Running on {DEVICE}"
        )

if st.session_state[
    "clip4clip_loading"
]:

    with st.spinner(
        "Loading CLIP4Clip..."
    ):

        processor, model = (
            get_clip4clip(HF_KEY)
        )

        st.session_state[
            "clip4clip_processor"
        ] = processor

        st.session_state[
            "clip4clip_model"
        ] = model

        st.session_state[
            "clip4clip_loading"
        ] = False

    st.rerun()

if st.session_state[
    "clap_loading"
]:

    with st.spinner(
        "Loading CLAP..."
    ):

        (
            model,
            tokenizer,
            extractor
        ) = get_clap(HF_KEY)

        st.session_state[
            "clap_model"
        ] = model

        st.session_state[
            "clap_tokenizer"
        ] = tokenizer

        st.session_state[
            "clap_extractor"
        ] = extractor

        st.session_state[
            "clap_loading"
        ] = False

    st.rerun()

if st.session_state[
    "tinyclip_loading"
]:

    with st.spinner(
        "Loading TinyCLIP..."
    ):

        (
            pipe,
            model,
            processor
        ) = get_tinyclip(HF_KEY)

        st.session_state[
            "tinyclip_pipe"
        ] = pipe

        st.session_state[
            "tinyclip_model"
        ] = model

        st.session_state[
            "tinyclip_processor"
        ] = processor

        st.session_state[
            "tinyclip_loading"
        ] = False

    st.rerun()

if st.session_state[
    "clip_loading"
]:

    with st.spinner(
        "Loading Original CLIP..."
    ):

        (
            model,
            processor
        ) = get_clip(HF_KEY)

        st.session_state[
            "clip_model"
        ] = model

        st.session_state[
            "clip_processor"
        ] = processor

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

    if (
        st.session_state
        .clip4clip_model
        is not None
    ):

        render_clip4clip_tab()

    else:

        st.warning(
            "Load CLIP4Clip first"
        )

# ==========================================
# CLAP TAB
# ==========================================

with tab2:

    if (
        st.session_state
        .clap_model
        is not None
    ):

        render_clap_tab()

    else:

        st.warning(
            "Load CLAP first"
        )

# ==========================================
# TinyCLIP TAB
# ==========================================

with tab3:

    if (
        st.session_state
        .tinyclip_pipe
        is not None
    ):

        render_tinyclip_tab()

    else:

        st.warning(
            "Load TinyCLIP first"
        )

# ==========================================
# ORIGINAL CLIP TAB
# ==========================================

with tab4:

    if (
        st.session_state
        .clip_model
        is not None
    ):

        render_clip_tab()

    else:

        st.warning(
            "Load Original CLIP first"
        )

# ==========================================
# AV CLIP TAB
# ==========================================
with tab5:

    if (
        st.session_state.clip4clip_model
        and
        st.session_state.clap_model
        is not None
    ):

        render_av_tab()

    else:

        st.warning(
            "Load CLIP4Clip and CLAP first"
        )
