import streamlit as st

from config import HF_TOKEN

from models.clip4clip_model import (
    load_clip4clip
)

from models.clap_model import (
    load_clap
)

from models.tinyclip_model import (
    load_tinyclip
)

from models.clip_model import (
    load_clip
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

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header(
        "Model Loader"
    )

    st.markdown(
        """
        Load only the models you need.

        Models download automatically
        on first use and are cached
        locally afterwards.
        """
    )

    st.markdown("---")

    # ======================================
    # CLIP4Clip
    # ======================================

    if st.button(
        "Load CLIP4Clip",
        use_container_width=True
    ):

        with st.spinner(
            "Loading CLIP4Clip..."
        ):

            processor, model = (
                load_clip4clip(
                    HF_TOKEN
                )
            )

            st.session_state[
                "clip4clip_processor"
            ] = processor

            st.session_state[
                "clip4clip_model"
            ] = model

        st.success(
            "CLIP4Clip Loaded"
        )

    # ======================================
    # CLAP
    # ======================================

    if st.button(
        "Load CLAP",
        use_container_width=True
    ):

        with st.spinner(
            "Loading CLAP..."
        ):

            (
                model,
                tokenizer,
                extractor
            ) = load_clap(HF_TOKEN)

            st.session_state[
                "clap_model"
            ] = model

            st.session_state[
                "clap_tokenizer"
            ] = tokenizer

            st.session_state[
                "clap_extractor"
            ] = extractor

        st.success(
            "CLAP Loaded"
        )

    # ======================================
    # TinyCLIP
    # ======================================

    if st.button(
        "Load TinyCLIP",
        use_container_width=True
    ):

        with st.spinner(
            "Loading TinyCLIP..."
        ):

            (
                pipe,
                model,
                processor
            ) = load_tinyclip(
                HF_TOKEN
            )

            st.session_state[
                "tinyclip_pipe"
            ] = pipe

            st.session_state[
                "tinyclip_model"
            ] = model

            st.session_state[
                "tinyclip_processor"
            ] = processor

        st.success(
            "TinyCLIP Loaded"
        )

    # ======================================
    # ORIGINAL CLIP
    # ======================================

    if st.button(
        "Load Original CLIP",
        use_container_width=True
    ):

        with st.spinner(
            "Loading Original CLIP..."
        ):

            model, processor = (
                load_clip(
                    HF_TOKEN
                )
            )

            st.session_state[
                "clip_model"
            ] = model

            st.session_state[
                "clip_processor"
            ] = processor

        st.success(
            "Original CLIP Loaded"
        )

# ==========================================
# TABS
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "CLIP4Clip",
        "CLAP",
        "TinyCLIP",
        "Original CLIP"
    ]
)

# ==========================================
# CLIP4Clip TAB
# ==========================================

with tab1:

    if (
        st.session_state
        .clip4clip_model
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
    ):

        render_clip_tab()

    else:

        st.warning(
            "Load Original CLIP first"
        )