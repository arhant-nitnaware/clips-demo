import os
import tempfile

import matplotlib.pyplot as plt
import streamlit as st

from inference.clip4clip_infer import (
    classify_video,
    query_video,
    encode_frames,
    encode_text
)

from utils.report import (
    show_report
)

from utils.video_utils import (
    extract_frames,
    get_video_frame_count
)

from inference.clip4clip_similarity import (
    compute_video_similarity
)

import torch
import torch.nn.functional as F





def render_clip4clip_tab():

    st.header("CLIP4Clip")

    uploaded = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded is None:
        return

    # ======================================
    # VIDEO DISPLAY
    # ======================================

    video_col1, video_col2, video_col3 = (
        st.columns([1, 3, 1])
    )

    with video_col2:

        st.video(uploaded)

    # ======================================
    # TEMP VIDEO
    # ======================================

    with tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    ) as tmp:

        tmp.write(uploaded.read())

        video_path = tmp.name

    # ======================================
    # FRAME SETTINGS
    # ======================================

    st.markdown(
        "### Frame Extraction Settings"
    )

    total_video_frames = (
        get_video_frame_count(
            video_path
        )
    )

    if total_video_frames <= 0:

        total_video_frames = 64

    slider_max = min(
        total_video_frames,
        128
    )

    default_frames = min(
        12,
        slider_max
    )

    max_frames = st.slider(
        "Number of Frames",
        min_value=2,
        max_value=slider_max,
        value=default_frames,
        step=1,
        help=(
            "Higher frame counts improve "
            "temporal representation but "
            "increase inference time."
        )
    )

    st.caption(
        f"""
        Total video frames detected:
        {total_video_frames}

        Selected frames for inference:
        {max_frames}
        """
    )

    frames = extract_frames(
        video_path,
        max_frames=max_frames
    )

    os.unlink(video_path)

    st.markdown("---")

    # ======================================
    # TASK MODES
    # ======================================

    mode = st.radio(
        "Task",
        [
            "Video Labeling",
            "Frame Retrieval",
            "Video Similarity"
        ],
        horizontal=True,
        key="clip4clip_mode"
    )

    # ======================================
    # MODE TRACKING
    # ======================================

    if (
        "clip4clip_current_mode"
        not in st.session_state
    ):

        st.session_state[
            "clip4clip_current_mode"
        ] = mode

    elif (
        st.session_state[
            "clip4clip_current_mode"
        ] != mode
    ):

        st.session_state[
            "clip4clip_label_result"
        ] = None

        st.session_state[
            "clip4clip_query_result"
        ] = None

        st.session_state[
            "clip4clip_similarity_result"
        ] = None

        st.session_state[
            "clip4clip_current_mode"
        ] = mode

    # ======================================
    # VIDEO LABELING
    # ======================================

    if mode == "Video Labeling":

        st.subheader(
            "Zero-shot Video Classification"
        )

        labels_raw = st.text_area(
            "Candidate Labels",
            value=(
                "car racing\n"
                "driving in city\n"
                "nature scene\n"
                "people talking\n"
                "sports event"
            ),
            height=180
        )

        if st.button(
            "Run Video Labeling"
        ):

            labels = [
                label.strip()
                for label in (
                    labels_raw.splitlines()
                )
                if label.strip()
            ]

            result = classify_video(
                st.session_state
                .clip4clip_processor,

                st.session_state
                .clip4clip_model,

                frames,

                labels
            )

            st.session_state[
                "clip4clip_label_result"
            ] = result

        result = st.session_state.get(
            "clip4clip_label_result"
        )

        if result is not None:

            st.markdown("---")

            show_report(
                "CLIP4Clip Video Labeling",
                result["input_details"],
                result["time_taken"],
                result["results"]
            )

    # ======================================
    # FRAME RETRIEVAL
    # ======================================

    elif mode == "Frame Retrieval":

        st.subheader(
            "Text-based Frame Retrieval"
        )

        query = st.text_input(
            "Query",
            value="a yellow race car"
        )

        top_k = st.slider(
            "Top Frames",
            min_value=1,
            max_value=12,
            value=4
        )

        if st.button(
            "Retrieve Frames"
        ):

            result = query_video(
                st.session_state
                .clip4clip_processor,

                st.session_state
                .clip4clip_model,

                frames,

                query
            )

            st.session_state[
                "clip4clip_query_result"
            ] = result

            st.session_state[
                "clip4clip_top_k"
            ] = top_k

        result = st.session_state.get(
            "clip4clip_query_result"
        )

        if result is not None:

            st.markdown("---")

            st.write(
                f"### Query: `{result['query']}`"
            )

            ranked_frames = result[
                "frame_scores"
            ]

            top_frames = ranked_frames[
                :st.session_state.get(
                    "clip4clip_top_k",
                    4
                )
            ]

            num_columns = 2

            for start_idx in range(
                0,
                len(top_frames),
                num_columns
            ):

                current_row = top_frames[
                    start_idx:
                    start_idx + num_columns
                ]

                columns = st.columns(
                    num_columns
                )

                for idx, frame_data in enumerate(
                    current_row
                ):

                    frame_idx, score = (
                        frame_data
                    )

                    columns[idx].image(
                        frames[frame_idx],
                        caption=(
                            f"Frame {frame_idx}\n"
                            f"Score: {score:.4f}"
                        ),
                        use_container_width=True
                    )

            st.markdown(
                "### Frame Retrieval Scores"
            )

            frame_indices = [
                frame_idx
                for frame_idx, _
                in ranked_frames
            ]

            scores = [
                score
                for _, score
                in ranked_frames
            ]

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            bars = ax.bar(
                frame_indices,
                scores
            )

            ax.set_xlabel(
                "Frame Index"
            )

            ax.set_ylabel(
                "Similarity Score"
            )

            ax.set_title(
                "Frame-wise Similarity"
            )

            ax.set_xticks(
                frame_indices
            )

            min_score = min(scores)
            max_score = max(scores)

            margin = max(
                (
                    max_score - min_score
                ) * 0.15,
                0.01
            )

            ax.set_ylim(
                min_score - margin,
                max_score + margin
            )

            for bar, score in zip(
                bars,
                scores
            ):

                ax.text(
                    bar.get_x() +
                    bar.get_width() / 2,

                    score,

                    f"{score:.3f}",

                    ha="center",
                    va="bottom",
                    fontsize=8
                )

            graph_col1, graph_col2, graph_col3 = (
                st.columns([1, 2, 1])
            )

            with graph_col2:

                st.pyplot(
                    fig,
                    use_container_width=True
                )

            st.markdown(
                "### Inference Details"
            )

            st.write(
                f"Time Taken: "
                f"{result['time_taken']:.4f}s"
            )

    # ======================================
    # VIDEO SIMILARITY
    # ======================================

    else:

        st.subheader(
            "Temporal Video Similarity"
        )

        prompts_raw = st.text_area(
            "Prompts",
            value=(
                "a car driving left to right\n"
                "a car driving right to left\n"
                "a parked car\n"
                "a racing vehicle"
            ),
            height=180
        )

        if st.button(
            "Compute Video Similarity"
        ):

            prompts = [
                prompt.strip()
                for prompt in (
                    prompts_raw.splitlines()
                )
                if prompt.strip()
            ]

            results = compute_video_similarity(
                st.session_state
                .clip4clip_processor,

                st.session_state
                .clip4clip_model,

                frames,

                prompts
            )

            st.session_state[
                "clip4clip_similarity_result"
            ] = results

        results = st.session_state.get(
            "clip4clip_similarity_result"
        )

        if results is not None:

            st.markdown("---")

            st.markdown(
                "### Video Similarity Scores"
            )

            labels = [
                item[0]
                for item in results
            ]

            scores = [
                item[1]
                for item in results
            ]

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            bars = ax.bar(
                labels,
                scores
            )

            ax.set_ylabel(
                "Similarity Score"
            )

            ax.set_title(
                "Video-Text Similarity"
            )

            ax.tick_params(
                axis="x",
                rotation=15
            )

            min_score = min(scores)
            max_score = max(scores)

            margin = max(
                (
                    max_score - min_score
                ) * 0.15,
                0.01
            )

            ax.set_ylim(
                min_score - margin,
                max_score + margin
            )

            for bar, score in zip(
                bars,
                scores
            ):

                ax.text(
                    bar.get_x() +
                    bar.get_width() / 2,

                    score,

                    f"{score:.4f}",

                    ha="center",
                    va="bottom",
                    fontsize=8
                )

            graph_col1, graph_col2, graph_col3 = (
                st.columns([1, 2, 1])
            )

            with graph_col2:

                st.pyplot(
                    fig,
                    use_container_width=True
                )

            st.markdown(
                "### Similarity Ranking"
            )

            for prompt, score in results:

                st.write(
                    f"- `{prompt}` → "
                    f"{score:.4f}"
                )

    # ======================================
    # MODEL INFO
    # ======================================

    st.markdown("---")

    st.markdown(
        """
        ### CLIP4Clip

        **Domain:** Video-Text Retrieval

        **Paper:**  
        https://arxiv.org/abs/2104.08860

        **Repository:**  
        https://github.com/ArrowLuo/CLIP4Clip

        **Organization:** Microsoft Research Asia

        **Overview:**  
        CLIP4Clip extends OpenAI CLIP for video understanding.
        Instead of processing a single image, it extracts frame-level
        visual embeddings from videos and aligns them with text
        embeddings inside a shared semantic space.

        **Training Strategy:**  
        The model is trained using contrastive learning on
        video-caption pairs. Video embeddings are constructed
        using sampled frame embeddings aggregated temporally.

        **Datasets Used:**  
        - MSR-VTT  
        - ActivityNet  
        - DiDeMo

        **Inference Pipeline:**  
        1. Sample frames from video  
        2. Encode frames using CLIP vision encoder  
        3. Aggregate temporal embeddings  
        4. Encode text query  
        5. Compute cosine similarity

        **Applications:**  
        - Video retrieval  
        - Video caption matching  
        - Semantic frame localization  
        - Video search
        """
    )