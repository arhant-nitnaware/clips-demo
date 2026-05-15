import os
import tempfile

import cv2
import matplotlib.pyplot as plt
import streamlit as st

from utils.video_utils import (
    extract_segment_frames,
    get_video_duration
)

from utils.audio_utils import (
    extract_audio_from_video
)

from inference.av_retrieval import (
    run_av_retrieval
)


def render_av_tab():

    st.header(
        "Multimodal Temporal Retrieval"
    )

    uploaded = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"],
        key="av_upload"
    )

    if uploaded is None:
        return

    # =====================================
    # VIDEO DISPLAY
    # =====================================

    video_col1, video_col2, video_col3 = (
        st.columns([1, 3, 1])
    )

    with video_col2:

        st.video(uploaded)

    # =====================================
    # SAVE TEMP VIDEO
    # =====================================

    with tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    ) as tmp:

        tmp.write(uploaded.read())

        video_path = tmp.name

    # =====================================
    # VIDEO METADATA
    # =====================================

    cap = cv2.VideoCapture(
        video_path
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    duration = get_video_duration(
        video_path
    )

    file_size_mb = (
        os.path.getsize(video_path)
        / (1024 * 1024)
    )

    # =====================================
    # AUDIO EXTRACTION
    # =====================================

    waveform, sample_rate = (
        extract_audio_from_video(
            video_path
        )
    )

    # =====================================
    # VIDEO INFORMATION
    # =====================================

    st.markdown("---")

    st.subheader(
        "Video Information"
    )

    info_col1, info_col2, info_col3 = (
        st.columns(3)
    )

    with info_col1:

        st.metric(
            "Duration",
            f"{duration:.2f}s"
        )

        st.metric(
            "FPS",
            f"{fps:.2f}"
        )

    with info_col2:

        st.metric(
            "Total Frames",
            frame_count
        )

        st.metric(
            "Resolution",
            f"{width}×{height}"
        )

    with info_col3:

        st.metric(
            "Audio Sample Rate",
            sample_rate
        )

        st.metric(
            "File Size",
            f"{file_size_mb:.2f} MB"
        )

    st.markdown("---")

    # =====================================
    # QUERY SETTINGS
    # =====================================

    st.subheader(
        "Retrieval Settings"
    )

    query = st.text_input(
        "Query",
        value=(
            "race car engine"
        )
    )

    settings_col1, settings_col2 = (
        st.columns(2)
    )

    with settings_col1:

        segment_seconds = st.slider(
            "Segment Duration (seconds)",
            min_value=1,
            max_value=10,
            value=3
        )

    with settings_col2:

        max_frames = st.slider(
            "Frames Per Segment",
            min_value=2,
            max_value=16,
            value=8
        )

    st.markdown("---")

    # =====================================
    # MODALITY WEIGHTS
    # =====================================

    st.subheader(
        "Modality Fusion"
    )

    visual_weight = st.slider(
        "Visual Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )

    audio_weight = (
        1.0 - visual_weight
    )

    st.caption(
        f"""
        Visual Contribution:
        {visual_weight:.2f}

        Audio Contribution:
        {audio_weight:.2f}
        """
    )

    st.markdown("---")

    # =====================================
    # RUN RETRIEVAL
    # =====================================

    if st.button(
        "Run Temporal Retrieval"
    ):

        segments = []

        current_t = 0.0

        while current_t < duration:

            end_t = min(
                current_t
                + segment_seconds,

                duration
            )

            frames = (
                extract_segment_frames(
                    video_path,
                    current_t,
                    end_t,
                    max_frames=
                    max_frames
                )
            )

            start_sample = int(
                current_t
                * sample_rate
            )

            end_sample = int(
                end_t
                * sample_rate
            )

            audio_segment = (
                waveform[
                    start_sample:
                    end_sample
                ]
            )

            if len(frames) == 0:

                current_t += (
                    segment_seconds
                )

                continue

            segments.append({

                "start_time":
                    current_t,

                "end_time":
                    end_t,

                "frames":
                    frames,

                "audio":
                    audio_segment,

                "sample_rate":
                    sample_rate
            })

            current_t += (
                segment_seconds
            )

        result = run_av_retrieval(

            st.session_state
            .clip4clip_processor,

            st.session_state
            .clip4clip_model,

            st.session_state
            .clap_model,

            st.session_state
            .clap_tokenizer,

            st.session_state
            .clap_extractor,

            segments,

            query,

            visual_weight,
            audio_weight
        )

        st.session_state[
            "av_result"
        ] = result

    # =====================================
    # RESULT HANDLING
    # =====================================

    result = st.session_state.get(
        "av_result"
    )

    if (
        result is not None
        and
        "results" not in result
    ):

        st.session_state[
            "av_result"
        ] = None

        result = None

    # =====================================
    # SHOW RESULTS
    # =====================================

    if result is not None:

        st.markdown("---")

        st.subheader(
            "Top Matching Segments"
        )

        top_results = result[
            "results"
        ][:5]

        timeline_x = []
        timeline_y = []

        for idx, segment in enumerate(
            top_results
        ):

            start_t = segment[
                "start_time"
            ]

            end_t = segment[
                "end_time"
            ]

            fused_score = segment[
                "fused_score"
            ]

            visual_score = segment[
                "visual_score"
            ]

            audio_score = segment[
                "audio_score"
            ]

            st.markdown(
                f"""
                ### Segment
                {start_t:.1f}s
                → {end_t:.1f}s
                """
            )

            score_col1, score_col2, score_col3 = (
                st.columns(3)
            )

            with score_col1:

                st.metric(
                    "Fused",
                    f"{fused_score:.4f}"
                )

            with score_col2:

                st.metric(
                    "Visual",
                    f"{visual_score:.4f}"
                )

            with score_col3:

                st.metric(
                    "Audio",
                    f"{audio_score:.4f}"
                )

            preview_frames = segment[
                "frames"
            ][:2]

            cols = st.columns(
                len(preview_frames)
            )

            for col, frame in zip(
                cols,
                preview_frames
            ):

                col.image(
                    frame,
                    use_container_width=True
                )

            timeline_x.append(
                f"{start_t:.0f}-"
                f"{end_t:.0f}s"
            )

            timeline_y.append(
                fused_score
            )

            st.markdown("---")

        # =================================
        # TIMELINE GRAPH
        # =================================

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        bars = ax.bar(
            timeline_x,
            timeline_y
        )

        ax.set_ylabel(
            "Similarity"
        )

        ax.set_xlabel(
            "Video Segment"
        )

        ax.set_title(
            "Temporal Retrieval Scores"
        )

        min_score = min(
            timeline_y
        )

        max_score = max(
            timeline_y
        )

        margin = max(
            (
                max_score
                - min_score
            ) * 0.15,
            0.01
        )

        ax.set_ylim(
            min_score - margin,
            max_score + margin
        )

        for bar, score in zip(
            bars,
            timeline_y
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,

                score,

                f"{score:.3f}",

                ha="center",
                va="bottom"
            )

        graph_col1, graph_col2, graph_col3 = (
            st.columns([1, 2, 1])
        )

        with graph_col2:

            st.pyplot(
                fig,
                use_container_width=True
            )

        st.markdown("---")

        st.subheader(
            "Inference Details"
        )

        st.write(
            f"Query: `{result['query']}`"
        )

        st.write(
            f"Inference Time: "
            f"{result['time_taken']:.4f}s"
        )

    os.unlink(video_path)