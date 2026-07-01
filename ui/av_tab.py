import matplotlib.pyplot as plt
import streamlit as st
import requests
import base64


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
    # VIDEO METADATA (via API)
    # =====================================

    API_URL = "http://localhost:8000"

    uploaded.seek(0)
    files_payload = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    
    duration = 0.0
    fps = 0.0
    frame_count = 0
    resolution = "N/A"
    sample_rate = 0
    file_size_mb = 0.0

    try:
        response = requests.post(f"{API_URL}/media/info", files=files_payload)
        if response.status_code == 200:
            media_data = response.json()
            duration = media_data.get("duration", 0.0)
            fps = media_data.get("fps", 0.0)
            frame_count = media_data.get("frame_count", 0)
            resolution = media_data.get("resolution", "N/A")
            sample_rate = media_data.get("sample_rate", 0)
            file_size_mb = media_data.get("file_size_mb", 0.0)
        else:
            st.error(f"Error getting video info: {response.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")

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
            resolution
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

    settings_col1, settings_col2, settings_col3 = (
        st.columns(3)
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

    with settings_col3:

        top_k = st.number_input(
            "Top Segments",
            min_value=1,
            max_value=12,
            value=3,
            step=1
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
        value=0.5,
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

        API_URL = "http://localhost:8000"
        uploaded.seek(0)
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        data = {
            "query": query,
            "visual_weight": float(visual_weight),
            "audio_weight": float(audio_weight),
            "segment_seconds": float(segment_seconds),
            "top_k": int(top_k)
        }
        try:
            response = requests.post(f"{API_URL}/av/search", files=files, data=data)
            if response.status_code == 200:
                api_result = response.json()
                st.session_state["av_result"] = api_result
            else:
                st.error(f"Error from API: {response.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

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

    if result is not None:

        st.markdown("---")

        st.subheader(
            "Top Matching Segments"
        )

        top_results = result[
            "results"
        ][:int(top_k)]

        timeline_results = sorted(
            result["results"],
            key=lambda x: x["start_time"]
        )

        timeline_x = [
            (
                f"{segment['start_time']:.0f}-"
                f"{segment['end_time']:.0f}s"
            )
            for segment in timeline_results
        ]

        timeline_y = [
            segment["fused_score"]
            for segment in timeline_results
        ]

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

            b64_image = segment.get("image")
            if b64_image:
                import base64
                import io
                from PIL import Image

                try:
                    image_bytes = base64.b64decode(b64_image.split(",")[1])
                    pil_img = Image.open(io.BytesIO(image_bytes))
                except Exception:
                    pil_img = b64_image
                    image_bytes = None

                st.image(pil_img, use_container_width=True)

                if image_bytes is not None:
                    st.download_button(
                        label=f"📥 Download Representative Frame ({start_t:.1f}s - {end_t:.1f}s)",
                        data=image_bytes,
                        file_name=f"segment_{start_t:.1f}_{end_t:.1f}.jpg",
                        mime="image/jpeg",
                        key=f"dl_av_{idx}"
                    )

            st.markdown("---")

        # =================================
        # TIMELINE GRAPH
        # =================================

        fig, ax = plt.subplots(
            figsize=(16, 6)
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

        # ======================================
        # SHOW ONLY MAJOR TICKS
        # ======================================

        step = max(
            1,
            len(timeline_x) // 10
        )

        tick_positions = list(
            range(
                0,
                len(timeline_x),
                step
            )
        )

        ax.set_xticks(
            tick_positions
        )

        ax.set_xticklabels(
            [
                timeline_x[i]
                for i in tick_positions
            ],
            rotation=45,
            ha="right"
        )

        ax.minorticks_off()

        # ======================================
        # Y LIMITS
        # ======================================

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

        # ======================================
        # CLEANUP
        # ======================================

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3
        )

        fig.tight_layout()

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