import matplotlib.pyplot as plt
import streamlit as st
import requests
import base64

from utils.report import (
    show_report
)


def render_clap_tab():

    st.header("CLAP")

    uploaded = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3", "flac"]
    )

    if uploaded is None:
        return

    audio_col1, audio_col2, audio_col3 = (
        st.columns([1, 2, 1])
    )

    with audio_col2:

        st.audio(uploaded)

    # ==========================================
    # TASK SELECTOR
    # ==========================================

    mode = st.radio(
        "Task",
        [
            "Audio Labeling",
            "Audio Retrieval"
        ],
        horizontal=True
    )

    # ==========================================
    # MODE TRACKING
    # ==========================================

    if (
        "clap_current_mode"
        not in st.session_state
    ):

        st.session_state[
            "clap_current_mode"
        ] = mode

    elif (
        st.session_state[
            "clap_current_mode"
        ] != mode
    ):

        st.session_state[
            "clap_label_result"
        ] = None

        st.session_state[
            "clap_retrieval_result"
        ] = None

        st.session_state[
            "clap_current_mode"
        ] = mode

    # ==========================================
    # AUDIO LABELING
    # ==========================================

    if mode == "Audio Labeling":

        st.subheader(
            "Zero-shot Audio Classification"
        )

        texts_raw = st.text_area(
            "Descriptions",
            value=(
                "dog barking\n"
                "birds chirping\n"
                "music\n"
                "car engine\n"
                "people talking"
            ),
            height=180
        )

        if st.button(
            "Run Audio Labeling"
        ):

            texts = [
                text.strip()
                for text in (
                    texts_raw.splitlines()
                )
                if text.strip()
            ]

            API_URL = "http://localhost:8000"
            uploaded.seek(0)
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            data = {"labels_str": ",".join(texts)}
            try:
                response = requests.post(f"{API_URL}/clap/label", files=files, data=data)
                if response.status_code == 200:
                    api_result = response.json()
                    result = {
                        "time_taken": api_result["time_taken"],
                        "results": api_result["results"],
                        "input_details": {
                            "Audio File": uploaded.name,
                            "Number of Descriptions": len(texts)
                        }
                    }
                    st.session_state["clap_label_result"] = result
                else:
                    st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

        # ======================================
        # RESULT DISPLAY
        # ======================================

        result = st.session_state.get(
            "clap_label_result"
        )

        if result is not None:

            st.markdown("---")

            show_report(
                "CLAP Audio Labeling",
                result["input_details"],
                result["time_taken"],
                result["results"]
            )

    # ==========================================
    # AUDIO RETRIEVAL
    # ==========================================

    else:

        st.subheader(
            "Semantic Audio Retrieval"
        )

        query = st.text_input(
            "Query",
            value="car engine reving"
        )

        segment_seconds = st.slider(
            "Segment Length (seconds)",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help=(
                "Smaller segments improve "
                "retrieval localization "
                "but increase inference time."
            )
        )

        col1, col2 = st.columns([1, 4])

        with col1:

            top_k = int(st.number_input(
                "Top Segments",
                min_value=1,
                max_value=12    ,
                value=4,
                step=1
            ))

        if st.button(
            "Retrieve Audio Segments"
        ):

            API_URL = "http://localhost:8000"
            uploaded.seek(0)
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            data = {"query": query, "segment_seconds": float(segment_seconds)}
            try:
                response = requests.post(f"{API_URL}/clap/search", files=files, data=data)
                if response.status_code == 200:
                    api_result = response.json()
                    st.session_state["clap_retrieval_result"] = api_result
                    st.session_state["clap_top_k"] = top_k
                else:
                    st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

        # ======================================
        # RESULT DISPLAY
        # ======================================

        result = st.session_state.get(
            "clap_retrieval_result"
        )

        if result is not None:

            st.markdown("---")

            st.write(
                f"### Query: "
                f"`{result['query']}`"
            )

            ranked_segments = result[
                "results"
            ]

            top_segments = ranked_segments[
                :st.session_state.get(
                    "clap_top_k",
                    4
                )
            ]

            # ==================================
            # TOP SEGMENTS
            # ==================================

            st.markdown(
                "### Top Matching Segments"
            )

            for idx, segment in enumerate(top_segments):
                start_t = segment["start_time"]
                end_t = segment["end_time"]
                score = segment["score"]
                b64_audio = segment["audio"]

                st.markdown(
                    f"""
                    **Segment {idx + 1}**

                    Time:
                    {start_t:.2f}s
                    → {end_t:.2f}s

                    Similarity:
                    {score:.4f}
                    """
                )

                if b64_audio:
                    audio_bytes = base64.b64decode(b64_audio.split(",")[1])
                    st.audio(
                        audio_bytes,
                        format="audio/wav"
                    )

        # ==================================
        # BAR GRAPH
        # ==================================

        st.markdown(
            "### Segment Similarity Graph"
        )

        timeline_segments = sorted(
            ranked_segments,
            key=lambda x: x["start_time"]
        )

        segment_labels = [
            (
                f"{item['start_time']:.0f}-"
                f"{item['end_time']:.0f}s"
            )
            for item in timeline_segments
        ]

        scores = [
            item["score"]
            for item in timeline_segments
        ]

        x = list(
            range(
                len(scores)
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        bars = ax.bar(
            x,
            scores
        )

        ax.set_xlabel(
            "Audio Segments"
        )

        ax.set_ylabel(
            "Similarity Score"
        )

        ax.set_title(
            "Semantic Retrieval Scores"
        )

        # ======================================
        # SHOW ONLY MAJOR TICKS
        # ======================================

        step = max(
            1,
            len(segment_labels) // 10
        )

        tick_positions = list(
            range(
                0,
                len(segment_labels),
                step
            )
        )

        ax.set_xticks(
            tick_positions
        )

        ax.set_xticklabels(
            [
                segment_labels[i]
                for i in tick_positions
            ],
            rotation=45,
            ha="right"
        )

        ax.minorticks_off()

        # ======================================
        # Y AXIS LIMITS
        # ======================================

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

        # ======================================
        # CLEANUP
        # ======================================

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3
        )

        fig.tight_layout()

        graph_col1, graph_col2, graph_col3 = (
            st.columns([1, 4, 1])
        )

        with graph_col2:

            st.pyplot(
                fig,
                use_container_width=True
            )

        # ==================================
        # INFERENCE DETAILS
        # ==================================

        st.markdown(
            "### Inference Details"
        )

        st.write(
            f"Time Taken: "
            f"{result['time_taken']:.4f}s"
        )