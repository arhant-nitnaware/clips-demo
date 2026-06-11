import os
import tempfile

import matplotlib.pyplot as plt
import soundfile as sf
import streamlit as st

from inference.clap_infer import (
    run_clap
)

from inference.clap_retrieval import (
    retrieve_audio_segments
)

from utils.audio_utils import (
    split_audio_segments
)

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

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as tmp:

        tmp.write(uploaded.read())

        audio_path = tmp.name

    waveform, sample_rate = sf.read(
        audio_path,
        always_2d=False
    )

    os.unlink(audio_path)

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

            result = run_clap(
                st.session_state
                .clap_model,

                st.session_state
                .clap_tokenizer,

                st.session_state
                .clap_extractor,

                waveform,

                sample_rate,

                texts
            )

            st.session_state[
                "clap_label_result"
            ] = result

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

            segments = split_audio_segments(
                waveform,
                sample_rate,
                segment_seconds
            )

            result = retrieve_audio_segments(
                st.session_state
                .clap_model,

                st.session_state
                .clap_tokenizer,

                st.session_state
                .clap_extractor,

                waveform,

                sample_rate,

                query,

                segments
            )

            st.session_state[
                "clap_retrieval_result"
            ] = result

            st.session_state[
                "clap_segments"
            ] = segments

            st.session_state[
                "clap_top_k"
            ] = top_k

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

            for idx, (
                start_t,
                end_t,
                score
            ) in enumerate(top_segments):

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

                start_sample = int(
                    start_t * sample_rate
                )

                end_sample = int(
                    end_t * sample_rate
                )
                
                #ccccccccccccccccccccc
                segment_audio = waveform[
                    start_sample:end_sample
                ]

                # ensure mono
                if segment_audio.ndim > 1:

                    segment_audio = segment_audio.mean(
                        axis=1
                    )

                # convert dtype
                segment_audio = (
                    segment_audio.astype("float32")
                )

                # normalize safely
                max_val = abs(segment_audio).max()

                if max_val > 1.0:

                    segment_audio = (
                        segment_audio / max_val
                    )

                st.audio(
                    segment_audio,
                    sample_rate=sample_rate
                )

        # ==================================
        # BAR GRAPH
        # ==================================

        st.markdown(
            "### Segment Similarity Graph"
        )

        timeline_segments = sorted(
            ranked_segments,
            key=lambda x: x[0]
        )

        segment_labels = [
            (
                f"{start:.0f}-"
                f"{end:.0f}s"
            )
            for start, end, _
            in timeline_segments
        ]

        scores = [
            score
            for _, _, score
            in timeline_segments
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