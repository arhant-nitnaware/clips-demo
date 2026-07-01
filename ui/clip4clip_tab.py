import matplotlib.pyplot as plt
import streamlit as st
import requests

from utils.report import (
    show_report
)





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
    # VIDEO METADATA (via API)
    # ======================================

    API_URL = "http://localhost:8000"

    uploaded.seek(0)
    files_payload = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    total_video_frames = 0
    try:
        response = requests.post(f"{API_URL}/media/info", files=files_payload)
        if response.status_code == 200:
            media_data = response.json()
            total_video_frames = media_data.get("frame_count", 0)
        else:
            st.error(f"Error getting video info: {response.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")

    if total_video_frames <= 0:
        total_video_frames = 64

    # ======================================
    # FRAME SETTINGS
    # ======================================

    st.markdown(
        "### Frame Extraction Settings"
    )

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
        max_value=total_video_frames,
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

    st.markdown("---")

    # ======================================
    # TASK MODES
    # ======================================

    mode = st.radio(
        "Task",
        [
            "Video Labeling",
            "Frame Retrieval",
            #"Video Similarity"
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

            API_URL = "http://localhost:8000"
            uploaded.seek(0)
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            data = {"labels_str": ",".join(labels), "max_frames": max_frames}
            try:
                response = requests.post(f"{API_URL}/clip4clip/label", files=files, data=data)
                if response.status_code == 200:
                    api_result = response.json()
                    result = {
                        "time_taken": api_result["time_taken"],
                        "results": api_result["results"],
                        "input_details": {
                            "Number of Frames Used for Inference": max_frames,
                            "Number of Labels": len(labels)
                        }
                    }
                    st.session_state["clip4clip_label_result"] = result
                else:
                    st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

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
        
        col1, _ = st.columns([1, 4])

        with col1:

            top_k = st.number_input(
                "Top Frames",
                min_value=1,
                max_value=20,
                value=4,
                step=1
                )

        if st.button(
            "Retrieve Frames"
        ):

            API_URL = "http://localhost:8000"
            uploaded.seek(0)
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            data = {"query": query, "max_frames": max_frames}
            try:
                response = requests.post(f"{API_URL}/clip4clip/search", files=files, data=data)
                if response.status_code == 200:
                    api_result = response.json()
                    st.session_state["clip4clip_query_result"] = api_result
                    st.session_state["clip4clip_top_k"] = top_k
                else:
                    st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

        result = st.session_state.get(
            "clip4clip_query_result"
        )

        if result is not None:

            st.markdown("---")

            st.write(
                f"### Query: `{result['query']}`"
            )

            # Results returned from FastAPI contains top matches with base64 images
            top_frames = result["results"][:st.session_state.get("clip4clip_top_k", 4)]

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

                    frame_idx = frame_data["frame_index"]
                    score = frame_data["score"]
                    b64_image = frame_data["image"]

                    columns[idx].image(
                        b64_image,
                        caption=(
                            f"Frame {frame_idx}\n"
                            f"Score: {score:.4f}"
                        ),
                        use_container_width=True
                    )

            st.markdown(
                "### Frame Retrieval Scores"
            )

            # Reconstruct ranked_frames from all_scores for the graph plotting
            ranked_frames = [
                (item["frame_index"], item["score"])
                for item in result.get("all_scores", [])
            ]

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

            #ax.set_xticks(frame_indices)

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

            '''
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
            '''

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

            API_URL = "http://localhost:8000"
            uploaded.seek(0)
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            data = {"prompts_str": ",".join(prompts), "max_frames": max_frames}
            try:
                response = requests.post(f"{API_URL}/clip4clip/similarity", files=files, data=data)
                if response.status_code == 200:
                    api_result = response.json()
                    results = [
                        (item["prompt"], item["score"])
                        for item in api_result["results"]
                    ]
                    st.session_state["clip4clip_similarity_result"] = results
                else:
                    st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

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

    
