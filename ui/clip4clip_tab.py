import matplotlib.pyplot as plt
import streamlit as st
import requests

from utils.report import (
    show_report
)





def render_clip4clip_tab():

    st.header("CLIP4Clip")

    search_mode = st.radio(
        "Search Mode",
        options=["Just One Video", "Batch of Videos"],
        index=0,
        horizontal=True,
        key="clip4clip_search_mode"
    )

    # Clean up results if mode changed
    if "prev_clip4clip_search_mode" not in st.session_state:
        st.session_state["prev_clip4clip_search_mode"] = search_mode
    elif st.session_state["prev_clip4clip_search_mode"] != search_mode:
        st.session_state["clip4clip_label_result"] = None
        st.session_state["clip4clip_query_result"] = None
        st.session_state["prev_clip4clip_search_mode"] = search_mode

    if search_mode == "Just One Video":
        uploaded = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov"],
            key="clip4clip_upload"
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
        duration = 0.0
        fps = 0.0
        resolution = "N/A"
        file_size_mb = 0.0
        try:
            response = requests.post(f"{API_URL}/media/info", files=files_payload)
            if response.status_code == 200:
                media_data = response.json()
                total_video_frames = media_data.get("frame_count", 0)
                duration = media_data.get("duration", 0.0)
                fps = media_data.get("fps", 0.0)
                resolution = media_data.get("resolution", "N/A")
                file_size_mb = media_data.get("file_size_mb", 0.0)
            else:
                st.error(f"Error getting video info: {response.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

        if total_video_frames <= 0:
            total_video_frames = 64

        # ======================================
        # VIDEO INFORMATION
        # ======================================

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
                total_video_frames
            )

            st.metric(
                "Resolution",
                resolution
            )

        with info_col3:

            st.metric(
                "File Size",
                f"{file_size_mb:.2f} MB"
            )

    else:
        uploaded_files = st.file_uploader(
            "Upload Videos",
            type=["mp4", "avi", "mov"],
            accept_multiple_files=True,
            key="clip4clip_upload_batch"
        )

        if not uploaded_files:
            return

        st.write(f"Uploaded {len(uploaded_files)} video(s).")
        file_details = []
        for idx, f in enumerate(uploaded_files):
            file_details.append({
                "Filename": f.name,
                "Size (MB)": f"{f.size / (1024 * 1024):.2f}"
            })
        st.table(file_details)

        st.subheader("Play Uploaded Video")
        selected_video_name = st.selectbox(
            "Select video to play",
            options=[f.name for f in uploaded_files],
            key="clip4clip_selected_play_video"
        )
        selected_file = next(f for f in uploaded_files if f.name == selected_video_name)
        
        video_col1, video_col2, video_col3 = (
            st.columns([1, 3, 1])
        )
        with video_col2:
            st.video(selected_file)

    st.markdown("---")

    # ======================================
    # FRAME SETTINGS
    # ======================================

    st.markdown(
        "### Frame Extraction Settings"
    )

    if search_mode == "Just One Video":
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
    else:
        max_frames = st.slider(
            "Number of Frames (per Video)",
            min_value=2,
            max_value=64,
            value=12,
            step=1,
            help=(
                "Number of frames to extract from each video "
                "in the batch for search inference."
            )
        )

        st.caption(
            f"""
            Selected frames per video for inference:
            {max_frames}
            """
        )

    st.markdown("---")

    # ======================================
    # TASK MODES
    # ======================================

    if search_mode == "Just One Video":
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
    else:
        mode = "Frame Retrieval"
        st.markdown("**Task:** Frame Retrieval (Searching among multiple videos)")

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
            value="a yellow race car",
            key="clip4clip_query"
        )
        
        col1, _ = st.columns([1, 4])

        with col1:

            top_k = st.number_input(
                "Top Frames",
                min_value=1,
                max_value=20,
                value=4,
                step=1,
                key="clip4clip_top_k_input"
                )

        if st.button(
            "Retrieve Frames"
        ):
            API_URL = "http://localhost:8000"
            if search_mode == "Just One Video":
                uploaded.seek(0)
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                data = {"query": query, "max_frames": max_frames, "top_k": int(top_k)}
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
            else:
                files_payload = []
                for f in uploaded_files:
                    f.seek(0)
                    files_payload.append(
                        ("files", (f.name, f.getvalue(), f.type))
                    )
                data = {"query": query, "max_frames": max_frames, "top_k": int(top_k)}
                try:
                    with st.spinner("Processing batch retrieval..."):
                        response = requests.post(f"{API_URL}/clip4clip/batch_search", files=files_payload, data=data)
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
            top_frames = result["results"][:int(top_k)]

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

                    import base64
                    import io
                    from PIL import Image

                    try:
                        image_bytes = base64.b64decode(b64_image.split(",")[1])
                        pil_img = Image.open(io.BytesIO(image_bytes))
                    except Exception:
                        pil_img = b64_image

                    video_name = frame_data.get("video_name", "")
                    timestamp = frame_data.get("timestamp", None)
                    
                    if search_mode == "Batch of Videos" and video_name:
                        caption_text = (
                            f"Video: {video_name}\n"
                            f"Frame: {frame_idx}\n"
                            f"Time: {timestamp:.2f}s\n"
                            f"Score: {score:.4f}"
                        )
                    else:
                        caption_text = (
                            f"Frame: {frame_idx}\n"
                            f"Score: {score:.4f}"
                        )

                    columns[idx].image(
                        pil_img,
                        caption=caption_text,
                        width=280
                    )

            st.markdown(
                "### Frame Retrieval Scores"
            )

            if search_mode == "Just One Video":
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
            else:
                ranked_frames = top_frames
                frame_labels = [
                    f"{item.get('video_name', '')[:12]}...\nF{item['frame_index']} ({item['timestamp']:.1f}s)"
                    for item in ranked_frames
                ]
                scores = [
                    item["score"]
                    for item in ranked_frames
                ]

                fig, ax = plt.subplots(
                    figsize=(10, 4)
                )

                bars = ax.bar(
                    frame_labels,
                    scores
                )

                ax.set_xlabel(
                    "Frame Source"
                )

                ax.set_ylabel(
                    "Similarity Score"
                )

                ax.set_title(
                    f"Top {len(ranked_frames)} Matching Frames Across Videos"
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



    
