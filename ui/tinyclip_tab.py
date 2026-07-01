import matplotlib.pyplot as plt
import streamlit as st
import requests

from PIL import Image

from utils.report import (
    show_report
)


def render_tinyclip_tab():

    st.header("TinyCLIP")

    mode = st.radio(
        "Task",
        [
            "Image Labeling",
            "Image Retrieval"
        ],
        horizontal=True,
        key="tinyclip_mode"
    )

    # ======================================
    # LABELING
    # ======================================

    if mode == "Image Labeling":

        uploaded = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg"],
            key="tinyclip_label_upload"
        )

        if uploaded is None:
            return

        image = Image.open(
            uploaded
        ).convert("RGB")

        image_col1, image_col2, image_col3 = (
            st.columns([1, 3, 1])
        )

        with image_col2:

            st.image(
                image,
                use_container_width=True
            )

        # Extract image info
        img_width, img_height = image.size
        uploaded.seek(0)
        raw_img = Image.open(uploaded)
        img_format = raw_img.format or "Unknown"
        file_size_mb = uploaded.size / (1024 * 1024)

        st.markdown("---")
        st.subheader("Image Information")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Resolution", f"{img_width}x{img_height}")
        with info_col2:
            st.metric("Format", img_format)
        with info_col3:
            st.metric("File Size", f"{file_size_mb:.2f} MB")
        st.markdown("---")

        labels_raw = st.text_area(
            "Labels",
            value=(
                "animal\n"
                "vehicle\n"
                "food\n"
                "building"
            ),
            height=160
        )

        if st.button(
            "Run TinyCLIP",
            key="tinyclip_run_button"
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
            data = {"labels_str": ",".join(labels)}
            try:
                response = requests.post(f"{API_URL}/tinyclip/label", files=files, data=data)
                if response.status_code == 200:
                    api_result = response.json()
                    result = {
                        "time_taken": api_result["time_taken"],
                        "results": api_result["results"],
                        "input_details": {
                            "Image Size": image.size
                        }
                    }
                    st.session_state["tinyclip_result"] = result
                else:
                    st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

        result = st.session_state.get(
            "tinyclip_result"
        )

        if result is not None:

            st.markdown("---")

            show_report(
                "TinyCLIP",
                result["input_details"],
                result["time_taken"],
                result["results"]
            )

    # ======================================
    # RETRIEVAL
    # ======================================

    else:

        uploaded_images = (
            st.file_uploader(
                "Upload Retrieval Images",

                type=[
                    "png",
                    "jpg",
                    "jpeg"
                ],

                accept_multiple_files=True,

                key="tinyclip_retrieval_upload"
            )
        )

        if not uploaded_images:
            return

        total_images = len(uploaded_images)
        total_size_mb = sum(img.size for img in uploaded_images) / (1024 * 1024)
        
        widths = []
        heights = []
        for img in uploaded_images:
            img.seek(0)
            with Image.open(img) as pimg:
                w, h = pimg.size
                widths.append(w)
                heights.append(h)
        avg_width = int(sum(widths) / len(widths)) if widths else 0
        avg_height = int(sum(heights) / len(heights)) if heights else 0

        st.markdown("---")
        st.subheader("Media Batch Information")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Total Images", total_images)
        with info_col2:
            st.metric("Average Resolution", f"{avg_width}x{avg_height}")
        with info_col3:
            st.metric("Total File Size", f"{total_size_mb:.2f} MB")
        st.markdown("---")

        query = st.text_input(
            "Query",
            value="vehicle",
            key="tinyclip_query"
        )

        top_k = st.slider(
            "Top Matches",
            min_value=1,
            max_value=min(
                len(uploaded_images),
                12
            ),
            value=min(
                len(uploaded_images),
                4
            ),
            key="tinyclip_topk"
        )

        if st.button(
            "Retrieve Images",
            key="tinyclip_retrieve_button"
        ):

            API_URL = "http://localhost:8000"
            files = []
            for idx, uploaded in enumerate(uploaded_images):
                uploaded.seek(0)
                files.append(("files", (uploaded.name, uploaded.getvalue(), uploaded.type)))
            
            data = {"query": query}
            try:
                response = requests.post(f"{API_URL}/tinyclip/search", files=files, data=data)
                if response.status_code == 200:
                    api_result = response.json()
                    st.session_state["tinyclip_retrieval_result"] = api_result
                    st.session_state["tinyclip_top_k"] = top_k
                else:
                    st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

        result = st.session_state.get(
            "tinyclip_retrieval_result"
        )

        if result is not None:

            st.markdown("---")

            # Map the results to (image_name, pil_image, score)
            image_lookup = {uploaded.name: uploaded for uploaded in uploaded_images}
            ranked_results = []
            for r in result["results"]:
                image_name = r["image_path"]
                score = r["score"]
                uploaded_file = image_lookup.get(image_name)
                if uploaded_file:
                    uploaded_file.seek(0)
                    img = Image.open(uploaded_file).convert("RGB")
                    ranked_results.append((image_name, img, score))

            top_results = ranked_results[:int(top_k)]

            num_columns = 4

            for start_idx in range(
                0,
                len(top_results),
                num_columns
            ):

                row = top_results[
                    start_idx:
                    start_idx +
                    num_columns
                ]

                columns = st.columns(
                    len(row)
                )

                for idx, (column, (
                    image_name,
                    image,
                    score
                )) in enumerate(zip(
                    columns,
                    row
                )):

                    column.image(
                        image,
                        caption=(
                            f"{image_name}\n"
                            f"{score:.4f}"
                        ),
                        use_container_width=True
                    )

                    uploaded_file = image_lookup.get(image_name)
                    if uploaded_file:
                        uploaded_file.seek(0)
                        orig_bytes = uploaded_file.getvalue()
                        column.download_button(
                            label="📥 Download",
                            data=orig_bytes,
                            file_name=image_name,
                            mime=uploaded_file.type,
                            key=f"dl_tinyclip_{image_name}_{start_idx}_{idx}"
                        )

            # ==============================
            # SCORE GRAPH
            # ==============================

            labels = [
                item[0]
                for item
                in ranked_results
            ]

            scores = [
                item[2]
                for item
                in ranked_results
            ]

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            bars = ax.bar(
                labels,
                scores
            )

            ax.set_ylabel(
                "Similarity"
            )

            ax.set_title(
                "TinyCLIP Retrieval Scores"
            )

            ax.tick_params(
                axis="x",
                rotation=45
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
            
            # ==============================
            # INFERENCE DETAILS
            # ==============================

            st.markdown(
                "### Inference Details"
            )

            st.write(
                f"Time Taken: "
                f"{result['time_taken']:.4f}s"
            )

            st.write(
                f"Images Processed: "
                f"{len(ranked_results)}"
            )

            st.write(
                f"Query: "
                f"`{result['query']}`"
            )