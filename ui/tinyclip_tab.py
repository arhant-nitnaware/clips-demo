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
                width=800
            )

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

            top_results = ranked_results[
                :st.session_state.get(
                    "tinyclip_top_k",
                    4
                )
            ]

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

                for column, (
                    image_name,
                    image,
                    score
                ) in zip(
                    columns,
                    row
                ):

                    column.image(
                        image,
                        caption=(
                            f"{image_name}\n"
                            f"{score:.4f}"
                        ),
                        width="stretch",
                        
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