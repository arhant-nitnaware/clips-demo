import matplotlib.pyplot as plt
import streamlit as st

from PIL import Image

from inference.clip_infer import (
    run_clip
)

from inference.clip_retrieval import (
    retrieve_images
)

from utils.report import (
    show_report
)


def render_clip_tab():

    st.header("Original CLIP")

    mode = st.radio(
        "Task",
        [
            "Image Labeling",
            "Image Retrieval"
        ],
        horizontal=True
    )

    # ======================================
    # LABELING
    # ======================================

    if mode == "Image Labeling":

        uploaded = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg"],
            key="clip_label_upload"
        )

        if uploaded is None:
            return

        image = Image.open(
            uploaded
        ).convert("RGB")

        image_col1, image_col2, image_col3 = (
            st.columns([1, 2, 1])
        )

        with image_col2:

            st.image(
                image,
                width=350
            )

        labels_raw = st.text_area(
            "Labels",
            value=(
                "warship\n"
                "airport\n"
                "forest\n"
                "city\n"
                "bridge"
            ),
            height=180
        )

        if st.button(
            "Run CLIP"
        ):

            labels = [
                label.strip()
                for label in (
                    labels_raw
                    .splitlines()
                )
                if label.strip()
            ]

            result = run_clip(
                st.session_state
                .clip_model,

                st.session_state
                .clip_processor,

                image,

                labels
            )

            st.session_state[
                "clip_label_result"
            ] = result

        result = st.session_state.get(
            "clip_label_result"
        )

        if result is not None:

            st.markdown("---")

            show_report(
                "Original CLIP",
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

                key="clip_retrieval"
            )
        )

        if not uploaded_images:
            return

        query = st.text_input(
            "Query",
            value="warship"
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
            )
        )

        if st.button(
            "Retrieve Images"
        ):

            result = retrieve_images(
                st.session_state
                .clip_model,

                st.session_state
                .clip_processor,

                query,

                uploaded_images
            )

            st.session_state[
                "clip_retrieval_result"
            ] = result

            st.session_state[
                "clip_top_k"
            ] = top_k

        result = st.session_state.get(
            "clip_retrieval_result"
        )

        if result is not None:

            st.markdown("---")

            st.write(
                f"### Query: "
                f"`{result['query']}`"
            )

            ranked_results = result[
                "results"
            ]

            top_results = ranked_results[
                :st.session_state.get(
                    "clip_top_k",
                    4
                )
            ]

            num_columns = 4

            for start_idx in range(
                0,
                len(top_results),
                num_columns
            ):

                current_row = (
                    top_results[
                        start_idx:
                        start_idx +
                        num_columns
                    ]
                )

                columns = st.columns(
                    len(current_row)
                )

                for column, (
                    image_name,
                    image,
                    score
                ) in zip(
                    columns,
                    current_row
                ):

                    column.image(
                        image,
                        caption=(
                            f"{image_name}\n"
                            f"{score:.4f}"
                        ),
                        width=350
                    )

            # ==============================
            # BAR GRAPH
            # ==============================

            st.markdown(
                "### Retrieval Scores"
            )

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
                "Semantic Retrieval Scores"
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

            st.markdown(
                "### Inference Details"
            )

            st.write(
                f"Time Taken: "
                f"{result['time_taken']:.4f}s"
            )