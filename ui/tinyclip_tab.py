import matplotlib.pyplot as plt
import streamlit as st

from PIL import Image

from inference.tinyclip_infer import (
    run_tinyclip
)

from inference.tinyclip_retrieval import (
    retrieve_tinyclip_images
)

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

            result = run_tinyclip(
                st.session_state
                .tinyclip_pipe,

                image,

                labels
            )

            st.session_state[
                "tinyclip_result"
            ] = result

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

        # ==============================
        # SAFETY CHECK
        # ==============================

        if (
            st.session_state
            .tinyclip_model is None
            or
            st.session_state
            .tinyclip_processor is None
        ):

            st.error(
                "Reload TinyCLIP model."
            )

            return

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

            result = (
                retrieve_tinyclip_images(
                    st.session_state
                    .tinyclip_model,

                    st.session_state
                    .tinyclip_processor,

                    query,

                    uploaded_images
                )
            )

            st.session_state[
                "tinyclip_retrieval_result"
            ] = result

            st.session_state[
                "tinyclip_top_k"
            ] = top_k

        result = st.session_state.get(
            "tinyclip_retrieval_result"
        )

        if result is not None:

            st.markdown("---")

            ranked_results = result[
                "results"
            ]

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
                        width=350
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