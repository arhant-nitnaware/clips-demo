import streamlit as st

from PIL import Image

from inference.tinyclip_infer import (
    run_tinyclip
)

from utils.report import (
    show_report
)


def render_tinyclip_tab():

    st.header("TinyCLIP")

    uploaded = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
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

    st.subheader(
        "Zero-shot Image Classification"
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
        "Run TinyCLIP"
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