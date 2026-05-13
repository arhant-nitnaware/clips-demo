import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results):

    import textwrap

    labels = [
        item[0]
        if not isinstance(item, dict)
        else item["label"]
        for item in results
    ]

    scores = [
        item[1]
        if not isinstance(item, dict)
        else item["score"]
        for item in results
    ]

    # ======================================
    # Wrap long labels
    # ======================================

    wrapped_labels = [
        "\n".join(
            textwrap.wrap(
                label,
                width=14
            )
        )
        for label in labels
    ]

    scores = np.array(scores)

    # ======================================
    # Dynamic Y-axis scaling
    # ======================================

    min_score = scores.min()
    max_score = scores.max()

    margin = max(
        (max_score - min_score) * 0.15,
        0.01
    )

    y_min = min_score - margin
    y_max = max_score + margin

    # ======================================
    # Smaller figure
    # ======================================

    fig, ax = plt.subplots(
        figsize=(8, 3.8)
    )

    bars = ax.bar(
        wrapped_labels,
        scores
    )

    ax.set_ylim(
        y_min,
        y_max
    )

    ax.set_ylabel(
        "Similarity"
    )

    ax.set_title(
        "Label Similarity Scores"
    )

    ax.tick_params(
        axis="x",
        labelsize=9
    )

    ax.tick_params(
        axis="y",
        labelsize=9
    )

    # ======================================
    # Value labels
    # ======================================

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

    plt.tight_layout()

    graph_col1, graph_col2, graph_col3 = st.columns(
    [1, 3, 1]
    )   

    with graph_col2:

        st.pyplot(
            fig,
            use_container_width=True
        )


def show_report(
    model_name,
    input_details,
    inference_time,
    results
):

    st.markdown(
        f"## {model_name}"
    )

    st.markdown(
        "### Input Details"
    )

    for key, value in (
        input_details.items()
    ):

        st.write(
            f"**{key}**: {value}"
        )

    st.markdown(
        "### Inference Details"
    )

    st.write(
        "**Device**: CPU"
    )

    st.write(
        f"**Inference Time**: "
        f"{inference_time:.4f} seconds"
    )

    st.markdown(
        "### Similarity Graph"
    )

    plot_results(results)

    st.markdown(
        "### Raw Output"
    )

    for idx, item in enumerate(
        results,
        start=1
    ):

        if isinstance(item, dict):

            label = item["label"]
            score = item["score"]

        else:

            label, score = item

        st.write(
            f"{idx}. "
            f"{label} "
            f"→ {score:.4f}"
        )