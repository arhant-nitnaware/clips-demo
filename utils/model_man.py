import gc
import torch
import streamlit as st


def unload_model(*keys):

    for key in keys:

        if key in st.session_state:

            st.session_state[key] = None

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()