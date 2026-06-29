import gc
import torch
import streamlit as st

def unload_model(*keys):
    for key in keys:
        if key in st.session_state:
            st.session_state[key] = None

    # Map session state keys to model manager cache keys
    from models import model_manager
    for key in keys:
        if "clip4clip" in key:
            model_manager.unload_model_instance("clip4clip")
        elif "tinyclip" in key:
            model_manager.unload_model_instance("tinyclip")
        elif "clap" in key:
            model_manager.unload_model_instance("clap")
        elif "clip" in key:  # Ensure this is after clip4clip and tinyclip checks
            model_manager.unload_model_instance("clip")

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()