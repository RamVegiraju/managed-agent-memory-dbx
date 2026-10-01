"""Launch the local managed sessions and memory Streamlit demo."""

import streamlit as st


def main() -> None:
    """Configure Streamlit before importing cached UI resources."""
    st.set_page_config(page_title="Managed Sessions + Memory", page_icon="🧠", layout="wide")
    from memory_demo.streamlit_ui import render

    render()


main()
