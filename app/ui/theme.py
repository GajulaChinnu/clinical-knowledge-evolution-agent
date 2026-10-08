"""Enterprise stylesheet loader for the CKEA Streamlit workspace."""

from functools import lru_cache
from pathlib import Path

import streamlit as st

THEME_CSS_PATH = Path(__file__).with_name("theme.css")


@lru_cache(maxsize=1)
def _theme_css() -> str:
    return THEME_CSS_PATH.read_text(encoding="utf-8")


def inject_theme() -> None:
    """Inject the shared stylesheet into the current page."""
    st.markdown(f"<style>\n{_theme_css()}\n</style>", unsafe_allow_html=True)
