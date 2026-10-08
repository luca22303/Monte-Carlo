"""Streamlit entry point: ``streamlit run app/streamlit_app.py``."""

from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent

st.set_page_config(page_title="Monte Carlo wealth engine", page_icon="📈", layout="wide")

pages = [
    st.Page(HERE / "views/setup.py", title="Setup", icon="⚙️"),
    st.Page(HERE / "views/simulation.py", title="Simulation", icon="📈", default=True),
    st.Page(HERE / "views/strategies.py", title="Strategies", icon="⚖️"),
    st.Page(HERE / "views/explorer.py", title="Allocation explorer", icon="🧭"),
    st.Page(HERE / "views/etfs.py", title="ETFs & savings plan", icon="🧾"),
    st.Page(HERE / "views/goal.py", title="Goal planner", icon="🎯"),
    st.Page(HERE / "views/review.py", title="Annual review", icon="🗓️"),
    st.Page(HERE / "views/assumptions.py", title="Assumptions & history", icon="📚"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.caption(
        "Educational planning tool, not investment advice. Results are only as good as the assumptions; "
        "see *Assumptions & history*."
    )

nav.run()
