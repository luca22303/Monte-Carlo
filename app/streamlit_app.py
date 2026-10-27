"""Streamlit entry point: ``streamlit run app/streamlit_app.py``."""

from pathlib import Path

import streamlit as st
from common import get_cfg, setup_summary, show_flash

HERE = Path(__file__).parent

st.set_page_config(page_title="Monte Carlo wealth engine", page_icon="📈", layout="wide")

pages = [
    st.Page(HERE / "views/setup.py", title="Setup", icon="⚙️"),
    st.Page(HERE / "views/simulation.py", title="Simulation", icon="📈", default=True),
    st.Page(HERE / "views/strategies.py", title="Strategies", icon="⚖️"),
    st.Page(HERE / "views/explorer.py", title="Allocation explorer", icon="🧭"),
    st.Page(HERE / "views/etfs.py", title="ETFs & savings plan", icon="🧾"),
    st.Page(HERE / "views/goal.py", title="Goal planner", icon="🎯"),
    st.Page(HERE / "views/stress.py", title="Stress & behaviour", icon="🌪️"),
    st.Page(HERE / "views/review.py", title="Annual review", icon="🗓️"),
    st.Page(HERE / "views/assumptions.py", title="Assumptions & history", icon="📚"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("**Active setup**")
    st.caption(setup_summary(get_cfg()))
    st.page_link(pages[0], label="Change in Setup", icon="⚙️")
    st.caption("💾 Your inputs are saved automatically on this computer (folder *my_data*).")
    with st.popover("⏻ Quit dashboard"):
        st.caption("Stops the dashboard running in the background. Your inputs are saved. Start it again with "
                   "the Monte Carlo app or start_dashboard.command.")
        if st.button("Quit now", type="primary"):
            import os
            import signal
            import threading

            st.success("Dashboard stopped. You can close this tab.")
            threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    st.divider()
    st.caption(
        "Educational planning tool, not investment advice. Results are only as good as the assumptions; "
        "see *Assumptions & history*."
    )

show_flash()
nav.run()
