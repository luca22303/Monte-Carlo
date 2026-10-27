#!/bin/bash
# Stop the background dashboard (also possible with "Quit dashboard" in the app's sidebar).
pkill -f "streamlit run app/streamlit_app.py" && echo "Dashboard stopped." || echo "Dashboard was not running."
