#!/bin/bash
# Removes the "Monte Carlo" app and the login autostart (your data in my_data/ is kept).
PLIST="$HOME/Library/LaunchAgents/com.montecarlo.dashboard.plist"
[ -f "$PLIST" ] && launchctl unload "$PLIST" 2>/dev/null; rm -f "$PLIST"
rm -rf "$HOME/Applications/Monte Carlo.app"
pkill -f "streamlit run app/streamlit_app.py" 2>/dev/null
echo "Removed the app and the autostart. Your saved data in my_data/ is untouched."
