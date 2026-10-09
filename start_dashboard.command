#!/bin/bash
# Double-click this file in Finder (macOS) to update and open the dashboard.
# Close the Terminal window or press Ctrl+C to stop it.
cd "$(dirname "$0")" || exit 1

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
  echo "Python 3.11 or newer is needed. Install it from https://www.python.org/downloads/ and try again."
  read -r -p "Press Enter to close."
  exit 1
fi

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate

echo "Checking for updates…"
git pull --ff-only || echo "Could not update – starting the current version."

echo "Installing/updating libraries (first run takes a minute)…"
pip install -q -r requirements.txt

echo "Starting the dashboard – it opens in your browser."
exec streamlit run app/streamlit_app.py
