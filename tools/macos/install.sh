#!/bin/bash
# Creates the "Monte Carlo" app (no terminal window) in ~/Applications.
#   bash tools/macos/install.sh              app only
#   bash tools/macos/install.sh --autostart  app + start the dashboard automatically at login
set -eu
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$HOME/Applications/Monte Carlo.app"
LAUNCH="$REPO/tools/macos/launch.sh"
chmod +x "$LAUNCH" "$REPO/tools/macos/stop.sh"
mkdir -p "$HOME/Applications"

rm -rf "$APP"
osacompile -o "$APP" -e "do shell script \"/bin/bash \" & quoted form of \"$LAUNCH\" & \" > /dev/null 2>&1 &\""
echo "Created: $APP"

if [ "${1:-}" = "--autostart" ]; then
  PLIST="$HOME/Library/LaunchAgents/com.montecarlo.dashboard.plist"
  mkdir -p "$HOME/Library/LaunchAgents" "$REPO/logs"
  cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.montecarlo.dashboard</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$LAUNCH</string><string>--service</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>$REPO/logs/launchd.log</string>
  <key>StandardErrorPath</key><string>$REPO/logs/launchd.log</string>
</dict>
</plist>
PLISTEOF
  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load "$PLIST"
  echo "Autostart enabled: the dashboard now runs in the background after every login."
  echo "Bookmark http://localhost:8501 in your browser."
fi

echo
echo "Done. Open 'Monte Carlo' from Launchpad or ~/Applications (drag it to your Dock to keep it there)."
