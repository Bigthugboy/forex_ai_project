#!/bin/bash

# Forex AI Continuous Learning Daemon Installation Script for macOS
# This script installs the daemon as a launchd service for automatic startup

set -e

echo "=== Forex AI Continuous Learning Daemon Installation (macOS) ==="

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMON_SCRIPT="$PROJECT_DIR/continuous_learning_daemon.py"
SERVICE_NAME="com.forexai.daemon"
USER=$(whoami)

echo "Project directory: $PROJECT_DIR"
echo "Daemon script: $DAEMON_SCRIPT"
echo "User: $USER"

# Check if daemon script exists
if [[ ! -f "$DAEMON_SCRIPT" ]]; then
    echo "Error: Daemon script not found at $DAEMON_SCRIPT"
    exit 1
fi

# Make daemon script executable
chmod +x "$DAEMON_SCRIPT"

# Create launchd plist file
PLIST_FILE="$HOME/Library/LaunchAgents/${SERVICE_NAME}.plist"
mkdir -p "$(dirname "$PLIST_FILE")"

cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$SERVICE_NAME</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>$PROJECT_DIR/venv/bin/python</string>
        <string>$DAEMON_SCRIPT</string>
        <string>--daemon</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <true/>
    
    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/logs/daemon_stdout.log</string>
    
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/logs/daemon_stderr.log</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>$PROJECT_DIR</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
    
    <key>ProcessType</key>
    <string>Background</string>
    
    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
EOF

echo "Created launchd plist file: $PLIST_FILE"

# Create log directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Create monitoring script for macOS
MONITOR_SCRIPT="$PROJECT_DIR/monitor_daemon_macos.sh"
cat > "$MONITOR_SCRIPT" << 'EOF'
#!/bin/bash

# Forex AI Daemon Monitoring Script for macOS
# Monitors the daemon status and restarts if needed

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="com.forexai.daemon"
LOG_FILE="$PROJECT_DIR/logs/daemon_monitor.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

check_daemon() {
    if launchctl list | grep -q "$SERVICE_NAME"; then
        log "Daemon is running"
        return 0
    else
        log "Daemon is not running, attempting to start..."
        launchctl load ~/Library/LaunchAgents/${SERVICE_NAME}.plist
        sleep 5
        
        if launchctl list | grep -q "$SERVICE_NAME"; then
            log "Daemon started successfully"
            return 0
        else
            log "Failed to start daemon"
            return 1
        fi
    fi
}

# Check daemon status
check_daemon

# If running as cron job, exit here
if [[ "$1" == "cron" ]]; then
    exit 0
fi

# If running interactively, show status
echo "=== Forex AI Daemon Status ==="
launchctl list | grep "$SERVICE_NAME" || echo "Service not found"

echo ""
echo "=== Recent Logs ==="
if [[ -f "$PROJECT_DIR/logs/continuous_learning_daemon.log" ]]; then
    tail -20 "$PROJECT_DIR/logs/continuous_learning_daemon.log"
else
    echo "No log file found"
fi

echo ""
echo "=== Monitoring Commands ==="
echo "Check status: launchctl list | grep $SERVICE_NAME"
echo "View logs: tail -f $PROJECT_DIR/logs/continuous_learning_daemon.log"
echo "Restart: launchctl unload ~/Library/LaunchAgents/${SERVICE_NAME}.plist && launchctl load ~/Library/LaunchAgents/${SERVICE_NAME}.plist"
echo "Stop: launchctl unload ~/Library/LaunchAgents/${SERVICE_NAME}.plist"
echo "Start: launchctl load ~/Library/LaunchAgents/${SERVICE_NAME}.plist"
EOF

chmod +x "$MONITOR_SCRIPT"

# Create cron job for monitoring (every 5 minutes)
CRON_JOB="*/5 * * * * $MONITOR_SCRIPT cron"
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo ""
echo "=== Installation Complete ==="
echo "Service name: $SERVICE_NAME"
echo "Plist file: $PLIST_FILE"
echo "Monitoring script: $MONITOR_SCRIPT"
echo "Log file: $PROJECT_DIR/logs/continuous_learning_daemon.log"
echo ""
echo "=== Next Steps ==="
echo "1. Start the daemon: launchctl load ~/Library/LaunchAgents/${SERVICE_NAME}.plist"
echo "2. Check status: launchctl list | grep $SERVICE_NAME"
echo "3. View logs: tail -f $PROJECT_DIR/logs/continuous_learning_daemon.log"
echo "4. Monitor: $MONITOR_SCRIPT"
echo ""
echo "=== Service Management ==="
echo "Start:   launchctl load ~/Library/LaunchAgents/${SERVICE_NAME}.plist"
echo "Stop:    launchctl unload ~/Library/LaunchAgents/${SERVICE_NAME}.plist"
echo "Restart: launchctl unload ~/Library/LaunchAgents/${SERVICE_NAME}.plist && launchctl load ~/Library/LaunchAgents/${SERVICE_NAME}.plist"
echo "Status:  launchctl list | grep $SERVICE_NAME"
echo "Logs:    tail -f $PROJECT_DIR/logs/continuous_learning_daemon.log"
echo ""
echo "The daemon will automatically start on system boot and restart if it crashes."
echo "Monitoring script will check every 5 minutes and restart if needed." 