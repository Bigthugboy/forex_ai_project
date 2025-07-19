#!/bin/bash

# Forex AI Continuous Learning Daemon Installation Script
# This script installs the daemon as a system service for automatic startup

set -e

echo "=== Forex AI Continuous Learning Daemon Installation ==="

# Check if running as root (needed for systemd service installation)
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root for security reasons."
   echo "Please run as a regular user and use sudo when prompted."
   exit 1
fi

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMON_SCRIPT="$PROJECT_DIR/continuous_learning_daemon.py"
SERVICE_NAME="forex-ai-daemon"
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

# Create systemd service file
SERVICE_FILE="/tmp/${SERVICE_NAME}.service"
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Forex AI Continuous Learning Daemon
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python $DAEMON_SCRIPT --daemon
ExecStop=$PROJECT_DIR/venv/bin/python $DAEMON_SCRIPT --stop
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Environment variables
Environment=PYTHONPATH=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_DIR/logs
ReadWritePaths=$PROJECT_DIR/models/saved_models

[Install]
WantedBy=multi-user.target
EOF

echo "Created systemd service file: $SERVICE_FILE"

# Install the service
echo "Installing systemd service..."
sudo cp "$SERVICE_FILE" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload

# Enable the service for automatic startup
echo "Enabling service for automatic startup..."
sudo systemctl enable "$SERVICE_NAME"

# Create log directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Create monitoring script
MONITOR_SCRIPT="$PROJECT_DIR/monitor_daemon.sh"
cat > "$MONITOR_SCRIPT" << 'EOF'
#!/bin/bash

# Forex AI Daemon Monitoring Script
# Monitors the daemon status and restarts if needed

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="forex-ai-daemon"
LOG_FILE="$PROJECT_DIR/logs/daemon_monitor.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

check_daemon() {
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log "Daemon is running"
        return 0
    else
        log "Daemon is not running, attempting to start..."
        sudo systemctl start "$SERVICE_NAME"
        sleep 5
        
        if systemctl is-active --quiet "$SERVICE_NAME"; then
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
systemctl status "$SERVICE_NAME" --no-pager -l

echo ""
echo "=== Recent Logs ==="
journalctl -u "$SERVICE_NAME" --no-pager -l -n 20

echo ""
echo "=== Monitoring Commands ==="
echo "Check status: systemctl status $SERVICE_NAME"
echo "View logs: journalctl -u $SERVICE_NAME -f"
echo "Restart: sudo systemctl restart $SERVICE_NAME"
echo "Stop: sudo systemctl stop $SERVICE_NAME"
echo "Enable/disable: sudo systemctl enable/disable $SERVICE_NAME"
EOF

chmod +x "$MONITOR_SCRIPT"

# Create cron job for monitoring (every 5 minutes)
CRON_JOB="*/5 * * * * $MONITOR_SCRIPT cron"
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo ""
echo "=== Installation Complete ==="
echo "Service name: $SERVICE_NAME"
echo "Monitoring script: $MONITOR_SCRIPT"
echo "Log file: $PROJECT_DIR/logs/continuous_learning_daemon.log"
echo ""
echo "=== Next Steps ==="
echo "1. Start the daemon: sudo systemctl start $SERVICE_NAME"
echo "2. Check status: systemctl status $SERVICE_NAME"
echo "3. View logs: journalctl -u $SERVICE_NAME -f"
echo "4. Monitor: $MONITOR_SCRIPT"
echo ""
echo "=== Service Management ==="
echo "Start:   sudo systemctl start $SERVICE_NAME"
echo "Stop:    sudo systemctl stop $SERVICE_NAME"
echo "Restart: sudo systemctl restart $SERVICE_NAME"
echo "Status:  systemctl status $SERVICE_NAME"
echo "Logs:    journalctl -u $SERVICE_NAME -f"
echo ""
echo "The daemon will automatically start on system boot and restart if it crashes."
echo "Monitoring script will check every 5 minutes and restart if needed." 