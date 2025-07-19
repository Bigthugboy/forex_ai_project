# 🤖 Forex AI 24/7 Continuous Learning System

## Overview

This system provides **true 24/7 continuous learning** for your Forex AI trading system, even when the main trading process is offline. The daemon runs in the background, continuously analyzing markets, learning patterns, and improving models.

## 🚀 Features

### ✅ **Continuous Market Analysis**
- **Every 5 minutes**: Analyzes all trading pairs for patterns
- **Real-time pattern detection**: Technical patterns, trends, volatility
- **News sentiment integration**: Economic calendar and crypto news
- **Volume analysis**: Volume trends and spikes

### ✅ **Intelligent Pattern Learning**
- **Every 30 minutes**: Learns from accumulated pattern data
- **Success rate analysis**: Which patterns lead to profitable outcomes
- **Pattern memory**: Persistent storage of learned insights
- **Adaptive learning**: Updates based on market conditions

### ✅ **Automatic Model Updates**
- **Every hour**: Checks if models need updating
- **Performance-based triggers**: Updates when accuracy drops
- **Insight-driven retraining**: Uses accumulated learning insights
- **Version control**: Tracks model versions and changes

### ✅ **Performance Monitoring**
- **Every 2 hours**: Checks performance metrics
- **Win rate tracking**: Monitors signal success rates
- **PnL analysis**: Tracks average profit/loss
- **Health checks**: Alerts if no signals generated

### ✅ **Persistent State Management**
- **State persistence**: Survives system reboots
- **Learning insights**: Stores market analysis data
- **Pattern memory**: Remembers successful patterns
- **Performance history**: Tracks metrics over time

## 📁 System Components

### 1. **Continuous Learning Daemon** (`continuous_learning_daemon.py`)
- **Main daemon process** that runs 24/7
- **Background scheduling** of all learning tasks
- **State management** and persistence
- **Error handling** and recovery

### 2. **System Service Installation**
- **Linux**: `install_daemon.sh` (systemd service)
- **macOS**: `install_daemon_macos.sh` (launchd service)
- **Automatic startup** on system boot
- **Automatic restart** on crashes

### 3. **Monitoring Dashboard** (`daemon_dashboard.py`)
- **Web-based dashboard** at `http://localhost:5000`
- **Real-time status** monitoring
- **Performance metrics** visualization
- **Daemon control** (start/stop)
- **Live logs** viewing

### 4. **Monitoring Scripts**
- **Linux**: `monitor_daemon.sh`
- **macOS**: `monitor_daemon_macos.sh`
- **Cron-based monitoring** every 5 minutes
- **Automatic restart** if daemon stops

## 🛠️ Installation

### Prerequisites
```bash
# Install required Python packages
pip install python-daemon flask flask-socketio schedule
```

### Step 1: Install the Daemon Service

**For Linux (systemd):**
```bash
chmod +x install_daemon.sh
./install_daemon.sh
```

**For macOS (launchd):**
```bash
chmod +x install_daemon_macos.sh
./install_daemon_macos.sh
```

### Step 2: Start the Daemon

**Linux:**
```bash
sudo systemctl start forex-ai-daemon
sudo systemctl enable forex-ai-daemon  # Auto-start on boot
```

**macOS:**
```bash
launchctl load ~/Library/LaunchAgents/com.forexai.daemon.plist
```

### Step 3: Start the Dashboard

```bash
python daemon_dashboard.py
```

Then open: http://localhost:5000

## 📊 Dashboard Features

### 🔄 **Daemon Status**
- **Real-time status**: Running/Stopped/Error
- **Process ID**: Shows daemon PID
- **Control buttons**: Start/Stop daemon
- **Auto-refresh**: Updates every 10 seconds

### 📈 **Performance Metrics**
- **Total signals**: Number of signals generated
- **Win rate**: Percentage of successful signals
- **Average PnL**: Average profit/loss per signal
- **Last update**: Timestamp of last performance check

### 🧠 **Market Analysis**
- **Per-pair analysis**: EURUSD, USDJPY, BTCUSD
- **Pattern detection**: Active technical patterns
- **Sentiment scores**: News sentiment analysis
- **Volatility levels**: Current market volatility
- **Volume trends**: Increasing/decreasing/stable

### 🧠 **Pattern Memory**
- **Success rates**: Which patterns work best
- **Per-pair patterns**: Pattern effectiveness by pair
- **Learning insights**: Accumulated knowledge
- **Adaptive learning**: Pattern evolution over time

### 📝 **Live Logs**
- **Real-time logs**: Daemon activity logs
- **Auto-scroll**: Latest entries visible
- **Error highlighting**: Easy error identification
- **Log filtering**: Configurable log levels

## 🔧 Configuration

### Learning Intervals
Edit `continuous_learning_daemon.py` to adjust intervals:

```python
# Learning intervals (in seconds)
self.market_analysis_interval = 300    # 5 minutes
self.pattern_learning_interval = 1800  # 30 minutes
self.model_update_interval = 3600      # 1 hour
self.performance_check_interval = 7200 # 2 hours
```

### Trading Pairs
Configure pairs in `config.py`:

```python
TRADING_PAIRS = [
    'EURUSD', 'USDJPY', 'USDCHF', 'NZDUSD',
    'EURJPY', 'NZDJPY', 'BTCUSD'
]
```

### Log Levels
Adjust logging in the daemon:

```python
self.logger = get_logger('continuous_learning_daemon', 
                        log_file='logs/continuous_learning_daemon.log',
                        level=logging.INFO)
```

## 📋 Management Commands

### Daemon Control

**Linux:**
```bash
# Check status
systemctl status forex-ai-daemon

# Start daemon
sudo systemctl start forex-ai-daemon

# Stop daemon
sudo systemctl stop forex-ai-daemon

# Restart daemon
sudo systemctl restart forex-ai-daemon

# View logs
journalctl -u forex-ai-daemon -f

# Enable/disable auto-start
sudo systemctl enable forex-ai-daemon
sudo systemctl disable forex-ai-daemon
```

**macOS:**
```bash
# Check status
launchctl list | grep com.forexai.daemon

# Start daemon
launchctl load ~/Library/LaunchAgents/com.forexai.daemon.plist

# Stop daemon
launchctl unload ~/Library/LaunchAgents/com.forexai.daemon.plist

# View logs
tail -f logs/continuous_learning_daemon.log
```

### Manual Daemon Control
```bash
# Run in foreground (for testing)
python continuous_learning_daemon.py --foreground

# Run as daemon
python continuous_learning_daemon.py --daemon

# Stop daemon
python continuous_learning_daemon.py --stop

# Check status
python continuous_learning_daemon.py --status
```

### Dashboard Control
```bash
# Start dashboard
python daemon_dashboard.py

# Access dashboard
open http://localhost:5000
```

## 📊 Monitoring and Alerts

### Automatic Monitoring
- **Cron job**: Checks daemon every 5 minutes
- **Auto-restart**: Restarts if daemon stops
- **Log monitoring**: Tracks errors and warnings
- **Performance alerts**: Notifies of performance drops

### Manual Monitoring
```bash
# Check daemon status
./monitor_daemon.sh          # Linux
./monitor_daemon_macos.sh    # macOS

# View recent logs
tail -50 logs/continuous_learning_daemon.log

# Check learning insights
cat logs/learning_insights.json

# Check pattern memory
cat logs/pattern_memory.json
```

## 🔍 Troubleshooting

### Common Issues

**1. Daemon won't start**
```bash
# Check logs
tail -f logs/continuous_learning_daemon.log

# Check permissions
ls -la continuous_learning_daemon.py

# Check Python environment
which python
python --version
```

**2. Dashboard won't load**
```bash
# Check if port 5000 is available
lsof -i :5000

# Check Flask installation
pip list | grep flask

# Check firewall settings
sudo ufw status  # Linux
```

**3. No learning insights**
```bash
# Check if data fetching works
python -c "from data.data_fetcher import DataFetcher; df = DataFetcher().get_price_data('BTCUSD', '1h', 48); print('Data OK' if df is not None else 'Data failed')"

# Check API keys
cat .env | grep API
```

**4. Service won't start on boot**
```bash
# Linux: Check systemd
sudo systemctl status forex-ai-daemon
sudo journalctl -u forex-ai-daemon

# macOS: Check launchd
launchctl list | grep com.forexai.daemon
```

### Log Analysis

**Key log entries to watch for:**
- `"Starting Continuous Learning Daemon"` - Daemon started successfully
- `"Market analysis completed"` - Pattern analysis working
- `"Pattern learning for"` - Learning cycle active
- `"Model retraining completed"` - Model updates working
- `"Performance (7 days)"` - Performance monitoring active

**Error patterns:**
- `"Error analyzing market patterns"` - Data fetching issues
- `"Error in pattern learning"` - Learning algorithm issues
- `"Error updating models"` - Model training issues
- `"Error checking performance"` - Performance tracking issues

## 📈 Performance Optimization

### Resource Usage
- **CPU**: ~5-10% during analysis cycles
- **Memory**: ~200-500MB depending on data size
- **Disk**: ~50-100MB for logs and state files
- **Network**: Minimal (API calls for data)

### Optimization Tips
1. **Adjust intervals** for your system capabilities
2. **Monitor resource usage** with `htop` or `top`
3. **Clean old logs** periodically
4. **Use SSD storage** for better I/O performance
5. **Optimize API calls** by adjusting lookback periods

## 🔐 Security Considerations

### File Permissions
```bash
# Secure daemon files
chmod 600 continuous_learning_daemon.py
chmod 700 logs/
chmod 600 logs/*.json
```

### Network Security
- **Dashboard**: Only accessible on localhost by default
- **API keys**: Stored in `.env` file with restricted permissions
- **Service isolation**: Runs with minimal privileges

### Data Privacy
- **Local storage**: All data stored locally
- **No external sharing**: Learning insights stay private
- **Encrypted storage**: Consider encrypting sensitive data

## 🚀 Advanced Features

### Custom Learning Algorithms
Extend the daemon with custom learning:

```python
def custom_pattern_analyzer(self, features_df):
    """Custom pattern analysis logic"""
    # Your custom analysis here
    return custom_patterns
```

### Integration with External Systems
```python
def send_to_external_system(self, insights):
    """Send insights to external system"""
    # API calls, database updates, etc.
    pass
```

### Custom Alerts
```python
def custom_alert_trigger(self, performance_data):
    """Custom alert logic"""
    if performance_data['win_rate'] < 0.5:
        self.send_alert("Low win rate detected!")
```

## 📞 Support

### Getting Help
1. **Check logs**: `tail -f logs/continuous_learning_daemon.log`
2. **Use dashboard**: http://localhost:5000
3. **Review documentation**: This README
4. **Check system status**: Use monitoring scripts

### Reporting Issues
When reporting issues, include:
- **System**: Linux/macOS version
- **Python version**: `python --version`
- **Error logs**: Relevant log entries
- **Steps to reproduce**: What you were doing
- **Expected behavior**: What should happen

---

## 🎯 Summary

This 24/7 continuous learning system provides:

✅ **True 24/7 operation** - Works even when main process is offline  
✅ **Intelligent learning** - Adapts to market conditions  
✅ **Automatic management** - Self-monitoring and recovery  
✅ **Real-time monitoring** - Web dashboard for oversight  
✅ **Persistent learning** - Remembers and builds on insights  
✅ **Performance tracking** - Monitors and improves over time  

Your Forex AI system will now continuously learn and improve, even when you're not actively trading! 🚀 