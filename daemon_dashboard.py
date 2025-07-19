#!/usr/bin/env python3
"""
Forex AI Daemon Dashboard
Real-time monitoring dashboard for the continuous learning daemon
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-socketio"])
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'forex_ai_daemon_dashboard'
socketio = SocketIO(app, cors_allowed_origins="*")

class DaemonDashboard:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.state_file = os.path.join(self.project_dir, 'logs', 'daemon_state.json')
        self.insights_file = os.path.join(self.project_dir, 'logs', 'learning_insights.json')
        self.pattern_file = os.path.join(self.project_dir, 'logs', 'pattern_memory.json')
        self.daemon_log = os.path.join(self.project_dir, 'logs', 'continuous_learning_daemon.log')
        
    def get_daemon_status(self):
        """Get daemon status"""
        try:
            # Check if daemon is running
            pid_file = '/tmp/forex_ai_daemon.pid'
            if os.path.exists(pid_file):
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                try:
                    os.kill(pid, 0)  # Check if process exists
                    return {'status': 'running', 'pid': pid}
                except OSError:
                    return {'status': 'stopped', 'pid': None}
            else:
                return {'status': 'stopped', 'pid': None}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def get_learning_insights(self):
        """Get learning insights"""
        try:
            if os.path.exists(self.insights_file):
                with open(self.insights_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            return {'error': str(e)}
    
    def get_pattern_memory(self):
        """Get pattern memory"""
        try:
            if os.path.exists(self.pattern_file):
                with open(self.pattern_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            return {'error': str(e)}
    
    def get_recent_logs(self, lines=50):
        """Get recent daemon logs"""
        try:
            if os.path.exists(self.daemon_log):
                with open(self.daemon_log, 'r') as f:
                    all_lines = f.readlines()
                    return all_lines[-lines:] if len(all_lines) > lines else all_lines
            return []
        except Exception as e:
            return [f"Error reading logs: {e}"]
    
    def get_performance_metrics(self):
        """Get performance metrics"""
        try:
            insights = self.get_learning_insights()
            performance_data = insights.get('performance', [])
            
            if performance_data:
                latest = performance_data[-1]
                return {
                    'total_signals': latest.get('total_signals', 0),
                    'win_rate': latest.get('win_rate', 0),
                    'avg_pnl': latest.get('avg_pnl', 0),
                    'last_update': latest.get('timestamp', '')
                }
            return {}
        except Exception as e:
            return {'error': str(e)}
    
    def get_market_analysis(self):
        """Get current market analysis"""
        try:
            insights = self.get_learning_insights()
            market_data = {}
            
            for pair in ['EURUSD', 'USDJPY', 'BTCUSD']:
                if pair in insights and insights[pair]:
                    latest = insights[pair][-1]
                    market_data[pair] = {
                        'patterns': latest.get('patterns', {}),
                        'sentiment': latest.get('sentiment', 0),
                        'volatility': latest.get('volatility', 0),
                        'volume_trend': latest.get('volume_trend', 'unknown'),
                        'last_update': latest.get('timestamp', '')
                    }
            
            return market_data
        except Exception as e:
            return {'error': str(e)}

dashboard = DaemonDashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """API endpoint for daemon status"""
    return jsonify(dashboard.get_daemon_status())

@app.route('/api/insights')
def api_insights():
    """API endpoint for learning insights"""
    return jsonify(dashboard.get_learning_insights())

@app.route('/api/patterns')
def api_patterns():
    """API endpoint for pattern memory"""
    return jsonify(dashboard.get_pattern_memory())

@app.route('/api/logs')
def api_logs():
    """API endpoint for recent logs"""
    lines = request.args.get('lines', 50, type=int)
    return jsonify({'logs': dashboard.get_recent_logs(lines)})

@app.route('/api/performance')
def api_performance():
    """API endpoint for performance metrics"""
    return jsonify(dashboard.get_performance_metrics())

@app.route('/api/market')
def api_market():
    """API endpoint for market analysis"""
    return jsonify(dashboard.get_market_analysis())

@app.route('/api/start_daemon')
def api_start_daemon():
    """API endpoint to start daemon"""
    try:
        daemon_script = os.path.join(dashboard.project_dir, 'continuous_learning_daemon.py')
        subprocess.Popen([sys.executable, daemon_script, '--daemon'])
        return jsonify({'success': True, 'message': 'Daemon started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_daemon')
def api_stop_daemon():
    """API endpoint to stop daemon"""
    try:
        daemon_script = os.path.join(dashboard.project_dir, 'continuous_learning_daemon.py')
        subprocess.run([sys.executable, daemon_script, '--stop'])
        return jsonify({'success': True, 'message': 'Daemon stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def background_update():
    """Background thread to update dashboard data"""
    while True:
        try:
            # Emit updates to connected clients
            socketio.emit('status_update', dashboard.get_daemon_status())
            socketio.emit('performance_update', dashboard.get_performance_metrics())
            socketio.emit('market_update', dashboard.get_market_analysis())
            time.sleep(10)  # Update every 10 seconds
        except Exception as e:
            print(f"Error in background update: {e}")
            time.sleep(30)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status_update', dashboard.get_daemon_status())

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

# Create templates directory and HTML template
templates_dir = os.path.join(dashboard.project_dir, 'templates')
os.makedirs(templates_dir, exist_ok=True)

dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forex AI Daemon Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .status {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }
        .status-running { background-color: #4CAF50; }
        .status-stopped { background-color: #f44336; }
        .status-error { background-color: #ff9800; }
        .button {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        .button:hover { background: #5a6fd8; }
        .button:disabled { background: #ccc; cursor: not-allowed; }
        .logs {
            background: #1e1e1e;
            color: #00ff00;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .metric-value {
            font-weight: bold;
            color: #667eea;
        }
        .pattern-item {
            background: #f8f9fa;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        .full-width {
            grid-column: 1 / -1;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Forex AI Continuous Learning Daemon</h1>
            <p>Real-time monitoring and control dashboard</p>
        </div>

        <div class="grid">
            <!-- Daemon Status -->
            <div class="card">
                <h3>🔄 Daemon Status</h3>
                <div id="status-display">
                    <div class="status">
                        <span class="status-indicator" id="status-indicator"></span>
                        <span id="status-text">Loading...</span>
                    </div>
                    <div id="status-details"></div>
                </div>
                <div style="margin-top: 15px;">
                    <button class="button" onclick="startDaemon()" id="start-btn">Start Daemon</button>
                    <button class="button" onclick="stopDaemon()" id="stop-btn">Stop Daemon</button>
                </div>
            </div>

            <!-- Performance Metrics -->
            <div class="card">
                <h3>📊 Performance Metrics</h3>
                <div id="performance-display">
                    <div class="metric">
                        <span>Total Signals:</span>
                        <span class="metric-value" id="total-signals">-</span>
                    </div>
                    <div class="metric">
                        <span>Win Rate:</span>
                        <span class="metric-value" id="win-rate">-</span>
                    </div>
                    <div class="metric">
                        <span>Avg PnL:</span>
                        <span class="metric-value" id="avg-pnl">-</span>
                    </div>
                    <div class="metric">
                        <span>Last Update:</span>
                        <span class="metric-value" id="last-update">-</span>
                    </div>
                </div>
            </div>

            <!-- Market Analysis -->
            <div class="card">
                <h3>📈 Market Analysis</h3>
                <div id="market-display">
                    <div id="market-pairs"></div>
                </div>
            </div>

            <!-- Pattern Memory -->
            <div class="card">
                <h3>🧠 Pattern Memory</h3>
                <div id="pattern-display">
                    <div id="pattern-list"></div>
                </div>
            </div>
        </div>

        <!-- Logs -->
        <div class="card full-width">
            <h3>📝 Recent Logs</h3>
            <div class="logs" id="logs-display">Loading logs...</div>
        </div>
    </div>

    <script>
        const socket = io();
        
        // Update status display
        function updateStatus(data) {
            const indicator = document.getElementById('status-indicator');
            const text = document.getElementById('status-text');
            const details = document.getElementById('status-details');
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');
            
            indicator.className = 'status-indicator status-' + data.status;
            text.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
            
            if (data.pid) {
                details.innerHTML = `<small>PID: ${data.pid}</small>`;
            } else {
                details.innerHTML = '';
            }
            
            startBtn.disabled = data.status === 'running';
            stopBtn.disabled = data.status === 'stopped';
        }
        
        // Update performance metrics
        function updatePerformance(data) {
            if (data.error) {
                document.getElementById('performance-display').innerHTML = 
                    `<div style="color: red;">Error: ${data.error}</div>`;
                return;
            }
            
            document.getElementById('total-signals').textContent = data.total_signals || '-';
            document.getElementById('win-rate').textContent = 
                data.win_rate ? (data.win_rate * 100).toFixed(1) + '%' : '-';
            document.getElementById('avg-pnl').textContent = 
                data.avg_pnl ? data.avg_pnl.toFixed(2) : '-';
            document.getElementById('last-update').textContent = 
                data.last_update ? new Date(data.last_update).toLocaleString() : '-';
        }
        
        // Update market analysis
        function updateMarket(data) {
            if (data.error) {
                document.getElementById('market-display').innerHTML = 
                    `<div style="color: red;">Error: ${data.error}</div>`;
                return;
            }
            
            let html = '';
            for (const [pair, info] of Object.entries(data)) {
                if (pair === 'performance') continue;
                
                html += `
                    <div class="pattern-item">
                        <strong>${pair}</strong><br>
                        <small>Patterns: ${Object.keys(info.patterns || {}).length}</small><br>
                        <small>Sentiment: ${info.sentiment?.toFixed(3) || 'N/A'}</small><br>
                        <small>Volatility: ${info.volatility?.toFixed(3) || 'N/A'}</small><br>
                        <small>Volume: ${info.volume_trend || 'N/A'}</small>
                    </div>
                `;
            }
            document.getElementById('market-pairs').innerHTML = html || 'No market data available';
        }
        
        // Update pattern memory
        function updatePatterns(data) {
            if (data.error) {
                document.getElementById('pattern-display').innerHTML = 
                    `<div style="color: red;">Error: ${data.error}</div>`;
                return;
            }
            
            let html = '';
            for (const [pair, patterns] of Object.entries(data)) {
                html += `<div class="pattern-item"><strong>${pair}</strong><br>`;
                for (const [pattern, stats] of Object.entries(patterns)) {
                    html += `<small>${pattern}: ${(stats.success_rate * 100).toFixed(1)}% success</small><br>`;
                }
                html += '</div>';
            }
            document.getElementById('pattern-list').innerHTML = html || 'No pattern data available';
        }
        
        // Update logs
        function updateLogs() {
            fetch('/api/logs')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('logs-display').textContent = data.logs.join('');
                })
                .catch(error => {
                    document.getElementById('logs-display').textContent = 'Error loading logs: ' + error;
                });
        }
        
        // Daemon control functions
        function startDaemon() {
            fetch('/api/start_daemon')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Daemon started successfully');
                        setTimeout(() => location.reload(), 2000);
                    } else {
                        alert('Failed to start daemon: ' + data.error);
                    }
                });
        }
        
        function stopDaemon() {
            fetch('/api/stop_daemon')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Daemon stopped successfully');
                        setTimeout(() => location.reload(), 2000);
                    } else {
                        alert('Failed to stop daemon: ' + data.error);
                    }
                });
        }
        
        // Socket event handlers
        socket.on('status_update', updateStatus);
        socket.on('performance_update', updatePerformance);
        socket.on('market_update', updateMarket);
        
        // Initial load
        fetch('/api/status').then(r => r.json()).then(updateStatus);
        fetch('/api/performance').then(r => r.json()).then(updatePerformance);
        fetch('/api/market').then(r => r.json()).then(updateMarket);
        fetch('/api/patterns').then(r => r.json()).then(updatePatterns);
        updateLogs();
        
        // Refresh logs every 30 seconds
        setInterval(updateLogs, 30000);
    </script>
</body>
</html>'''

# Write the HTML template
with open(os.path.join(templates_dir, 'dashboard.html'), 'w') as f:
    f.write(dashboard_html)

if __name__ == '__main__':
    print("Starting Forex AI Daemon Dashboard...")
    print("Dashboard will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    # Start background update thread
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()
    
    # Start Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=False) 