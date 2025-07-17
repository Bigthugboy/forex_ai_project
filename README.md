# AI-Powered Forex & Crypto Signal Generator

An advanced AI-driven trading system that generates real-time signals for forex and cryptocurrency pairs using multi-timeframe analysis, technical indicators, and news sentiment analysis.

---

## 🚦 Pipeline Orchestration (NEW)

All signal generation, scheduling, and continuous learning is now managed by a robust `PipelineOrchestrator` class. This orchestrator:
- Schedules and runs clustered signal generation sessions for all configured pairs
- Handles all data fetching, feature engineering, model training, and signal output
- Manages continuous learning and retraining cycles
- Supports dependency injection for testability and integration testing

**Main entry points:**
- `main.py`: Runs the orchestrator in clustered/scheduled mode (production)
- `core.py`: Runs a single session and a continuous learning cycle (for UI, CLI, or ad-hoc use)
- `auto_retrain.py`: Runs orchestrator-based retraining on a schedule

**Advanced usage:**
You can use the orchestrator directly in Python:
```python
from pipeline.pipeline_orchestrator import PipelineOrchestrator
from utils.notify import send_email

orchestrator = PipelineOrchestrator()
# Run a single session (all pairs)
results = orchestrator.run_session(send_email_func=send_email)
# Run continuous learning
orchestrator.run_continuous_learning()
```

---

## 🛠️ Installation & Setup

### 1. Environment Setup
```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration
Edit `config.py` to add your API keys:
```python
# Required APIs
NEWS_API_KEY = 'your_newsapi_key'
ALPHA_VANTAGE_API_KEY = 'your_alphavantage_key'

# Optional APIs (for enhanced features)
CRYPTOPANIC_API_KEY = 'your_cryptopanic_key'
```

### 3. Run the System
```bash
# Run main orchestrated analysis (recommended)
python main.py

# Run a single session and continuous learning (for UI/CLI)
python core.py

# Run orchestrator-based retraining (scheduled)
python auto_retrain.py

# Test multi-timeframe functionality (standalone)
python test_multi_timeframe.py
```

---

## 📁 Project Structure

```
forex_ai_project/
├── config.py                 # Configuration and API keys
├── main.py                   # Main execution script (uses PipelineOrchestrator)
├── core.py                   # Single session + learning (uses PipelineOrchestrator)
├── auto_retrain.py           # Scheduled retraining (uses PipelineOrchestrator)
├── requirements.txt          # Python dependencies
├── pipeline/
│   ├── pipeline_orchestrator.py # Central pipeline orchestration logic
│   └── signal_pipeline.py       # Signal pipeline for a single pair
│
├── data/
│   ├── fetch_market.py       # Market data collection
│   ├── fetch_news.py         # News sentiment analysis
│   ├── preprocess.py         # Feature engineering
│   └── multi_timeframe.py    # Multi-timeframe analysis
│
├── models/
│   ├── train_model.py        # AI model training
│   ├── predict.py            # Signal prediction
│   └── saved_models/         # Trained model files
│
├── signals/
│   └── signal_generator.py   # Signal generation logic
│
├── utils/
│   ├── logger.py             # Logging utilities
│   └── helpers.py            # Helper functions
│
└── logs/                     # System logs
```

---

## 🧪 Integration Testing

Integration tests for the orchestrator and pipeline are provided in `tests/test_pipeline_integration.py`. These tests:
- Run the orchestrator for a single session and continuous learning cycle
- Mock email and logger dependencies for testability
- Ensure the full pipeline runs without error and produces expected results

---

## 🔧 Configuration Options

### Trading Parameters
```python
RISK_PER_TRADE = 2.0          # USD risk per trade
SL_PIPS = 20                  # Stop loss in pips (20-30 range)
MIN_LOT_SIZE = 0.02           # Minimum position size
MAX_LOT_SIZE = 0.05           # Maximum position size
```

### Model Parameters
```python
LOOKBACK_PERIOD = 90         # Days of historical data
CONFIDENCE_THRESHOLD = 0.75   # Minimum confidence for signals
CONFLUENCE_REQUIREMENT = 4    # Minimum aligned indicators
```

### Timeframes
```python
TIMEFRAMES = ['15m', '1h', '4h']  # Available timeframes
PREDICTION_WINDOW = 5         # Hours ahead to predict
```

---

## 📈 Performance Monitoring

The system logs comprehensive information:
- Feature engineering progress
- Model training metrics
- Signal generation details
- Confluence factor analysis
- Multi-timeframe validation

Check `logs/` directory for detailed analysis.

---

## ⚠️ Risk Disclaimer

This is an AI-powered trading system for educational and research purposes. Trading forex and cryptocurrencies involves substantial risk of loss. Always:
- Test thoroughly on demo accounts
- Start with small position sizes
- Never risk more than you can afford to lose
- Consider consulting with financial advisors

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional technical indicators
- More news data sources
- Enhanced risk management
- Backtesting framework
- Real-time data streaming

---

## 📄 License

MIT License - see LICENSE file for details.
