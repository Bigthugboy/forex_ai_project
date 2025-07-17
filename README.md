# AI-Powered Forex & Crypto Signal Generator

An advanced AI-driven trading system that generates real-time signals for forex and cryptocurrency pairs using multi-timeframe analysis, technical indicators, and news sentiment analysis.

## 🚀 Features

### Multi-Timeframe Analysis
- **15-minute data**: Short-term momentum and entry timing
- **1-hour data**: Base timeframe for signal generation
- **4-hour data**: Long-term trend analysis and support/resistance
- **Cross-timeframe confluence**: Validates signals across timeframes

### Technical Analysis
- **50+ Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic
- **Advanced Patterns**: Head & Shoulders, Double Tops/Bottoms, Wedges, Fakeouts
- **Market Structure**: Supply/Demand zones, Wyckoff phases, Trend/Range detection
- **Price Action**: Pin bars, Engulfing patterns, Breakouts, Inside bars

### AI & Machine Learning
- **Ensemble Model**: XGBoost + LightGBM with probability calibration
- **Multi-timeframe Features**: 28 additional indicators from different timeframes
- **News Sentiment Integration**: Real-time sentiment analysis from multiple sources
- **Confluence Validation**: Requires 4+ aligned indicators for signal generation

### Risk Management
- **Dynamic Position Sizing**: Based on account risk and stop-loss distance
- **Pip-based SL/TP**: Configurable stop-loss and take-profit levels
- **Confidence Scoring**: 75% threshold for signal generation
- **Economic Calendar**: Suppresses signals during high-impact news events

## 🏗️ How the Project Works

### 1. Data Collection Pipeline

```
Market Data Sources:
├── Yahoo Finance (15m, 1h, 4h OHLCV data)
├── Alpha Vantage (Real-time forex rates)
└── CryptoCompare (Crypto data)

News Data Sources:
├── NewsAPI (Forex news)
├── CryptoPanic (Crypto news)
├── CoinGecko (Crypto news)
└── RSS Feeds (FXStreet, ForexCrunch, Economic Calendar)
```

### 2. Multi-Timeframe Feature Engineering

The system fetches data from three timeframes and creates comprehensive features:

```python
# 15-minute timeframe features (14 indicators)
sma_20_15m, sma_50_15m, ema_12_15m, ema_26_15m
rsi_14_15m, macd_15m, macd_signal_15m
bb_high_15m, bb_low_15m, atr_14_15m
volatility_20_15m, price_vs_sma20_15m, price_vs_sma50_15m
trend_strength_15m

# 4-hour timeframe features (14 indicators)
sma_20_4h, sma_50_4h, ema_12_4h, ema_26_4h
rsi_14_4h, macd_4h, macd_signal_4h
bb_high_4h, bb_low_4h, atr_14_4h
volatility_20_4h, price_vs_sma20_4h, price_vs_sma50_4h
trend_strength_4h
```

### 3. AI Model Training Process

#### Target Variable Creation
```python
# Calculate 5-hour future returns
df['future_return'] = df['Close'].shift(-5) / df['Close'] - 1

# Binary classification: 1 if price increases >0.1%, 0 otherwise
df['target'] = (df['future_return'] > 0.001).astype(int)
```

#### Model Architecture
```python
# Ensemble of two powerful algorithms
xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
lgb_model = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)

# Soft voting ensemble
ensemble = VotingClassifier(estimators=[('xgb', xgb_model), ('lgb', lgb_model)], voting='soft')

# Probability calibration for better confidence scores
calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
```

#### Training Process
1. **Data Preparation**: Multi-timeframe features + news sentiment
2. **Feature Selection**: Remove price columns, keep technical indicators
3. **Class Balancing**: Upsample minority class for balanced training
4. **Train/Test Split**: 80/20 split with stratification
5. **Model Training**: Ensemble training with cross-validation
6. **Probability Calibration**: Improve confidence score accuracy

### 4. Signal Generation Logic

#### Confluence Requirements (4+ Indicators Must Align)

The system requires **at least 4 confirming factors** for signal generation:

```python
confluence_factors = []

# 1. Trend Alignment
if trendline_up and prediction == BUY: confluence_factors.append('trend_up')
if trendline_down and prediction == SELL: confluence_factors.append('trend_down')

# 2. Price Action Patterns
if breakout_high and prediction == BUY: confluence_factors.append('breakout_high')
if bullish_engulfing and prediction == BUY: confluence_factors.append('bullish_engulfing')

# 3. Technical Indicators
if rsi_14 < 30 and prediction == BUY: confluence_factors.append('rsi_oversold')
if macd > macd_signal and prediction == BUY: confluence_factors.append('macd_bull')

# 4. News Sentiment
if news_sentiment > 0.2 and prediction == BUY: confluence_factors.append('news_bull')

# Signal Generation
if len(confluence_factors) >= 4 and confidence >= 0.75:
    generate_signal()
```

#### Multi-Timeframe Validation

```python
# Cross-timeframe momentum alignment
if momentum_15m > 0 and momentum_4h > 0:
    confluence_factors.append('momentum_aligned')

# RSI conditions across timeframes
if rsi_15m < 30 and rsi_4h < 40:
    confluence_factors.append('multi_tf_oversold')
```

### 5. Signal Output

Each generated signal includes:

```python
signal = {
    'pair': 'USDJPY',
    'trade_type': 'Buy Stop',  # Instant, Buy Stop, Sell Stop, Buy Limit, Sell Limit
    'entry': 148.250,
    'signal': 'BUY',
    'confidence': 0.82,  # Must be >= 75%
    'confluence': 5,     # Number of aligned factors (must be >= 4)
    'confluence_factors': ['trend_up', 'breakout_high', 'rsi_oversold', 'macd_bull', 'news_bull'],
    'stop_loss': 147.850,
    'take_profit_1': 148.650,
    'take_profit_2': 149.050,
    'take_profit_3': 149.450,
    'position_size': 0.02  # Lot size based on risk management
}
```

## 📊 Trading Strategy

### Optimal Trading Style: Day Trading with Swing Capabilities

**Timeframe**: 2-8 hour holds
**Entry**: Based on 1h signals with 15m precision
**Exit**: Within 5 hours (prediction window)
**Confidence**: 75% threshold
**Risk**: Moderate with proper stop-losses

### Why This Configuration Works

1. **15m data**: Captures short-term momentum and entry timing
2. **1h data**: Primary signal generation timeframe
3. **4h data**: Confirms long-term trend direction
4. **5-hour prediction**: Allows trades to develop without being too long
5. **75% confidence**: Filters weak signals while allowing enough opportunities

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
# Run main analysis
python main.py

# Test multi-timeframe functionality
python test_multi_timeframe.py

# Run core analysis (without training)
python core.py
```

## 📁 Project Structure

```
forex_ai_project/
├── config.py                 # Configuration and API keys
├── main.py                   # Main execution script
├── core.py                   # Core analysis functions
├── requirements.txt          # Python dependencies
├── test_multi_timeframe.py   # Multi-timeframe testing
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

## 📈 Performance Monitoring

The system logs comprehensive information:
- Feature engineering progress
- Model training metrics
- Signal generation details
- Confluence factor analysis
- Multi-timeframe validation

Check `logs/` directory for detailed analysis.

## ⚠️ Risk Disclaimer

This is an AI-powered trading system for educational and research purposes. Trading forex and cryptocurrencies involves substantial risk of loss. Always:
- Test thoroughly on demo accounts
- Start with small position sizes
- Never risk more than you can afford to lose
- Consider consulting with financial advisors

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional technical indicators
- More news data sources
- Enhanced risk management
- Backtesting framework
- Real-time data streaming

## 📄 License

MIT License - see LICENSE file for details.
