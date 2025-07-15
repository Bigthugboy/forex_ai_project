# AI-Powered Forex & Crypto Signal Generator

This project provides real-time, AI-driven trading signals for USD/JPY and BTC/USD pairs, blending technical and fundamental analysis for actionable insights.

## Features
- Real-time market and news data analysis
- Technical indicators (SMA, EMA, RSI, MACD, etc.)
- News sentiment analysis
- 1-hour timeframe signal generation
- Trade type, confidence score, SL/TP output
- Configurable risk/reward and model parameters

## Project Structure
```
forex_ai_project/
├── config.py
├── requirements.txt
├── main.py
├── data/
│   ├── fetch_market.py
│   ├── fetch_news.py
│   └── preprocess.py
├── models/
│   ├── train_model.py
│   ├── predict.py
│   └── saved_models/
├── signals/
│   ├── signal_generator.py
│   └── risk_management.py
├── utils/
│   ├── logger.py
│   └── helpers.py
├── logs/
└── README.md
```

## Usage
1. Install dependencies in a Python 3.10 virtual environment:
   ```sh
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configure API keys and parameters in `config.py`.
3. Run the main script:
   ```sh
   python main.py
   ```

## Customization
- Add more trading pairs or indicators in `config.py`.
- Extend data sources in `data/` modules.
- Train new models in `models/train_model.py`.

## License
MIT
