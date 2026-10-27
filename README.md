# AlphaX Signal Desk v0.3

A Streamlit-based trading analysis dashboard for crypto, Forex, and gold.

## Features

- Public Bybit Spot candle data for BTCUSDT, ETHUSDT, and SOLUSDT
- 1H and 15M EMA trend-bias checks
- Preliminary 5M liquidity-sweep, displacement, and structure-break checks
- Manual trade setup validation
- Maximum risk of 1% per setup, with 0.5% as the preferred risk setting
- Minimum risk-to-reward ratio of 1:2
- Session-only paper-trading journal with CSV export
- CSV candle analysis for Forex and gold
- Optional automatic refresh

## Run locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

## Important limitations

- This application does not place live trades.
- The automated setup checks are simplified heuristics, not complete or independently validated Smart Money Concepts detection.
- Bybit data is public Spot market data; it does not automatically provide Forex or gold prices.
- The paper journal is held in the active app session and is not permanent storage.
- Always verify prices, position sizing, and risk independently.

## Version

Development version: 0.3.0
