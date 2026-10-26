# AlphaX Signal Desk — v0.1

A phone-friendly, signal-only SMC/price-action review prototype.

## What it does
- Lets you manually enter a proposed Forex, gold, or crypto setup.
- Checks basic price ordering, direction vs. bias, a five-item SMC checklist,
  planned risk percentage, and minimum 1:2 reward-to-risk.
- Records approve/reject decisions for paper tracking.
- Exports the in-session journal to CSV.

## What it does NOT do
- No MT5 or Bybit connection.
- No live market data.
- No automated SMC detection.
- No broker orders, exchange orders, or API credentials.
- "Approve" means approve for paper tracking only.

## Deploy from Android with GitHub + Streamlit Community Cloud
1. Create a GitHub account and a new repository named `alphax-signal-desk`.
2. Add `app.py`, `requirements.txt`, and this `README.md` to the repository.
3. Open https://share.streamlit.io/ and connect GitHub.
4. Create an app from your repository; select `app.py` as the main file.
5. Open the app URL on your Android phone.
6. Test only with manually verified chart values. Download the CSV journal before leaving the session.

## Strategy defaults in this prototype
- Bias: 1H and 15M.
- Entry timeframe: 5M.
- Checklist: liquidity sweep, displacement, BOS/CHoCH, FVG/order block, acceptable conditions.
- Risk guideline: 0.5% by default, with the prototype blocking values above 1%.
- Minimum reward-to-risk: 1:2.

These are initial filters for testing, not proof of a profitable edge. Tune them only after reviewing documented examples and backtests.

## Next safe development steps
1. Add persistent storage and login protection.
2. Add public Bybit OHLCV candle data for analysis only.
3. Add chart-data import for MT5 from a CSV exported by a desktop terminal.
4. Write explicit and testable definitions for swing points, liquidity sweeps, displacement,
   BOS/CHoCH, FVG, and order blocks.
5. Add paper-trade outcome tracking and backtests.
6. Only consider demo execution after separate safety review. Live trading is out of scope.
