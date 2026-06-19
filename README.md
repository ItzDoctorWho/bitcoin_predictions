# Bitcoin Price Forecasting & Strategy Deployment

This project builds, compares, and deploys multiple machine learning and statistical models to forecast daily Bitcoin closing prices. It incorporates leakage-aware feature engineering, chronological evaluation splits, and standardizes comparisons to **Price Space (USD)**. 

Additionally, the project features a premium **Streamlit web application** dashboard that fetches real-time prices via the Binance API, trains forecasting models on-the-fly, and backtests a trading strategy net of transaction fees.

---

## 📁 Repository Structure

```text
├── data/
│   └── btc_historical.csv         # Local cache of historical Bitcoin daily data
├── notebook/
│   ├── Bitcoin_Price_Forecasting_main.ipynb  # Clean, annotated final project notebook
│   └── drafts/                    # Directory containing all experimental notebooks & drafts
├── requirements.txt               # Dependencies list for Streamlit app
├── streamlit_app.py               # Main Streamlit web application code
└── README.md                      # Professional project summary & documentation
```

---

## 📊 Methodology & Comparative Analysis

The project implements a strict, leakage-aware pipeline to ensure realistic time-series evaluations:
- **Chronological Split:** Train-test split is date-bounded (Jan 1, 2024) to avoid look-ahead bias inherent in random shuffling.
- **Leakage Prevention:** Rolling indicators, lags, volatility, and momentum indicators are strictly shifted by 1 day so that no future closing prices leak into today's feature set.
- **Price Space Standardization:** Models are compared in absolute USD prices rather than returns to ensure a fair evaluation against the Naive (`tomorrow = today`) baseline.

### Model Performance Comparison

| Model | MAE (USD) | RMSE (USD) | MAPE (%) | R² Score | Directional Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | **$1,152.02** | **$1,672.20** | **1.71%** | **0.992** | **82.20%** |
| **Linear Regression**| $1,154.50 | $1,675.00 | $1.73% | 0.991 | 76.80% |
| **XGBoost** | $1,280.40 | $1,842.10 | 1.90% | 0.988 | 63.50% |
| **Random Forest** | $1,340.20 | $1,910.40 | 1.95% | 0.986 | 50.90% |
| **Naive (t-1) Baseline**| $1,180.50 | $1,712.10 | 1.76% | 0.990 | 0.00% (Degenerate) |

*Note: The naive model's 0% directional accuracy is a structural artifact of always predicting no change (`PctNonZeroSignal = 0%`). When compared to a fair coin flip (~50%), Ridge Regression demonstrates a genuine predictive edge.*

---

## 🪙 Streamlit App & Deployment 

The Streamlit app is fully responsive, styled with a custom **Bitcoin Gold Dark Theme**, and optimized for zero-configuration cloud deployment:
1. **API Integration:** Connects to the **Binance API** (`BTCUSDT`) for real-time daily data downloads that bypass cloud IP blocks common with Yahoo Finance.
2. **On-the-Fly Training:** Fits Linear, Ridge, or Random Forest models dynamically based on user configurations in the sidebar.
3. **Trading Backtest:** Simulates a long-only trading strategy driven by model direction, net of a **0.1% transaction fee**, and plots it against a Buy & Hold baseline.
4. **Tomorrow's Forecast:** Displays a live card predicting tomorrow's price level and market direction (Bullish 📈 / Bearish 📉).

### How to Run Locally

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```
