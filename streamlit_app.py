import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import xgboost; fall back gracefully if not available
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# =====================================================================
# SEO & PAGE SETUP
# =====================================================================
st.set_page_config(
    page_title="Bitcoin Price Forecasting Dashboard",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Gold Premium Theme)
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0d0f12;
        color: #e2e8f0;
    }
    
    /* Headers & Text colors */
    h1, h2, h3 {
        color: #f7931a !important; /* Bitcoin Gold */
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #14171c;
        border-right: 1px solid #2d3748;
    }
    
    /* Cards and Containers styling */
    div[data-testid="stMetricValue"] {
        color: #f7931a !important;
        font-weight: bold;
    }
    
    .metric-card {
        background-color: #1b1f26;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    .gold-border {
        border-left: 5px solid #f7931a;
    }
    
    .green-border {
        border-left: 5px solid #00c805;
    }
    
    .red-border {
        border-left: 5px solid #ff3b30;
    }
</style>
""", unsafe_allow_html=True)

# Title Block
st.title("🪙 Bitcoin Price Forecasting & Strategy Dashboard")
st.markdown("A premium machine learning deployment predicting daily closing prices with leakage-aware feature engineering.")

# =====================================================================
# DATA RETRIEVAL (WITH CACHING)
# =====================================================================
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_bitcoin_data(start_date, end_date):
    # Try fetching from Binance API first (unblocked on Cloud providers like AWS)
    try:
        import time
        # Convert start_date and end_date string to timestamps in milliseconds
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
        
        symbol = "BTCUSDT"
        interval = "1d"
        url = "https://api.binance.com/api/v3/klines"
        
        data = []
        current_start = start_ts
        while True:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "limit": 1000
            }
            res = requests.get(url, params=params, timeout=10).json()
            if not res:
                break
            data.extend(res)
            last_close_time = res[-1][6]
            current_start = last_close_time + 1
            if len(res) < 1000 or last_close_time >= end_ts or last_close_time > int(time.time() * 1000) - 86400000:
                break
            time.sleep(0.05)
            
        if data:
            df = pd.DataFrame(data)
            df = df[[0, 4]].copy()
            df.columns = ['Date', 'close']
            df['Date'] = pd.to_datetime(df['Date'], unit='ms')
            df.set_index('Date', inplace=True)
            df['close'] = df['close'].astype(float)
            # Filter to start_date and end_date
            df = df.loc[start_date:end_date]
            if not df.empty:
                return df
    except Exception as binance_err:
        pass # Fallback to yfinance if Binance fails
        
    # Fallback to yfinance
    ticker = "BTC-USD"
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, with Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    df = yf.download(ticker, start=start_date, end=end_date, session=session, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Close']].copy()
    df.columns = ['close']
    return df



# =====================================================================
# FEATURE ENGINEERING
# =====================================================================
def engineer_features(df):
    feat = df.copy()
    
    # 1. Price-based features (shifted by 1 to prevent leakage)
    feat['returns'] = feat['close'].pct_change().shift(1)
    feat['log_returns'] = np.log(feat['close'] / feat['close'].shift(1)).shift(1)
    
    # 2. Lag features (past values)
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        feat[f'lag_{lag}'] = feat['close'].shift(lag)
        
    # 3. Rolling statistics (shifted by 1 to prevent leakage)
    for window in [7, 14, 21, 30, 60]:
        feat[f'rolling_mean_{window}'] = feat['close'].rolling(window).mean().shift(1)
        feat[f'rolling_std_{window}'] = feat['close'].rolling(window).std().shift(1)
        feat[f'rolling_min_{window}'] = feat['close'].rolling(window).min().shift(1)
        feat[f'rolling_max_{window}'] = feat['close'].rolling(window).max().shift(1)
        
    # 4. Momentum indicators (yesterday's close vs rolling mean)
    for window in [7, 14, 30]:
        feat[f'price_vs_ma_{window}'] = (
            (feat['lag_1'] - feat[f'rolling_mean_{window}']) / feat[f'rolling_mean_{window}']
        )
        
    # 5. Volatility features
    feat['volatility_7'] = feat['returns'].rolling(7).std().shift(1)
    feat['volatility_30'] = feat['returns'].rolling(30).std().shift(1)
    
    # 6. Calendar features
    feat['dayofweek'] = feat.index.dayofweek
    feat['month'] = feat.index.month
    feat['quarter'] = feat.index.quarter
    feat['year'] = feat.index.year
    
    feat['dayofweek_sin'] = np.sin(2 * np.pi * feat['dayofweek'] / 7)
    feat['dayofweek_cos'] = np.cos(2 * np.pi * feat['dayofweek'] / 7)
    feat['month_sin'] = np.sin(2 * np.pi * feat['month'] / 12)
    feat['month_cos'] = np.cos(2 * np.pi * feat['month'] / 12)
    
    feat['days_from_start'] = (feat.index - feat.index[0]).days
    feat['dayofyear'] = feat.index.dayofyear
    
    # Target: next day's return
    feat['target_return'] = feat['close'].pct_change().shift(-1)
    
    return feat

# =====================================================================
# SIDEBAR CONTROLS
# =====================================================================
st.sidebar.header("⚙️ Configuration")

start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now() + timedelta(days=1))
split_date = st.sidebar.date_input("Train/Test Split Date", datetime(2024, 1, 1))

model_options = ["Ridge Regression", "Linear Regression", "Random Forest"]
if XGB_AVAILABLE:
    model_options.append("XGBoost")
selected_model_name = st.sidebar.selectbox("Choose Main Forecasting Model", model_options)

st.sidebar.markdown("""
---
### Model Parameters
""")
if selected_model_name == "Ridge Regression":
    ridge_alpha = st.sidebar.slider("Ridge Alpha (L2 Regularization)", 0.01, 1000.0, 10.0)
elif selected_model_name == "Random Forest":
    rf_estimators = st.sidebar.slider("N Estimators", 50, 300, 100, step=50)
    rf_depth = st.sidebar.slider("Max Depth", 5, 30, 15)
elif selected_model_name == "XGBoost" and XGB_AVAILABLE:
    xgb_lr = st.sidebar.slider("Learning Rate", 0.01, 0.2, 0.05, step=0.01)
    xgb_estimators = st.sidebar.slider("N Estimators", 50, 300, 100, step=50)

# Load data
with st.spinner("Fetching Bitcoin data..."):
    df_raw = load_bitcoin_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

# Validate data presence
if df_raw.empty:
    st.error("❌ No data downloaded from Yahoo Finance. Please check the start and end dates or your internet connection.")
    st.stop()

if len(df_raw) < 60:
    st.error("❌ Insufficient data. Please select a larger date range (at least 60 days of historical data are required for rolling features).")
    st.stop()

# Engineer features
feat_all = engineer_features(df_raw)


# Keep track of latest row (for tomorrow's prediction) before dropping NA targets
latest_row = feat_all.iloc[[-1]].copy()
feat = feat_all.dropna().copy()

FEATURE_COLS = [col for col in feat.columns if col not in ['close', 'target_return']]

# Chronological Train-Test Split
split_date_str = split_date.strftime("%Y-%m-%d")
train_data = feat.loc[feat.index < split_date_str].copy()
test_data = feat.loc[feat.index >= split_date_str].copy()

X_train = train_data[FEATURE_COLS]
y_train = train_data['target_return']
X_test = test_data[FEATURE_COLS]
y_test = test_data['target_return']
y_test_price = test_data['close'].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
latest_features_scaled = scaler.transform(latest_row[FEATURE_COLS])

# =====================================================================
# MODEL FITTING & EVALUATION (Price Space Standardized)
# =====================================================================
def get_model(name):
    if name == "Ridge Regression":
        return Ridge(alpha=ridge_alpha, random_state=42)
    elif name == "Linear Regression":
        return LinearRegression()
    elif name == "Random Forest":
        return RandomForestRegressor(n_estimators=rf_estimators, max_depth=rf_depth, random_state=42, n_jobs=-1)
    elif name == "XGBoost" and XGB_AVAILABLE:
        return xgb.XGBRegressor(learning_rate=xgb_lr, n_estimators=xgb_estimators, random_state=42, n_jobs=-1)
    else:
        return Ridge(alpha=10.0, random_state=42)

# Train all available models to build comparison table
models_to_run = ["Linear Regression", "Ridge Regression", "Random Forest"]
if XGB_AVAILABLE:
    models_to_run.append("XGBoost")

results = []
predictions_dict = {}

# 1. Naive Model Baseline
naive_pred_price = test_data['lag_1'].values  # tomorrow's price = today's price
# In return space, naive prediction is 0. But we evaluate everything in Price Space!

def evaluate_price_space(actual_prices, predicted_prices, name, prev_day_prices):
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    r2 = r2_score(actual_prices, predicted_prices)
    
    # Directional Accuracy
    actual_dir = np.sign(actual_prices - prev_day_prices)
    pred_dir = np.sign(predicted_prices - prev_day_prices)
    dir_acc = np.mean(actual_dir == pred_dir) * 100
    pct_nonzero = np.mean(pred_dir != 0) * 100
    
    return {
        "Model": name,
        "MAE (USD)": mae,
        "RMSE (USD)": rmse,
        "MAPE (%)": mape,
        "R2 Score": r2,
        "Dir. Accuracy (%)": dir_acc,
        "Pct Non-Zero Signal (%)": pct_nonzero
    }

# Evaluate Naive
results.append(evaluate_price_space(y_test_price, naive_pred_price, "Naive (t-1)", test_data['lag_1'].values))
predictions_dict["Naive (t-1)"] = naive_pred_price

# Evaluate ML Models
for name in models_to_run:
    model = get_model(name)
    # Fit in return space
    if name in ["Linear Regression", "Ridge Regression"]:
        model.fit(X_train_scaled, y_train)
        pred_return = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred_return = model.predict(X_test)
        
    # Convert return to price space
    pred_price = test_data['lag_1'].values * (1 + pred_return)
    predictions_dict[name] = pred_price
    
    results.append(evaluate_price_space(
        y_test_price, pred_price, name, test_data['lag_1'].values
    ))

results_df = pd.DataFrame(results).set_index("Model")

# Get chosen model details
selected_model = get_model(selected_model_name)
if selected_model_name in ["Linear Regression", "Ridge Regression"]:
    selected_model.fit(X_train_scaled, y_train)
    selected_pred_return = selected_model.predict(X_test_scaled)
    tomorrow_pred_return = selected_model.predict(latest_features_scaled)[0]
else:
    selected_model.fit(X_train, y_train)
    selected_pred_return = selected_model.predict(X_test)
    tomorrow_pred_return = selected_model.predict(latest_row[FEATURE_COLS])[0]

selected_pred_price = test_data['lag_1'].values * (1 + selected_pred_return)

# Tomorrow's Forecast
today_close = df_raw['close'].iloc[-1]
tomorrow_predicted_price = today_close * (1 + tomorrow_pred_return)
tomorrow_direction = "Bullish 📈" if tomorrow_pred_return > 0 else "Bearish 📉"
direction_color = "#00c805" if tomorrow_pred_return > 0 else "#ff3b30"
border_class = "green-border" if tomorrow_pred_return > 0 else "red-border"

# =====================================================================
# DASHBOARD LAYOUT
# =====================================================================

# Row 1: Key Metrics & Next Day Prediction
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card gold-border">
        <p style="margin: 0; color: #a0aec0; font-size: 14px;">Bitcoin Current Price</p>
        <h2 style="margin: 5px 0 0 0;">${today_close:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    selected_mae = results_df.loc[selected_model_name, "MAE (USD)"]
    st.markdown(f"""
    <div class="metric-card gold-border">
        <p style="margin: 0; color: #a0aec0; font-size: 14px;">{selected_model_name} MAE</p>
        <h2 style="margin: 5px 0 0 0;">${selected_mae:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    selected_acc = results_df.loc[selected_model_name, "Dir. Accuracy (%)"]
    st.markdown(f"""
    <div class="metric-card gold-border">
        <p style="margin: 0; color: #a0aec0; font-size: 14px;">Directional Accuracy</p>
        <h2 style="margin: 5px 0 0 0;">{selected_acc:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card {border_class}">
        <p style="margin: 0; color: #a0aec0; font-size: 14px;">Tomorrow's Forecast ({selected_model_name})</p>
        <h2 style="margin: 5px 0 0 0; color: {direction_color} !important;">${tomorrow_predicted_price:,.2f}</h2>
        <span style="font-size: 14px; font-weight: bold; color: {direction_color};">{tomorrow_direction} ({tomorrow_pred_return*100:+.2f}%)</span>
    </div>
    """, unsafe_allow_html=True)

# Row 2: Charts
tab1, tab2 = st.tabs(["📊 Price Predictions & Backtests", "🏆 Model Performance Comparison"])

with tab1:
    # 1. Price Forecast Chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=test_data.index, y=y_test_price, name="Actual Price", line=dict(color="#ffffff", width=2)))
    fig_price.add_trace(go.Scatter(x=test_data.index, y=selected_pred_price, name=f"{selected_model_name} Forecast", line=dict(color="#f7931a", width=1.5, dash='dash')))
    fig_price.add_trace(go.Scatter(x=test_data.index, y=predictions_dict["Naive (t-1)"], name="Naive Baseline", line=dict(color="#718096", width=1, dash='dot'), opacity=0.7))
    
    fig_price.update_layout(
        title=f"Bitcoin Close Price: Forecast vs Actuals (Test Period)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        paper_bgcolor="#0d0f12",
        plot_bgcolor="#0d0f12",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified"
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # 2. Cumulative Trading Returns Backtest
    st.subheader("📈 Trading Strategy Backtest")
    st.markdown("A simple trading strategy: Go long if the model predicts a positive return for the next day, otherwise stay flat. net of a 0.1% fee per transaction.")
    
    actual_returns = pd.Series(y_test_price, index=test_data.index).pct_change()
    predicted_direction = (pd.Series(selected_pred_price, index=test_data.index) > test_data['lag_1']).astype(int)
    
    strategy_returns = actual_returns * predicted_direction
    FEE = 0.001
    trades = predicted_direction.diff().abs() > 0
    strategy_returns_net = strategy_returns - FEE * trades
    
    cum_strategy = (1 + strategy_returns.fillna(0)).cumprod()
    cum_strategy_net = (1 + strategy_returns_net.fillna(0)).cumprod()
    cum_buy_hold = (1 + actual_returns.fillna(0)).cumprod()
    
    fig_strat = go.Figure()
    fig_strat.add_trace(go.Scatter(x=test_data.index, y=cum_buy_hold, name="Buy & Hold", line=dict(color="#718096", width=2)))
    fig_strat.add_trace(go.Scatter(x=test_data.index, y=cum_strategy, name="Model Strategy (Gross)", line=dict(color="#f7931a", width=1.5, dash='dash')))
    fig_strat.add_trace(go.Scatter(x=test_data.index, y=cum_strategy_net, name="Model Strategy (Net of Fees)", line=dict(color="#00c805", width=2)))
    
    fig_strat.update_layout(
        title="Cumulative Strategy Returns vs Buy & Hold",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (Multiplier)",
        template="plotly_dark",
        paper_bgcolor="#0d0f12",
        plot_bgcolor="#0d0f12",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified"
    )
    st.plotly_chart(fig_strat, use_container_width=True)

with tab2:
    st.subheader("Model Performance Comparison Table")
    st.dataframe(results_df.style.highlight_min(axis=0, subset=["MAE (USD)", "RMSE (USD)", "MAPE (%)"], color="#2a1215")
                                 .highlight_max(axis=0, subset=["R2 Score", "Dir. Accuracy (%)"], color="#122a15"),
                 use_container_width=True)
    
    # Metric comparison charts
    col_a, col_b = st.columns(2)
    with col_a:
        fig_mae = go.Figure(data=[
            go.Bar(name='MAE', x=results_df.index, y=results_df['MAE (USD)'], marker_color='#f7931a')
        ])
        fig_mae.update_layout(title="Mean Absolute Error Comparison (Lower is Better)", template="plotly_dark", paper_bgcolor="#0d0f12", plot_bgcolor="#0d0f12")
        st.plotly_chart(fig_mae, use_container_width=True)
        
    with col_b:
        fig_dir = go.Figure(data=[
            go.Bar(name='Directional Accuracy', x=results_df.index, y=results_df['Dir. Accuracy (%)'], marker_color='#00c805')
        ])
        fig_dir.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50% Coin Flip")
        fig_dir.update_layout(title="Directional Accuracy Comparison (Higher is Better)", template="plotly_dark", paper_bgcolor="#0d0f12", plot_bgcolor="#0d0f12")
        st.plotly_chart(fig_dir, use_container_width=True)

# Footer info & Warning
st.markdown("""
---
**⚠️ Disclaimer:** This application is for educational and forecasting demonstration purposes only. cryptocurrency markets are highly volatile and unpredictable. None of the statistics or predictions outputted by this app should be considered financial or investment advice.
""")
