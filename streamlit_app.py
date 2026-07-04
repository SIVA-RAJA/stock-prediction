"""

USAGE:

         streamlit run streamlit_app.py

"""


import sys
from pathlib import Path
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LSTM_DIR = ROOT / "LSTM"

for _p in (str(ROOT), str(DATA_DIR), str(LSTM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.config import TICKERS, COMPANY_NAMES, CONFIGS
from LSTM.predict import predict


st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="centered",
    )
st.title("Stock Price Prediction")
st.caption("This app predicts the next day's closing price of a stock using an LSTM model.")

col1, col2 = st.columns(2)
with col1:
    market = st.selectbox("Select Market", list(TICKERS.keys()))
with col2:
    region = st.selectbox("Select Region", list(TICKERS[market].keys()))

tickers_in_region = TICKERS[market][region]
ticker = st.selectbox("Select Ticker", tickers_in_region, format_func=lambda t: f"{t} - {COMPANY_NAMES.get(t, '')}" if COMPANY_NAMES.get(t) else t)
interval = st.selectbox("Select Interval", list(CONFIGS.keys()), index=list(CONFIGS.keys()).index("1d"))

run = st.button("Predict", type="primary", use_container_width=True)

if run:
    with st.spinner(f"Fetching latest data and running inference for {ticker} @ {interval}..."):
        try:
            result = predict(ticker=ticker, market=market, region=region, interval=interval)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
    st.success(f"Prediction as of {result['as_of']}")

    m1, m2, m3 = st.columns(3)

    m1.metric(label="Last Close Price", value=f"${result['last_close']:.2f}")
    m2.metric(label="Predicted Close Price", value=f"${result['predicted_next_close']:.2f}", delta=f"{result['predicted_change_pct']:.2f}%")
    m3.metric(label="Direction", value=result['direction'], delta=f"{result['direction_confidence']:.1%} confidence")
    
    st.subheader("Attention over input window")
    attn = result["attention_weights"]
    attn_df = pd.DataFrame({"timestep (0=oldest, -1=most recent)": list(range(len(attn))), "attention_weight": attn}).set_index("timestep (0=oldest, -1=most recent)")
    st.bar_chart(attn_df)

    with st.expander("Raw results"):
        st.json(result)

st.divider()
st.caption(
    "This tool re-run the exact trainning features pipeline (clean -> features -> saved scaler)"
    "on freshly downloaded data from Yahoo Finance, and then feeds the last window into the exported ONNX model."
    "This app is for educational purposes only."
    "It is not financial advice. Use at your own risk.")
