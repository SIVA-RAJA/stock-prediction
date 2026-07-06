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
st.title("Stock Direction Predictor")
st.caption("This app predicts the next day's price direction (UP/DOWN) using an LSTM model.")

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

    direction = result['direction']
    confidence = result['direction_confidence']

    color = "#16c784" if direction == "UP" else "#ea3943"   # green / red
    arrow = "▲" if direction == "UP" else "▼"


    m1, m2 = st.columns(2)

    m1.metric(label="Last Close Price", value=f"${result['last_close']:.2f}")
    m2.markdown(f"""
                    <div style="font-size: 0.9rem; color: gray;">Direction</div>
                    <div style="font-size: 2rem; font-weight: 600; color: {color};">
                    {arrow} {direction}
                    </div>
                """, unsafe_allow_html=True)

    st.caption(f"Model confidence: {confidence:.1%}")

    st.subheader("Attention over input window")
    attn = result["attention_weights"]
    attn_df = pd.DataFrame({"timestep (0=oldest, -1=most recent)": list(range(len(attn))), "attention_weight": attn}).set_index("timestep (0=oldest, -1=most recent)")
    st.bar_chart(attn_df)

    with st.expander("Raw results"):
        st.json(result)

st.divider()
st.caption(
    "This tool re-runs the exact training features pipeline (clean -> features -> saved scaler) "
    "on freshly downloaded data from Yahoo Finance, and feeds the last window into the exported ONNX model "
    "to predict next-period price direction. "
    "This app is for educational purposes only. It is not financial advice. Use at your own risk.")
