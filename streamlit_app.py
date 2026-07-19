"""
USAGE:
    streamlit run streamlit_app.py

This version calls your deployed Django API instead of loading the ONNX
model directly, so Streamlit Cloud stays lightweight and there's a single
source of truth for predictions.

Configure the API URL via Streamlit secrets (recommended for deployment)
or an environment variable:

  .streamlit/secrets.toml
  ----------------------
  API_BASE_URL = "https://your-django-app.onrender.com"

or locally:
  export API_BASE_URL=http://127.0.0.1:8000
"""

import os
import streamlit as st
import pandas as pd
import requests

try:
    _secret_url = st.secrets.get("API_BASE_URL")
except Exception:
    # No secrets.toml (or it's empty/malformed) — fine for local dev,
    # just fall back to the environment variable / default below.
    _secret_url = None

API_BASE_URL = _secret_url or os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="centered",
)
st.title("Stock Direction Predictor")
st.caption("This app predicts the next day's price direction (UP/DOWN) using an LSTM model.")


@st.cache_data(ttl=3600)
def fetch_meta():
    resp = requests.get(f"{API_BASE_URL}/api/meta/", timeout=15)
    resp.raise_for_status()
    return resp.json()


try:
    meta = fetch_meta()
except requests.RequestException as e:
    st.error(f"Couldn't reach the prediction API at {API_BASE_URL}: {e}")
    st.stop()

tickers = meta["tickers"]          # expected shape: {market: {region: [tickers]}}
intervals = meta["intervals"]
company_names = meta.get("company_names", {})

col1, col2 = st.columns(2)
with col1:
    market = st.selectbox("Select Market", list(tickers.keys()))
with col2:
    region = st.selectbox("Select Region", list(tickers[market].keys()))

tickers_in_region = tickers[market][region]
ticker = st.selectbox(
    "Select Ticker",
    tickers_in_region,
    format_func=lambda t: f"{t} - {company_names.get(t, '')}" if company_names.get(t) else t,
)
interval = st.selectbox("Select Interval", intervals, index=intervals.index("1d") if "1d" in intervals else 0)

run = st.button("Predict", type="primary", use_container_width=True)

if run:
    resp: requests.Response | None = None
    with st.spinner(f"Fetching latest data and running inference for {ticker} @ {interval}..."):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/api/predict/",
                json={"ticker": ticker, "market": market, "region": region, "interval": interval},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.HTTPError as e:
            detail = str(e)
            if resp is not None and resp.content:
                try:
                    detail = resp.json().get("error", detail)
                except requests.exceptions.JSONDecodeError:
                    # Server sent back something that isn't JSON at all
                    # (e.g. a raw Django/HTML error page) — show a preview
                    # of it instead of crashing.
                    detail = f"{detail} — server said: {resp.text[:300]!r}"
            st.error(f"Prediction failed: {detail}")
            st.stop()
        except requests.RequestException as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    direction = result["direction"]
    confidence = result["direction_confidence"]

    color = "#16c784" if direction == "UP" else "#ea3943"   # green / red
    arrow = "▲" if direction == "UP" else "▼"

    m1, m2 = st.columns(2)
    m1.metric(label="Last Close Price", value=f"{result['last_close']:.2f}")
    m2.markdown(f"""
                    <div style="font-size: 0.9rem; color: gray;">Direction</div>
                    <div style="font-size: 2rem; font-weight: 600; color: {color};">
                    {arrow} {direction}
                    </div>
                """, unsafe_allow_html=True)

    st.caption(f"Model confidence: {confidence:.1%}")

    st.subheader("Attention over input window")
    attn = result["attention_weights"]
    attn_df = pd.DataFrame(
        {"timestep (0=oldest, -1=most recent)": list(range(len(attn))), "attention_weight": attn}
    ).set_index("timestep (0=oldest, -1=most recent)")
    st.bar_chart(attn_df)

    with st.expander("Raw results"):
        st.json(result)

st.divider()
st.caption(
    "This tool re-runs the exact training features pipeline (clean -> features -> saved scaler) "
    "on freshly downloaded data from Yahoo Finance, and feeds the last window into the exported ONNX model "
    "to predict next-period price direction. "
    "This app is for educational purposes only. It is not financial advice. Use at your own risk.")
