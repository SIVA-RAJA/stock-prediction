from pathlib import Path

def _detect_project_root() -> Path:

    try:
        import google.colab             # noqa: F401
        in_colab = True
    except ImportError:
        in_colab = False

    if in_colab:
        colab_path = Path("/content/drive/MyDrive/Stock Predictor")
        if Path("/content/drive/MyDrive").exists():
            return colab_path

        return colab_path

    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _detect_project_root()
BASE_DIR = PROJECT_ROOT
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "dataset"
PARQUET_PATH = DATA_DIR / "market_data_full.parquet"
HDF5_PATH = DATA_DIR / "market_data.hd5"
SCALER_DIR = DATA_DIR / "scalers"

for d in [DATA_DIR, SCALER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "1m":  ("7d",   "1m"),
    "2m":  ("60d",  "2m"),
    "5m":  ("60d",  "5m"),
    "15m": ("60d",  "15m"),
    "30m": ("60d",  "30m"),
    "90m": ("60d",  "90m"),
    "1h":  ("730d", "1h"),
    "1d":  ("max",  "1d"),
    "5d": ("max",  "5d"),
    "1wk": ("max",  "1wk"),
    "1mo": ("max",  "1mo"),
    "3mo": ("max",  "3mo"),
}


TICKERS = {
    "STOCK" : {
        "USA": ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","BRK-B","JPM"],
        "INDIA": ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","BHARTIARTL.NS","SBIN.NS","LT.NS","HINDUNILVR.NS","SUNPHARMA.NS"],
        "JAPAN": ["7203.T","6758.T","8306.T","6501.T","7974.T","9984.T","6861.T","9983.T","8035.T","7267.T"],
        "CHINA": ["601398.SS","601318.SS","600519.SS","600036.SS","601857.SS","002594.SZ","300750.SZ","601088.SS","000333.SZ","000858.SZ"],
        "SAUDI ARABIA" :["2222.SR", "1120.SR", "1211.SR", "1180.SR", "7010.SR", "2010.SR", "2082.SR", "1010.SR", "1150.SR", "1060.SR"],
        "UK": ["SHEL.L","AZN.L","ULVR.L","HSBA.L","BP.L","RIO.L","GSK.L","REL.L","DGE.L","NG.L"],
        "GERMANY": ["BMW.DE","SAP.DE","SIE.DE","ALV.DE","MBG.DE","DTE.DE","BAS.DE","MUV2.DE","VOW3.DE","ADS.DE"],
        "FRANCE": ["AIR.PA","MC.PA","OR.PA","TTE.PA","SAN.PA","BNP.PA","CAP.PA","CS.PA","SU.PA","ENGI.PA"],
        "CANADA": ["SHOP.TO","RY.TO","TD.TO","ENB.TO","BNS.TO","CNQ.TO","CP.TO","TRI.TO","BMO.TO","SU.TO"],
        "AUSTRALIA": ["CBA.AX","BHP.AX","CSL.AX","WBC.AX","NAB.AX","ANZ.AX","MQG.AX","WES.AX","GMG.AX","RIO.AX"],
        "KOREA": ["005930.KS","000660.KS","035420.KS","005380.KS","035720.KS","051910.KS","006400.KS","028260.KS","012330.KS","068270.KS"],
        "SINGAPORE": ["D05.SI","AJBU.SI","U11.SI","Z74.SI","C6L.SI","S68.SI","BN4.SI","F34.SI","G13.SI","U96.SI"],
        "BRAZIL": ["PETR4.SA","VALE3.SA","ITUB4.SA","BBDC4.SA","ABEV3.SA","BBAS3.SA","WEGE3.SA","B3SA3.SA","RENT3.SA","SUZB3.SA"],
    },
    "CRYPTO" : {
        "CRYPTO": ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","DOGE-USD","TRX-USD","AVAX-USD","LINK-USD"],
    },
    "FOREX" : {
        "FOREX": ["EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X","USDCAD=X","USDCHF=X","NZDUSD=X","USDINR=X","USDCNY=X","USDHKD=X"],
    },
    "COMMODITIES" : {
        "COMMODITIES": ["GC=F","SI=F","CL=F","NG=F","HG=F","PL=F","PA=F","ZC=F","ZW=F","ZS=F"],
    },
    "INDICES" : {
        "INDICES": ["^GSPC","^IXIC","^DJI","^RUT","^NSEI","^BSESN","^FTSE","^GDAXI","^FCHI","^HSI"],
    }
}


COMPANY_NAMES = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA Corporation", "AMZN": "Amazon.com Inc.", "GOOGL": "Alphabet Inc.", "META": "Meta Platforms Inc.", "TSLA": "Tesla Inc.", "AVGO": "Broadcom Inc.", "BRK-B": "Berkshire Hathaway Inc.", "JPM": "JPMorgan Chase & Co.",
    "RELIANCE.NS": "Reliance Industries Limited", "TCS.NS": "Tata Consultancy Services Limited", "HDFCBANK.NS": "HDFC Bank Limited", "INFY.NS": "Infosys Limited", "ICICIBANK.NS": "ICICI Bank Limited", "BHARTIARTL.NS": "Bharti Airtel Limited", "SBIN.NS": "State Bank of India", "LT.NS": "Larsen & Toubro Limited", "HINDUNILVR.NS": "Hindustan Unilever Limited", "SUNPHARMA.NS": "Sun Pharmaceutical Industries Limited",
    "7203.T": "Toyota Motor Corporation", "6758.T": "Sony Group Corporation", "8306.T": "Mitsubishi UFJ Financial Group, Inc.", "6501.T": "Hitachi, Ltd.", "7974.T": "Nintendo Co., Ltd.", "9984.T": "SoftBank Group Corp.", "6861.T": "Keyence Corporation", "9983.T": "Fast Retailing Co., Ltd.", "8035.T": "Tokyo Electron Limited", "7267.T": "Honda Motor Co., Ltd.",
    "601398.SS" : "Industrial and Commercial Bank of China Limited", "601318.SS": "Ping An Insurance Group Company of China, Ltd.", "600519.SS": "Kweichow Moutai Co., Ltd.", "600036.SS": "China Merchants Co., Ltd.", "601857.SS": "PetroChina Company Limited", "002594.SZ": "BYD Company Limited", "300750.SZ": "Contemporary Amperex Technology Co., Limited", "601088.SS": "China Shenhua Energy Company Limited", "000333.SZ": "Midea Group Co., Ltd.", "000858.SZ": "Wuliangye Yibin Co., Ltd.",
    "2222.SR": "Saudi Aramco", "1120.SR": "Al Rajhi Banking and Investment Corporation", "1211.SR": "Saudi Arabian Mining Company (Maaden)", "1180.SR": "The Saudi National Bank", "7010.SR": "Saudi Telecom Company", "2010.SR": "Saudi Basic Industries Corporation", "2082.SR": "ACWA Power Company", "1010.SR": "Riyad Bank", "1150.SR": "Alinma Bank", "1060.SR": "Saudi Awwal Bank ",
    "SHEL.L": "Shell plc", "AZN.L": "AstraZeneca plc", "ULVR.L": "Unilever plc", "HSBA.L": "HSBC Holdings plc", "BP.L": "BP p.l.c.", "RIO.L": "Rio Tinto Group", "GSK.L": "GSK plc", "REL.L": "RELX plc", "DGE.L": "Diageo plc", "NG.L": "National Grid plc",
    "BMW.DE": "Bayerische Motoren Werke Aktiengesellschaft", "SAP.DE": "SAP SE", "SIE.DE": "Siemens Aktiengesellschaft", "ALV.DE": "Allianz SE", "MBG.DE": "Mercedes-Benz Group AG", "DTE.DE": "Deutsche Telekom AG", "BAS.DE": "BASF SE", "MUV2.DE": "Münchener Rückversicherungs-Gesellschaft Aktiengesellschaft in München", "VOW3.DE": "Volkswagen AG", "ADS.DE": "Adidas AG",
    "AIR.PA": "Airbus SE", "MC.PA": "LVMH Moët Hennessy – Louis Vuitton, Société Européenne", "OR.PA": "L'Oréal S.A.", "TTE.PA": "TotalEnergies", "SAN.PA": "Sanofi", "BNP.PA": "BNP Paribas SA", "CAP.PA": "Capgemini SE", "CS.PA": "AXA SA", "SU.PA": "Schneider Electric S.E.", "ENGI.PA": "Engie SA",
    "SHOP.TO": "Shopify Inc.", "RY.TO": "Royal Bank of Canada", "TD.TO": "The Toronto-Dominion Bank", "ENB.TO": "Enbridge Inc.", "BNS.TO": "The Bank of Nova Scotia", "CNQ.TO": "Canadian Natural Resources Limited", "CP.TO": "Canadian Pacific Kansas City Limited", "TRI.TO": "Thomson Reuters Corporation", "BMO.TO": "Bank of Montreal", "SU.TO": "Suncor Energy Inc.",
    "CBA.AX": "Commonwealth Bank of Australia", "BHP.AX": "BHP Group Limited", "CSL.AX": "CSL Limited", "WBC.AX": "Westpac Banking Corporation", "NAB.AX": "National Australia Bank Limited", "ANZ.AX": "ANZ Group Holdings Limited", "MQG.AX": "Macquarie Group Limited", "WES.AX": "Wesfarmers Limited", "GMG.AX": "Goodman Group", "RIO.AX": "Rio Tinto Group",
    "005930.KS": "Samsung Electronics C0., Ltd.", "000660.KS": "SK hynix Inc.", "035420.KS": "NAVER Corporation", "005380.KS": "Hyundai Motor Company", "035720.KS": "Kakao Corp.", "051910.KS": "LG Chem Ltd,", "006400.KS": "Samsung SDI Co., Ltd.", "028260.KS": "Samsung C&T Corporation", "012330.KS": "Hyundai Mobis Co., Ltd.", "068270.KS": "Celltrion, Inc.",
    "D05.SI": "DBS Group Holdings Limited", "AJBU.SI": "Keppel DC REIT","U11.SI": "United Overseas Bank Limited", "Z74.SI": "Singapore Telecommunications Limited", "C6L.SI": "Singapore Airlines Limited", "S68.SI": "Singapore Exchange Limited", "BN4.SI": "Keppel Ltd.", "F34.SI": "Wilmar International Limited", "G13.SI": "Genting Singapore Limited", "U96.SI": "Sembcorp Industries Ltd.",
    "PETR4.SA": "Petroleo Brasileiro S.A. - Petrobras", "VALE3.SA": "Vale S.A.", "ITUB4.SA": "Itaú Unibanco Holding S.A.", "BBDC4.SA": "Banco Bradesco S.A.", "ABEV3.SA": "Ambev S.A.", "BBAS3.SA": "Banco do Brasil S.A.", "WEGE3.SA": "WEG S.A.", "B3SA3.SA": "B3 S.A. - Brasil, Bolsa, Balcão", "RENT3.SA": "Localiza Rent a Car S.A.", "SUZB3.SA": "Suzano S.A.",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "BNB-USD": "Binance Coin", "SOL-USD": "Solana", "XRP-USD": "Ripple", "ADA-USD": "Cardano", "DOGE-USD": "Dogecoin", "TRX-USD": "TRON", "AVAX-USD": "Avalanche", "LINK-USD": "Chainlink",
    "EURUSD=X": "EUR/USD", "USDJPY=X": "USD/JPY", "GBPUSD=X": "GBP/USD", "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD", "USDCHF=X": "USD/CHF", "NZDUSD=X": "NZD/USD", "USDINR=X": "USD/INR", "USDCNY=X": "USD/CNY", "USDHKD=X": "USD/HKD",
    "GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil", "NG=F": "Natural Gas", "HG=F": "Copper", "PL=F": "Platinum", "PA=F": "Palladium", "ZC=F": "Corn", "ZW=F": "Wheat", "ZS=F": "Soybeans",
    "^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "DOW JONES", "^RUT": "RUSSELL 2000", "^NSEI": "NIFTY 50", "^BSESN": "BSE SENSEX", "^FTSE": "FTSE 100", "^GDAXI": "DAX", "^FCHI": "CAC 40", "^HSI": "HANG SENG",
}


def build_embeddings_maps():
    ticker_to_id , market_to_id, region_to_id = {}, {}, {}
    interval_to_id = {iv: i for i, iv in enumerate(CONFIGS.keys(), start=1)}
    t_idx = m_idx = r_idx = 1

    for market, regions in TICKERS.items():
        if market not in market_to_id:
            market_to_id[market] = m_idx
            m_idx +=1
        for region, tickers in regions.items():
            if region not in region_to_id:
                region_to_id[region] = r_idx
                r_idx += 1
            for tk in tickers:
                if tk not in ticker_to_id:
                    ticker_to_id[tk] = t_idx
                    t_idx +=1

    return ticker_to_id, market_to_id, region_to_id, interval_to_id

TICKER_TO_ID, MARKET_TO_ID, REGION_TO_ID, INTERVAL_TO_ID = build_embeddings_maps()

ID_TO_TICKER = {v: k for k, v in TICKER_TO_ID.items()}
ID_TO_INTERVAL = {v: k for k, v in INTERVAL_TO_ID.items()}


MIN_ROWS = 30

SMA_WINDOWS = [10, 20, 50, 200]
EMA_WINDOWS = [9, 12, 21, 26, 50]
LAG_WINDOWS = [1, 2, 3, 5, 10]
ATR_PERIOD = 14
RSI_PERIOD = 14
MFI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
BB_STD = 2
STOCH_WINDOW = 14
STOCH_SMOOTH = 3
WILLIAMS_R_PERIOD = 14
ROC_PERIOD = 10
CMF_PERIOD = 20
CCI_PERIOD = 20
VWAP_PERIOD = 14
MARKET_REGIME_WINDOW = 20
