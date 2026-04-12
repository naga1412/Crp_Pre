from fastapi import FastAPI, Query, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import asyncio
import json
import time
import xml.etree.ElementTree as ET
from typing import Optional
from datetime import datetime, timezone, timedelta

import websockets as ws_lib
import yfinance as yf

from ta_logic import (
    process_ohlcv_data, process_yfinance_data,
    generate_trade_setup, analyze_single_timeframe, generate_mtf_prediction,
)

app = FastAPI(title="SMC Pro — Crypto & Stock TA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

VALID_TIMEFRAMES = {
    "1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"
}

# Full top-down HTF map: covers all common lower timeframes
HTF_MAP = {
    '1m':  ['5m',  '15m', '1h'],
    '3m':  ['15m', '1h'],
    '5m':  ['15m', '1h', '4h'],
    '15m': ['1h',  '4h', '1d'],
    '30m': ['1h',  '4h', '1d'],
    '1h':  ['4h',  '1d', '1w'],
    '2h':  ['4h',  '1d', '1w'],
    '4h':  ['1d',  '1w'],
    '1d':  ['1w'],
    '1w':  [],
}

# yfinance interval + period per app timeframe
YF_TF_MAP = {
    '1m':  ('1m',  '7d'),
    '3m':  ('5m',  '60d'),
    '5m':  ('5m',  '60d'),
    '15m': ('15m', '60d'),
    '30m': ('30m', '60d'),
    '1h':  ('1h',  '730d'),
    '2h':  ('1h',  '730d'),
    '4h':  ('1h',  '730d'),   # resample
    '6h':  ('1h',  '730d'),
    '8h':  ('1d',  '5y'),
    '12h': ('1d',  '5y'),
    '1d':  ('1d',  '5y'),
    '1w':  ('1wk', 'max'),
}

# Popular stocks & ETFs for the search bar
POPULAR_STOCKS = [
    # ── US Mega-Cap Tech ──────────────────────────────────────────────
    {"symbol":"AAPL",  "name":"Apple Inc.",             "exchange":"NASDAQ","type":"stock"},
    {"symbol":"MSFT",  "name":"Microsoft Corp.",         "exchange":"NASDAQ","type":"stock"},
    {"symbol":"NVDA",  "name":"NVIDIA Corp.",            "exchange":"NASDAQ","type":"stock"},
    {"symbol":"GOOGL", "name":"Alphabet Inc. (Class A)", "exchange":"NASDAQ","type":"stock"},
    {"symbol":"GOOG",  "name":"Alphabet Inc. (Class C)", "exchange":"NASDAQ","type":"stock"},
    {"symbol":"AMZN",  "name":"Amazon.com Inc.",         "exchange":"NASDAQ","type":"stock"},
    {"symbol":"META",  "name":"Meta Platforms",          "exchange":"NASDAQ","type":"stock"},
    {"symbol":"TSLA",  "name":"Tesla Inc.",              "exchange":"NASDAQ","type":"stock"},
    {"symbol":"AMD",   "name":"Advanced Micro Devices",  "exchange":"NASDAQ","type":"stock"},
    {"symbol":"INTC",  "name":"Intel Corp.",             "exchange":"NASDAQ","type":"stock"},
    {"symbol":"AVGO",  "name":"Broadcom Inc.",           "exchange":"NASDAQ","type":"stock"},
    {"symbol":"ORCL",  "name":"Oracle Corp.",            "exchange":"NYSE",  "type":"stock"},
    {"symbol":"NFLX",  "name":"Netflix Inc.",            "exchange":"NASDAQ","type":"stock"},
    {"symbol":"ADBE",  "name":"Adobe Inc.",              "exchange":"NASDAQ","type":"stock"},
    {"symbol":"CRM",   "name":"Salesforce Inc.",         "exchange":"NYSE",  "type":"stock"},
    {"symbol":"QCOM",  "name":"Qualcomm Inc.",           "exchange":"NASDAQ","type":"stock"},
    {"symbol":"ARM",   "name":"Arm Holdings",            "exchange":"NASDAQ","type":"stock"},
    {"symbol":"PLTR",  "name":"Palantir Technologies",   "exchange":"NASDAQ","type":"stock"},
    {"symbol":"COIN",  "name":"Coinbase Global",         "exchange":"NASDAQ","type":"stock"},
    {"symbol":"MSTR",  "name":"MicroStrategy",           "exchange":"NASDAQ","type":"stock"},
    {"symbol":"HOOD",  "name":"Robinhood Markets",       "exchange":"NASDAQ","type":"stock"},
    {"symbol":"SHOP",  "name":"Shopify Inc.",            "exchange":"NYSE",  "type":"stock"},
    {"symbol":"ASML",  "name":"ASML Holding",            "exchange":"NASDAQ","type":"stock"},
    {"symbol":"TSM",   "name":"Taiwan Semiconductor",    "exchange":"NYSE",  "type":"stock"},
    {"symbol":"BABA",  "name":"Alibaba Group",           "exchange":"NYSE",  "type":"stock"},
    {"symbol":"NVO",   "name":"Novo Nordisk",            "exchange":"NYSE",  "type":"stock"},
    {"symbol":"UBER",  "name":"Uber Technologies",       "exchange":"NYSE",  "type":"stock"},
    {"symbol":"LYFT",  "name":"Lyft Inc.",               "exchange":"NASDAQ","type":"stock"},
    {"symbol":"SNAP",  "name":"Snap Inc.",               "exchange":"NYSE",  "type":"stock"},
    {"symbol":"PINS",  "name":"Pinterest Inc.",          "exchange":"NYSE",  "type":"stock"},
    {"symbol":"RBLX",  "name":"Roblox Corp.",            "exchange":"NYSE",  "type":"stock"},
    {"symbol":"DKNG",  "name":"DraftKings Inc.",         "exchange":"NASDAQ","type":"stock"},
    {"symbol":"SOFI",  "name":"SoFi Technologies",       "exchange":"NASDAQ","type":"stock"},
    {"symbol":"SMCI",  "name":"Super Micro Computer",    "exchange":"NASDAQ","type":"stock"},
    {"symbol":"APP",   "name":"AppLovin Corp.",          "exchange":"NASDAQ","type":"stock"},
    {"symbol":"DELL",  "name":"Dell Technologies",       "exchange":"NYSE",  "type":"stock"},
    {"symbol":"HIMS",  "name":"Hims & Hers Health",      "exchange":"NYSE",  "type":"stock"},
    {"symbol":"CELH",  "name":"Celsius Holdings",        "exchange":"NASDAQ","type":"stock"},
    {"symbol":"MARA",  "name":"Marathon Digital Holdings","exchange":"NASDAQ","type":"stock"},
    {"symbol":"RIOT",  "name":"Riot Platforms",          "exchange":"NASDAQ","type":"stock"},
    {"symbol":"CLSK",  "name":"CleanSpark Inc.",         "exchange":"NASDAQ","type":"stock"},
    # ── Finance ──────────────────────────────────────────────────────
    {"symbol":"JPM",   "name":"JPMorgan Chase",          "exchange":"NYSE",  "type":"stock"},
    {"symbol":"BAC",   "name":"Bank of America",         "exchange":"NYSE",  "type":"stock"},
    {"symbol":"GS",    "name":"Goldman Sachs",           "exchange":"NYSE",  "type":"stock"},
    {"symbol":"MS",    "name":"Morgan Stanley",          "exchange":"NYSE",  "type":"stock"},
    {"symbol":"V",     "name":"Visa Inc.",               "exchange":"NYSE",  "type":"stock"},
    {"symbol":"MA",    "name":"Mastercard Inc.",         "exchange":"NYSE",  "type":"stock"},
    {"symbol":"WFC",   "name":"Wells Fargo",             "exchange":"NYSE",  "type":"stock"},
    {"symbol":"C",     "name":"Citigroup Inc.",          "exchange":"NYSE",  "type":"stock"},
    {"symbol":"AXP",   "name":"American Express",        "exchange":"NYSE",  "type":"stock"},
    {"symbol":"BLK",   "name":"BlackRock Inc.",          "exchange":"NYSE",  "type":"stock"},
    # ── Healthcare ───────────────────────────────────────────────────
    {"symbol":"JNJ",   "name":"Johnson & Johnson",       "exchange":"NYSE",  "type":"stock"},
    {"symbol":"PFE",   "name":"Pfizer Inc.",             "exchange":"NYSE",  "type":"stock"},
    {"symbol":"MRNA",  "name":"Moderna Inc.",            "exchange":"NASDAQ","type":"stock"},
    {"symbol":"UNH",   "name":"UnitedHealth Group",      "exchange":"NYSE",  "type":"stock"},
    {"symbol":"ABBV",  "name":"AbbVie Inc.",             "exchange":"NYSE",  "type":"stock"},
    {"symbol":"LLY",   "name":"Eli Lilly & Co.",         "exchange":"NYSE",  "type":"stock"},
    # ── Energy ───────────────────────────────────────────────────────
    {"symbol":"XOM",   "name":"ExxonMobil",              "exchange":"NYSE",  "type":"stock"},
    {"symbol":"CVX",   "name":"Chevron Corp.",           "exchange":"NYSE",  "type":"stock"},
    {"symbol":"OXY",   "name":"Occidental Petroleum",    "exchange":"NYSE",  "type":"stock"},
    # ── Consumer ─────────────────────────────────────────────────────
    {"symbol":"WMT",   "name":"Walmart Inc.",            "exchange":"NYSE",  "type":"stock"},
    {"symbol":"HD",    "name":"Home Depot",              "exchange":"NYSE",  "type":"stock"},
    {"symbol":"MCD",   "name":"McDonald's Corp.",        "exchange":"NYSE",  "type":"stock"},
    {"symbol":"SBUX",  "name":"Starbucks Corp.",         "exchange":"NASDAQ","type":"stock"},
    {"symbol":"NKE",   "name":"Nike Inc.",               "exchange":"NYSE",  "type":"stock"},
    {"symbol":"KO",    "name":"Coca-Cola Co.",           "exchange":"NYSE",  "type":"stock"},
    {"symbol":"PEP",   "name":"PepsiCo Inc.",            "exchange":"NASDAQ","type":"stock"},
    {"symbol":"COST",  "name":"Costco Wholesale",        "exchange":"NASDAQ","type":"stock"},
    # ── Industrials ──────────────────────────────────────────────────
    {"symbol":"BA",    "name":"Boeing Co.",              "exchange":"NYSE",  "type":"stock"},
    {"symbol":"GE",    "name":"GE Aerospace",            "exchange":"NYSE",  "type":"stock"},
    {"symbol":"CAT",   "name":"Caterpillar Inc.",        "exchange":"NYSE",  "type":"stock"},
    {"symbol":"HON",   "name":"Honeywell Intl.",         "exchange":"NASDAQ","type":"stock"},
    {"symbol":"RTX",   "name":"RTX Corp. (Raytheon)",    "exchange":"NYSE",  "type":"stock"},
    # ── ETFs ─────────────────────────────────────────────────────────
    {"symbol":"SPY",   "name":"SPDR S&P 500 ETF",        "exchange":"NYSE",  "type":"etf"},
    {"symbol":"QQQ",   "name":"Invesco QQQ ETF",          "exchange":"NASDAQ","type":"etf"},
    {"symbol":"IWM",   "name":"iShares Russell 2000",     "exchange":"NYSE",  "type":"etf"},
    {"symbol":"GLD",   "name":"SPDR Gold ETF",            "exchange":"NYSE",  "type":"etf"},
    {"symbol":"SLV",   "name":"iShares Silver ETF",       "exchange":"NYSE",  "type":"etf"},
    {"symbol":"USO",   "name":"United States Oil Fund",   "exchange":"NYSE",  "type":"etf"},
    {"symbol":"GDX",   "name":"VanEck Gold Miners ETF",   "exchange":"NYSE",  "type":"etf"},
    {"symbol":"GDXJ",  "name":"VanEck Junior Gold Miners","exchange":"NYSE",  "type":"etf"},
    {"symbol":"VTI",   "name":"Vanguard Total Market",    "exchange":"NYSE",  "type":"etf"},
    {"symbol":"VOO",   "name":"Vanguard S&P 500 ETF",     "exchange":"NYSE",  "type":"etf"},
    {"symbol":"TLT",   "name":"iShares 20Y Treasury",     "exchange":"NASDAQ","type":"etf"},
    {"symbol":"DIA",   "name":"SPDR Dow Jones ETF",       "exchange":"NYSE",  "type":"etf"},
    {"symbol":"XLF",   "name":"Financial Select Sector",  "exchange":"NYSE",  "type":"etf"},
    {"symbol":"XLK",   "name":"Technology Select Sector", "exchange":"NYSE",  "type":"etf"},
    {"symbol":"XLE",   "name":"Energy Select Sector",     "exchange":"NYSE",  "type":"etf"},
    {"symbol":"XLV",   "name":"Health Care Select ETF",   "exchange":"NYSE",  "type":"etf"},
    {"symbol":"XLI",   "name":"Industrial Select Sector", "exchange":"NYSE",  "type":"etf"},
    {"symbol":"ARKK",  "name":"ARK Innovation ETF",       "exchange":"NYSE",  "type":"etf"},
    {"symbol":"ARKW",  "name":"ARK Next Gen Internet ETF","exchange":"NASDAQ","type":"etf"},
    {"symbol":"HYG",   "name":"iShares HY Corporate Bond","exchange":"NYSE",  "type":"etf"},
    {"symbol":"EEM",   "name":"iShares MSCI Emerg Mkts",  "exchange":"NYSE",  "type":"etf"},
    {"symbol":"VEA",   "name":"Vanguard FTSE Dev Markets", "exchange":"NYSE", "type":"etf"},
    # ── Commodities (via ETFs / futures tickers) ─────────────────────
    {"symbol":"GC=F",  "name":"Gold Futures",             "exchange":"COMEX","type":"commodity"},
    {"symbol":"SI=F",  "name":"Silver Futures",           "exchange":"COMEX","type":"commodity"},
    {"symbol":"CL=F",  "name":"Crude Oil Futures (WTI)",  "exchange":"NYMEX","type":"commodity"},
    {"symbol":"NG=F",  "name":"Natural Gas Futures",      "exchange":"NYMEX","type":"commodity"},
    {"symbol":"ZW=F",  "name":"Wheat Futures",            "exchange":"CBOT", "type":"commodity"},
    {"symbol":"ZC=F",  "name":"Corn Futures",             "exchange":"CBOT", "type":"commodity"},
    # ── FX / Indices ─────────────────────────────────────────────────
    {"symbol":"^GSPC", "name":"S&P 500 Index",            "exchange":"INDEX","type":"index"},
    {"symbol":"^IXIC", "name":"NASDAQ Composite",         "exchange":"INDEX","type":"index"},
    {"symbol":"^DJI",  "name":"Dow Jones Industrial",     "exchange":"INDEX","type":"index"},
    {"symbol":"^VIX",  "name":"CBOE Volatility Index",    "exchange":"INDEX","type":"index"},
    {"symbol":"DX-Y.NYB","name":"US Dollar Index",        "exchange":"ICE",  "type":"index"},
]

# Hardcoded crypto pairs (shown instantly + Binance cache fallback)
POPULAR_CRYPTO = [
    {"symbol":"BTC/USDT",   "name":"Bitcoin",          "exchange":"Binance","type":"crypto"},
    {"symbol":"ETH/USDT",   "name":"Ethereum",         "exchange":"Binance","type":"crypto"},
    {"symbol":"BNB/USDT",   "name":"BNB",              "exchange":"Binance","type":"crypto"},
    {"symbol":"SOL/USDT",   "name":"Solana",           "exchange":"Binance","type":"crypto"},
    {"symbol":"XRP/USDT",   "name":"Ripple (XRP)",     "exchange":"Binance","type":"crypto"},
    {"symbol":"DOGE/USDT",  "name":"Dogecoin",         "exchange":"Binance","type":"crypto"},
    {"symbol":"ADA/USDT",   "name":"Cardano",          "exchange":"Binance","type":"crypto"},
    {"symbol":"AVAX/USDT",  "name":"Avalanche",        "exchange":"Binance","type":"crypto"},
    {"symbol":"DOT/USDT",   "name":"Polkadot",         "exchange":"Binance","type":"crypto"},
    {"symbol":"LINK/USDT",  "name":"Chainlink",        "exchange":"Binance","type":"crypto"},
    {"symbol":"MATIC/USDT", "name":"Polygon (MATIC)",  "exchange":"Binance","type":"crypto"},
    {"symbol":"UNI/USDT",   "name":"Uniswap",          "exchange":"Binance","type":"crypto"},
    {"symbol":"LTC/USDT",   "name":"Litecoin",         "exchange":"Binance","type":"crypto"},
    {"symbol":"BCH/USDT",   "name":"Bitcoin Cash",     "exchange":"Binance","type":"crypto"},
    {"symbol":"ATOM/USDT",  "name":"Cosmos (ATOM)",    "exchange":"Binance","type":"crypto"},
    {"symbol":"FIL/USDT",   "name":"Filecoin",         "exchange":"Binance","type":"crypto"},
    {"symbol":"NEAR/USDT",  "name":"NEAR Protocol",    "exchange":"Binance","type":"crypto"},
    {"symbol":"ARB/USDT",   "name":"Arbitrum",         "exchange":"Binance","type":"crypto"},
    {"symbol":"OP/USDT",    "name":"Optimism",         "exchange":"Binance","type":"crypto"},
    {"symbol":"INJ/USDT",   "name":"Injective",        "exchange":"Binance","type":"crypto"},
    {"symbol":"SUI/USDT",   "name":"Sui Network",      "exchange":"Binance","type":"crypto"},
    {"symbol":"APT/USDT",   "name":"Aptos",            "exchange":"Binance","type":"crypto"},
    {"symbol":"TRX/USDT",   "name":"TRON",             "exchange":"Binance","type":"crypto"},
    {"symbol":"TON/USDT",   "name":"Toncoin",          "exchange":"Binance","type":"crypto"},
    {"symbol":"SHIB/USDT",  "name":"Shiba Inu",        "exchange":"Binance","type":"crypto"},
    {"symbol":"PEPE/USDT",  "name":"Pepe",             "exchange":"Binance","type":"crypto"},
    {"symbol":"WLD/USDT",   "name":"Worldcoin",        "exchange":"Binance","type":"crypto"},
    {"symbol":"JTO/USDT",   "name":"Jito",             "exchange":"Binance","type":"crypto"},
    {"symbol":"PYTH/USDT",  "name":"Pyth Network",     "exchange":"Binance","type":"crypto"},
    {"symbol":"JUP/USDT",   "name":"Jupiter",          "exchange":"Binance","type":"crypto"},
    {"symbol":"WIF/USDT",   "name":"dogwifhat",        "exchange":"Binance","type":"crypto"},
    {"symbol":"BONK/USDT",  "name":"Bonk",             "exchange":"Binance","type":"crypto"},
    {"symbol":"SEI/USDT",   "name":"Sei Network",      "exchange":"Binance","type":"crypto"},
    {"symbol":"RENDER/USDT","name":"Render Network",   "exchange":"Binance","type":"crypto"},
    {"symbol":"FET/USDT",   "name":"Fetch.ai",         "exchange":"Binance","type":"crypto"},
    {"symbol":"RNDR/USDT",  "name":"Render Token",     "exchange":"Binance","type":"crypto"},
    {"symbol":"GRT/USDT",   "name":"The Graph",        "exchange":"Binance","type":"crypto"},
    {"symbol":"SAND/USDT",  "name":"The Sandbox",      "exchange":"Binance","type":"crypto"},
    {"symbol":"MANA/USDT",  "name":"Decentraland",     "exchange":"Binance","type":"crypto"},
    {"symbol":"AXS/USDT",   "name":"Axie Infinity",    "exchange":"Binance","type":"crypto"},
    {"symbol":"IMX/USDT",   "name":"Immutable X",      "exchange":"Binance","type":"crypto"},
    {"symbol":"LDO/USDT",   "name":"Lido DAO",         "exchange":"Binance","type":"crypto"},
    {"symbol":"MKR/USDT",   "name":"Maker",            "exchange":"Binance","type":"crypto"},
    {"symbol":"AAVE/USDT",  "name":"Aave",             "exchange":"Binance","type":"crypto"},
    {"symbol":"COMP/USDT",  "name":"Compound",         "exchange":"Binance","type":"crypto"},
    {"symbol":"CRV/USDT",   "name":"Curve DAO",        "exchange":"Binance","type":"crypto"},
    {"symbol":"SUSHI/USDT", "name":"SushiSwap",        "exchange":"Binance","type":"crypto"},
    {"symbol":"1INCH/USDT", "name":"1inch Network",    "exchange":"Binance","type":"crypto"},
    {"symbol":"BLUR/USDT",  "name":"Blur",             "exchange":"Binance","type":"crypto"},
    {"symbol":"ETHFI/USDT", "name":"Ether.fi",         "exchange":"Binance","type":"crypto"},
    {"symbol":"ENA/USDT",   "name":"Ethena",           "exchange":"Binance","type":"crypto"},
    # BTC cross-pairs
    {"symbol":"ETH/BTC",    "name":"Ethereum / BTC",   "exchange":"Binance","type":"crypto"},
    {"symbol":"BNB/BTC",    "name":"BNB / BTC",        "exchange":"Binance","type":"crypto"},
    {"symbol":"SOL/BTC",    "name":"Solana / BTC",     "exchange":"Binance","type":"crypto"},
]

# Cached crypto symbols from Binance
_cached_crypto: list = []

# ─────────────────────────────────────────────
#  NEWS CACHE & SCORING
# ─────────────────────────────────────────────

_news_cache: dict = {}        # key → (timestamp_float, response_dict)
_global_news_cache: tuple = (0.0, [])   # (ts, items) — world market news
NEWS_CACHE_TTL        = 60    # 1 min per-symbol cache (near real-time)
GLOBAL_NEWS_CACHE_TTL = 60    # 1 min global news cache (near real-time)

# ── Keyword taxonomy ──────────────────────────────────────────────────────────
_KW_BULLISH = [
    'rate cut','fed cut','rate cuts','pivot','dovish','stimulus','easing','qe',
    'soft landing','beats expectations','record high','rally','surge','soars',
    'upgrade','etf approved','approves bitcoin','deal reached','ceasefire',
    'peace talks','trade deal','strong jobs','gdp growth','recovery',
]
_KW_BEARISH = [
    'rate hike','rate increase','rate hikes','hawkish','tightening','inflation',
    'recession','stagflation','war','attack','conflict','invasion','airstrike',
    'missile','sanctions','tariff','trade war','ban','crackdown','default',
    'banking crisis','collapse','crash','plunge','misses','downgrade',
    'layoffs','job cuts','geopolitical','hack','exploit','rug pull','sec charges',
    'lawsuit','debt ceiling','shutdown','nuclear',
]

# ── Category detection keywords ───────────────────────────────────────────────
_CATEGORIES = {
    'FED / RATES':    ['fed','federal reserve','fomc','powell','rate cut','rate hike',
                       'interest rate','basis points','monetary policy','cpi','inflation',
                       'pce','treasury yield','10-year','bond yield'],
    'WAR / GEO':      ['war','attack','conflict','invasion','airstrike','missile','troops',
                       'ukraine','russia','israel','gaza','iran','north korea','taiwan',
                       'china military','nuclear','nato','geopolit','sanction'],
    'CRYPTO REG':     ['sec','cftc','bitcoin etf','crypto regulation','crypto ban',
                       'blockchain','coinbase','binance charged','crypto law','cbdc',
                       'digital asset','crypto tax','senate crypto','congress crypto'],
    'EARNINGS':       ['earnings','revenue','profit','eps','quarterly','beats','misses',
                       'guidance','outlook','results','q1','q2','q3','q4'],
    'MACRO / ECONOMY':['gdp','unemployment','jobs report','nonfarm','retail sales',
                       'consumer sentiment','manufacturing','pmi','recession','debt',
                       'deficit','spending bill','budget','imf','world bank'],
    'CRYPTO MARKET':  ['bitcoin','btc','ethereum','eth','solana','sol','crypto market',
                       'altcoin','defi','nft','halving','blockchain','cryptocurrency'],
    'STOCK MARKET':   ['s&p 500','nasdaq','dow jones','stock market','wall street',
                       'market rally','market crash','ipo','merger','acquisition'],
}

# Free RSS feeds — all public, no API key required
_RSS_FEEDS = [
    # Yahoo Finance — global markets
    {'url': 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US',
     'source': 'Yahoo Finance'},
    {'url': 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US',
     'source': 'Yahoo Finance (Crypto)'},
    # CNBC RSS feeds
    {'url': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114',
     'source': 'CNBC Markets'},
    {'url': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664',
     'source': 'CNBC World Economy'},
    {'url': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362',
     'source': 'CNBC Crypto'},
    # MarketWatch
    {'url': 'https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines',
     'source': 'MarketWatch'},
    # CoinDesk
    {'url': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
     'source': 'CoinDesk'},
    # Reuters via Google News RSS (no auth needed)
    {'url': 'https://news.google.com/rss/search?q=federal+reserve+interest+rate&hl=en-US&gl=US&ceid=US:en',
     'source': 'Google News (Fed)'},
    {'url': 'https://news.google.com/rss/search?q=war+conflict+geopolitical+market&hl=en-US&gl=US&ceid=US:en',
     'source': 'Google News (Geo)'},
    {'url': 'https://news.google.com/rss/search?q=bitcoin+ethereum+crypto+market&hl=en-US&gl=US&ceid=US:en',
     'source': 'Google News (Crypto)'},
    {'url': 'https://news.google.com/rss/search?q=stock+market+economy+recession&hl=en-US&gl=US&ceid=US:en',
     'source': 'Google News (Markets)'},
]

COIN_NAMES = {
    'btc':'bitcoin','eth':'ethereum','bnb':'binance','sol':'solana',
    'xrp':'xrp ripple','doge':'dogecoin','ada':'cardano','avax':'avalanche',
    'dot':'polkadot','link':'chainlink','matic':'polygon','shib':'shiba inu',
    'uni':'uniswap','ltc':'litecoin','atom':'cosmos','near':'near protocol',
    'arb':'arbitrum','op':'optimism','sui':'sui network','apt':'aptos',
    'inj':'injective','ton':'toncoin','pepe':'pepe coin',
}


def _fetch_rss(url: str, timeout: int = 8) -> list:
    """Fetch RSS feed → list of {title, link, pubDate}."""
    try:
        r = requests.get(url, timeout=timeout,
                         headers={'User-Agent': 'Mozilla/5.0 (compatible; SMCPro/1.0)'})
        if r.status_code != 200:
            return []
        # Strip illegal XML chars that break ET parser
        content = r.content.replace(b'\x00', b'')
        root = ET.fromstring(content)
        items = []
        for item in root.iter('item'):
            title = (item.findtext('title') or '').strip()
            link  = (item.findtext('link')  or '').strip()
            pub   = (item.findtext('pubDate') or '').strip()
            if title:
                items.append({'title': title, 'link': link, 'pubDate': pub})
        return items[:25]
    except Exception:
        return []


def _detect_category(title: str) -> str:
    t = title.lower()
    for cat, keywords in _CATEGORIES.items():
        if any(kw in t for kw in keywords):
            return cat
    return 'MARKET NEWS'


def _score_sentiment(title: str) -> tuple:
    """Return (sentiment, impact_pct, impact_label)."""
    t = title.lower()
    bull = sum(1 for kw in _KW_BULLISH if kw in t)
    bear = sum(1 for kw in _KW_BEARISH if kw in t)

    # Boost for high-impact keywords
    HIGH_IMPACT = ['war','invasion','attack','rate cut','rate hike','recession',
                   'crash','collapse','record high','fomc','ceasefire','ban','default']
    boost = 2 if any(kw in t for kw in HIGH_IMPACT) else 1

    if bull > bear:
        pct = round(min(1.0 + bull * 0.7 * boost, 5.0), 1)
        lbl = 'HIGH' if pct >= 3.0 else 'MEDIUM' if pct >= 1.5 else 'LOW'
        return 'bullish', pct, lbl
    if bear > bull:
        pct = round(min(1.0 + bear * 0.7 * boost, 5.0), 1)
        lbl = 'HIGH' if pct >= 3.0 else 'MEDIUM' if pct >= 1.5 else 'LOW'
        return 'bearish', pct, lbl
    return 'neutral', 0.3, 'LOW'


def _parse_pub_date(pub: str) -> str:
    """Parse RSS pubDate → ISO-8601 UTC string."""
    for fmt in ('%a, %d %b %Y %H:%M:%S %z', '%a, %d %b %Y %H:%M:%S GMT',
                '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S%z'):
        try:
            dt = datetime.strptime(pub.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            continue
    return datetime.now(timezone.utc).isoformat()


def _fetch_global_news() -> list:
    """
    Fetch from all RSS feeds in parallel threads and return deduplicated list.
    Results cached for GLOBAL_NEWS_CACHE_TTL seconds.
    """
    global _global_news_cache
    ts, cached = _global_news_cache
    if time.time() - ts < GLOBAL_NEWS_CACHE_TTL:
        return cached

    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_items = []

    def _fetch_one(feed):
        items = _fetch_rss(feed['url'])
        out = []
        for it in items:
            sentiment, impact_pct, impact_lbl = _score_sentiment(it['title'])
            out.append({
                'headline':     it['title'],
                'url':          it['link'],
                'source':       feed['source'],
                'timestamp':    _parse_pub_date(it['pubDate']),
                'sentiment':    sentiment,
                'impact_pct':   impact_pct,
                'impact_label': impact_lbl,
                'category':     _detect_category(it['title']),
            })
        return out

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_fetch_one, f): f for f in _RSS_FEEDS}
        for fut in as_completed(futures, timeout=12):
            try:
                all_items.extend(fut.result())
            except Exception:
                pass

    # Deduplicate by first 80 chars of headline
    seen, deduped = set(), []
    for item in sorted(all_items, key=lambda x: x['timestamp'], reverse=True):
        key = item['headline'][:80].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    _global_news_cache = (time.time(), deduped[:60])
    return deduped[:60]


# ─────────────────────────────────────────────
#  SYMBOL DETECTION
# ─────────────────────────────────────────────

def is_crypto(symbol: str) -> bool:
    return '/' in symbol


# ─────────────────────────────────────────────
#  BINANCE HELPERS
# ─────────────────────────────────────────────

def fetch_binance_symbols() -> list:
    global _cached_crypto
    if _cached_crypto:
        return _cached_crypto
    resp = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        syms = [
            s['symbol'].replace('USDT', '/USDT')
            for s in data['symbols']
            if s['symbol'].endswith('USDT') and s['status'] == 'TRADING'
        ]
        _cached_crypto = syms
        return syms
    return [c['symbol'] for c in POPULAR_CRYPTO]


def fetch_binance_ohlcv(symbol: str, timeframe: str, limit: int) -> list:
    sym  = symbol.replace('/', '')
    url  = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval={timeframe}&limit={limit}"
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise Exception(f"Binance API error: {resp.text[:200]}")
    return [[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
            for r in resp.json()]


# ─────────────────────────────────────────────
#  YFINANCE HELPERS
# ─────────────────────────────────────────────

def fetch_stock_ohlcv(symbol: str, timeframe: str, limit: int):
    interval, period = YF_TF_MAP.get(timeframe, ('1d', '5y'))
    ticker = yf.Ticker(symbol)
    hist   = ticker.history(period=period, interval=interval, auto_adjust=True)
    if hist.empty:
        raise Exception(f"No data for '{symbol}'. Check the ticker symbol.")

    # Resample for multi-hour timeframes
    if timeframe == '4h':
        hist = hist.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
    elif timeframe == '2h':
        hist = hist.resample('2h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
    elif timeframe == '6h':
        hist = hist.resample('6h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()

    return hist.iloc[-limit:]


# ─────────────────────────────────────────────
#  SHARED MTF ANALYSIS
# ─────────────────────────────────────────────

def run_mtf_analysis(symbol: str, primary_tf: str, primary_df: pd.DataFrame):
    htf_tfs  = HTF_MAP.get(primary_tf, ['4h', '1d', '1w'])
    analyses  = [analyze_single_timeframe(primary_df.copy(), primary_tf)]
    warnings  = []

    for htf in htf_tfs:
        try:
            if is_crypto(symbol):
                ohlcv  = fetch_binance_ohlcv(symbol, htf, 200)
                htf_df = process_ohlcv_data(ohlcv)
            else:
                hist   = fetch_stock_ohlcv(symbol, htf, 200)
                htf_df = process_yfinance_data(hist)
            analyses.append(analyze_single_timeframe(htf_df.copy(), htf))
        except Exception as e:
            warnings.append(f"{htf}: {str(e)[:60]}")

    mtf = generate_mtf_prediction(analyses, primary_tf)
    if warnings:
        mtf['htf_warnings'] = warnings
    return mtf


# ─────────────────────────────────────────────
#  REST ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "SMC Pro TA API Running"}


@app.get("/api/market/symbols")
def get_symbols():
    """Returns all crypto symbols (Binance) + popular stocks/ETFs/commodities."""
    try:
        crypto_live = fetch_binance_symbols()
        crypto_set  = set(crypto_live)
        # Build meta: live Binance symbols first
        crypto_meta = [{"symbol": s,
                        "name": s.replace('/USDT','').replace('/BTC','').replace('/ETH',''),
                        "exchange": "Binance", "type": "crypto"} for s in crypto_live]
        # Add hardcoded crypto not already in live list
        extra_crypto = [c for c in POPULAR_CRYPTO if c['symbol'] not in crypto_set]
        all_meta = crypto_meta + extra_crypto + POPULAR_STOCKS
        return {"symbols": crypto_live, "meta": all_meta}
    except Exception:
        # Offline fallback
        all_meta = POPULAR_CRYPTO + POPULAR_STOCKS
        return {"symbols": [c['symbol'] for c in POPULAR_CRYPTO], "meta": all_meta}


@app.get("/api/market/data")
def get_market_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = Query(default=500, ge=10, le=1000),
):
    if timeframe not in VALID_TIMEFRAMES:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe '{timeframe}'")
    try:
        if is_crypto(symbol):
            ohlcv = fetch_binance_ohlcv(symbol, timeframe, limit)
            df    = process_ohlcv_data(ohlcv)
        else:
            hist  = fetch_stock_ohlcv(symbol, timeframe, limit)
            df    = process_yfinance_data(hist)

        df_out  = df.replace({np.nan: None})
        records = df_out.to_dict(orient='records')
        # Emit timestamps as UTC ISO strings
        for r in records:
            if r.get('timestamp') is not None:
                ts = r['timestamp']
                r['timestamp'] = (ts.isoformat()+'Z') if hasattr(ts, 'isoformat') else str(ts)+'Z'
        return {"symbol": symbol, "timeframe": timeframe, "data": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis/trade-setup")
def get_trade_setup(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = Query(default=200, ge=10, le=1000),
):
    if timeframe not in VALID_TIMEFRAMES:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe '{timeframe}'")
    try:
        if is_crypto(symbol):
            ohlcv = fetch_binance_ohlcv(symbol, timeframe, limit)
            df    = process_ohlcv_data(ohlcv)
        else:
            hist  = fetch_stock_ohlcv(symbol, timeframe, limit)
            df    = process_yfinance_data(hist)

        current_price = float(df.iloc[-1]['close'])
        mtf           = run_mtf_analysis(symbol, timeframe, df)
        setup         = generate_trade_setup(df, current_price,
                                             mtf_prediction=mtf,
                                             tf_label=timeframe)
        setup['mtf_prediction'] = mtf

        return {"symbol": symbol, "current_price": current_price, "setup": setup}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news")
def get_news(symbol: str = "BTC/USDT"):
    """
    Fetch world market news from 10+ free RSS sources (CNBC, Yahoo Finance,
    MarketWatch, CoinDesk, Google News) grouped by category.
    Symbol-specific headlines boosted to top.
    Cached 5 min per-symbol, 10 min global.
    """
    cache_key = symbol.lower().replace('/', '_').replace('=', '_').replace('^', '')
    now = time.time()
    if cache_key in _news_cache:
        cached_ts, cached_data = _news_cache[cache_key]
        if now - cached_ts < NEWS_CACHE_TTL:
            return cached_data

    # Fetch all global news (parallel, cached separately)
    all_news = _fetch_global_news()

    # ── Symbol-specific news boost ────────────────────────────────────
    # Build search terms for the current symbol
    if is_crypto(symbol):
        coin      = symbol.split('/')[0].lower()
        sym_terms = {coin, COIN_NAMES.get(coin, coin), coin.upper()}
    else:
        sym_terms = {symbol.upper(), symbol.split('=')[0].split('^')[-1].upper()}

    def _is_symbol_related(headline: str) -> bool:
        h = headline.lower()
        return any(t.lower() in h for t in sym_terms if len(t) > 1)

    symbol_news = [n for n in all_news if _is_symbol_related(n['headline'])]
    other_news  = [n for n in all_news if not _is_symbol_related(n['headline'])]

    # Tag symbol-specific items
    for n in symbol_news:
        n = dict(n)  # copy

    # Merge: symbol first, then rest (already sorted by timestamp desc)
    combined = symbol_news + other_news

    # ── Build category summary ────────────────────────────────────────
    cat_summary = {}
    for item in combined:
        cat = item['category']
        if cat not in cat_summary:
            cat_summary[cat] = {'bullish': 0, 'bearish': 0, 'neutral': 0, 'count': 0}
        cat_summary[cat][item['sentiment']] += 1
        cat_summary[cat]['count'] += 1

    # Overall market sentiment
    total_bull = sum(1 for n in combined if n['sentiment'] == 'bullish')
    total_bear = sum(1 for n in combined if n['sentiment'] == 'bearish')
    high_impact = [n for n in combined if n.get('impact_label') == 'HIGH']

    response = {
        "symbol":           symbol,
        "fetched_at":       datetime.now(timezone.utc).isoformat(),
        "news":             combined[:40],
        "symbol_news":      symbol_news[:8],
        "high_impact":      high_impact[:5],
        "category_summary": cat_summary,
        "overall_sentiment": {
            "bullish": total_bull,
            "bearish": total_bear,
            "neutral": len(combined) - total_bull - total_bear,
            "label":   "Risk-ON" if total_bull > total_bear else
                       "Risk-OFF" if total_bear > total_bull else "Neutral",
        },
    }
    _news_cache[cache_key] = (now, response)
    return response


# ─────────────────────────────────────────────
#  WEBSOCKET — REAL-TIME PRICE STREAM
# ─────────────────────────────────────────────

@app.websocket("/ws/{symbol:path}")
async def websocket_stream(websocket: WebSocket, symbol: str, timeframe: str = "1h"):
    await websocket.accept()
    try:
        if is_crypto(symbol):
            await _stream_crypto(websocket, symbol, timeframe)
        else:
            await _stream_stock(websocket, symbol, timeframe)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _stream_crypto(websocket: WebSocket, symbol: str, timeframe: str):
    """Proxy Binance kline WebSocket stream to client."""
    sym      = symbol.replace('/', '').lower()
    interval = timeframe if timeframe in VALID_TIMEFRAMES else '1h'
    url      = f"wss://stream.binance.com:9443/ws/{sym}@kline_{interval}"

    async with ws_lib.connect(url) as binance_ws:
        while True:
            try:
                raw = await asyncio.wait_for(binance_ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue

            msg   = json.loads(raw)
            kline = msg.get('k', {})
            await websocket.send_json({
                "type":      "price_update",
                "symbol":    symbol,
                "price":     float(kline.get('c', 0)),
                "open":      float(kline.get('o', 0)),
                "high":      float(kline.get('h', 0)),
                "low":       float(kline.get('l', 0)),
                "close":     float(kline.get('c', 0)),
                "volume":    float(kline.get('v', 0)),
                "timestamp": int(kline.get('t', 0)),   # ms UTC
                "is_closed": bool(kline.get('x', False)),
            })


async def _stream_stock(websocket: WebSocket, symbol: str, timeframe: str):
    """Poll yfinance every 5 s and push latest price (delayed ~15 min for free tier)."""
    while True:
        try:
            ticker = yf.Ticker(symbol)
            hist   = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                latest = hist.iloc[-1]
                # Ensure UTC ms timestamp
                if hasattr(latest.name, 'timestamp'):
                    ts_ms = int(latest.name.timestamp() * 1000)
                else:
                    ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                await websocket.send_json({
                    "type":      "price_update",
                    "symbol":    symbol,
                    "price":     float(latest['Close']),
                    "open":      float(latest['Open']),
                    "high":      float(latest['High']),
                    "low":       float(latest['Low']),
                    "close":     float(latest['Close']),
                    "volume":    float(latest['Volume']),
                    "timestamp": ts_ms,
                    "is_closed": False,
                    "delayed":   True,
                })
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
        await asyncio.sleep(5)
