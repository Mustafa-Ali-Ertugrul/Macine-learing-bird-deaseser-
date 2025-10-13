# eth_trend_bot.py - ETH Trend Değişimi Stratejisi
import os
import sys
import time
import json
import logging
from datetime import datetime
from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np

# Windows Unicode desteği
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# .env dosyası desteği
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Logging konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('eth_trend_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Konfigürasyon
API_KEY = os.getenv("BINANCE_KEY")
API_SECRET = os.getenv("BINANCE_SECRET")
SYMBOL = "ETHUSDT"
BASE_QUANTITY = 0.01  # ETH için uygun miktar
INTERVAL = Client.KLINE_INTERVAL_1HOUR
LIMIT = 100
STOP_LOSS_PERCENT = 0.02
MAX_POSITION_SIZE = 0.1
MIN_USDT_BALANCE = 20
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5

# Testnet kullan
client = Client(API_KEY, API_SECRET, testnet=True)

# BTC botundaki tüm fonksiyonları kopyalayın, sadece SYMBOL ve BASE_QUANTITY değiştirin
# ... (btc_trend_bot.py'deki tüm fonksiyonlar aynı)