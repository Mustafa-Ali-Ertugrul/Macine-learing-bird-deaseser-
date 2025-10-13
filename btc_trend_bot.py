# btc_trend_bot.py - Trend Değişimi Stratejisi
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
        logging.FileHandler('trend_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Konfigürasyon
API_KEY = os.getenv("BINANCE_KEY")
API_SECRET = os.getenv("BINANCE_SECRET")
SYMBOL = "BTCUSDT"
BASE_QUANTITY = 0.001
INTERVAL = Client.KLINE_INTERVAL_1HOUR
LIMIT = 100
STOP_LOSS_PERCENT = 0.02
MAX_POSITION_SIZE = 0.1
MIN_USDT_BALANCE = 20
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5

# Testnet kullan
client = Client(API_KEY, API_SECRET, testnet=True)

def get_klines():
    """Binance'den kline verilerini çek"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            candles = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT)
            df = pd.DataFrame(candles, columns=[
                'open_time','open','high','low','close','volume',
                'close_time','qav','trades','taker_base','taker_quote','ignore'
            ])
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['open'] = pd.to_numeric(df['open'])
            df['volume'] = pd.to_numeric(df['volume'])
            return df
        except Exception as e:
            logger.warning(f"Kline verisi alınamadı (deneme {attempt+1}/{RETRY_ATTEMPTS}): {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise e

def rsi(series, period=14):
    """RSI hesapla"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def moving_average(series, period=20):
    """Hareketli ortalama hesapla"""
    return series.rolling(period).mean()

def detect_trend_change(df):
    if len(df) < 3:
        return "NO_DATA"
    
    # Son 3 periyodun trend durumu
    recent_trends = []
    for i in range(-3, 0):
        ma20 = df['ma20'].iloc[i]
        ma50 = df['ma50'].iloc[i]
        if not pd.isna(ma20) and not pd.isna(ma50):
            recent_trends.append(ma20 > ma50)
    
    if len(recent_trends) < 2:
        return "NO_CHANGE"
    
    # Trend değişimi kontrolü
    current_trend = recent_trends[-1]
    previous_trend = recent_trends[-2]
    
    if not previous_trend and current_trend:
        return "BULLISH_REVERSAL"  # Düşüşten yükselişe
    elif previous_trend and not current_trend:
        return "BEARISH_REVERSAL"  # Yükselişten düşüşe
    
    return "NO_CHANGE"

def get_balance(asset='USDT'):
    """Bakiye sorgula"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            bal = client.get_asset_balance(asset=asset)
            return float(bal['free']) if bal else 0.0
        except Exception as e:
            logger.warning(f"Bakiye sorgulanamadı (deneme {attempt+1}/{RETRY_ATTEMPTS}): {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                return 0.0

def calculate_position_size(current_price, usdt_balance):
    """Güvenli pozisyon boyutu hesapla"""
    max_usdt_to_use = usdt_balance * MAX_POSITION_SIZE
    max_btc_quantity = max_usdt_to_use / current_price
    
    if usdt_balance < MIN_USDT_BALANCE:
        logger.warning(f"Yetersiz USDT bakiyesi: {usdt_balance}")
        return 0
    
    return min(BASE_QUANTITY, max_btc_quantity)

def place_order(side: str, qty: float, current_price: float):
    """Emir ver"""
    if qty <= 0:
        logger.warning("Gecersiz miktar, emir verilmedi")
        return None
        
    try:
        min_qty = 0.00001
        if qty < min_qty:
            logger.warning(f"Miktar cok kucuk: {qty} < {min_qty}")
            return None
            
        order = client.order_market(symbol=SYMBOL, side=side, quantity=qty)
        logger.info(f"{side} emri başarılı: {qty} BTC @ ~${current_price}")
        return order
    except Exception as e:
        logger.error(f"Emir hatasi: {e}")
        return None

def save_state(state):
    """Bot durumunu kaydet"""
    try:
        with open("trend_bot_state.json", "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Durum kaydedilemedi: {e}")

def load_state():
    """Bot durumunu yükle"""
    try:
        if os.path.exists("trend_bot_state.json"):
            with open("trend_bot_state.json") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Durum yüklenemedi: {e}")
    
    return {
        "position": None,
        "entry_price": None,
        "last_trade_time": None,
        "total_trades": 0,
        "profitable_trades": 0
    }

def trend_reversal_strategy():
    """Ana trend değişimi stratejisi"""
    try:
        # Teknik verileri al
        df = get_klines()
        df['ma20'] = moving_average(df['close'], 20)
        df['ma50'] = moving_average(df['close'], 50)
        df['rsi'] = rsi(df['close'])
        
        # Trend değişimini tespit et
        trend_change = detect_trend_change(df)
        
        current_price = float(df['close'].iloc[-1])
        current_rsi = float(df['rsi'].iloc[-1])
        current_trend = "UP" if df['ma20'].iloc[-1] > df['ma50'].iloc[-1] else "DOWN"
        
        logger.info(f"Fiyat: ${current_price:.2f} | RSI: {current_rsi:.2f} | Trend: {current_trend} | Değişim: {trend_change}")
        
        # ALIM SİNYALİ - Düşüşten yükselişe geçiş
        if (trend_change == "BULLISH_REVERSAL" and 
            current_rsi < 70 and 
            state["position"] != "LONG"):
            
            usdt_balance = get_balance('USDT')
            quantity = calculate_position_size(current_price, usdt_balance)
            
            if quantity > 0:
                order = place_order(SIDE_BUY, quantity, current_price)
                if order:
                    state["position"] = "LONG"
                    state["entry_price"] = current_price
                    state["last_trade_time"] = datetime.now().isoformat()
                    state["total_trades"] += 1
                    logger.info(f"[ALIM] TREND REVERSAL: ${current_price:.2f} | Miktar: {quantity}")
        
        # SATIM SİNYALİ - Yükselişten düşüşe geçiş
        elif (trend_change == "BEARISH_REVERSAL" and 
              state["position"] == "LONG"):
            
            btc_balance = get_balance('BTC')
            if btc_balance > 0.00001:
                order = place_order(SIDE_SELL, btc_balance, current_price)
                if order:
                    profit = (current_price - state["entry_price"]) / state["entry_price"] * 100
                    if profit > 0:
                        state["profitable_trades"] += 1
                    state["position"] = None
                    state["entry_price"] = None
                    logger.info(f"[SATIM] TREND REVERSAL: ${current_price:.2f} | Kar: {profit:.2f}%")
        
        # İstatistikleri logla
        if state["total_trades"] > 0:
            win_rate = (state["profitable_trades"] / state["total_trades"]) * 100
            logger.info(f"İstatistik: {state['total_trades']} işlem | Başarı: %{win_rate:.1f}")
        
        save_state(state)
        
    except Exception as e:
        logger.error(f"Strateji hatası: {e}")

def check_api_connection():
    """API bağlantısını kontrol et"""
    try:
        client.ping()
        logger.info("[OK] Binance API bağlantısı başarılı")
        return True
    except Exception as e:
        logger.error(f"[ERROR] API bağlantı hatası: {e}")
        return False

def main_loop():
    """Ana bot döngüsü"""
    logger.info("[START] Trend Reversal Bot başlatıldı")
    
    while True:
        try:
            trend_reversal_strategy()
            time.sleep(60 * 1)  # 1 dakika da bir kontrol
            
        except KeyboardInterrupt:
            logger.info("[STOP] Bot durduruldu")
            break
        except Exception as e:
            logger.error(f"Ana döngü hatası: {e}")
            time.sleep(60)

# Ana kod
state = load_state()

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        logger.error("[ERROR] API anahtarlari bulunamadi!")
        exit(1)

    if not check_api_connection():
        logger.error("[ERROR] API bağlantısı kurulamadı!")
        exit(1)

    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("[STOP] Bot güvenli şekilde durduruldu")
    except Exception as e:
        logger.error(f"[CRITICAL] Kritik hata: {e}")
        exit(1)