# btc_trading_bot.py
import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np

# Windows Unicode desteği
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# .env dosyası desteği (isteğe bağlı)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv yüklü değilse devam et

# Logging konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1) Konfigürasyon
API_KEY    = os.getenv("BINANCE_KEY")    # export BINANCE_KEY=xxx
API_SECRET = os.getenv("BINANCE_SECRET") # export BINANCE_SECRET=xxx
SYMBOL     = "BTCUSDT"
BASE_QUANTITY = 0.001     # her işlemde kaç BTC al/sat
INTERVAL   = Client.KLINE_INTERVAL_1HOUR
LIMIT      = 100
RSI_BUY    = 40
RSI_SELL   = 60
STOP_LOSS_PERCENT = 0.02  # %2 stop-loss
MAX_POSITION_SIZE = 0.1   # Portföyün max %10'u
MIN_USDT_BALANCE = 20     # Minimum USDT bakiyesi
RETRY_ATTEMPTS = 3        # Hata durumunda yeniden deneme
RETRY_DELAY = 5           # Yeniden deneme arası bekleme (saniye)
# ─────────────────────────────────────────────

# GÜVENLİK: Testnet kullan!
client = Client(API_KEY, API_SECRET, testnet=True)  # testnet=True → GÜVENLİ TEST

def get_klines():
    """Binance'den kline verilerini çek ve DataFrame'e dönüştür"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            candles = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT)
            df = pd.DataFrame(candles, columns=[
                'open_time','open','high','low','close','volume',
                'close_time','qav','trades','taker_base','taker_quote','ignore'
            ])
            df['close'] = pd.to_numeric(df['close'])
            df['high']  = pd.to_numeric(df['high'])
            df['low']   = pd.to_numeric(df['low'])
            df['open']  = pd.to_numeric(df['open'])
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

def get_technical_signals():
    """Teknik analiz sinyallerini hesapla"""
    df = get_klines()
    df['rsi'] = rsi(df['close'])
    df['ma20'] = moving_average(df['close'], 20)
    df['ma50'] = moving_average(df['close'], 50)
    
    current_price = float(df['close'].iloc[-1])
    current_rsi = float(df['rsi'].iloc[-1])
    ma20_current = float(df['ma20'].iloc[-1])
    ma50_current = float(df['ma50'].iloc[-1])
    
    # Trend analizi
    trend_bullish = ma20_current > ma50_current
    price_above_ma20 = current_price > ma20_current
    
    return {
        'price': current_price,
        'rsi': current_rsi,
        'ma20': ma20_current,
        'ma50': ma50_current,
        'trend_bullish': trend_bullish,
        'price_above_ma20': price_above_ma20
    }

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
    
    # Minimum bakiye kontrolü
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
        # Minimum emir miktarı kontrolü
        min_qty = 0.00001  # Binance minimum BTC miktarı
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
        with open("bot_state.json", "w") as f:
            json.dump(state, f, indent=2)
        logger.debug("Durum kaydedildi")
    except Exception as e:
        logger.error(f"Durum kaydedilemedi: {e}")

def load_state():
    """Bot durumunu yükle"""
    try:
        if os.path.exists("bot_state.json"):
            with open("bot_state.json") as f:
                state = json.load(f)
                logger.info("Önceki durum yüklendi")
                return state
    except Exception as e:
        logger.error(f"Durum yüklenemedi: {e}")
    
    return {
        "position": None,
        "entry_price": None,
        "stop_loss": None,
        "last_trade_time": None
    }

def should_stop_loss(current_price, entry_price, position):
    """Stop-loss kontrolü"""
    if position != "LONG" or not entry_price:
        return False
    
    loss_percent = (entry_price - current_price) / entry_price
    if loss_percent >= STOP_LOSS_PERCENT:
        logger.warning(f"Stop-loss tetiklendi! Kayıp: {loss_percent:.2%}")
        return True
    return False

# ─────────────────────────────────────────────
state = load_state()

def main_loop():
    """Ana trading döngüsü"""
    logger.info("[START] BTC Trading Bot başlatıldı")
    
    while True:
        try:
            # Teknik analiz sinyallerini al
            signals = get_technical_signals()
            current_price = signals['price']
            rsi_now = signals['rsi']
            
            logger.info(f"Fiyat: ${current_price:.2f} | RSI: {rsi_now:.2f} | Trend: {'UP' if signals['trend_bullish'] else 'DOWN'}")
            
            # Stop-loss kontrolü
            if should_stop_loss(current_price, state.get("entry_price"), state.get("position")):
                btc_balance = get_balance("BTC")
                if btc_balance > 0:
                    order = place_order(SIDE_SELL, btc_balance, current_price)
                    if order:
                        state["position"] = None
                        state["entry_price"] = None
                        state["stop_loss"] = None
                        save_state(state)
                        logger.info("Stop-loss emri verildi")
            
            # Alım sinyali
            elif (rsi_now <= RSI_BUY and 
                  state["position"] != "LONG" and 
                  signals['trend_bullish'] and 
                  signals['price_above_ma20']):
                
                usdt_balance = get_balance("USDT")
                quantity = calculate_position_size(current_price, usdt_balance)
                
                if quantity > 0:
                    order = place_order(SIDE_BUY, quantity, current_price)
                    if order:
                        state["position"] = "LONG"
                        state["entry_price"] = current_price
                        state["stop_loss"] = current_price * (1 - STOP_LOSS_PERCENT)
                        state["last_trade_time"] = datetime.now(timezone.utc).isoformat()
                        save_state(state)
                        logger.info(f"LONG pozisyon açıldı @ ${current_price:.2f}")
            
            # Satım sinyali
            elif (rsi_now >= RSI_SELL and 
                  state["position"] == "LONG"):
                
                btc_balance = get_balance("BTC")
                if btc_balance > 0:
                    order = place_order(SIDE_SELL, btc_balance, current_price)
                    if order:
                        entry_price = state.get("entry_price", current_price)
                        profit_percent = (current_price - entry_price) / entry_price if entry_price else 0
                        
                        state["position"] = None
                        state["entry_price"] = None
                        state["stop_loss"] = None
                        state["last_trade_time"] = datetime.now(timezone.utc).isoformat()
                        save_state(state)
                        
                        logger.info(f"LONG pozisyon kapatıldı @ ${current_price:.2f} | Kar: {profit_percent:.2%}")
            
            # Durum raporu
            if state["position"] == "LONG":
                entry_price = state.get("entry_price", current_price)
                unrealized_pnl = (current_price - entry_price) / entry_price if entry_price else 0
                logger.info(f"Açık pozisyon: LONG @ ${entry_price:.2f} | Gerçekleşmemiş P&L: {unrealized_pnl:.2%}")
        
        except KeyboardInterrupt:
            logger.info("Bot durduruldu")
            break
        except Exception as e:
            logger.error(f"Ana döngü hatası: {e}")
        
        # Market saatleri kontrolü (7/24 kripto için gerekli değil ama isteğe bağlı)
        time.sleep(60 * 30)  # 30 dakika bekle

def check_api_connection():
    """API bağlantısını test et"""
    try:
        client.ping()
        logger.info("[OK] Binance API bağlantısı başarılı")
        return True
    except Exception as e:
        logger.error(f"[ERROR] API bağlantı hatası: {e}")
        return False

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        logger.error("[ERROR] API anahtarlari bulunamadi! BINANCE_KEY ve BINANCE_SECRET environment variables'larini ayarlayin.")
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