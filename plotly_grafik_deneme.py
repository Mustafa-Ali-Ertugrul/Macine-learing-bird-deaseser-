#!/usr/bin/env python3
# btc_daily_pnl.py
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

# 1) Binance'ten günlük kapanış verisini çek
symbol, interval, limit = 'BTCUSDT', '1d', 365  # son 1 yıl
url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
resp = requests.get(url)
resp.raise_for_status()

cols = ['ot','o','h','l','c','v','ct','qv','trades','tb','tq','ig']
df = pd.DataFrame(resp.json(), columns=cols)
df['date']  = pd.to_datetime(df['ot'], unit='ms')
df['close'] = df['c'].astype(float)
df = df[['date', 'close']].sort_values('date').reset_index(drop=True)

# 2) Günlük kar / zarar (kapanış - önceki kapanış)
df['pnl'] = df['close'].diff()

# 3) Günlük kar/zarar grafiği
fig = go.Figure()
colors = ['green' if v >= 0 else 'red' for v in df['pnl']]
fig.add_trace(go.Bar(
    x=df['date'],
    y=df['pnl'],
    marker_color=colors,
    name='Günlük Kar/Zarar',
    hovertemplate='%{x|%Y-%m-%d}<br>ΔPrice: %{y:.2f} USDT<extra></extra>'
))
fig.update_layout(
    title=f"{symbol} – Günlük Kar / Zarar (USDT)",
    xaxis_title="Tarih",
    yaxis_title="Günlük Değişim (USDT)",
    template="plotly_white",
    height=500
)

# 4) HTML olarak tek dosya
html_str = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')
with open("btc_daily_pnl.html", "w", encoding="utf-8") as f:
    f.write('<html><head><meta charset="UTF-8"><title>BTC Günlük Kar/Zarar</title></head><body>')
    f.write(html_str)
    f.write('</body></html>')

print("✅ btc_daily_pnl.html oluşturuldu.")
import webbrowser, os
webbrowser.open("file://" + os.path.abspath("btc_daily_pnl.html"))