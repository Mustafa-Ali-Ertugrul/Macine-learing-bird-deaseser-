import matplotlib.pyplot as plt
import numpy as np

# Veri oluşturma
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

# Grafik oluşturma
plt.figure(figsize=(10, 8))

# Alt grafik 1: Sinüs dalgası
plt.subplot(2, 2, 1)
plt.plot(x, y1, 'b-', linewidth=2)
plt.title('Sinüs Dalgası')
plt.xlabel('X değerleri')
plt.ylabel('sin(x)')
plt.grid(True, alpha=0.3)

# Alt grafik 2: Kosinüs dalgası
plt.subplot(2, 2, 2)
plt.plot(x, y2, 'r-', linewidth=2)
plt.title('Kosinüs Dalgası')
plt.xlabel('X değerleri')
plt.ylabel('cos(x)')
plt.grid(True, alpha=0.3)

# Alt grafik 3: Sinüs * Kosinüs
plt.subplot(2, 2, 3)
plt.plot(x, y3, 'g-', linewidth=2)
plt.title('Sinüs × Kosinüs')
plt.xlabel('X değerleri')
plt.ylabel('sin(x) × cos(x)')
plt.grid(True, alpha=0.3)

# Alt grafik 4: Hepsi bir arada
plt.subplot(2, 2, 4)
plt.plot(x, y1, 'b-', label='sin(x)', linewidth=2)
plt.plot(x, y2, 'r-', label='cos(x)', linewidth=2)
plt.plot(x, y3, 'g-', label='sin(x)×cos(x)', linewidth=2)
plt.title('Tüm Fonksiyonlar')
plt.xlabel('X değerleri')
plt.ylabel('Y değerleri')
plt.legend()
plt.grid(True, alpha=0.3)

# Grafikleri düzenle ve göster
plt.tight_layout()
plt.show()

# Basit bar grafiği örneği
plt.figure(figsize=(8, 6))
kategoriler = ['Python', 'JavaScript', 'Java', 'C++', 'C#']
değerler = [85, 75, 70, 65, 60]
renkler = ['#3776ab', '#f7df1e', '#007396', '#00599c', '#239120']

plt.bar(kategoriler, değerler, color=renkler)
plt.title('Programlama Dilleri Popülerlik Skoru')
plt.xlabel('Diller')
plt.ylabel('Popülerlik (%)')
plt.ylim(0, 100)

# Her çubuğun üstüne değer yaz
for i, v in enumerate(değerler):
    plt.text(i, v + 1, str(v), ha='center', va='bottom')


# Eğitim sürecini görselleştir
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Eğitim Süreci')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (MSE)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------- 6) Recursive forecasting ----------
last60 = scaled[-look_back:].reshape(1, look_back, 1)
preds_scaled = []
steps = 30
for _ in range(steps):
    p = model.predict(last60, verbose=0)[0, 0]
    preds_scaled.append(p)
    last60 = np.append(last60[:, 1:, :], [[[p]]], axis=1)

preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
future_idx = pd.date_range(df.index[-1] + dt.timedelta(days=1), periods=steps, freq='D')

# ---------- 7) Görselleştir ----------
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['close'], label='Gerçek Fiyat')
plt.plot(future_idx, preds, label='30-Günlük Tahmin', linestyle='--', color='red')
plt.title('BTCUSDT – LSTM Tahmin')
plt.xlabel('Tarih')
plt.ylabel('Fiyat (USDT)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------- 8) Opsiyonel: tahminleri CSV'ye yaz ----------
out = pd.DataFrame({'date': future_idx, 'predicted_close': preds})
out.to_csv('btc_30d_forecast.csv', index=False)
print("Tahminler btc_30d_forecast.csv dosyasına yazıldı.")