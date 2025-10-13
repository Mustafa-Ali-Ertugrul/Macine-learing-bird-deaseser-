#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bitcoin Daily PnL Analyzer Demo
Bu script, düzeltilmiş BTC analizörünün nasıl kullanılacağını gösterir.
"""

import btc_daily_pnl
import numpy as np
import pandas as pd

def main():
    print("🚀 Bitcoin Daily PnL Analyzer Demo")
    print("=" * 50)
    
    # Analyzer'ı başlat
    print("📊 Analyzer başlatılıyor...")
    analyzer = btc_daily_pnl.AdvancedBTCAnalyzer()
    
    # Veri çek
    print("\n📈 Bitcoin verisi çekiliyor...")
    df = analyzer.fetch_data()
    # Veri çekildi mesajı btc_daily_pnl.py'de zaten gösteriliyor
    
    # Özellik mühendisliği
    print("\n🔧 Gelişmiş özellik mühendisliği yapılıyor...")
    X, y, feature_names = analyzer.create_advanced_features(df)
    # Özellik oluşturma mesajı btc_daily_pnl.py'de zaten gösteriliyor
    
    # Model eğitimi (sadece modeller yüklü değilse)
    if not analyzer.ensemble_models:
        print("\n🤖 Ensemble modeller eğitiliyor...")
        results, best_model_name, X_test, y_test = analyzer.train_ensemble_models(X, y)
        print(f"✅ En iyi model: {best_model_name}")
    else:
        print("\n✅ Önceden eğitilmiş modeller yüklendi")
    
    # Tahmin yap
    print("\n🔮 Ensemble tahmin yapılıyor...")
    latest_features = X[-1:, :]  # En son veri noktası
    prediction = analyzer.create_ensemble_prediction(latest_features)
    
    # Sonuçları göster
    current_price = df['close'].iloc[-1]
    predicted_change = prediction - current_price
    change_percent = (predicted_change / current_price) * 100
    
    print("\n📊 SONUÇLAR:")
    print(f"💰 Mevcut BTC Fiyatı: ${current_price:,.2f}")
    print(f"🎯 Tahmin Edilen Fiyat: ${prediction:,.2f}")
    print(f"📈 Beklenen Değişim: ${predicted_change:,.2f} ({change_percent:+.2f}%)")
    
    if change_percent > 0:
        print("🟢 Yükseliş bekleniyor")
    else:
        print("🔴 Düşüş bekleniyor")
    
    # Risk metrikleri
    print("\n📊 Risk Analizi:")
    risk_metrics = analyzer.calculate_advanced_risk_metrics(df['close'])
    
    print(f"📉 Maksimum Düşüş: {risk_metrics['max_drawdown']:.2f}%")
    print(f"⚡ Volatilite: {risk_metrics['volatility']:.2f}%")
    print(f"📊 Sharpe Oranı: {risk_metrics['sharpe_ratio']:.3f}")
    print(f"🎯 VaR (95%): {risk_metrics['var_95']:.2f}%")
    print(f"📈 Toplam Getiri: {risk_metrics['total_return']:.2f}%")
    print(f"🎯 Kazanma Oranı: {risk_metrics['win_rate']:.2f}%")
    print(f"🟢 En İyi Gün: {risk_metrics['best_day']:.2f}%")
    print(f"🔴 En Kötü Gün: {risk_metrics['worst_day']:.2f}%")
    
    print("\n✅ Analiz tamamlandı!")

if __name__ == "__main__":
    main()