@echo off
echo ========================================
echo    CRYPTO TREND BOT LAUNCHER
echo ========================================
echo.
echo Tüm botlar başlatılıyor...
echo.

REM BTC Bot (mevcut)
if exist "btc_trend_bot.py" (
    echo [1/6] BTC Bot başlatılıyor...
    start "BTC Trend Bot" cmd /k "python btc_trend_bot.py"
    timeout /t 2 /nobreak >nul
) else (
    echo [HATA] btc_trend_bot.py bulunamadı!
)

REM ETH Bot
if exist "eth_trend_bot.py" (
    echo [2/6] ETH Bot başlatılıyor...
    start "ETH Trend Bot" cmd /k "python eth_trend_bot.py"
    timeout /t 2 /nobreak >nul
) else (
    echo [UYARI] eth_trend_bot.py bulunamadı - atlanıyor
)

REM ADA Bot
if exist "ada_trend_bot.py" (
    echo [3/6] ADA Bot başlatılıyor...
    start "ADA Trend Bot" cmd /k "python ada_trend_bot.py"
    timeout /t 2 /nobreak >nul
) else (
    echo [UYARI] ada_trend_bot.py bulunamadı - atlanıyor
)

REM SOL Bot
if exist "sol_trend_bot.py" (
    echo [4/6] SOL Bot başlatılıyor...
    start "SOL Trend Bot" cmd /k "python sol_trend_bot.py"
    timeout /t 2 /nobreak >nul
) else (
    echo [UYARI] sol_trend_bot.py bulunamadı - atlanıyor
)

REM DOT Bot
if exist "dot_trend_bot.py" (
    echo [5/6] DOT Bot başlatılıyor...
    start "DOT Trend Bot" cmd /k "python dot_trend_bot.py"
    timeout /t 2 /nobreak >nul
) else (
    echo [UYARI] dot_trend_bot.py bulunamadı - atlanıyor
)

REM MATIC Bot
if exist "matic_trend_bot.py" (
    echo [6/6] MATIC Bot başlatılıyor...
    start "MATIC Trend Bot" cmd /k "python matic_trend_bot.py"
    timeout /t 2 /nobreak >nul
) else (
    echo [UYARI] matic_trend_bot.py bulunamadı - atlanıyor
)

echo.
echo ========================================
echo Tüm mevcut botlar başlatıldı!
echo Her bot ayrı bir terminal penceresinde çalışıyor.
echo Botları durdurmak için terminal pencerelerini kapatın.
echo ========================================
echo.
pause