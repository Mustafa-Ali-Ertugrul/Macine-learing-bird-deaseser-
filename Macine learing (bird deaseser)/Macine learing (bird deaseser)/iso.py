from flask import Flask, render_template, redirect, url_for, request
import time
import socket
from ftplib import FTP
import smtplib
import pandas as pd
import os
from dotenv import load_dotenv
import requests
import traceback
import ping3

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

excel_filename = "protokol_verileri.xlsx"
columns = ["Timestamp", "Country", "HTTP_Latency", "DNS_Latency", "DNS_IP", "FTP_Latency", "SMTP_Latency"]

# Country-specific target servers
COUNTRY_SERVERS = {
    "Turkiye": {
        "http": "https://www.trendyol.com",
        "dns": "dns.google.com",  # Using Google DNS for all regions
        "ping": "www.trendyol.com"
    },
    "ABD": {
        "http": "https://www.amazon.com",
        "dns": "dns.google.com",
        "ping": "www.amazon.com"
    },
    "Cin": {
        "http": "https://www.baidu.com",
        "dns": "dns.google.com",
        "ping": "www.baidu.com"
    }
}

# Default country
DEFAULT_COUNTRY = "Turkiye"

try:
    df = pd.read_excel(excel_filename)
    # Add Country column if it doesn't exist
    if 'Country' not in df.columns:
        df['Country'] = DEFAULT_COUNTRY
        df.to_excel(excel_filename, index=False)
except FileNotFoundError:
    df = pd.DataFrame(columns=columns)
    df.to_excel(excel_filename, index=False)

def get_http_latency(country):
    try:
        target_url = COUNTRY_SERVERS.get(country, COUNTRY_SERVERS[DEFAULT_COUNTRY])["http"]
        start_time = time.time()
        response = requests.get(target_url, timeout=10)
        response.raise_for_status()
        latency = time.time() - start_time
        return latency
    except Exception as e:
        print(f"HTTP Hatası ({country}):", traceback.format_exc())
        return None

def measure_dns_latency(country):
    try:
        dns_server = COUNTRY_SERVERS.get(country, COUNTRY_SERVERS[DEFAULT_COUNTRY])["dns"]
        start = time.time()
        socket.setdefaulttimeout(5)
        ip = socket.gethostbyname(dns_server)
        latency = time.time() - start
        return latency, ip
    except Exception as e:
        print(f"DNS Hatası ({country}):", traceback.format_exc())
        return None, "DNS Hatası"

def ftp_latency():
    retries = 3
    for attempt in range(retries):
        try:
            ftp_host = os.getenv('FTP_HOST', 'ftp.dlptest.com')
            ftp_user = os.getenv('FTP_USER', 'dlpuser')
            ftp_password = os.getenv('FTP_PASSWORD', 'rNrKYTX9g7z3RgJRmxWuGHbeu')
            ftp_timeout = int(os.getenv('FTP_TIMEOUT', 10))
            ftp = FTP(ftp_host, timeout=ftp_timeout)
            ftp.set_pasv(True)
            ftp.login(ftp_user, ftp_password)
            start_time = time.time()
            ftp.retrlines('LIST')
            latency = time.time() - start_time
            ftp.quit()
            return latency
        except Exception as e:
            print(f"FTP Deneme {attempt + 1} Hatası:", traceback.format_exc())
            if attempt == retries - 1:
                return None
            time.sleep(5)

def smtp_latency():
    try:
        email = os.getenv('EMAIL')
        app_password = os.getenv('APP_PASSWORD')
        smtp_timeout = int(os.getenv('SMTP_TIMEOUT', 10))

        if not email or not app_password:
            raise ValueError("SMTP bilgileri eksik.")

        start_time = time.time()
        server = smtplib.SMTP('smtp.gmail.com', 587, timeout=smtp_timeout)
        server.starttls()
        server.login(email, app_password)
        msg = "Subject: Test\n\nTest email."
        server.sendmail(email, email, msg)
        latency = time.time() - start_time
        server.quit()
        return latency
    except Exception as e:
        print("SMTP Hatası:", traceback.format_exc())
        return None

def ping_server(country):
    try:
        target = COUNTRY_SERVERS.get(country, COUNTRY_SERVERS[DEFAULT_COUNTRY])["ping"]
        result = ping3.ping(target, timeout=5)
        return result  # Returns time in seconds or None if failed
    except Exception as e:
        print(f"Ping Hatası ({country}):", traceback.format_exc())
        return None

def save_data(country=DEFAULT_COUNTRY):
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"{country} için ölçüm başlatiliyor...")
        
        print("HTTP ölçülüyor...")
        http_latency = get_http_latency(country)
        print("DNS ölçülüyor...")
        dns_latency, dns_ip = measure_dns_latency(country)
        print("FTP ölçülüyor...")
        ftp_latency_value = ftp_latency()
        print("SMTP ölçülüyor...")
        smtp_latency_value = smtp_latency()
        print("Ping ölçülüyor...")
        ping_latency = ping_server(country)

        new_data = {
            "Timestamp": timestamp,
            "Country": country,
            "HTTP_Latency": f"{http_latency:.4f}" if http_latency is not None else "Veri Yok",
            "DNS_Latency": f"{dns_latency:.4f}" if dns_latency is not None else "Veri Yok",
            "DNS_IP": dns_ip if dns_ip else "Veri Yok",
            "FTP_Latency": f"{ftp_latency_value:.4f}" if ftp_latency_value is not None else "Veri Yok",
            "SMTP_Latency": f"{smtp_latency_value:.4f}" if smtp_latency_value is not None else "Veri Yok",
            "Ping_Latency": f"{ping_latency:.4f}" if ping_latency is not None else "Veri Yok"
        }

        df = pd.read_excel(excel_filename)
        
        # Ensure all required columns exist
        for column in columns + ["Ping_Latency"]:
            if column not in df.columns:
                df[column] = "Veri Yok"

        new_row = pd.DataFrame([new_data])
        df = pd.concat([df, new_row], ignore_index=True).fillna("Veri Yok")
        df.to_excel(excel_filename, index=False)
        print(f"Veri kaydedildi: {timestamp} - {country}")

    except Exception as e:
        print("save_data Hatası:", traceback.format_exc())

def clear_data():
    try:
        df = pd.read_excel(excel_filename)
        df = df.iloc[0:0]
        df.to_excel(excel_filename, index=False)
        print("Veriler temizlendi.")
        return True
    except Exception as e:
        print("clear_data Hatası:", traceback.format_exc())
        return False

@app.route('/')
def show_data():
    current_country = request.args.get('country', DEFAULT_COUNTRY)
    
    try:
        df = pd.read_excel(excel_filename)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
        df.to_excel(excel_filename, index=False)

    # Ensure the Country column exists
    if 'Country' not in df.columns:
        df['Country'] = DEFAULT_COUNTRY
        df.to_excel(excel_filename, index=False)

    numeric_cols = ['HTTP_Latency', 'DNS_Latency', 'FTP_Latency', 'SMTP_Latency', 'Ping_Latency']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0
    
    return render_template('index.html', 
                           data=df.to_dict('records'),
                           last_update=time.strftime("%Y-%m-%d %H:%M:%S"),
                           current_country=current_country)

@app.route('/yeni_olcum')
def yeni_olcum():
    country = request.args.get('country', DEFAULT_COUNTRY)
    print(f"Yeni ölçüm başlatıldı... Ülke: {country}")
    save_data(country)
    return redirect(url_for('show_data', country=country))

@app.route('/clear_data', methods=['POST'])
def clear_data_route():
    country = request.form.get('country', DEFAULT_COUNTRY)
    if clear_data():
        return redirect(url_for('show_data', country=country))
    else:
        return "Temizleme hatası", 500

@app.route("/grafikler")
def grafik_goster():
    country = request.args.get('country', DEFAULT_COUNTRY)
    return render_template("grafikler.html", 
                           data=veri_yukle(), 
                           last_update=get_last_update(),
                           current_country=country)

def veri_yukle():
    try:
        df = pd.read_excel("protokol_verileri.xlsx")
        return df.to_dict(orient="records")
    except Exception as e:
        print("Veri yüklenemedi:", e)
        return []

def get_last_update():
    try:
        mod_time = os.path.getmtime("protokol_verileri.xlsx")
        return datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "Bilinmiyor"

# Add missing import
from datetime import datetime

if __name__ == "__main__":
    if not os.path.exists(excel_filename):
        df = pd.DataFrame(columns=columns)
        df.to_excel(excel_filename, index=False)
    app.run(host="0.0.0.0", port=5000, debug=True)


