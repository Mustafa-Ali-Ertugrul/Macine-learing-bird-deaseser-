from flask import Flask, render_template
import time
import socket
from ftplib import FTP
import smtplib
import pandas as pd

app = Flask(__name__)

# Excel dosyasının oluşturulması ve başlıkların yazılması
excel_filename = "protokol_verileri.xlsx"
columns = ["Timestamp", "HTTP_Latency", "DNS_Latency", "FTP_Latency", "SMTP_Latency"]

# Eğer dosya yoksa başlıkları oluşturuyoruz
try:
    df = pd.read_excel(excel_filename)
except FileNotFoundError:
    df = pd.DataFrame(columns=columns)
    df.to_excel(excel_filename, index=False)


@app.route('/')
def show_data():
    df = pd.read_excel(excel_filename)

    # Sayısal sütunları float'a çevirme ve saniye cinsinden düzenleme
    numeric_cols = ['HTTP_Latency', 'DNS_Latency', 'FTP_Latency', 'SMTP_Latency']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Gecikmeleri saniye olarak formatlama
    df['HTTP_Latency'] = df['HTTP_Latency'].astype(float)
    df['DNS_Latency'] = df['DNS_Latency'].astype(float)
    df['FTP_Latency'] = df['FTP_Latency'].astype(float)
    df['SMTP_Latency'] = df['SMTP_Latency'].astype(float)

    return render_template('index.html',
                           data=df.to_dict('records'),
                           last_update=time.strftime("%Y-%m-%d %H:%M:%S"))


# HTTP gecikmesi
@app.route('/data', methods=['GET'])
def send_data():
    start_time = time.time()
    response = {"message": "Hello, this is the HTTP server response!"}
    latency = time.time() - start_time
    response["latency"] = latency
    return response


# DNS adresini döndüren fonksiyon
def dns_lookup(domain):
    try:
        start_time = time.time()
        ip = socket.gethostbyname(domain)
        latency = time.time() - start_time
        return latency  # DNS gecikmesini döndürür
    except socket.timeout as e:
        print(f"DNS Timeout: {e}")  # Hata mesajı yazdırılır
        return "DNS Timeout"  # Timeout olduğunda geçici bir değer döndürülür
    except Exception as e:
        print(f"DNS Error: {e}")  # Diğer hatalar yazdırılır
        return "DNS Error"  # Diğer hatalarda geçici bir değer döndürülür


# FTP gecikmesi
def ftp_latency():
    try:
        ftp = FTP('ftp.dlptest.com', timeout=60)
        ftp.login('dlpuser', 'rNrKYTX9g7z3RgJRmxWuGHbeu')
        start_time = time.time()
        ftp.retrlines('LIST')
        latency = time.time() - start_time
        ftp.quit()
        return latency
    except Exception as e:
        print(f"FTP Error: {e}")
        return None


# SMTP gecikmesi
def smtp_latency():
    try:
        start_time = time.time()  # Start time before login
        server = smtplib.SMTP('smtp.gmail.com', 587, timeout=60)
        server.starttls()
        server.login("mustafali1230@hotmail.com", "1qaz2wsx3edC%56")
        msg = "Subject: Test Mail\n\nThis is a test email."
        server.sendmail("mustafali1230@hotmail.com", "receiver_email@gmail.com", msg)
        latency = time.time() - start_time
        server.quit()
        return latency
    except Exception as e:
        print(f"SMTP Error: {e}")
        return None


# Veri kaydetme fonksiyonu
@app.route('/save_data', methods=['GET'])
def save_data():
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # HTTP, DNS, FTP ve SMTP gecikme verilerini alıyoruz
        http_latency = send_data()["latency"]
        dns_latency = dns_lookup("google.com")  # DNS gecikmesi yerine IP adresini alıyoruz
        ftp_latency_value = ftp_latency()
        smtp_latency_value = smtp_latency()

        # Yeni veriyi oluştururken
        new_data = {
            "Timestamp": timestamp,
            "HTTP_Latency": f"{http_latency:.2f}",
            "DNS_Latency": f"{dns_latency:.2f}" if isinstance(dns_latency, (int, float)) else dns_latency,
            "FTP_Latency": f"{ftp_latency_value:.2f}" if isinstance(ftp_latency_value,
                                                                    (int, float)) else "FTP Not Available",
            "SMTP_Latency": f"{smtp_latency_value:.2f}" if isinstance(smtp_latency_value,
                                                                      (int, float)) else "SMTP Not Available"
        }

        print(f"Yeni veri: {new_data}")  # Hata ayıklama için veri kontrolü

        # Veriyi Excel dosyasına ekle
        df = pd.read_excel(excel_filename)

        # new_row adında yeni bir DataFrame oluşturuyoruz
        new_row = pd.DataFrame([new_data])

        # df ve new_row'u birleştiriyoruz
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.fillna("Veri Yok")  # Tüm NaN değerlerini "Veri Yok" ile değiştirir
        # Veriyi tekrar Excel dosyasına yaz
        df.to_excel(excel_filename, index=False)
        print(f"Veri başarıyla {excel_filename} dosyasına kaydedildi.")

        return "Veri başarıyla kaydedildi!"
    except Exception as e:
        print(f"Hata oluştu: {e}")  # Hata mesajını yazdır
        return f"Veri kaydedilirken hata oluştu: {e}"


# Excel temizleme fonksiyonu
def clean_excel():
    try:
        df = pd.read_excel(excel_filename)
        numeric_cols = ['HTTP_Latency', 'DNS_Latency', 'FTP_Latency', 'SMTP_Latency']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        df.to_excel(excel_filename, index=False)
        print(f"{excel_filename} dosyası başarıyla temizlendi.")
    except Exception as e:
        print(f"Excel dosyası temizlenirken hata oluştu: {e}")


# Uygulama başlangıcında temizlik yapın
if __name__ == "__main__":
    clean_excel()  # Excel temizliği
    app.run(host="0.0.0.0", port=5000, debug=True)  # Flask uygulaması çalıştırılıyor