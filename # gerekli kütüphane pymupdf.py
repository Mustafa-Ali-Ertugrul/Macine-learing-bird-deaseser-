# gerekli kütüphane pymupdf

import fitz  # pymupdf modülü

def extract_text_from_pdf(pdf_path):
    """
    Verilen bir PDF dosyasından metni çıkartır.
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        full_text += page.get_text()

    doc.close()
    return full_text

# Metni Temizlemek

import re

def clean_text(text):
    """
    Metni temizler ve kelimeler arası boşlukları kaldırır.
    """
    text = re.sub(r'\s+', '', text)  # Tüm boşlukları kaldırır
    return text

# JSON Formatına Dönüştürmek

import json

def save_to_json(text, output_path):
    """
    Metni JSON formatına dönüştürüp dosyaya kaydeder.
    """
    data = {"content": text}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

#Tüm Süreci Birleştirme

import os

def process_pdf_to_json(pdf_folder, output_folder):
    """
    Bir klasördeki tüm PDF dosyalarını temizleyip JSON formatına çevirir.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".pdf", ".json"))

            print(f"İşleniyor: {filename}")
            # PDF'den metni çıkar
            raw_text = extract_text_from_pdf(pdf_path)
            # Metni temizle
            cleaned_text = clean_text(raw_text)
            # JSON olarak kaydet
            save_to_json(cleaned_text, output_path)
            print(f"Tamamlandı: {output_path}")

#Kullanım

pdf_folder = "C:\Users\Ali\OneDrive\Masaüstü\mikroişlemciler"  # PDF dosyalarının olduğu klasör
output_folder = "C:\Users\Ali\OneDrive\Masaüstü\AAjSON"  # JSON dosyalarının kaydedileceği klasör

process_pdf_to_json(pdf_folder, output_folder)
