"""
URL listesinden toplu görsel indirici.

Kullanım:
    1. urls.txt dosyasına her satıra bir URL yaz
    2. python scripts/collection/download_from_urls.py

Kurulum:
    pip install requests
"""

import os
import time
import hashlib
from pathlib import Path
from urllib.parse import urlparse
import requests

URLS_FILE = "urls.txt"
OUTPUT_DIR = "dataset_images"
DELAY_SECONDS = 1.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; dataset-downloader/1.0)"
}


def guess_extension(url: str, content_type: str | None) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]:
        return suffix

    if content_type:
        mapping = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
        }
        return mapping.get(content_type.split(";")[0].strip().lower(), ".jpg")

    return ".jpg"


def safe_filename(url: str, index: int, ext: str) -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
    return f"img_{index:05d}_{h}{ext}"


def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    session = requests.Session()
    session.headers.update(HEADERS)

    success = 0
    fail = 0

    for i, url in enumerate(urls, start=1):
        try:
            r = session.get(url, timeout=30, stream=True)
            r.raise_for_status()

            content_type = r.headers.get("Content-Type")
            if not content_type or not content_type.startswith("image/"):
                print(f"[SKIP] Görsel değil: {url}")
                fail += 1
                continue

            ext = guess_extension(url, content_type)
            filename = safe_filename(url, i, ext)
            filepath = out / filename

            with open(filepath, "wb") as img_file:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        img_file.write(chunk)

            print(f"[OK] {filepath}")
            success += 1

        except Exception as e:
            print(f"[ERR] {url} -> {e}")
            fail += 1

        time.sleep(DELAY_SECONDS)

    print(f"\nBitti. Başarılı: {success} | Hatalı: {fail}")


if __name__ == "__main__":
    main()
