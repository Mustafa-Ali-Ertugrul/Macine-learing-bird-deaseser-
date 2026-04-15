"""
Hastalık bazlı klasörlü dataset indirici.

Her hastalık için ayrı klasöre indirir.
DATASET sözlüğüne hastalık adı ve URL listesini yaz.

Kullanım:
    python scripts/collection/disease_downloader.py

Kurulum:
    pip install requests
"""

import time
import hashlib
from pathlib import Path
from urllib.parse import urlparse
import requests

# ─────────────────────────────────────────
# Duck + Goose hastalık URL şablonu
# URL'leri kendi topladığın linklerle doldur
# ─────────────────────────────────────────
DATASET = {
    # ── Duck Hastalıkları ──
    "duck/Avian_Influenza": [
        # "https://example.com/duck-ai-1.jpg",
    ],
    "duck/Coccidiosis": [],
    "duck/Fowl_Pox": [],
    "duck/Healthy": [],
    "duck/Histomoniasis": [],
    "duck/Infectious_Bronchitis": [],
    "duck/Infectious_Bursal_Disease": [],
    "duck/Mareks_Disease": [],
    "duck/Newcastle_Disease": [],
    "duck/Salmonella": [],

    # ── Goose Hastalıkları ──
    "goose/Avian_Influenza": [],
    "goose/Coccidiosis": [],
    "goose/Fowl_Pox": [],
    "goose/Healthy": [],
    "goose/Histomoniasis": [],
    "goose/Infectious_Bronchitis": [],
    "goose/Infectious_Bursal_Disease": [],
    "goose/Mareks_Disease": [],
    "goose/Newcastle_Disease": [],
    "goose/Salmonella": [],
}

OUTPUT_ROOT = "downloaded_dataset"
DELAY_SECONDS = 1.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; dataset-builder/1.0)"
}


def ext_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in [".jpg", ".jpeg", ".png", ".webp"] else ".jpg"


def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    root = Path(OUTPUT_ROOT)
    root.mkdir(exist_ok=True)

    total_success = 0
    total_fail = 0

    for label, urls in DATASET.items():
        if not urls:
            continue

        label_dir = root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        label_name = label.split("/")[-1]
        print(f"\n[{label}] -> {len(urls)} görsel")

        success = 0
        for i, url in enumerate(urls, start=1):
            try:
                r = session.get(url, timeout=30, stream=True)
                r.raise_for_status()

                if not r.headers.get("Content-Type", "").startswith("image/"):
                    print(f"  [SKIP] Görsel değil: {url}")
                    total_fail += 1
                    continue

                ext = ext_from_url(url)
                h = hashlib.md5(url.encode()).hexdigest()[:10]
                filename = label_dir / f"{label_name}_{i:05d}_{h}{ext}"

                with open(filename, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)

                print(f"  [OK] {filename}")
                success += 1

            except Exception as e:
                print(f"  [ERR] {url} -> {e}")
                total_fail += 1

            time.sleep(DELAY_SECONDS)

        total_success += success

    print(f"\n{'='*50}")
    print(f"Toplam başarılı: {total_success}")
    print(f"Toplam hatalı:   {total_fail}")


if __name__ == "__main__":
    main()
