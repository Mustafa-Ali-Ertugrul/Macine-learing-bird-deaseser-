"""
Tek bir web sayfasındaki görselleri çekip indirir.

Kullanım:
    PAGE_URL değişkenini hedef sayfaya ayarla
    python scripts/collection/download_images_from_page.py

Kurulum:
    pip install requests beautifulsoup4
"""

import time
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

PAGE_URL = "https://example.com/some-disease-page"
OUTPUT_DIR = "page_images"
DELAY_SECONDS = 1.0
MIN_WIDTH_HINT = 200  # çok küçük ikonları elemek için kaba filtre

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; image-scraper/1.0)"
}


def can_fetch(url: str, user_agent: str = "*") -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


def looks_too_small(img_tag) -> bool:
    width = img_tag.get("width")
    height = img_tag.get("height")
    try:
        if width and int(width) < MIN_WIDTH_HINT:
            return True
        if height and int(height) < MIN_WIDTH_HINT:
            return True
    except Exception:
        pass
    return False


def collect_image_urls(page_url: str) -> list[str]:
    r = requests.get(page_url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    urls = set()

    for img in soup.find_all("img"):
        if looks_too_small(img):
            continue

        candidates = [
            img.get("src"),
            img.get("data-src"),
            img.get("data-lazy-src"),
            img.get("data-original"),
        ]

        srcset = img.get("srcset")
        if srcset:
            parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
            candidates.extend(parts)

        for c in candidates:
            if not c:
                continue
            full = urljoin(page_url, c)
            if full.startswith("data:"):
                continue
            urls.add(full)

    return sorted(urls)


def guess_extension(url: str, content_type: str | None) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
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


def save_images(image_urls: list[str], output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    success = 0
    for idx, img_url in enumerate(image_urls, start=1):
        try:
            if not can_fetch(img_url):
                print(f"[ROBOTS SKIP] {img_url}")
                continue

            r = session.get(img_url, timeout=30, stream=True)
            r.raise_for_status()

            content_type = r.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                print(f"[SKIP] Görsel değil: {img_url}")
                continue

            ext = guess_extension(img_url, content_type)
            name_hash = hashlib.md5(img_url.encode("utf-8")).hexdigest()[:12]
            filename = f"img_{idx:05d}_{name_hash}{ext}"
            filepath = out / filename

            with open(filepath, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

            print(f"[OK] {filepath}")
            success += 1

        except Exception as e:
            print(f"[ERR] {img_url} -> {e}")

        time.sleep(DELAY_SECONDS)

    print(f"\nToplam indirilen: {success}/{len(image_urls)}")


def main():
    image_urls = collect_image_urls(PAGE_URL)
    print(f"Bulunan görsel sayısı: {len(image_urls)}")
    save_images(image_urls, OUTPUT_DIR)


if __name__ == "__main__":
    main()
