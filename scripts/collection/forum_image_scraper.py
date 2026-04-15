"""
Çok sayfalı forum / thread sayfalarından görsel toplama.

Mantık:
    - Thread / kategori sayfalarını liste halinde verirsin
    - Her sayfada img etiketlerini toplar
    - Avatar, logo, emoji gibi çöpü filtreler
    - Sonra indirir

Kullanım:
    START_URLS listesini düzenle
    python scripts/collection/forum_image_scraper.py

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

START_URLS = [
    "https://example.com/forum/thread-1",
    "https://example.com/forum/thread-2",
]

OUTPUT_DIR = "forum_dataset"
DELAY_SECONDS = 1.2
MAX_PAGES = 50

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; forum-image-scraper/1.0)"
}

BAD_KEYWORDS = [
    "avatar", "logo", "icon", "emoji", "smilie", "banner", "sprite",
    "thumbnail", "thumb", "profile"
]


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


def is_bad_image_url(url: str) -> bool:
    lowered = url.lower()
    return any(k in lowered for k in BAD_KEYWORDS)


def get_soup(url: str, session: requests.Session) -> BeautifulSoup:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def extract_images(page_url: str, soup: BeautifulSoup) -> set[str]:
    images = set()

    for img in soup.find_all("img"):
        src = (
            img.get("data-src")
            or img.get("data-lazy-src")
            or img.get("src")
        )
        if not src:
            continue

        full = urljoin(page_url, src)
        if full.startswith("data:"):
            continue
        if is_bad_image_url(full):
            continue

        alt = (img.get("alt") or "").lower()
        cls = " ".join(img.get("class", [])).lower()

        if any(k in alt for k in ["avatar", "emoji", "icon"]):
            continue
        if any(k in cls for k in ["avatar", "emoji", "icon"]):
            continue

        images.add(full)

    return images


def find_next_page(page_url: str, soup: BeautifulSoup) -> str | None:
    link_text_candidates = ["next", "sonraki", "›", "»"]

    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True).lower()
        rel = " ".join(a.get("rel", [])).lower()
        href = urljoin(page_url, a["href"])

        if "next" in rel or text in link_text_candidates or "next" in text:
            return href

    return None


def download_image(
    img_url: str, session: requests.Session, output_dir: Path, index: int
):
    r = session.get(img_url, timeout=30, stream=True)
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise ValueError("Görsel içerik değil")

    ext_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
    }
    ext = ext_map.get(content_type.split(";")[0].strip().lower(), ".jpg")

    name_hash = hashlib.md5(img_url.encode("utf-8")).hexdigest()[:12]
    filepath = output_dir / f"img_{index:06d}_{name_hash}{ext}"

    with open(filepath, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

    print(f"[OK] {filepath}")


def crawl():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    visited_pages = set()
    found_images = set()

    for start_url in START_URLS:
        current_url = start_url
        page_count = 0

        while (
            current_url
            and current_url not in visited_pages
            and page_count < MAX_PAGES
        ):
            if not can_fetch(current_url):
                print(f"[ROBOTS SKIP PAGE] {current_url}")
                break

            print(f"[PAGE] {current_url}")
            visited_pages.add(current_url)

            try:
                soup = get_soup(current_url, session)
                page_images = extract_images(current_url, soup)
                found_images.update(page_images)
                print(f"   + {len(page_images)} görsel bulundu")
                current_url = find_next_page(current_url, soup)
            except Exception as e:
                print(f"[ERR PAGE] {current_url} -> {e}")
                break

            page_count += 1
            time.sleep(DELAY_SECONDS)

    print(f"\nToplam benzersiz görsel: {len(found_images)}")

    for idx, img_url in enumerate(sorted(found_images), start=1):
        try:
            if not can_fetch(img_url):
                print(f"[ROBOTS SKIP IMG] {img_url}")
                continue
            download_image(img_url, session, output_dir, idx)
        except Exception as e:
            print(f"[ERR IMG] {img_url} -> {e}")

        time.sleep(DELAY_SECONDS)


if __name__ == "__main__":
    crawl()
