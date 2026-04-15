"""
Duck & Goose hastalık görsellerini belirli kaynak sayfalardan indirir.
Her sınıf için ayrı klasör, metadata JSON oluşturur.
"""

import json
import time
import hashlib
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

# ─────────────────────────────────────────
# Kaynak Yapılandırması
# ─────────────────────────────────────────
SOURCES = {
    "duck_plague": [
        "https://www.thepoultrysite.com/disease-guide/duck-virus-enteritis-duck-plague",
    ],
    "goose_parvovirus": [
        "https://www.thepoultrysite.com/disease-guide/goose-parvovirus-derzsys-disease",
        "https://www.thepoultrysite.com/articles/goose-parvovirus",
    ],
    "duck_viral_hepatitis": [
        "https://www.thepoultrysite.com/disease-guide/duck-viral-hepatitis",
    ],
    "avian_pox": [
        "https://cwhl.vet.cornell.edu/resource/avian-pox",
        "https://cwhl.vet.cornell.edu/resources/disease-fact-sheets",
    ],
    "bumblefoot": [
        "https://www.backyardchickens.com/articles/treating-bumble-foot-in-ducks.74245/",
        "https://www.backyardchickens.com/threads/treating-ducks-for-bumble-foot.1491181/",
        "https://www.backyardchickens.com/threads/possible-bumblefoot.1646259/",
        "https://www.backyardchickens.com/threads/duck-foot-sores-bumblefoot.1648968/",
        "https://www.backyardchickens.com/articles/bumblefoot-what-is-it.74647/",
    ],
    "waterfowl_viral_disease": [
        "https://www.thepoultrysite.com/articles/major-viral-diseases-of-waterfowl-and-their-control",
    ],
}

OUTPUT_ROOT = "downloaded_disease_images"
DELAY_SECONDS = 2.0
MIN_IMAGE_SIZE = 150  # minimum px
MAX_RETRIES = 2

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Filtreleme
BAD_URL_KEYWORDS = [
    "avatar", "logo", "icon", "emoji", "smilie", "banner", "sprite",
    "thumbnail", "thumb", "profile", "button", "widget", "badge",
    "social", "facebook", "twitter", "youtube", "pinterest",
    "advertisement", "ad-", "ads/", "advert", "pixel", "tracking",
    "gravatar", "wp-emoji", "smiley", "emoticon",
    "footer", "header-logo", "site-logo", "nav-",
]

BAD_ALT_KEYWORDS = [
    "avatar", "logo", "icon", "emoji", "profile",
    "advertisement", "banner", "social", "share",
    "illustration", "diagram", "chart", "infographic",
    "stock photo", "shutterstock", "getty", "istock",
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_source_name(url: str) -> str:
    """URL'den kaynak site adını çıkar."""
    netloc = urlparse(url).netloc.lower()
    netloc = netloc.replace("www.", "")
    return netloc.replace(".", "_")


def is_bad_url(url: str) -> bool:
    lowered = url.lower()
    return any(kw in lowered for kw in BAD_URL_KEYWORDS)


def is_bad_alt(alt_text: str) -> bool:
    lowered = alt_text.lower()
    return any(kw in lowered for kw in BAD_ALT_KEYWORDS)


def get_image_extension(url: str, content_type: str = "") -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in VALID_EXTENSIONS:
        return suffix

    ct = content_type.split(";")[0].strip().lower()
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    return mapping.get(ct, ".jpg")


def validate_image(data: bytes) -> tuple[bool, int, int]:
    """Görselin geçerli ve yeterli boyutta olup olmadığını kontrol et."""
    try:
        img = Image.open(BytesIO(data))
        w, h = img.size
        if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
            return False, w, h
        # Çizim/illüstrasyon tespiti (çok az renk = muhtemelen ikon/çizim)
        if img.mode in ("P", "1"):
            return False, w, h
        return True, w, h
    except Exception:
        return False, 0, 0


def extract_image_urls(page_url: str, session: requests.Session) -> list[dict]:
    """Sayfadaki tüm potansiyel hastalık görsellerini bul."""
    try:
        r = session.get(page_url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  [SAYFA HATASI] {page_url}: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    seen_urls = set()

    for img in soup.find_all("img"):
        # Tüm potansiyel src'leri topla
        candidates = []
        for attr in ["data-src", "data-lazy-src", "data-original", "data-full-src", "src"]:
            val = img.get(attr)
            if val:
                candidates.append(val)

        # srcset'ten en büyük versiyonu al
        srcset = img.get("srcset")
        if srcset:
            parts = []
            for entry in srcset.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                tokens = entry.split()
                if tokens:
                    url_part = tokens[0]
                    width = 0
                    if len(tokens) > 1 and tokens[1].endswith("w"):
                        try:
                            width = int(tokens[1][:-1])
                        except ValueError:
                            pass
                    parts.append((url_part, width))
            if parts:
                parts.sort(key=lambda x: x[1], reverse=True)
                candidates.insert(0, parts[0][0])  # En büyük versiyonu önce ekle

        alt_text = (img.get("alt") or "").strip()
        title_text = (img.get("title") or "").strip()
        css_class = " ".join(img.get("class", [])).lower()

        # CSS class filtreleme
        if any(kw in css_class for kw in ["avatar", "emoji", "icon", "logo", "social"]):
            continue

        # Alt text filtreleme
        if is_bad_alt(alt_text):
            continue

        for candidate in candidates:
            if not candidate or candidate.startswith("data:"):
                continue

            full_url = urljoin(page_url, candidate)

            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            if is_bad_url(full_url):
                continue

            # Sadece görsel uzantıları
            parsed_path = urlparse(full_url).path.lower()
            has_image_ext = any(parsed_path.endswith(ext) for ext in VALID_EXTENSIONS)
            # Bazı URL'lerde uzantı olmayabilir (CDN), bunları da deneyelim
            if not has_image_ext and not any(kw in parsed_path for kw in ["/image", "/photo", "/img", "/media", "/uploads"]):
                # Uzantısız ve görsel yolu değilse, atla
                if "." in parsed_path.split("/")[-1]:
                    continue

            results.append({
                "url": full_url,
                "alt": alt_text,
                "title": title_text,
                "note": "symptom photo" if alt_text else "needs manual review",
            })
            break  # Aynı img tag'ından sadece en iyi URL'yi al

    return results


def download_and_save(
    img_info: dict,
    disease_class: str,
    source_name: str,
    output_dir: Path,
    session: requests.Session,
    index: int,
) -> bool:
    """Görseli indir, doğrula ve kaydet."""
    url = img_info["url"]

    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=30, stream=True)
            r.raise_for_status()

            content_type = r.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                return False

            data = r.content
            valid, w, h = validate_image(data)
            if not valid:
                if w > 0:
                    print(f"  [KÜÇÜK/GEÇERSİZ] {w}x{h} - {url}")
                return False

            ext = get_image_extension(url, content_type)
            url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
            filename = f"{disease_class}_{source_name}_{url_hash}{ext}"
            filepath = output_dir / filename

            with open(filepath, "wb") as f:
                f.write(data)

            # Metadata JSON
            metadata = {
                "class": disease_class,
                "source_page": img_info.get("source_page", ""),
                "image_url": url,
                "source_site": source_name,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "dimensions": f"{w}x{h}",
                "file_size_kb": round(len(data) / 1024, 1),
                "alt_text": img_info.get("alt", ""),
                "notes": img_info.get("note", "needs manual review"),
            }

            meta_path = output_dir / f"{disease_class}_{source_name}_{url_hash}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"  [OK] {filename} ({w}x{h}, {metadata['file_size_kb']}KB)")
            return True

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            print(f"  [ERR] {url}: {e}")
            return False
        except Exception as e:
            print(f"  [ERR] {url}: {e}")
            return False

    return False


def main():
    root = Path(OUTPUT_ROOT)
    root.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    total_downloaded = 0
    total_skipped = 0
    class_stats = {}

    print("=" * 60)
    print("  DUCK & GOOSE HASTALIK GÖRSELİ İNDİRİCİ")
    print("=" * 60)
    print(f"  Çıktı klasörü: {root.absolute()}")
    print(f"  Sınıf sayısı:  {len(SOURCES)}")
    print(f"  Toplam URL:    {sum(len(v) for v in SOURCES.values())}")
    print("=" * 60)

    for disease_class, urls in SOURCES.items():
        print(f"\n{'─'*50}")
        print(f"📋 Sınıf: {disease_class} ({len(urls)} kaynak)")
        print(f"{'─'*50}")

        class_dir = root / disease_class
        class_dir.mkdir(exist_ok=True)

        class_downloaded = 0

        for page_url in urls:
            source_name = get_source_name(page_url)
            source_dir = class_dir / source_name
            source_dir.mkdir(exist_ok=True)

            print(f"\n  🌐 Sayfa: {page_url}")
            print(f"     Kaynak: {source_name}")

            # Görselleri bul
            image_infos = extract_image_urls(page_url, session)
            print(f"     Bulunan görsel: {len(image_infos)}")

            if not image_infos:
                continue

            # İndir
            for idx, img_info in enumerate(image_infos, start=1):
                img_info["source_page"] = page_url
                success = download_and_save(
                    img_info, disease_class, source_name,
                    source_dir, session, idx,
                )
                if success:
                    class_downloaded += 1
                    total_downloaded += 1
                else:
                    total_skipped += 1

                time.sleep(DELAY_SECONDS)

        class_stats[disease_class] = class_downloaded
        print(f"\n  ✅ {disease_class}: {class_downloaded} görsel indirildi")

    # ── Özet ──
    print(f"\n{'='*60}")
    print("  ÖZET")
    print(f"{'='*60}")
    for cls, count in class_stats.items():
        status = "✅" if count > 0 else "⚠️"
        print(f"  {status} {cls}: {count} görsel")
    print(f"\n  Toplam indirilen: {total_downloaded}")
    print(f"  Atlanan/Hatalı:   {total_skipped}")
    print(f"  Çıktı: {root.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
