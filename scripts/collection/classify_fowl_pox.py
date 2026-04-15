"""
general/Fowl_Pox içindeki görselleri inceler:
  - Gerçek avian pox görselleri ayıklanır
  - İlgisiz fact sheet thumbnail'ları kaldırılır
  - Tür tespiti yapılır (duck/goose/unknown)
"""

import json
import shutil
import os
from pathlib import Path
from collections import defaultdict

SOURCE_DIR = Path("dataset/general/Fowl_Pox")
DATASET_DIR = Path("dataset")

# Avian pox ile İLGİLİ anahtar kelimeler (URL veya alt text'te)
AVIAN_POX_KEYWORDS = [
    "avian-pox", "avian_pox", "fowl-pox", "fowl_pox",
    "pox", "poxvirus",
]

# Kesinlikle İLGİSİZ fact sheet konuları
UNRELATED_KEYWORDS = [
    "rabies", "salmonella", "botulism", "tularemia", "cwd",
    "lead", "mercury", "pesticide", "rodenticide", "pcb",
    "mange", "heartworm", "brucellosis", "neosporosis",
    "toxoplasmosis", "winter-tick", "rhdv", "bsal", "parvo",
    "sarcocyst", "ranavirus", "echinococcosis", "bordetella",
    "morbillivirus", "pigeon", "fibromas", "abscess", "ahd",
    "aspergillosis", "avulavir", "lpdv", "hd-deer", "sfd",
    "eee", "myco-conj", "molting", "nws", "lept", "cdv",
    "p-tenuis", "liver-fluke", "trichomonosis", "bdv", "bd_",
    "sphaeridiotrema", "adenovirus", "organophosphate",
    "neonicotinoid", "baylisascaris", "baylis", "pasteurella",
    "reovirus", "reo_", "dermatophilosis", "wnv", "wns",
    "shutterstock", "deer",
]

# Duck tespiti
DUCK_URL_KEYWORDS = ["duck", "ördek"]
# Goose tespiti
GOOSE_URL_KEYWORDS = ["goose", "kaz", "gosling"]


def is_avian_pox_related(metadata: dict) -> bool:
    """Görselin gerçekten avian pox ile ilgili olup olmadığını kontrol et."""
    alt = (metadata.get("alt_text") or "").lower()
    url = (metadata.get("image_url") or "").lower()
    combined = f"{alt} {url}"

    # Önce ilgisiz olanları ele
    for kw in UNRELATED_KEYWORDS:
        if kw in combined:
            return False

    # Avian pox ile ilgili mi?
    for kw in AVIAN_POX_KEYWORDS:
        if kw in combined:
            return True

    # Generic "fact sheet" thumbnail'ı ama pox değilse ilgisiz
    if "fact sheet" in alt or "thumbnail" in alt:
        return False

    # Belirlenemedi, conservative olarak ilgisiz say
    return False


def detect_species_from_metadata(metadata: dict) -> str:
    """Metadata'dan tür tespiti."""
    alt = (metadata.get("alt_text") or "").lower()
    url = (metadata.get("image_url") or "").lower()
    source = (metadata.get("source_page") or "").lower()
    combined = f"{alt} {url} {source}"

    duck_score = sum(1 for kw in DUCK_URL_KEYWORDS if kw in combined)
    goose_score = sum(1 for kw in GOOSE_URL_KEYWORDS if kw in combined)

    if duck_score > goose_score:
        return "duck"
    if goose_score > duck_score:
        return "goose"

    return "unknown"


def main():
    if not SOURCE_DIR.exists():
        print(f"[HATA] Kaynak klasör bulunamadı: {SOURCE_DIR}")
        return

    # Tüm görsel + metadata çiftlerini bul
    images = sorted(SOURCE_DIR.glob("*.jpg")) + sorted(SOURCE_DIR.glob("*.png")) + sorted(SOURCE_DIR.glob("*.webp"))

    print("=" * 60)
    print("  FOWL_POX GÖRSEL SINIFLANDIRMA")
    print("=" * 60)
    print(f"  Toplam görsel: {len(images)}")
    print()

    kept_pox = []
    removed_unrelated = []
    species_counts = defaultdict(int)

    for img_path in images:
        json_path = img_path.with_suffix(".json")
        metadata = {}
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        alt = (metadata.get("alt_text") or "bilinmiyor")
        url = (metadata.get("image_url") or "")

        if is_avian_pox_related(metadata):
            species = detect_species_from_metadata(metadata)
            kept_pox.append({
                "path": img_path,
                "json_path": json_path,
                "species": species,
                "alt": alt,
                "url": url,
            })
            print(f"  [POX ✅] {img_path.name} | tür: {species} | alt: {alt[:50]}")
        else:
            removed_unrelated.append({
                "path": img_path,
                "json_path": json_path,
                "alt": alt,
                "url": url,
            })
            print(f"  [İLGİSİZ ❌] {img_path.name} | alt: {alt[:50]}")

    print(f"\n{'─'*60}")
    print(f"  Avian Pox ile ilgili: {len(kept_pox)}")
    print(f"  İlgisiz (silinecek): {len(removed_unrelated)}")

    # İlgisiz görselleri sil
    removed_dir = DATASET_DIR / "removed_unrelated"
    removed_dir.mkdir(exist_ok=True)

    for item in removed_unrelated:
        # Yedek olarak removed_unrelated'a taşı
        dest = removed_dir / item["path"].name
        shutil.move(str(item["path"]), str(dest))
        if item["json_path"].exists():
            shutil.move(str(item["json_path"]), str(removed_dir / item["json_path"].name))

    print(f"\n  İlgisiz görseller -> {removed_dir}/")

    # Avian pox görsellerini tür bazlı taşı
    for item in kept_pox:
        species = item["species"]

        if species in ("duck", "goose"):
            dest_dir = DATASET_DIR / species / "Fowl_Pox"
        else:
            # unknown -> general'da kal
            dest_dir = SOURCE_DIR  # zaten burada
            species_counts["general/Fowl_Pox"] += 1
            # Metadata güncelle
            if item["json_path"].exists():
                with open(item["json_path"], "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["species_detected"] = "unknown"
                with open(item["json_path"], "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / item["path"].name

        if not dest_path.exists():
            shutil.move(str(item["path"]), str(dest_path))
            species_counts[f"{species}/Fowl_Pox"] += 1

            if item["json_path"].exists():
                with open(item["json_path"], "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["species_detected"] = species
                dest_json = dest_path.with_suffix(".json")
                with open(dest_json, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
                item["json_path"].unlink()

            print(f"  [TAŞINDI] {item['path'].name} -> {species}/Fowl_Pox/")

    # Temp dosyayı temizle
    meta_dump = SOURCE_DIR / "_meta_dump.txt"
    if meta_dump.exists():
        meta_dump.unlink()

    # Rapor
    print(f"\n{'='*60}")
    print("  SONUÇ")
    print(f"{'='*60}")
    for key, count in sorted(species_counts.items()):
        print(f"  {key}: {count}")

    remaining_general = len(list(SOURCE_DIR.glob("*.jpg"))) + len(list(SOURCE_DIR.glob("*.png")))
    print(f"\n  duck/Fowl_Pox: {species_counts.get('duck/Fowl_Pox', 0)}")
    print(f"  goose/Fowl_Pox: {species_counts.get('goose/Fowl_Pox', 0)}")
    print(f"  general/Fowl_Pox: {remaining_general}")
    print(f"  İlgisiz silinen: {len(removed_unrelated)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
