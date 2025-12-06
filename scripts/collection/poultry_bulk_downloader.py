import os
import zipfile
import requests
import pandas as pd
from tqdm import tqdm

# --------------------------
# 🔧 KLASÖR YAPISI
# --------------------------
DATA_DIR = "data"
FECAL_DIR = os.path.join(DATA_DIR, "zenodo_fecal")
PCR_DIR = os.path.join(DATA_DIR, "zenodo_pcr")
os.makedirs(FECAL_DIR, exist_ok=True)
os.makedirs(PCR_DIR, exist_ok=True)

# --------------------------
# 🔗 VERİSETİ LİNKLERİ
# --------------------------
fecal_links = {
    "cocci": "https://zenodo.org/records/4628934/files/cocci.zip?download=1",
    "healthy": "https://zenodo.org/records/4628934/files/healthy.zip?download=1",
    "ncd": "https://zenodo.org/records/4628934/files/ncd.zip?download=1",
    "salmo": "https://zenodo.org/records/4628934/files/salmo.zip?download=1",
}

pcr_links = {
    "pcrcocci": "https://zenodo.org/records/5801834/files/pcrcocci.zip?download=1",
    "pcrhealthy": "https://zenodo.org/records/5801834/files/pcrhealthy.zip?download=1",
    "pcrncd": "https://zenodo.org/records/5801834/files/pcrncd.zip?download=1",
    "pcrsalmo": "https://zenodo.org/records/5801834/files/pcrsalmo.zip?download=1",
}

# --------------------------
# 📥 İNDİRME FONKSİYONU
# --------------------------
def download_and_extract(url, dest_dir):
    local_zip = os.path.join(dest_dir, os.path.basename(url).split("?")[0])
    if not os.path.exists(local_zip):
        print(f"⬇️ Downloading {url} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(local_zip, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=os.path.basename(local_zip)
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
    else:
        print(f"✅ Already downloaded: {local_zip}")

    print(f"📦 Extracting {local_zip} ...")
    with zipfile.ZipFile(local_zip, "r") as zip_ref:
        zip_ref.extractall(dest_dir)

# --------------------------
# 🔁 TÜM DOSYALARI İNDİR
# --------------------------
for label, url in fecal_links.items():
    download_and_extract(url, os.path.join(FECAL_DIR, label))

for label, url in pcr_links.items():
    download_and_extract(url, os.path.join(PCR_DIR, label))

# --------------------------
# 🧾 CSV OLUŞTUR
# --------------------------
records = []
for base_dir, source, tissue in [
    (FECAL_DIR, "zenodo", "feces"),
    (PCR_DIR, "zenodo", "feces_pcr"),
]:
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                label = os.path.basename(root)
                img_path = os.path.join(root, f)
                records.append((img_path, label, source, tissue))

df = pd.DataFrame(records, columns=["image_path", "label", "source", "tissue"])
df.to_csv("poultry_labeled_12k.csv", index=False)
print(f"✅ Dataset saved to poultry_labeled_12k.csv with {len(df)} entries.")

# Kaggle modülünü opsiyonel hale getiriyoruz; varsa kullanacağız
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    HAS_KAGGLE = True
except Exception:
    HAS_KAGGLE = False

OUT_DIR = "poultry_microscopy"
CSV_OUT = "poultry_labeled_12k.csv"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Kaggle veri setleri (birden fazla kaynak desteklenir)
KAGGLE_DATASETS = [
    # Kullanıcı tarafından belirtilen aktif Kaggle veri setleri
    'allandclive/chicken-disease-1',
    'efoeetienneblavo/chicken-disease-dataset',
    'kausthubkannan/poultry-diseases-detection',
]

def _infer_label_from_path(img_path: pathlib.Path, base_dir: pathlib.Path):
    """Görüntü yolundan sınıf etiketini çıkar (train/test/val kalıplarını destekler)."""
    try:
        rel_parts = img_path.relative_to(base_dir).parts
    except Exception:
        # base_dir tahmin edilemedi, üst klasör adını kullan
        return img_path.parent.name.lower().replace(' ', '_')

    # Olabilecek bölümler: ['train', '<label>', ...] veya ['test', '<label>', ...]
    for split_name in ('train', 'test', 'validation', 'val'):
        if split_name in rel_parts:
            idx = rel_parts.index(split_name)
            if idx + 1 < len(rel_parts):
                return rel_parts[idx + 1].lower().replace(' ', '_')
    # Aksi halde ilk klasör veya üst klasör adı etiket kabul edilir
    if len(rel_parts) >= 2:
        return rel_parts[0].lower().replace(' ', '_')
    return img_path.parent.name.lower().replace(' ', '_')

def kaggle_chicken():
    if not HAS_KAGGLE:
        print("⚠️ Kaggle modülü bulunamadı, Kaggle verisi atlanıyor.")
        return []
    rows = []
    try:
        api = KaggleApi()
        api.authenticate()
        kaggle_root = pathlib.Path(OUT_DIR) / "kaggle"
        kaggle_root.mkdir(parents=True, exist_ok=True)

        for slug in KAGGLE_DATASETS:
            print(f"⬇️  Kaggle indiriliyor: {slug}")
            dest_dir = kaggle_root / slug.replace('/', '_')
            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                api.dataset_download_files(slug, path=str(dest_dir), unzip=True)
            except Exception as e:
                print(f"⚠️ {slug} indirilemedi: {e}. Atlanıyor.")
                continue

            # Görüntüleri tara (jpg/png/tif)
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif'):
                for img in dest_dir.rglob(ext):
                    label = _infer_label_from_path(img, dest_dir)
                    rows.append({
                        'image_path': str(img),
                        'disease': label,
                        'tissue': 'viscera',
                        'source': 'kaggle',
                        'width': 512,
                        'height': 512,
                    })
        print(f"Kaggle eklendi: {len(rows)} kayıt")
        return rows
    except Exception as e:
        print(f"⚠️ Kaggle verisi indirilemedi: {e}. Atlanıyor.")
        return []

# 2) Zenodo 512×512 histopatoloji (820 img, 4 sınıf)
# Not: eski 'record' URL 404 döndürüyor; yeni 'records' + ?download=1 kullanılmalı
ZENODO_820 = "https://zenodo.org/records/7504927/files/poultry_histopath_512.zip?download=1"

def zenodo_820():
    try:
        zip_path = f"{OUT_DIR}/zenodo_512.zip"
        extract_dir = f"{OUT_DIR}/zenodo_512"
        if not os.path.exists(zip_path):
            print("⬇️  Zenodo 820 indiriliyor...")
            r = requests.get(ZENODO_820, stream=True, timeout=60)
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm.tqdm(total=total, unit='B') as bar:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        rows = []
        for img in pathlib.Path(extract_dir).rglob("*.png"):
            # dosya adı: class_001.png → class = {IB,ND,Healthy,IBD}
            cls = img.stem.split('_')[0].lower()
            rows.append({
                'image_path': str(img),
                'disease': cls,
                'tissue': 'mixed',  # her sınıf farklı doku
                'source': 'zenodo',
                'width': 512,
                'height': 512,
            })
        print(f"Zenodo eklendi: {len(rows)} kayıt")
        return rows
    except Exception as e:
        print(f"⚠️ Zenodo verisi indirilemedi: {e}. Atlanıyor.")
        return []

# 3) Mevcut Figshare verisini oku
def load_prev():
    try:
        old = pd.read_csv('poultry_labeled.csv')
        old['source'] = 'figshare'
        rows = old.to_dict('records')
        print(f"Figshare eklendi: {len(rows)} kayıt")
        return rows
    except Exception:
        print("⚠️ Mevcut Figshare CSV bulunamadı (poultry_labeled.csv). Atlanıyor.")
        return []

# 4) Hepsini birleştir
if __name__ == '__main__':
    all_rows = []
    print("1) Kaggle indiriliyor...")
    all_rows += kaggle_chicken()
    print("2) Zenodo indiriliyor...")
    all_rows += zenodo_820()
    print("3) Önceki Figshare ekleniyor...")
    all_rows += load_prev()

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        print("⚠️ Hiç kayıt toplanamadı. Kaggle kurulumu ve Zenodo linkini kontrol edin.")
    df.to_csv(CSV_OUT, index=False)
    print("\n✅ Tamamlandı!")
    if 'disease' in df.columns:
        print(df['disease'].value_counts())
    print(f"Toplam görüntü: {len(df)} → {CSV_OUT}")