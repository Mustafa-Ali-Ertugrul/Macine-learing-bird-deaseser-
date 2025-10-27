# poultry_bulk_downloader.py
import os, json, zipfile, requests, pandas as pd, tqdm, pathlib, shutil

# Kaggle modülünü opsiyonel hale getir
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    HAS_KAGGLE = True
except Exception:
    HAS_KAGGLE = False

OUT_DIR = "poultry_microscopy"
CSV_OUT = "poultry_labeled_12k.csv"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Kaggle Chicken Disease Classification (11 470 img, 6 sınıf)
def kaggle_chicken():
    if not HAS_KAGGLE:
        print("⚠️ Kaggle modülü bulunamadı, Kaggle verisi atlanıyor.")
        return []
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('berkayalan/comprehensive-chicken-disease-dataset',
                                   path=OUT_DIR, unzip=True)
        # klasik yapı: train/, test/ altında klasör adı = etiket
        rows = []
        for split in ('train','test'):
            for label_path in pathlib.Path(f"{OUT_DIR}/{split}").glob("*"):
                if not label_path.is_dir():
                    continue
                label = label_path.name          # 'Newcastle', 'Salmonella', ...
                for img in label_path.rglob("*.jpg"):
                    rows.append({'image_path': str(img),
                                 'disease': label.lower().replace(' ','_'),
                                 'tissue': 'viscera',      # makro görüntü
                                 'source': 'kaggle',
                                 'width': 512, 'height': 512})
        return rows
    except Exception as e:
        print(f"⚠️ Kaggle verisi indirilemedi: {e}. Atlanıyor.")
        return []

# 2) Zenodo 512×512 histopatoloji (820 img, 4 sınıf)
ZENODO_820 = "https://zenodo.org/records/7504927/files/poultry_histopath_512.zip"
def zenodo_820():
    try:
        zip_path = f"{OUT_DIR}/zenodo_512.zip"
        if not os.path.exists(zip_path):
            print("⬇️  Zenodo 820 indiriliyor...")
            r = requests.get(ZENODO_820, stream=True, timeout=30)
            r.raise_for_status()
            total = int(r.headers.get('content-length',0))
            with open(zip_path,'wb') as f, tqdm.tqdm(total=total,unit='B') as bar:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        with zipfile.ZipFile(zip_path,'r') as z:
            z.extractall(f"{OUT_DIR}/zenodo_512")
        rows = []
        for img in pathlib.Path(f"{OUT_DIR}/zenodo_512").rglob("*.png"):
            # dosya adı: class_001.png → class = {IB,ND,Healthy,IBD}
            cls = img.stem.split('_')[0].lower()
            rows.append({'image_path': str(img),
                         'disease': cls,
                         'tissue': 'mixed',   # her sınıf farklı doku
                         'source': 'zenodo',
                         'width': 512, 'height': 512})
        return rows
    except Exception as e:
        print(f"⚠️ Zenodo verisi indirilemedi: {e}. Atlanıyor.")
        return []

# 3) Mevcut Figshare verisini (194) oku
def load_prev():
    try:
        old = pd.read_csv('poultry_labeled.csv')
        old['source'] = 'figshare'
        return old.to_dict('records')
    except:
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
    df.to_csv(CSV_OUT, index=False)
    print("\n✅ Tamamlandı!")
    print(df['disease'].value_counts())
    print(f"Toplam görüntü: {len(df)} → {CSV_OUT}")