# auto_label.py
import pandas as pd, re, json, pathlib, shutil

df = pd.read_csv('poultry_dataset.csv')

# Anahtar kelime -> etiket haritaları
# Not: Hastalık etiketleri HTML aracıyla uyumlu olacak şekilde küçük harf yapılmıştır.
DISEASE_MAP = {
    r'\binfectious bronchitis\b|\bibv\b|\bib\b': 'ib',
    r'\binfectious bursal disease\b|\bibd\b': 'ibd',
    r'\bcoccidiosis\b|\beimeria\b': 'coccidiosis',
    r'\bsalmonella\b': 'salmonella',
    r'\bfatty liver\b|\bhepatosis\b': 'fatty_liver',
    r'\bhistomoniasis\b': 'histomoniasis',
    r'\bnewcastle\b|\bndv\b|\bnewcastle disease\b': 'newcastle',
    r'\bmarek\b|\bmarek\'s\b|\bmd\b': 'marek',
    r'\bavian influenza\b|\bavian flu\b|\bai\b|\bhpai\b|\blpai\b': 'avian_influenza',
    r'\bnegative control\b|\bnc\b|\bcontrol\b|\bhealthy\b': 'healthy'
}

TISSUE_MAP = {
    r'\btrachea\b': 'trachea',
    r'\bbursa\b|\bfabricius\b': 'bursa',
    r'\bliver\b|\bhepatic\b': 'liver',
    r'\bkidney\b|\brenal\b': 'kidney',
    r'\bspleen\b': 'spleen',
    r'\bthymus\b': 'thymus',
    r'\blung\b|\bpulmonary\b': 'lung',
    r'\bpancreas\b': 'pancreas',
    r'\bhair follicle\b': 'hair_follicle',
    r'\bstomach\b|\bgizzard\b': 'stomach',
    r'\bintestine\b|\bcecum\b': 'intestine',
    r'\bhead\b': 'head',
    r'\bback\b': 'back'
}


def extract(txt):
    txt = txt.lower()
    # Hastalık çıkarımı: İlk eşleşen kural kullanılır, yoksa 'unknown'
    disease = next((v for k, v in DISEASE_MAP.items() if re.search(k, txt)), 'unknown')
    # Doku çıkarımı: İlk eşleşen kural kullanılır, yoksa 'unknown'
    tissue = next((v for k, v in TISSUE_MAP.items() if re.search(k, txt)), 'unknown')
    # Ek kural: dosya adında 'nc' varsa ve disease hâlâ unknown ise healthy yap
    if disease == 'unknown' and re.search(r'\bnc\b', txt):
        disease = 'healthy'
    return disease, tissue


def build_text(row: pd.Series) -> str:
    """Satırdan uygun metni birleştir: title yoksa filename/image_path/source/url ile devam et."""
    fields = ['title', 'filename', 'image_path', 'source', 'url']
    parts = []
    for f in fields:
        # Series.get mevcut değilse None döndürür; NaN değerleri atla
        val = row.get(f, None)
        if val is not None and pd.notna(val):
            parts.append(str(val))
    return ' '.join(parts)


# Etiket çıkarımı uygula (title kolonu şart değil)
df[['disease', 'tissue']] = df.apply(
    lambda r: extract(build_text(r)), axis=1, result_type='expand'
)

# Özet yazdır
print('Disease distribution:')
print(df['disease'].value_counts(dropna=False))
print('\nTissue distribution:')
print(df['tissue'].value_counts(dropna=False))

# Sonucu diske yaz
df.to_csv('poultry_labeled.csv', index=False)