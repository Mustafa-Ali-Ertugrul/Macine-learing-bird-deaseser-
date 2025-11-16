#!/usr/bin/env python3
"""
Figshare ve Zenodo'dan Kanatlı Patoloji Veri Setleri İndirme
Rate limit sorunu olmayan, garantili indirme sistemi
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import zipfile
import io
from tqdm import tqdm
import hashlib
import json
import time
from PIL import Image

# Konfigürasyon
CONFIG = {
    'output_dir': 'poultry_microscopy',
    'metadata_csv': 'poultry_dataset.csv',
    'min_image_size': (224, 224),
    'timeout': 60,
    'chunk_size': 8192,
}


class FigshareDownloader:
    """Figshare API ile veri seti indirme"""
    
    BASE_URL = 'https://api.figshare.com/v2'
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research/Educational Use)'
        })
    
    def search_datasets(self, query: str, limit: int = 20) -> List[Dict]:
        """Figshare'de veri seti ara"""
        print(f"🔍 Figshare aranıyor: '{query}'")
        
        search_url = f"{self.BASE_URL}/articles/search"
        params = {
            'search_for': query,
            'item_type': 3,  # Dataset
            'page_size': limit
        }
        
        try:
            response = self.session.post(search_url, json=params, timeout=30)
            response.raise_for_status()
            results = response.json()
            
            print(f"✅ {len(results)} veri seti bulundu")
            return results
        except Exception as e:
            print(f"⚠️ Figshare arama hatası: {e}")
            return []
    
    def get_dataset_files(self, article_id: int) -> List[Dict]:
        """Veri setindeki dosyaları al"""
        url = f"{self.BASE_URL}/articles/{article_id}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return [{
                'name': f['name'],
                'size': f['size'],
                'download_url': f['download_url'],
                'id': f['id']
            } for f in data.get('files', [])]
        except Exception as e:
            print(f"⚠️ Dosya listesi alınamadı: {e}")
            return []
    
    def download_file(self, url: str, output_path: Path, desc: str = "İndiriliyor") -> bool:
        """Dosyayı indir (progress bar ile)"""
        try:
            response = self.session.get(url, stream=True, timeout=CONFIG['timeout'])
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=CONFIG['chunk_size']):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
        except Exception as e:
            print(f"❌ İndirme hatası: {e}")
            return False


class ZenodoDownloader:
    """Zenodo API ile veri seti indirme"""
    
    BASE_URL = 'https://zenodo.org/api'
    
    def __init__(self):
        self.session = requests.Session()
    
    def search_datasets(self, query: str, limit: int = 20) -> List[Dict]:
        """Zenodo'da veri seti ara"""
        print(f"🔍 Zenodo aranıyor: '{query}'")
        
        search_url = f"{self.BASE_URL}/records"
        params = {
            'q': query,
            'type': 'dataset',
            'size': limit
        }
        
        try:
            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = data.get('hits', {}).get('hits', [])
            print(f"✅ {len(results)} veri seti bulundu")
            return results
        except Exception as e:
            print(f"⚠️ Zenodo arama hatası: {e}")
            return []
    
    def get_dataset_files(self, record_id: str) -> List[Dict]:
        """Veri setindeki dosyaları al"""
        url = f"{self.BASE_URL}/records/{record_id}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return [{
                'name': f['key'],
                'size': f['size'],
                'download_url': f['links']['self'],
                'checksum': f.get('checksum', '')
            } for f in data.get('files', [])]
        except Exception as e:
            print(f"⚠️ Dosya listesi alınamadı: {e}")
            return []
    
    def download_file(self, url: str, output_path: Path, desc: str = "İndiriliyor") -> bool:
        """Dosyayı indir"""
        try:
            response = self.session.get(url, stream=True, timeout=CONFIG['timeout'])
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=CONFIG['chunk_size']):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
        except Exception as e:
            print(f"❌ İndirme hatası: {e}")
            return False


class ImageProcessor:
    """İndirilen görüntüleri işle ve organize et"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_zip(self, zip_path: Path) -> List[Path]:
        """ZIP dosyasını aç ve görüntüleri çıkar"""
        print(f"📦 ZIP açılıyor: {zip_path.name}")
        extracted = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if self._is_image_file(member):
                        # Organize klasör yapısı
                        target_path = self.output_dir / Path(member).name
                        
                        # Dosyayı çıkar
                        with zip_ref.open(member) as source:
                            with open(target_path, 'wb') as target:
                                target.write(source.read())
                        
                        extracted.append(target_path)
            
            print(f"✅ {len(extracted)} görüntü çıkarıldı")
            return extracted
            
        except Exception as e:
            print(f"⚠️ ZIP açma hatası: {e}")
            return []
    
    def _is_image_file(self, filename: str) -> bool:
        """Dosya görüntü mü kontrol et"""
        ext = Path(filename).suffix.lower()
        return ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    
    def validate_and_resize(self, image_path: Path) -> Optional[Dict]:
        """Görüntüyü kontrol et ve gerekirse resize yap"""
        try:
            img = Image.open(image_path)
            
            # Minimum boyut kontrolü
            if img.size[0] < CONFIG['min_image_size'][0] or img.size[1] < CONFIG['min_image_size'][1]:
                return None
            
            # RGB'ye çevir
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize (max 2048x2048)
            if img.size[0] > 2048 or img.size[1] > 2048:
                img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                img.save(image_path, 'JPEG', quality=95)
            
            return {
                'image_path': str(image_path),
                'width': img.size[0],
                'height': img.size[1],
                'format': img.format,
                'filename': image_path.name
            }
            
        except Exception as e:
            print(f"⚠️ Görüntü işleme hatası ({image_path.name}): {e}")
            return None
    
    def create_metadata(self, image_infos: List[Dict], dataset_info: Dict) -> pd.DataFrame:
        """Metadata CSV oluştur"""
        df = pd.DataFrame(image_infos)
        
        # Dataset bilgilerini ekle
        df['dataset_source'] = dataset_info.get('source', 'unknown')
        df['dataset_title'] = dataset_info.get('title', 'unknown')
        df['dataset_url'] = dataset_info.get('url', '')
        
        # Dosya adından hastalık bilgisi çıkarmayı dene
        df['disease_hint'] = df['filename'].apply(self._extract_disease_hint)
        
        return df
    
    def _extract_disease_hint(self, filename: str) -> str:
        """Dosya adından hastalık ipucu çıkar"""
        fname_lower = filename.lower()
        
        disease_keywords = {
            'ib': ['ib', 'bronchitis', 'respiratory'],
            'ibd': ['ibd', 'bursa', 'gumboro'],
            'nd': ['nd', 'newcastle'],
            'coccidiosis': ['cocci', 'eimeria'],
            'salmonella': ['salmonella'],
            'fatty_liver': ['fatty', 'liver', 'hepatic'],
            'histomoniasis': ['histomon', 'blackhead'],
            'newcastle': ['newcastle', 'ndv', 'paramyxovirus'],
            'marek': ['marek', 'mdv', 'herpes'],
            'avian_influenza': ['influenza', 'flu', 'h5n1', 'h7n9', 'hpai', 'lpai'],
            'healthy': ['healthy', 'normal', 'control']
        }
        
        for disease, keywords in disease_keywords.items():
            if any(kw in fname_lower for kw in keywords):
                return disease
        
        return 'unknown'


def main():
    """Ana veri toplama pipeline"""
    
    print("🐔 Kanatlı Patoloji Veri Toplama (Figshare + Zenodo)\n")
    
    # Arama sorguları
    queries = [
        'chicken histopathology',
        'poultry disease microscopy',
        'avian pathology images',
        'broiler tissue microscopy',
        'newcastle disease poultry',
        'marek disease chicken',
        'avian influenza histology'
    ]
    
    # Downloaders
    figshare = FigshareDownloader()
    zenodo = ZenodoDownloader()
    processor = ImageProcessor(CONFIG['output_dir'])
    
    all_image_info = []
    
    # 1. Figshare'den ara ve indir
    print("\n" + "="*60)
    print("📚 FIGSHARE VERİ SETLERİ")
    print("="*60)
    
    for query in queries[:2]:  # İlk 2 sorgu
        datasets = figshare.search_datasets(query, limit=10)
        
        for idx, dataset in enumerate(datasets[:3], 1):  # İlk 3 sonuç
            print(f"\n📦 Dataset {idx}: {dataset['title'][:70]}...")
            print(f"   👤 Yazar: {dataset['authors'][0]['full_name'] if dataset.get('authors') else 'N/A'}")
            print(f"   📏 Boyut: {dataset.get('size', 0) / 1024 / 1024:.1f} MB")
            
            # Dosyaları al
            files = figshare.get_dataset_files(dataset['id'])
            
            for file in files:
                if file['name'].endswith('.zip') or processor._is_image_file(file['name']):
                    print(f"   ⬇️ İndiriliyor: {file['name']}")
                    
                    output_path = Path(CONFIG['output_dir']) / 'downloads' / file['name']
                    
                    if figshare.download_file(file['download_url'], output_path, file['name']):
                        # ZIP ise aç
                        if file['name'].endswith('.zip'):
                            images = processor.extract_zip(output_path)
                            
                            # Her görüntüyü işle
                            for img_path in images:
                                info = processor.validate_and_resize(img_path)
                                if info:
                                    info.update({
                                        'source': 'figshare',
                                        'title': dataset['title'],
                                        'url': dataset['url']
                                    })
                                    all_image_info.append(info)
                        else:
                            # Tekil görüntü
                            info = processor.validate_and_resize(output_path)
                            if info:
                                info.update({
                                    'source': 'figshare',
                                    'title': dataset['title'],
                                    'url': dataset['url']
                                })
                                all_image_info.append(info)
        
        time.sleep(2)  # Rate limiting
    
    # 2. Zenodo'dan ara ve indir
    print("\n" + "="*60)
    print("🌐 ZENODO VERİ SETLERİ")
    print("="*60)
    
    for query in queries[2:]:  # Son 2 sorgu
        datasets = zenodo.search_datasets(query, limit=10)
        
        for idx, dataset in enumerate(datasets[:2], 1):
            metadata = dataset.get('metadata', {})
            print(f"\n📦 Dataset {idx}: {metadata.get('title', 'N/A')[:70]}...")
            
            record_id = dataset['id']
            files = zenodo.get_dataset_files(record_id)
            
            for file in files[:5]:  # Max 5 dosya per dataset
                if file['name'].endswith('.zip') or processor._is_image_file(file['name']):
                    print(f"   ⬇️ İndiriliyor: {file['name']}")
                    
                    output_path = Path(CONFIG['output_dir']) / 'downloads' / file['name']
                    
                    if zenodo.download_file(file['download_url'], output_path, file['name']):
                        if file['name'].endswith('.zip'):
                            images = processor.extract_zip(output_path)
                            
                            for img_path in images:
                                info = processor.validate_and_resize(img_path)
                                if info:
                                    info.update({
                                        'source': 'zenodo',
                                        'title': metadata.get('title', 'N/A'),
                                        'url': dataset.get('links', {}).get('html', '')
                                    })
                                    all_image_info.append(info)
                        else:
                            info = processor.validate_and_resize(output_path)
                            if info:
                                info.update({
                                    'source': 'zenodo',
                                    'title': metadata.get('title', 'N/A'),
                                    'url': dataset.get('links', {}).get('html', '')
                                })
                                all_image_info.append(info)
        
        time.sleep(2)
    
    # 3. Metadata kaydet
    if all_image_info:
        df = pd.DataFrame(all_image_info)
        df.to_csv(CONFIG['metadata_csv'], index=False)
        
        print("\n" + "="*60)
        print("✅ TAMAMLANDI!")
        print("="*60)
        print(f"📊 Toplam görüntü: {len(df)}")
        print(f"💾 Metadata: {CONFIG['metadata_csv']}")
        print(f"📁 Görüntüler: {CONFIG['output_dir']}/")
        
        if 'disease_hint' in df.columns:
            print(f"\n🔬 Hastalık dağılımı (tahmini):")
            print(df['disease_hint'].value_counts())
        
        print(f"\n📏 Kaynak dağılımı:")
        print(df['source'].value_counts())
    else:
        print("\n⚠️ Hiç görüntü indirilemedi!")
        print("💡 Manuel indirme önerileri:")
        print("   1. https://figshare.com/search?q=chicken%20histopathology")
        print("   2. https://zenodo.org/search?q=poultry%20pathology")
        print("   3. https://www.kaggle.com/datasets?search=chicken+disease")


if __name__ == '__main__':
    # Pip paketi → modül adı eşlemesi (özellikle Pillow → PIL)
    required = {
        'requests': 'requests',
        'pandas': 'pandas',
        'tqdm': 'tqdm',
        'Pillow': 'PIL'
    }
    missing = []
    
    for pip_name, module_name in required.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
    
    if missing:
        print(f"⚠️ Eksik kütüphaneler: {', '.join(missing)}")
        print(f"Yüklemek için: pip install {' '.join(missing)}")
    else:
        main()