#!/usr/bin/env python3
"""
Yeni Eklenen Hastalıklar için Veri Seti İndirici
Newcastle Disease, Marek's Disease, Avian Influenza

Bu script özel kaynaklar ve veri setlerini bulur ve indirir.
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import time
from tqdm import tqdm
from PIL import Image
import io
import hashlib

# Konfigürasyon
CONFIG = {
    'output_dir': 'new_diseases_dataset',
    'metadata_csv': 'new_diseases_metadata.csv',
    'timeout': 60,
    'min_image_size': (512, 512),
    'max_retries': 3
}

# Yeni hastalıklar için özel kaynaklar
DISEASE_SOURCES = {
    'newcastle': {
        'keywords': ['newcastle disease', 'ndv', 'paramyxovirus', 'velogenic'],
        'pubmed_queries': [
            'newcastle disease chicken histopathology',
            'ndv poultry tissue microscopy',
            'paramyxovirus avian pathology'
        ],
        'kaggle_dataset': 'chandrashekarnatesh/poultry-diseases',  # Newcastle sınıfı içeriyor
        'description': 'Newcastle Disease - Viral disease causing respiratory, nervous, and digestive symptoms'
    },
    'marek': {
        'keywords': ["marek's disease", 'mdv', 'herpesvirus', 'lymphoid tumor'],
        'pubmed_queries': [
            'marek disease chicken histopathology',
            'mdv lymphoid tumor microscopy',
            'marek disease nerve lesion'
        ],
        'kaggle_dataset': None,  # Roboflow'dan bulunabilir
        'description': "Marek's Disease - Herpesvirus causing tumors and nerve damage"
    },
    'avian_influenza': {
        'keywords': ['avian influenza', 'bird flu', 'h5n1', 'h7n9', 'hpai'],
        'pubmed_queries': [
            'avian influenza h5n1 histopathology',
            'highly pathogenic avian influenza tissue',
            'bird flu chicken lung microscopy'
        ],
        'kaggle_dataset': None,
        'description': 'Avian Influenza - Highly contagious viral respiratory disease'
    }
}


class NewDiseaseDownloader:
    """Yeni hastalıklar için özel veri indirici"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.metadata = []
    
    def search_pubmed_central(self, query: str, max_results: int = 20) -> List[str]:
        """PubMed Central'da ara"""
        print(f"🔍 PubMed aranıyor: '{query}'")
        
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pmc',
            'term': f'{query} AND open access[filter] AND hasimages[text]',
            'retmax': max_results,
            'retmode': 'json'
        }
        
        try:
            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            pmc_ids = data.get('esearchresult', {}).get('idlist', [])
            print(f"✅ {len(pmc_ids)} makale bulundu")
            return pmc_ids
        except Exception as e:
            print(f"⚠️ PubMed arama hatası: {e}")
            return []
    
    def get_pmc_article_images(self, pmc_id: str) -> List[Dict]:
        """PMC makalesinden görüntüleri al"""
        article_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
        
        try:
            time.sleep(1)  # Rate limiting
            response = self.session.get(article_url, timeout=30)
            
            if response.status_code != 200:
                return []
            
            # HTML'den görüntü URL'lerini bul
            html = response.text
            images = []
            
            # Figure pattern'leri
            import re
            fig_pattern = r'/pmc/articles/PMC\d+/bin/([^"\']+\.(?:jpg|png|tif))'
            matches = re.findall(fig_pattern, html)
            
            for img_name in matches[:5]:  # Max 5 görüntü
                img_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/bin/{img_name}"
                images.append({
                    'url': img_url,
                    'pmc_id': pmc_id,
                    'filename': img_name
                })
            
            return images
            
        except Exception as e:
            print(f"⚠️ PMC{pmc_id} görüntü hatası: {str(e)[:50]}")
            return []
    
    def download_image(self, url: str, disease: str, metadata: Dict) -> Optional[Dict]:
        """Görüntü indir ve doğrula"""
        for attempt in range(CONFIG['max_retries']):
            try:
                time.sleep(0.5)  # Rate limiting
                
                response = self.session.get(url, timeout=CONFIG['timeout'], stream=True)
                
                if response.status_code != 200:
                    if attempt < CONFIG['max_retries'] - 1:
                        continue
                    return None
                
                # Görüntü kalite kontrolü
                img_data = response.content
                
                if len(img_data) < 10000:  # 10KB minimum
                    return None
                
                img = Image.open(io.BytesIO(img_data))
                
                if img.size[0] < CONFIG['min_image_size'][0] or img.size[1] < CONFIG['min_image_size'][1]:
                    return None
                
                # Dosya adı oluştur
                file_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"{disease}_{metadata.get('pmc_id', 'unknown')}_{file_hash}.jpg"
                
                # Hastalık klasörü oluştur
                disease_dir = self.output_dir / disease
                disease_dir.mkdir(exist_ok=True)
                
                filepath = disease_dir / filename
                
                # Kaydet
                img.convert('RGB').save(filepath, 'JPEG', quality=95)
                
                print(f"✓ İndirildi: {disease}/{filename}")
                
                return {
                    'image_path': str(filepath),
                    'filename': filename,
                    'disease': disease,
                    'source_url': url,
                    'pmc_id': metadata.get('pmc_id', 'unknown'),
                    'width': img.size[0],
                    'height': img.size[1],
                    'source': 'pubmed_central'
                }
                
            except Exception as e:
                if attempt == CONFIG['max_retries'] - 1:
                    pass  # Sessizce atla
                else:
                    time.sleep(2 ** attempt)
        
        return None
    
    def download_kaggle_dataset(self, dataset_name: str, disease: str) -> bool:
        """Kaggle veri setini indir (Manuel talimat)"""
        print(f"\n📦 Kaggle Veri Seti: {dataset_name}")
        print(f"💡 Manuel indirme gerekli:")
        print(f"   1. https://www.kaggle.com/datasets/{dataset_name} adresine gidin")
        print(f"   2. 'Download' butonuna tıklayın")
        print(f"   3. ZIP dosyasını {self.output_dir / disease} klasörüne çıkarın")
        print(f"   4. Sadece '{disease}' klasöründeki görüntüleri kullanın\n")
        return False


def main():
    """Ana veri toplama pipeline"""
    
    print("🐔 Yeni Hastalıklar için Veri Seti İndirici")
    print("=" * 60)
    print("Hastalıklar: Newcastle, Marek's Disease, Avian Influenza\n")
    
    downloader = NewDiseaseDownloader(CONFIG['output_dir'])
    all_metadata = []
    
    for disease, info in DISEASE_SOURCES.items():
        print("\n" + "=" * 60)
        print(f"📊 {disease.upper()}")
        print(f"📝 {info['description']}")
        print("=" * 60)
        
        # 1. Kaggle veri seti önerisi
        if info['kaggle_dataset']:
            downloader.download_kaggle_dataset(info['kaggle_dataset'], disease)
        
        # 2. PubMed Central'dan indir
        print(f"\n🔬 PubMed Central'dan aranıyor...")
        
        for query in info['pubmed_queries'][:2]:  # İlk 2 sorgu
            pmc_ids = downloader.search_pubmed_central(query, max_results=10)
            
            if not pmc_ids:
                continue
            
            print(f"\n⬇️ {len(pmc_ids)} makaleden görüntüler indiriliyor...")
            
            for pmc_id in tqdm(pmc_ids[:5], desc=f"{disease}"):  # Max 5 makale
                images = downloader.get_pmc_article_images(pmc_id)
                
                for img_data in images:
                    result = downloader.download_image(
                        img_data['url'],
                        disease,
                        {'pmc_id': pmc_id}
                    )
                    if result:
                        all_metadata.append(result)
            
            time.sleep(2)  # Sorgular arası bekleme
    
    # Sonuçları kaydet
    if all_metadata:
        df = pd.DataFrame(all_metadata)
        csv_path = Path(CONFIG['output_dir']) / CONFIG['metadata_csv']
        df.to_csv(csv_path, index=False)
        
        print("\n" + "=" * 60)
        print("✅ TAMAMLANDI!")
        print("=" * 60)
        print(f"📊 Toplam indirilen görüntü: {len(df)}")
        print(f"💾 Metadata: {csv_path}")
        print(f"\n📁 Hastalık dağılımı:")
        print(df['disease'].value_counts())
        
        # Her hastalık için özet
        for disease in df['disease'].unique():
            disease_count = len(df[df['disease'] == disease])
            print(f"\n   {disease}: {disease_count} görüntü")
            print(f"   Klasör: {CONFIG['output_dir']}/{disease}/")
    else:
        print("\n⚠️ PubMed'den hiç görüntü indirilemedi!")
    
    # Manuel indirme önerileri
    print("\n" + "=" * 60)
    print("💡 EK KAYNAKLAR - Manuel İndirme Önerileri")
    print("=" * 60)
    
    print("\n📦 KAGGLE:")
    print("   1. Newcastle Disease: https://www.kaggle.com/datasets/chandrashekarnatesh/poultry-diseases")
    print("   2. Poultry Diseases: https://www.kaggle.com/datasets?search=poultry+disease")
    
    print("\n🔬 PUBMED CENTRAL:")
    print("   - Newcastle: https://www.ncbi.nlm.nih.gov/pmc/?term=newcastle+disease+chicken+histopathology")
    print("   - Marek: https://www.ncbi.nlm.nih.gov/pmc/?term=marek+disease+histopathology")
    print("   - Avian Flu: https://www.ncbi.nlm.nih.gov/pmc/?term=avian+influenza+h5n1+pathology")
    
    print("\n🌐 FIGSHARE:")
    print("   - https://figshare.com/search?q=newcastle%20disease%20chicken")
    print("   - https://figshare.com/search?q=marek%20disease%20poultry")
    print("   - https://figshare.com/search?q=avian%20influenza%20pathology")
    
    print("\n🎓 ZENODO:")
    print("   - https://zenodo.org/search?q=newcastle%20disease%20poultry")
    print("   - https://zenodo.org/search?q=marek%20disease")
    
    print("\n🤖 ROBOFLOW:")
    print("   - https://universe.roboflow.com/search?q=chicken%20disease")
    print("   - https://universe.roboflow.com/goat-dataset/hen-and-its-diseases")


if __name__ == '__main__':
    # Gerekli kütüphaneleri kontrol et
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
        print(f"Yüklemek için: python -m pip install {' '.join(missing)}")
    else:
        main()
