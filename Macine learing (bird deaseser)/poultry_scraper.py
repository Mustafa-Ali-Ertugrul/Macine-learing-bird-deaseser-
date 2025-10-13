#!/usr/bin/env python3
"""
Kanatlı Patoloji Histopatoloji Görüntü Toplama Sistemi v2.0
403 hatası düzeltmesi + Alternatif kaynaklar
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, quote
import re
from tqdm import tqdm
import hashlib
from PIL import Image
import io
import json

# Konfigürasyon
CONFIG = {
    'output_dir': 'poultry_microscopy',
    'metadata_csv': 'poultry_dataset.csv',
    'max_workers': 5,  # Azaltıldı (rate limit için)
    'min_image_size': (512, 512),
    'max_retries': 3,
    'timeout': 30,
    'rate_limit_delay': 1.0,  # Artırıldı
}

DISEASE_KEYWORDS = {
    'ib': ['infectious bronchitis', 'ib virus', 'ibv', 'tracheal ciliostasis'],
    'ibd': ['infectious bursal disease', 'gumboro', 'bursal', 'lymphoid depletion'],
    'nd': ['newcastle disease', 'ndv', 'paramyxovirus', 'viscerotropic'],
    'coccidiosis': ['coccidi', 'eimeria', 'oocyst', 'intestinal lesion'],
    'fatty_liver': ['hepatic lipidosis', 'fatty liver', 'hepatic steatosis', 'ketoacidosis'],
    'histomoniasis': ['histomona', 'blackhead', 'cecal lesion', 'hepatic necrosis'],
    'healthy': ['normal', 'healthy', 'control', 'no lesion']
}

TISSUE_KEYWORDS = {
    'trachea': ['trachea', 'respiratory epithelium'],
    'bursa': ['bursa', 'bursa fabricius', 'bursal'],
    'liver': ['liver', 'hepatic', 'hepatocyte'],
    'intestine': ['intestine', 'intestinal', 'cecum', 'ileum'],
    'lung': ['lung', 'pulmonary', 'air sac'],
}


class EuropePMCScraper:
    """Europe PMC API - Alternatif kaynak (daha liberal)"""
    
    BASE_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/'
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_articles(self, query: str, max_results: int = 200) -> List[Dict]:
        """Europe PMC'de ara"""
        search_url = f"{self.BASE_URL}search"
        params = {
            'query': f'{query} AND (OPEN_ACCESS:y)',
            'resultType': 'core',
            'pageSize': min(max_results, 1000),
            'format': 'json'
        }
        
        print(f"🔍 Europe PMC aranıyor: '{query}'")
        
        try:
            response = self.session.get(search_url, params=params, timeout=CONFIG['timeout'])
            response.raise_for_status()
            data = response.json()
            
            results = data.get('resultList', {}).get('result', [])
            print(f"✅ {len(results)} makale bulundu")
            
            return results
            
        except Exception as e:
            print(f"⚠️ Europe PMC hatası: {e}")
            return []
    
    def get_article_images(self, article: Dict) -> List[Dict]:
        """Makale görüntülerini al"""
        images = []
        pmcid = article.get('pmcid', '')
        
        if not pmcid:
            return images
        
        # Full text XML'i al
        fulltext_url = f"{self.BASE_URL}{pmcid}/fullTextXML"
        
        time.sleep(CONFIG['rate_limit_delay'])
        
        try:
            response = self.session.get(fulltext_url, timeout=CONFIG['timeout'])
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            # Figure'ları bul
            for fig in root.findall('.//fig'):
                graphic = fig.find('.//graphic')
                if graphic is not None:
                    href = graphic.get('{http://www.w3.org/1999/xlink}href')
                    caption = ' '.join(fig.itertext())
                    
                    if href and self._is_histopathology_image(caption):
                        # Europe PMC image URL formatı
                        img_url = f"https://europepmc.org/articles/{pmcid}/bin/{href}.jpg"
                        
                        images.append({
                            'url': img_url,
                            'caption': caption,
                            'pmcid': pmcid,
                            'title': article.get('title', ''),
                            'fig_id': fig.get('id', '')
                        })
        
        except Exception as e:
            pass  # Sessizce atla
        
        return images
    
    def _is_histopathology_image(self, caption: str) -> bool:
        caption_lower = caption.lower()
        keywords = ['histopathology', 'microscop', 'tissue', 'h&e', 
                   'hematoxylin', 'stain', 'lesion', 'magnification', 
                   'section', 'trachea', 'liver', 'bursa']
        return any(kw in caption_lower for kw in keywords)


class BioRxivScraper:
    """bioRxiv preprints - Açık erişim"""
    
    BASE_URL = 'https://api.biorxiv.org/details/biorxiv/'
    
    def __init__(self):
        self.session = requests.Session()
    
    def search_articles(self, query: str) -> List[Dict]:
        """bioRxiv'de makale ara (tarih aralığı ile)"""
        # Son 2 yıl
        start_date = '2023-01-01'
        end_date = '2025-12-31'
        
        url = f"{self.BASE_URL}{start_date}/{end_date}/0/json"
        
        print(f"🔍 bioRxiv aranıyor...")
        
        try:
            response = self.session.get(url, timeout=CONFIG['timeout'])
            response.raise_for_status()
            data = response.json()
            
            articles = data.get('collection', [])
            
            # Query ile filtrele
            query_lower = query.lower()
            filtered = [
                a for a in articles 
                if query_lower in a.get('title', '').lower() or 
                   query_lower in a.get('abstract', '').lower()
            ]
            
            print(f"✅ {len(filtered)} makale bulundu")
            return filtered[:100]
            
        except Exception as e:
            print(f"⚠️ bioRxiv hatası: {e}")
            return []


class ImageDownloader:
    """Geliştirilmiş görüntü indirme"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Referer': 'https://europepmc.org/'
        })
    
    def download_image(self, url: str, metadata: Dict) -> Optional[Dict]:
        """Görüntü indir - geliştirilmiş hata yönetimi"""
        
        # URL temizleme
        url = url.replace('.jpg.jpg', '.jpg')
        
        for attempt in range(CONFIG['max_retries']):
            try:
                # Yavaş indirme (rate limit için)
                time.sleep(CONFIG['rate_limit_delay'])
                
                response = self.session.get(
                    url, 
                    timeout=CONFIG['timeout'], 
                    stream=True,
                    allow_redirects=True
                )
                
                if response.status_code == 403:
                    # Alternatif URL dene
                    alt_url = self._get_alternative_url(url, metadata)
                    if alt_url and alt_url != url:
                        response = self.session.get(alt_url, timeout=CONFIG['timeout'], stream=True)
                
                response.raise_for_status()
                
                # Görüntü kontrolü
                img_data = response.content
                img = Image.open(io.BytesIO(img_data))
                
                if img.size[0] < CONFIG['min_image_size'][0] or img.size[1] < CONFIG['min_image_size'][1]:
                    return None
                
                # Kaydet
                file_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                pmcid = metadata.get('pmcid', 'unknown')
                filename = f"{pmcid}_{metadata.get('fig_id', 'fig')}_{file_hash}.jpg"
                filepath = self.output_dir / filename
                
                img.convert('RGB').save(filepath, 'JPEG', quality=95)
                
                return {
                    'image_path': str(filepath),
                    'source_url': url,
                    'pmcid': pmcid,
                    'title': metadata.get('title', ''),
                    'caption': metadata.get('caption', ''),
                    'width': img.size[0],
                    'height': img.size[1],
                    'disease': self._detect_disease(metadata.get('caption', '') + ' ' + metadata.get('title', '')),
                    'tissue': self._detect_tissue(metadata.get('caption', '')),
                    'magnification': self._extract_magnification(metadata.get('caption', ''))
                }
                
            except Exception as e:
                if attempt == CONFIG['max_retries'] - 1:
                    return None
                time.sleep(2 ** attempt)
        
        return None
    
    def _get_alternative_url(self, url: str, metadata: Dict) -> str:
        """403 hatası için alternatif URL dene"""
        pmcid = metadata.get('pmcid', '')
        fig_id = metadata.get('fig_id', '')
        
        if 'europepmc.org' in url:
            # Farklı format dene
            return f"https://europepmc.org/articles/{pmcid}/bin/{fig_id}"
        
        return url
    
    def _detect_disease(self, text: str) -> str:
        text_lower = text.lower()
        for disease, keywords in DISEASE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return disease
        return 'unknown'
    
    def _detect_tissue(self, text: str) -> str:
        text_lower = text.lower()
        for tissue, keywords in TISSUE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return tissue
        return 'unknown'
    
    def _extract_magnification(self, text: str) -> str:
        match = re.search(r'(\d+)×|×(\d+)|magnification[:\s]+(\d+)', text.lower())
        if match:
            mag = next(g for g in match.groups() if g)
            return f"{mag}x"
        return 'unknown'


def main():
    """Ana veri toplama pipeline - Geliştirilmiş versiyon"""
    
    print("🐔 Kanatlı Patoloji Veri Toplama v2.0\n")
    print("⚙️ Europe PMC + bioRxiv kaynaklarından toplama yapılacak\n")
    
    # Arama sorguları
    queries = [
        'chicken histopathology microscopy',
        'poultry infectious disease pathology',
        'avian respiratory histology',
        'broiler intestinal lesion microscopy',
    ]
    
    # Scraper başlat
    epmc_scraper = EuropePMCScraper()
    downloader = ImageDownloader(CONFIG['output_dir'])
    
    all_metadata = []
    all_images = []
    
    # Europe PMC'den topla
    for query in queries:
        articles = epmc_scraper.search_articles(query, max_results=100)
        
        print(f"\n📄 {len(articles)} makale için görüntü taranıyor...")
        
        for article in tqdm(articles[:50], desc="Tarama"):  # İlk 50 makale
            images = epmc_scraper.get_article_images(article)
            all_images.extend(images)
        
        print(f"✅ Toplam {len(all_images)} görüntü bulundu")
        
        time.sleep(2)
    
    # Görüntüleri indir
    if all_images:
        print(f"\n⬇️ {len(all_images)} görüntü indiriliyor...")
        
        with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
            futures = {
                executor.submit(downloader.download_image, img['url'], img): img
                for img in all_images
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="İndirme"):
                result = future.result()
                if result:
                    all_metadata.append(result)
    
    # Sonuçları kaydet
    if all_metadata:
        df = pd.DataFrame(all_metadata)
        df.to_csv(CONFIG['metadata_csv'], index=False)
        
        print(f"\n✅ TAMAMLANDI!")
        print(f"📊 Başarılı indirme: {len(df)} / {len(all_images)} görüntü")
        print(f"💾 Metadata: {CONFIG['metadata_csv']}")
        
        if len(df) > 0:
            print(f"\n📁 Hastalık dağılımı:")
            print(df['disease'].value_counts())
            print(f"\n🔬 Doku dağılımı:")
            print(df['tissue'].value_counts())
            print(f"\n📏 Çözünürlük ortalaması: {df['width'].mean():.0f}x{df['height'].mean():.0f}")
    else:
        print("\n⚠️ Hiç görüntü indirilemedi!")
        print("\n💡 ÖNERİLER:")
        print("1. Manuel kaynak kullanın: Cornell Vet Atlas, ISPAH DVD")
        print("2. Kaggle'daki 'Chicken Disease' veri setini indirin")
        print("3. Üniversite patoloji arşivlerine başvurun")


if __name__ == '__main__':
    # Pip paketi → modül adı eşlemesi (özellikle Pillow için)
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
        print(f"⚠️ Eksik: {', '.join(missing)}")
        print(f"Yükle: pip install {' '.join(missing)}")
    else:
        main()