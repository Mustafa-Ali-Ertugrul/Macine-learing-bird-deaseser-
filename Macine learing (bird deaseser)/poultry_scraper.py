#!/usr/bin/env python3
"""
Kanatlı Patoloji Histopatoloji Görüntü Toplama Sistemi
PubMed Central API ile doğru URL formatı kullanarak indirme
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
import re
from tqdm import tqdm
import hashlib
from PIL import Image
import io

# Konfigürasyon
CONFIG = {
    'output_dir': 'poultry_microscopy',
    'metadata_csv': 'poultry_dataset.csv',
    'max_workers': 4,  # Rate limit için daha düşük eşzamanlılık
    'min_image_size': (512, 512),
    'max_retries': 3,
    'timeout': 30,
    'rate_limit_delay': 2.0,  # PubMed eutils için bekleme süresi
}

# Hastalık keywords
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


class PubMedScraper:
    """PubMed Central API ile görüntü toplama (FTP erişimi ile)"""
    
    BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    
    def __init__(self, email: str = 'researcher@example.com'):
        # Ortam değişkeninden e-posta al (PUBMED_EMAIL) yoksa verilen değeri kullan
        self.email = os.environ.get('PUBMED_EMAIL', email)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; PoultryResearch/1.0)'
        })
    
    def search_articles(self, query: str, max_results: int = 30) -> List[str]:
        """PubMed'de makale ara ve PMC ID'leri döndür"""
        search_url = f"{self.BASE_URL}esearch.fcgi"
        params = {
            'db': 'pmc',
            'term': query + ' AND open access[filter] AND hasimages[text]',
            'retmax': max_results,
            'retmode': 'json',
            'email': self.email,
            'sort': 'pub+date'
        }
        
        print(f"[Search] PubMed aranıyor: '{query}'")
        response = self._get_with_backoff(search_url, params)
        
        data = response.json()
        pmc_ids = data.get('esearchresult', {}).get('idlist', [])
        print(f"[Success] {len(pmc_ids)} makale bulundu")
        
        return pmc_ids
    
    def get_article_metadata(self, pmc_id: str) -> Optional[Dict]:
        """Makale metadata ve görüntü URL'lerini al (OAS endpoint kullanarak)"""
        # OAS (Open Access Subset) endpoint'ini kullan
        oas_url = f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
        params = {
            'verb': 'GetRecord',
            'identifier': f'oai:pubmedcentral.nih.gov:{pmc_id}',
            'metadataPrefix': 'pmc'
        }
        
        time.sleep(CONFIG['rate_limit_delay'])
        
        try:
            response = self._get_with_backoff(oas_url, params)
            
            root = ET.fromstring(response.content)
            
            # Namespace tanımla
            ns = {'oai': 'http://www.openarchives.org/OAI/2.0/'}
            
            # Article metadata
            record = root.find('.//oai:record', ns)
            if not record:
                return None
            
            # PMC article XML'ini çek
            article_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
            article_response = self._get_with_backoff(article_url, {})
            
            if article_response.status_code != 200:
                return None
            
            # HTML'den görüntü URL'lerini extract et
            html_content = article_response.text
            images = self._extract_images_from_html(html_content, pmc_id)
            
            if not images:
                return None
            
            # Title ve abstract'ı XML'den al
            title = self._extract_text_safe(root, './/article-title')
            abstract = self._extract_text_safe(root, './/abstract')
            
            return {
                'pmc_id': pmc_id,
                'title': title,
                'abstract': abstract,
                'images': images
            }
            
        except Exception as e:
            print(f"[Warning] PMC{pmc_id} metadata hatası: {str(e)[:100]}")
            return None

    def _get_with_backoff(self, url: str, params: Dict) -> requests.Response:
        """429 (Too Many Requests) ve geçici hatalarda artan bekleme ile yeniden dene"""
        delay = CONFIG['rate_limit_delay']
        for attempt in range(CONFIG['max_retries']):
            try:
                resp = self.session.get(url, params=params, timeout=CONFIG['timeout'])
                if resp.status_code == 429:
                    if attempt == CONFIG['max_retries'] - 1:
                        resp.raise_for_status()
                    time.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                return resp
            except requests.exceptions.RequestException:
                if attempt == CONFIG['max_retries'] - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError('Beklenmeyen backoff akışı')
    
    def _extract_text_safe(self, root, xpath):
        """XML'den text güvenli şekilde çıkar"""
        elem = root.find(xpath)
        if elem is not None:
            return ''.join(elem.itertext()).strip()
        return ''
    
    def _extract_images_from_html(self, html: str, pmc_id: str) -> List[Dict]:
        """HTML'den görüntü URL'lerini extract et"""
        images = []
        
        # Figure pattern'leri
        fig_pattern = r'<figure[^>]*>.*?<img[^>]*src="([^"]+)"[^>]*>.*?<figcaption[^>]*>(.*?)</figcaption>.*?</figure>'
        matches = re.findall(fig_pattern, html, re.DOTALL | re.IGNORECASE)
        
        for img_url, caption in matches:
            # Caption'ı temizle
            caption_clean = re.sub(r'<[^>]+>', '', caption).strip()
            
            if not self._is_histopathology_image(caption_clean):
                continue
            
            # URL'yi düzelt
            if img_url.startswith('/'):
                img_url = f'https://www.ncbi.nlm.nih.gov{img_url}'
            elif not img_url.startswith('http'):
                img_url = f'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/bin/{img_url}'
            
            # .jpg.jpg gibi çift uzantıyı düzelt
            img_url = re.sub(r'\.jpg\.jpg$', '.jpg', img_url)
            
            images.append({
                'url': img_url,
                'caption': caption_clean[:500],  # Limit caption length
                'fig_id': hashlib.md5(img_url.encode()).hexdigest()[:8]
            })
        
        return images[:5]  # Max 5 görüntü per makale
    
    def _is_histopathology_image(self, caption: str) -> bool:
        """Caption'dan histopatoloji görüntüsü olup olmadığını kontrol et"""
        if not caption or len(caption) < 10:
            return False
        
        caption_lower = caption.lower()
        keywords = ['histopatholog', 'microscop', 'tissue', 'section', 'h&e', 
                   'hematoxylin', 'stain', 'lesion', 'magnification', 'μm',
                   'trachea', 'liver', 'bursa', 'intestine', 'epithelium']
        
        return any(kw in caption_lower for kw in keywords)


class ImageDownloader:
    """Görüntü indirme ve kalite kontrolü"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.ncbi.nlm.nih.gov/'
        })
    
    def download_image(self, url: str, metadata: Dict) -> Optional[Dict]:
        """Görüntü indir ve metadata döndür"""
        for attempt in range(CONFIG['max_retries']):
            try:
                time.sleep(0.5)  # Rate limiting
                
                response = self.session.get(url, timeout=CONFIG['timeout'], stream=True)
                
                if response.status_code == 403:
                    # Alternative URL dene
                    alt_url = url.replace('/bin/', '/bin/').replace('.jpg', '.png')
                    response = self.session.get(alt_url, timeout=CONFIG['timeout'], stream=True)
                
                if response.status_code != 200:
                    return None
                
                response.raise_for_status()
                
                # Görüntü kalite kontrolü
                img_data = response.content
                
                if len(img_data) < 10000:  # 10KB minimum
                    return None
                
                img = Image.open(io.BytesIO(img_data))
                
                if img.size[0] < CONFIG['min_image_size'][0] or img.size[1] < CONFIG['min_image_size'][1]:
                    return None
                
                # Dosya adı oluştur
                file_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"{metadata['pmc_id']}_{metadata['fig_id']}_{file_hash}.jpg"
                filepath = self.output_dir / filename
                
                # Kaydet
                img.convert('RGB').save(filepath, 'JPEG', quality=95)
                
                # Metadata döndür
                return {
                    'image_path': str(filepath),
                    'source_url': url,
                    'pmc_id': metadata['pmc_id'],
                    'title': metadata['title'][:200],
                    'caption': metadata['caption'][:500],
                    'width': img.size[0],
                    'height': img.size[1],
                    'disease': self._detect_disease(metadata['caption'] + ' ' + metadata.get('abstract', '')[:500]),
                    'tissue': self._detect_tissue(metadata['caption']),
                    'magnification': self._extract_magnification(metadata['caption'])
                }
                
            except Exception as e:
                if attempt == CONFIG['max_retries'] - 1:
                    pass  # Sessizce atla
                else:
                    time.sleep(2 ** attempt)
        
        return None
    
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
        match = re.search(r'(\d+)\s*[×x]|[×x]\s*(\d+)|magnification[:\s]+(\d+)', text.lower())
        if match:
            mag = next(g for g in match.groups() if g)
            return f"{mag}x"
        return 'unknown'


def main():
    """Ana veri toplama pipeline"""
    
    print("[Chicken] Kanatlı Patoloji Veri Toplama Başlatılıyor...\n")
    
    # Odaklanmış sorgular
    queries = [
        'chicken trachea histology infectious bronchitis',
        'poultry bursa fabricius histopathology',
        'broiler liver histology fatty',
        'chicken intestine coccidiosis microscopy',
    ]
    
    scraper = PubMedScraper()
    downloader = ImageDownloader(CONFIG['output_dir'])
    
    all_metadata = []
    
    for query in queries:
        pmc_ids = scraper.search_articles(query, max_results=50)  # Azaltıldı
        
        print(f"\n[Article] {len(pmc_ids)} makale için metadata toplanıyor...")
        articles_with_images = []
        
        for pmc_id in tqdm(pmc_ids[:30], desc="Metadata"):  # Max 30 makale
            article_data = scraper.get_article_metadata(pmc_id)
            if article_data and article_data['images']:
                articles_with_images.append(article_data)
        
        print(f"[Success] {len(articles_with_images)} makalede görüntü bulundu")
        
        if not articles_with_images:
            continue
        
        # Görüntüleri indir (sequential - rate limiting için)
        print(f"\n[Download] Görüntüler indiriliyor...")
        
        for article in tqdm(articles_with_images, desc="Makaleler"):
            for img_data in article['images']:
                task_metadata = {
                    'pmc_id': article['pmc_id'],
                    'title': article['title'],
                    'abstract': article['abstract'],
                    'caption': img_data['caption'],
                    'fig_id': img_data['fig_id']
                }
                
                result = downloader.download_image(img_data['url'], task_metadata)
                if result:
                    all_metadata.append(result)
                    print(f"[Downloaded] {result['disease']} - {result['tissue']}")
        
        time.sleep(3)  # Sorgular arası bekleme
    
    # Sonuçları kaydet
    if all_metadata:
        df = pd.DataFrame(all_metadata)
        df.to_csv(CONFIG['metadata_csv'], index=False)
        
        print(f"\n[Complete] TAMAMLANDI!")
        print(f"[Stats] Toplam indirilen görüntü: {len(df)}")
        print(f"[Save] Metadata: {CONFIG['metadata_csv']}")
        print(f"\n[Disease] Hastalık dağılımı:")
        print(df['disease'].value_counts())
        print(f"\n[Tissue] Doku dağılımı:")
        print(df['tissue'].value_counts())
    else:
        print("[Warning] Hiç görüntü indirilemedi!")
        print("[Tip] Alternatif: Figshare veya Zenodo'dan veri seti arayın")
        print("   Örnek: https://figshare.com/search?q=chicken%20histopathology")


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
        print(f"[Missing] Eksik kütüphaneler: {', '.join(missing)}")
        print(f"Yüklemek için: pip install {' '.join(missing)}")
    else:
        main()