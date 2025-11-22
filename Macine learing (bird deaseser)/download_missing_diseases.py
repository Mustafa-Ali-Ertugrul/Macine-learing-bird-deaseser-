import os
import requests
import time
from pathlib import Path
import pandas as pd
import json

# Eksik hastalıklar
MISSING_DISEASES = {
    'mareks': 'Marek\'s Disease',
    'avian_influenza': 'Avian Influenza',
    'infectious_bronchitis': 'Infectious Bronchitis (IB)',
    'infectious_bursal': 'Infectious Bursal Disease (IBD)',
    'histomoniasis': 'Histomoniasis'
}

# İndirme dizini
OUTPUT_DIR = Path('poultry_microscopy')
OUTPUT_DIR.mkdir(exist_ok=True)

# CSV kayıt dosyası
csv_file = 'downloaded_missing_diseases.csv'

class DiseaseDatasetDownloader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.downloaded = []
        
    def download_image(self, url, filename, disease):
        """Tek bir resim indir"""
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                disease_dir = OUTPUT_DIR / disease
                disease_dir.mkdir(exist_ok=True)
                
                filepath = disease_dir / filename
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"✓ İndirildi: {filename}")
                return str(filepath)
            else:
                print(f"✗ Hata {response.status_code}: {url}")
                return None
        except Exception as e:
            print(f"✗ İndirme hatası: {e}")
            return None
    
    def download_from_kaggle_api(self):
        """Kaggle API ile veri seti indirme"""
        print("\n=== KAGGLE VERİ SETLERİ ===")
        print("Kaggle API kullanımı için:")
        print("1. Kaggle hesabınızdan API token indirin (kaggle.json)")
        print("2. Dosyayı ~/.kaggle/ klasörüne koyun")
        print("\nKomutlar:")
        
        kaggle_datasets = [
            {
                'name': 'chandrashekarnatesh/poultry-diseases',
                'diseases': ['mareks', 'avian_influenza', 'infectious_bronchitis']
            },
            {
                'name': 'allandclive/poultry-disease-detection',
                'diseases': ['mareks', 'infectious_bursal']
            }
        ]
        
        for dataset in kaggle_datasets:
            cmd = f"kaggle datasets download -d {dataset['name']} -p poultry_microscopy --unzip"
            print(f"\n{cmd}")
            print(f"İçerik: {', '.join([MISSING_DISEASES[d] for d in dataset['diseases']])}")
    
    def download_from_roboflow(self):
        """Roboflow'dan veri seti indirme talimatları"""
        print("\n=== ROBOFLOW VERİ SETLERİ ===")
        
        roboflow_datasets = [
            {
                'url': 'https://universe.roboflow.com/goat-dataset/hen-and-its-diseases',
                'name': 'Hen and Its Diseases',
                'diseases': ['mareks', 'infectious_bronchitis', 'infectious_bursal']
            },
            {
                'url': 'https://universe.roboflow.com/disease-detection-lcm9s/poultry-diseases-w3qsh',
                'name': 'Poultry Diseases Detection',
                'diseases': ['avian_influenza', 'mareks']
            }
        ]
        
        print("\nRoboflow'dan indirmek için:")
        for ds in roboflow_datasets:
            print(f"\n{ds['name']}: {ds['url']}")
            print(f"Hastalıklar: {', '.join([MISSING_DISEASES[d] for d in ds['diseases']])}")
            print("Export > Format: JPEG > Download")
    
    def search_zenodo(self, disease_name, disease_key):
        """Zenodo'da hastalık araması"""
        print(f"\n=== {disease_name} - ZENODO ARAMA ===")
        
        search_terms = [
            f'poultry {disease_name.lower()}',
            f'chicken {disease_name.lower()}',
            f'avian {disease_name.lower()}',
            f'{disease_name.lower()} histopathology'
        ]
        
        found_datasets = []
        
        for term in search_terms:
            try:
                url = f'https://zenodo.org/api/records/'
                params = {
                    'q': term,
                    'size': 10,
                    'type': 'dataset'
                }
                
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    hits = data.get('hits', {}).get('hits', [])
                    
                    for hit in hits:
                        title = hit.get('metadata', {}).get('title', '')
                        record_id = hit.get('id', '')
                        files = hit.get('files', [])
                        
                        if files:
                            found_datasets.append({
                                'title': title,
                                'id': record_id,
                                'url': f'https://zenodo.org/record/{record_id}',
                                'files': len(files)
                            })
                
                time.sleep(1)
            except Exception as e:
                print(f"Zenodo arama hatası: {e}")
        
        if found_datasets:
            print(f"\n{len(found_datasets)} veri seti bulundu:")
            for ds in found_datasets[:5]:
                print(f"\n  • {ds['title']}")
                print(f"    {ds['url']}")
                print(f"    Dosya sayısı: {ds['files']}")
        else:
            print("Sonuç bulunamadı.")
        
        return found_datasets
    
    def search_figshare(self, disease_name, disease_key):
        """Figshare'de hastalık araması"""
        print(f"\n=== {disease_name} - FIGSHARE ARAMA ===")
        
        try:
            url = 'https://api.figshare.com/v2/articles/search'
            params = {
                'search_for': f'poultry {disease_name.lower()}',
                'page_size': 10
            }
            
            response = self.session.post(url, json=params, timeout=15)
            if response.status_code == 200:
                results = response.json()
                
                if results:
                    print(f"\n{len(results)} sonuç bulundu:")
                    for item in results[:5]:
                        print(f"\n  • {item.get('title')}")
                        print(f"    {item.get('url')}")
                        print(f"    Dosya sayısı: {len(item.get('files', []))}")
                else:
                    print("Sonuç bulunamadı.")
            
            time.sleep(1)
        except Exception as e:
            print(f"Figshare arama hatası: {e}")
    
    def download_from_pubmed_central(self, disease_name, disease_key):
        """PubMed Central'dan hastalık görselleri bulma"""
        print(f"\n=== {disease_name} - PUBMED CENTRAL ===")
        
        search_url = f'https://www.ncbi.nlm.nih.gov/pmc/?term="{disease_name.lower()}"+AND+"poultry"+AND+"histopathology"'
        print(f"\nManuel arama: {search_url}")
        print("Açık erişimli makalelerdeki görselleri indirin")
    
    def generate_download_report(self):
        """İndirme raporu oluştur"""
        print("\n" + "="*60)
        print("EKSİK VERİ SETLERİ İNDİRME KILAVUZU")
        print("="*60)
        
        for disease_key, disease_name in MISSING_DISEASES.items():
            print(f"\n{'='*60}")
            print(f"{disease_name.upper()}")
            print(f"{'='*60}")
            
            # Zenodo ara
            self.search_zenodo(disease_name, disease_key)
            
            # Figshare ara
            self.search_figshare(disease_name, disease_key)
            
            # PubMed Central
            self.download_from_pubmed_central(disease_name, disease_key)
            
            time.sleep(2)
        
        # Kaggle talimatları
        self.download_from_kaggle_api()
        
        # Roboflow talimatları
        self.download_from_roboflow()
        
        print("\n" + "="*60)
        print("EK KAYNAKLAR")
        print("="*60)
        print("\n1. Google Dataset Search:")
        print("   https://datasetsearch.research.google.com/")
        print("   Arama: 'poultry disease' veya her hastalık ismi")
        
        print("\n2. Mendeley Data:")
        print("   https://data.mendeley.com/")
        print("   Arama: 'avian disease' 'poultry pathology'")
        
        print("\n3. IEEE DataPort:")
        print("   https://ieee-dataport.org/")
        print("   Arama: 'poultry disease detection'")
        
        print("\n4. University Repositories:")
        print("   - Cornell VetMed Image Library")
        print("   - UC Davis Veterinary Pathology")
        
        print("\n" + "="*60)
        print("MANUEL İNDİRME TALİMATLARI")
        print("="*60)
        print("\nBulunan veri setlerini şu klasörlere kaydedin:")
        for disease_key, disease_name in MISSING_DISEASES.items():
            print(f"  • poultry_microscopy/{disease_key}/")
        
        print("\n✓ Script tamamlandı!")

def main():
    print("="*60)
    print("EKSİK HASTALIK VERİ SETLERİ İNDİRİCİ")
    print("="*60)
    print("\nEksik hastalıklar:")
    for key, name in MISSING_DISEASES.items():
        print(f"  • {name}")
    
    downloader = DiseaseDatasetDownloader()
    downloader.generate_download_report()
    
    # Özet
    print("\n" + "="*60)
    print("ÖNEMLİ NOTLAR")
    print("="*60)
    print("\n1. Kaggle API için 'pip install kaggle' çalıştırın")
    print("2. Zenodo ve Figshare bağlantılarını tarayıcıda açın")
    print("3. Manuel indirilen dosyaları uygun klasörlere taşıyın")
    print("4. İndirme tamamlandıktan sonra 'python analyze_dataset.py' çalıştırın")

if __name__ == '__main__':
    main()
