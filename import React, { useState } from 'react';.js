&^qw!+%e'^& !+%AEQWCimport React, { useState } from 'react';
import { Download, Search, Database, CheckCircle, AlertCircle, FileText, Image } from 'lucide-react';

const PoultryDataCollector = () => {
  const [activeTab, setActiveTab] = useState('pubmed');
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState([]);
  const [downloading, setDownloading] = useState(false);
  const [stats, setStats] = useState({
    total: 0,
    downloaded: 0,
    labeled: 0,
    failed: 0
  });

  // Örnek hastalık kategorileri
  const diseases = [
    { id: 'healthy', name: 'Sağlıklı Doku', color: 'bg-green-100 text-green-800' },
    { id: 'ib', name: 'Infectious Bronchitis (IB)', color: 'bg-red-100 text-red-800' },
    { id: 'ibd', name: 'Infectious Bursal Disease (IBD)', color: 'bg-orange-100 text-orange-800' },
    { id: 'nd', name: 'Newcastle Disease', color: 'bg-purple-100 text-purple-800' },
    { id: 'coccidiosis', name: 'Coccidiosis', color: 'bg-yellow-100 text-yellow-800' },
    { id: 'fatty_liver', name: 'Fatty Liver Syndrome', color: 'bg-blue-100 text-blue-800' },
    { id: 'histomoniasis', name: 'Histomoniasis', color: 'bg-pink-100 text-pink-800' }
  ];

  // Veri kaynakları
  const dataSources = [
    {
      id: 'pubmed',
      name: 'PubMed Central (PMC)',
      icon: FileText,
      description: 'Open-access makalelerden supplemental görüntüler',
      estimatedImages: '~1,600',
      query: 'poultry histopathology microscopy'
    },
    {
      id: 'cornell',
      name: 'Cornell Vet Atlas',
      icon: Database,
      description: 'CC-BY-NC lisanslı yüksek çözünürlüklü histopatoloji',
      estimatedImages: '~1,100',
      query: 'avian diseases atlas'
    },
    {
      id: 'ispah',
      name: 'ISPAH Slide Set',
      icon: Image,
      description: 'Kişisel bilimsel kullanım için DVD set',
      estimatedImages: '~2,400',
      query: 'poultry pathology slides'
    }
  ];

  // Örnek arama sonuçları
  const mockSearchResults = [
    {
      id: 1,
      title: 'Histopathological changes in broiler chickens with IB',
      source: 'PMC8745632',
      images: 12,
      tissue: 'Trachea',
      magnification: '200x',
      license: 'CC-BY',
      disease: 'ib'
    },
    {
      id: 2,
      title: 'Bursal lesions in IBD-infected chickens',
      source: 'PMC9234567',
      images: 8,
      tissue: 'Bursa Fabricius',
      magnification: '100x, 400x',
      license: 'CC-BY-NC',
      disease: 'ibd'
    },
    {
      id: 3,
      title: 'Hepatic lipidosis in layer hens',
      source: 'Cornell Atlas #247',
      images: 15,
      tissue: 'Liver',
      magnification: '40x-400x',
      license: 'CC-BY-NC',
      disease: 'fatty_liver'
    },
    {
      id: 4,
      title: 'Coccidial oocysts in intestinal mucosa',
      source: 'ISPAH-2020-134',
      images: 6,
      tissue: 'Intestine',
      magnification: '400x',
      license: 'Educational',
      disease: 'coccidiosis'
    }
  ];

  const handleSearch = () => {
    setDownloading(true);
    setTimeout(() => {
      setResults(mockSearchResults);
      setStats({
        total: mockSearchResults.reduce((sum, r) => sum + r.images, 0),
        downloaded: 0,
        labeled: 0,
        failed: 0
      });
      setDownloading(false);
    }, 1500);
  };

  const handleDownload = (resultId) => {
    const result = results.find(r => r.id === resultId);
    if (result) {
      setStats(prev => ({
        ...prev,
        downloaded: prev.downloaded + result.images
      }));
    }
  };

  const getCurrentSource = () => dataSources.find(s => s.id === activeTab);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                <Database className="text-indigo-600" size={36} />
                Kanatlı Patoloji Veri Toplama
              </h1>
              <p className="text-gray-600 mt-2">
                Transformer model eğitimi için histopatoloji görüntüleri toplama ve organizasyon
              </p>
            </div>
          </div>
        </div>

        {/* İstatistikler */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Toplam Görüntü</p>
                <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
              </div>
              <Image className="text-blue-500" size={32} />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">İndirildi</p>
                <p className="text-2xl font-bold text-green-600">{stats.downloaded}</p>
              </div>
              <Download className="text-green-500" size={32} />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Etiketlendi</p>
                <p className="text-2xl font-bold text-indigo-600">{stats.labeled}</p>
              </div>
              <CheckCircle className="text-indigo-500" size={32} />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Başarısız</p>
                <p className="text-2xl font-bold text-red-600">{stats.failed}</p>
              </div>
              <AlertCircle className="text-red-500" size={32} />
            </div>
          </div>
        </div>

        {/* Veri Kaynağı Seçimi */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Veri Kaynağı Seçin</h2>
          <div className="flex gap-2 mb-6">
            {dataSources.map(source => {
              const Icon = source.icon;
              return (
                <button
                  key={source.id}
                  onClick={() => setActiveTab(source.id)}
                  className={`flex-1 p-4 rounded-lg border-2 transition-all ${
                    activeTab === source.id
                      ? 'border-indigo-500 bg-indigo-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <Icon className={activeTab === source.id ? 'text-indigo-600' : 'text-gray-400'} size={24} />
                  <p className="font-semibold mt-2">{source.name}</p>
                  <p className="text-sm text-gray-600">{source.estimatedImages}</p>
                </button>
              );
            })}
          </div>

          {/* Kaynak Detayları */}
          {getCurrentSource() && (
            <div className="bg-gray-50 rounded-lg p-4 mb-4">
              <p className="text-gray-700">{getCurrentSource().description}</p>
              <p className="text-sm text-gray-500 mt-2">
                Önerilen sorgu: <code className="bg-white px-2 py-1 rounded">{getCurrentSource().query}</code>
              </p>
            </div>
          )}

          {/* Arama */}
          <div className="flex gap-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Arama terimi girin (örn: chicken trachea histology)"
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
            <button
              onClick={handleSearch}
              disabled={downloading}
              className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 flex items-center gap-2"
            >
              {downloading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                  Aranıyor...
                </>
              ) : (
                <>
                  <Search size={20} />
                  Ara
                </>
              )}
            </button>
          </div>
        </div>

        {/* Hastalık Kategorileri */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Hedef Hastalıklar (7 Sınıf)</h2>
          <div className="grid grid-cols-4 gap-3">
            {diseases.map(disease => (
              <div key={disease.id} className={`${disease.color} rounded-lg p-3`}>
                <p className="font-semibold text-sm">{disease.name}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Sonuçlar */}
        {results.length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">
              Bulunan Kaynaklar ({results.length})
            </h2>
            <div className="space-y-3">
              {results.map(result => {
                const disease = diseases.find(d => d.id === result.disease);
                return (
                  <div key={result.id} className="border border-gray-200 rounded-lg p-4 hover:border-indigo-300 transition-colors">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h3 className="font-semibold text-gray-900">{result.title}</h3>
                        <div className="flex items-center gap-4 mt-2 text-sm text-gray-600">
                          <span className="flex items-center gap-1">
                            <FileText size={14} />
                            {result.source}
                          </span>
                          <span className="flex items-center gap-1">
                            <Image size={14} />
                            {result.images} görüntü
                          </span>
                          <span>Doku: {result.tissue}</span>
                          <span>Büyütme: {result.magnification}</span>
                        </div>
                        <div className="flex items-center gap-2 mt-2">
                          <span className={`${disease?.color} px-2 py-1 rounded text-xs font-medium`}>
                            {disease?.name}
                          </span>
                          <span className="bg-gray-100 text-gray-700 px-2 py-1 rounded text-xs">
                            {result.license}
                          </span>
                        </div>
                      </div>
                      <button
                        onClick={() => handleDownload(result.id)}
                        className="ml-4 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
                      >
                        <Download size={16} />
                        İndir
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Kullanım Talimatları */}
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-900 mb-2">📋 Sonraki Adımlar</h3>
          <ol className="text-sm text-blue-800 space-y-1 ml-4 list-decimal">
            <li>Yukarıdaki kaynakları arayın ve indirin</li>
            <li>İndirilen görüntüleri <code className="bg-white px-1 rounded">poultry_microscopy/</code> klasörüne kaydedin</li>
            <li>Her görüntü için metadata CSV dosyası oluşturun (image_path, disease, tissue, magnification)</li>
            <li>Etiketleme arayüzüne geçin (sonraki artifact)</li>
          </ol>
        </div>
      </div>
    </div>
  );
};

export default PoultryDataCollector;