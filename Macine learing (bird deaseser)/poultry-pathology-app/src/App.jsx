import React, { useState, useEffect, useCallback } from "react";

const API_BASE = "http://localhost:8000";

// Tür tanımları
const SPECIES_OPTIONS = [
  { id: "chicken", label: "Tavuk", emoji: "🐔" },
  { id: "goose",   label: "Kaz",   emoji: "🦢" },
  { id: "duck",    label: "Ördek", emoji: "🦆" },
];

const MODEL_OPTIONS = [
  { id: "vit_b16",         label: "ViT-B/16" },
  { id: "resnet50",        label: "ResNet-50" },
  { id: "efficientnet_b0", label: "EfficientNet-B0" },
  { id: "mobilenet_v2",    label: "MobileNetV2" },
];

function App() {
  // State
  const [selectedSpecies, setSelectedSpecies] = useState("chicken");
  const [selectedModel, setSelectedModel]     = useState("vit_b16");
  const [selectedFile, setSelectedFile]       = useState(null);
  const [previewUrl, setPreviewUrl]           = useState(null);
  const [result, setResult]                   = useState(null);
  const [loading, setLoading]                 = useState(false);
  const [error, setError]                     = useState(null);
  const [speciesInfo, setSpeciesInfo]         = useState({});

  // Tür bilgilerini yükle
  useEffect(() => {
    fetch(`${API_BASE}/species`)
      .then((res) => res.json())
      .then((data) => {
        const info = {};
        data.species.forEach((sp) => { info[sp.id] = sp; });
        setSpeciesInfo(info);
      })
      .catch((err) => console.error("Species bilgisi yüklenemedi:", err));
  }, []);

  // Dosya seçimi
  const handleFileChange = useCallback((e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  }, []);

  // Tahmin gönder
  const handlePredict = useCallback(async () => {
    if (!selectedFile) {
      setError("Lütfen bir görüntü seçin.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    const url = new URL(`${API_BASE}/predict`);
    url.searchParams.append("species", selectedSpecies);
    url.searchParams.append("model", selectedModel);
    url.searchParams.append("top_k", "5");

    try {
      const response = await fetch(url.toString(), {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Tahmin başarısız.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [selectedFile, selectedSpecies, selectedModel]);

  // Sıfırla
  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  }, []);

  const currentSpecies = SPECIES_OPTIONS.find((s) => s.id === selectedSpecies);
  const currentSpeciesInfo = speciesInfo[selectedSpecies];

  return (
    <div style={styles.container}>
      {/* Header */}
      <header style={styles.header}>
        <h1 style={styles.title}>
          🐦 Kümes Hayvanı Hastalık Sınıflandırma
        </h1>
        <p style={styles.subtitle}>
          Derin öğrenme ile tavuk, kaz ve ördek hastalıklarını tespit edin
        </p>
      </header>

      <div style={styles.mainGrid}>
        {/* Sol Panel — Kontroller */}
        <div style={styles.panel}>
          <h2 style={styles.panelTitle}>⚙️ Ayarlar</h2>

          {/* Tür Seçimi */}
          <div style={styles.formGroup}>
            <label style={styles.label}>Hayvan Türü</label>
            <div style={styles.speciesGrid}>
              {SPECIES_OPTIONS.map((sp) => {
                const info = speciesInfo[sp.id];
                const isSelected = selectedSpecies === sp.id;
                const isAvailable = info?.model_available;

                return (
                  <button
                    key={sp.id}
                    onClick={() => {
                      setSelectedSpecies(sp.id);
                      setResult(null);
                    }}
                    style={{
                      ...styles.speciesButton,
                      ...(isSelected ? styles.speciesButtonActive : {}),
                      opacity: isAvailable === false ? 0.6 : 1,
                    }}
                  >
                    <span style={styles.speciesEmoji}>{sp.emoji}</span>
                    <span style={styles.speciesLabel}>{sp.label}</span>
                    {isAvailable === false && (
                      <span style={styles.speciesBadge}>Model Yok</span>
                    )}
                    {isAvailable === true && (
                      <span style={styles.speciesBadgeReady}>Hazır ✓</span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Model Seçimi */}
          <div style={styles.formGroup}>
            <label style={styles.label}>Model Mimarisi</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={styles.select}
            >
              {MODEL_OPTIONS.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          </div>

          {/* Görüntü Yükleme */}
          <div style={styles.formGroup}>
            <label style={styles.label}>Görüntü</label>
            <div style={styles.uploadArea}>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                style={styles.fileInput}
                id="imageUpload"
              />
              <label htmlFor="imageUpload" style={styles.uploadLabel}>
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Önizleme"
                    style={styles.preview}
                  />
                ) : (
                  <div style={styles.uploadPlaceholder}>
                    <span style={{ fontSize: "2rem" }}>📷</span>
                    <span>Görüntü seçin veya sürükleyin</span>
                  </div>
                )}
              </label>
            </div>
          </div>

          {/* Butonlar */}
          <div style={styles.buttonGroup}>
            <button
              onClick={handlePredict}
              disabled={!selectedFile || loading}
              style={{
                ...styles.predictButton,
                opacity: !selectedFile || loading ? 0.5 : 1,
              }}
            >
              {loading ? "⏳ Analiz ediliyor..." : `${currentSpecies.emoji} Analiz Et`}
            </button>
            <button onClick={handleReset} style={styles.resetButton}>
              🔄 Sıfırla
            </button>
          </div>

          {/* Tür bilgisi */}
          {currentSpeciesInfo && (
            <div style={styles.infoBox}>
              <strong>{currentSpecies.emoji} {currentSpecies.label} Durumu:</strong>
              <br />
              Toplam görüntü: {currentSpeciesInfo.total_images}
              <br />
              Dataset: {currentSpeciesInfo.dataset_ready ? "✅ Hazır" : "⏳ Veri bekleniyor"}
              <br />
              Model: {currentSpeciesInfo.model_available ? "✅ Mevcut" : "⏳ Eğitilmedi"}
            </div>
          )}
        </div>

        {/* Sağ Panel — Sonuçlar */}
        <div style={styles.panel}>
          <h2 style={styles.panelTitle}>📊 Sonuçlar</h2>

          {error && (
            <div style={styles.errorBox}>
              ❌ {error}
            </div>
          )}

          {result && (
            <div>
              {/* Üst sonuç */}
              <div style={{
                ...styles.topResult,
                borderColor: result.top_prediction === "Healthy"
                  ? "#22c55e" : "#ef4444",
              }}>
                <div style={styles.topResultEmoji}>
                  {result.top_prediction === "Healthy" ? "🟢" : "🔴"}
                </div>
                <div>
                  <div style={styles.topResultClass}>
                    {result.top_prediction.replace(/_/g, " ")}
                  </div>
                  <div style={styles.topResultConf}>
                    Güven: {(result.confidence * 100).toFixed(1)}%
                  </div>
                  <div style={styles.topResultSpecies}>
                    Tür: {result.species_display}
                  </div>
                </div>
              </div>

              {/* Tüm tahminler */}
              <div style={styles.predictionsContainer}>
                <h3 style={{ marginBottom: "0.5rem" }}>Top Tahminler</h3>
                {result.predictions.map((pred, idx) => (
                  <div key={idx} style={styles.predictionRow}>
                    <div style={styles.predictionInfo}>
                      <span style={styles.predictionRank}>#{idx + 1}</span>
                      <span>{pred.class.replace(/_/g, " ")}</span>
                    </div>
                    <div style={styles.predictionBarContainer}>
                      <div
                        style={{
                          ...styles.predictionBar,
                          width: `${pred.confidence * 100}%`,
                          backgroundColor:
                            pred.class === "Healthy" ? "#22c55e" : "#ef4444",
                        }}
                      />
                    </div>
                    <span style={styles.predictionPercent}>
                      {(pred.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>

              {/* Meta bilgi */}
              <div style={styles.metaInfo}>
                <span>⏱ {result.inference_time_ms}ms</span>
                <span>🧠 {result.model}</span>
                <span>{currentSpecies.emoji} {result.species}</span>
              </div>
            </div>
          )}

          {!result && !error && (
            <div style={styles.placeholder}>
              <span style={{ fontSize: "3rem" }}>🔬</span>
              <p>Bir görüntü seçin ve analiz edin</p>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer style={styles.footer}>
        Desteklenen hastalıklar: {" "}
        {[
          "Avian Influenza", "Coccidiosis", "Fowl Pox", "Healthy",
          "Histomoniasis", "Infectious Bronchitis", "IBD",
          "Marek's Disease", "Newcastle Disease", "Salmonella",
        ].join(" • ")}
      </footer>
    </div>
  );
}

// ─────────────────────────────────────────
// Styles
// ─────────────────────────────────────────
const styles = {
  container: {
    maxWidth: "1100px",
    margin: "0 auto",
    padding: "1.5rem",
    fontFamily: "'Segoe UI', system-ui, sans-serif",
    color: "#1f2937",
  },
  header: {
    textAlign: "center",
    marginBottom: "2rem",
  },
  title: {
    fontSize: "1.8rem",
    marginBottom: "0.3rem",
  },
  subtitle: {
    color: "#6b7280",
    fontSize: "1rem",
  },
  mainGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "1.5rem",
  },
  panel: {
    background: "#fff",
    borderRadius: "12px",
    padding: "1.5rem",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
    border: "1px solid #e5e7eb",
  },
  panelTitle: {
    fontSize: "1.2rem",
    marginBottom: "1rem",
    paddingBottom: "0.5rem",
    borderBottom: "2px solid #e5e7eb",
  },
  formGroup: {
    marginBottom: "1.2rem",
  },
  label: {
    display: "block",
    fontWeight: "600",
    marginBottom: "0.4rem",
    fontSize: "0.9rem",
  },
  speciesGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr 1fr",
    gap: "0.5rem",
  },
  speciesButton: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "0.8rem 0.5rem",
    border: "2px solid #e5e7eb",
    borderRadius: "10px",
    background: "#fff",
    cursor: "pointer",
    transition: "all 0.2s",
    position: "relative",
  },
  speciesButtonActive: {
    borderColor: "#3b82f6",
    background: "#eff6ff",
  },
  speciesEmoji: {
    fontSize: "1.5rem",
  },
  speciesLabel: {
    fontSize: "0.85rem",
    fontWeight: "600",
    marginTop: "0.2rem",
  },
  speciesBadge: {
    fontSize: "0.6rem",
    background: "#fef3c7",
    color: "#92400e",
    padding: "1px 6px",
    borderRadius: "4px",
    marginTop: "0.2rem",
  },
  speciesBadgeReady: {
    fontSize: "0.6rem",
    background: "#d1fae5",
    color: "#065f46",
    padding: "1px 6px",
    borderRadius: "4px",
    marginTop: "0.2rem",
  },
  select: {
    width: "100%",
    padding: "0.6rem",
    borderRadius: "8px",
    border: "1px solid #d1d5db",
    fontSize: "0.9rem",
  },
  uploadArea: {
    position: "relative",
  },
  fileInput: {
    position: "absolute",
    opacity: 0,
    width: "100%",
    height: "100%",
    cursor: "pointer",
  },
  uploadLabel: {
    display: "block",
    cursor: "pointer",
  },
  uploadPlaceholder: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "150px",
    border: "2px dashed #d1d5db",
    borderRadius: "10px",
    color: "#9ca3af",
    gap: "0.5rem",
  },
  preview: {
    width: "100%",
    maxHeight: "200px",
    objectFit: "contain",
    borderRadius: "10px",
    border: "1px solid #e5e7eb",
  },
  buttonGroup: {
    display: "flex",
    gap: "0.5rem",
    marginBottom: "1rem",
  },
  predictButton: {
    flex: 1,
    padding: "0.8rem",
    borderRadius: "8px",
    border: "none",
    background: "#3b82f6",
    color: "#fff",
    fontWeight: "600",
    fontSize: "0.95rem",
    cursor: "pointer",
  },
  resetButton: {
    padding: "0.8rem 1.2rem",
    borderRadius: "8px",
    border: "1px solid #d1d5db",
    background: "#fff",
    cursor: "pointer",
    fontSize: "0.95rem",
  },
  infoBox: {
    background: "#f9fafb",
    padding: "0.8rem",
    borderRadius: "8px",
    fontSize: "0.85rem",
    lineHeight: "1.6",
    border: "1px solid #e5e7eb",
  },
  errorBox: {
    background: "#fef2f2",
    color: "#dc2626",
    padding: "1rem",
    borderRadius: "8px",
    border: "1px solid #fecaca",
  },
  topResult: {
    display: "flex",
    alignItems: "center",
    gap: "1rem",
    padding: "1.2rem",
    borderRadius: "10px",
    border: "2px solid",
    marginBottom: "1rem",
    background: "#fafafa",
  },
  topResultEmoji: {
    fontSize: "2.5rem",
  },
  topResultClass: {
    fontSize: "1.2rem",
    fontWeight: "700",
  },
  topResultConf: {
    fontSize: "0.95rem",
    color: "#6b7280",
  },
  topResultSpecies: {
    fontSize: "0.85rem",
    color: "#9ca3af",
  },
  predictionsContainer: {
    marginBottom: "1rem",
  },
  predictionRow: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    marginBottom: "0.4rem",
  },
  predictionInfo: {
    width: "180px",
    display: "flex",
    gap: "0.4rem",
    fontSize: "0.85rem",
  },
  predictionRank: {
    color: "#9ca3af",
    fontWeight: "600",
  },
  predictionBarContainer: {
    flex: 1,
    height: "8px",
    background: "#f3f4f6",
    borderRadius: "4px",
    overflow: "hidden",
  },
  predictionBar: {
    height: "100%",
    borderRadius: "4px",
    transition: "width 0.5s ease",
  },
  predictionPercent: {
    width: "50px",
    textAlign: "right",
    fontSize: "0.85rem",
    fontWeight: "600",
  },
  metaInfo: {
    display: "flex",
    justifyContent: "center",
    gap: "1.5rem",
    fontSize: "0.8rem",
    color: "#9ca3af",
    paddingTop: "0.5rem",
    borderTop: "1px solid #f3f4f6",
  },
  placeholder: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "300px",
    color: "#9ca3af",
    gap: "0.5rem",
  },
  footer: {
    textAlign: "center",
    marginTop: "2rem",
    fontSize: "0.75rem",
    color: "#9ca3af",
    lineHeight: "1.6",
  },
};

export default App;
