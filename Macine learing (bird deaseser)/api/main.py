"""
Çok-türlü hastalık sınıflandırma REST API.

Kullanım:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /                  → API bilgisi
    GET  /species           → Desteklenen türler
    GET  /species/{name}    → Tür detayı ve dataset durumu
    POST /predict           → Hastalık tahmini
    GET  /health            → Sağlık kontrolü
"""

import os
import sys
import io
import time
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# Add project root to path
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    SPECIES_CONFIG,
    SUPPORTED_SPECIES,
    DISEASE_CLASSES,
    get_config,
    check_dataset_exists,
)
from predict_single import predict as predict_image

# ─────────────────────────────────────────
# App
# ─────────────────────────────────────────
app = FastAPI(
    title="🐦 Kümes Hayvanı Hastalık Sınıflandırma API",
    description=(
        "Tavuk, kaz ve ördek türleri için derin öğrenme tabanlı "
        "hastalık sınıflandırma servisi."
    ),
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Model cache (her tür için ayrı)
# ─────────────────────────────────────────
_loaded_models = {}


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "Kümes Hayvanı Hastalık Sınıflandırma API",
        "version": "2.0.0",
        "supported_species": SUPPORTED_SPECIES,
        "disease_classes": DISEASE_CLASSES,
        "endpoints": {
            "/species": "Desteklenen türleri listele",
            "/predict": "Hastalık tahmini yap (POST)",
            "/health": "Sağlık kontrolü",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "loaded_models": list(_loaded_models.keys()),
        "timestamp": time.time(),
    }


@app.get("/species")
async def list_species():
    """Desteklenen tüm türleri listele."""
    species_list = []
    for sp_key, sp_conf in SPECIES_CONFIG.items():
        dataset_status = check_dataset_exists(sp_key)
        model_available = os.path.exists(
            get_config("vit_b16", sp_key)["model_save_path"]
        )
        species_list.append({
            "id": sp_key,
            "display_name": sp_conf["display_name"],
            "dataset_ready": dataset_status["ready_for_training"],
            "total_images": dataset_status["total_images"],
            "model_available": model_available,
        })
    return {"species": species_list}


@app.get("/species/{species_name}")
async def get_species_detail(species_name: str):
    """Belirli bir türün detaylı bilgisini döndür."""
    if species_name not in SUPPORTED_SPECIES:
        raise HTTPException(
            status_code=404,
            detail=f"Tür bulunamadı: '{species_name}'. "
                   f"Desteklenen: {SUPPORTED_SPECIES}",
        )

    dataset_status = check_dataset_exists(species_name)
    config = get_config("vit_b16", species_name)

    available_models = []
    for model_name in ["vit_b16", "resnext50", "resnest50d", "convnext_tiny", "cvt_13", "resnet50", "efficientnet_b0", "mobilenet_v2"]:
        model_path = get_config(model_name, species_name)["model_save_path"]
        if os.path.exists(model_path):
            available_models.append(model_name)

    return {
        "species": species_name,
        "display_name": SPECIES_CONFIG[species_name]["display_name"],
        "disease_classes": DISEASE_CLASSES,
        "dataset": dataset_status,
        "available_models": available_models,
        "config": {
            "raw_data_dir": config["raw_data_dir"],
            "model_dir": config["model_dir"],
            "results_dir": config["results_dir"],
        },
    }


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(..., description="Görüntü dosyası"),
    species: str = Query(
        default="chicken",
        description="Hayvan türü",
    ),
    model: str = Query(
        default="vit_b16",
        description="Model mimarisi",
    ),
    top_k: int = Query(default=5, ge=1, le=10, description="Top-K tahmin sayısı"),
):
    """
    Yüklenen görüntü için hastalık tahmini yap.

    - **file**: Kuş görüntüsü (JPG, PNG)
    - **species**: chicken, goose veya duck
    - **model**: Model mimarisi
    - **top_k**: Kaç tahmin döndürülsün
    """
    # Tür doğrulama
    if species not in SUPPORTED_SPECIES:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen tür: '{species}'. Desteklenen: {SUPPORTED_SPECIES}",
        )

    # Dosya tipi kontrolü
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Yalnızca görüntü dosyaları kabul edilir.",
        )

    # Model varlık kontrolü
    config = get_config(model, species)
    if not os.path.exists(config["model_save_path"]):
        raise HTTPException(
            status_code=404,
            detail=(
                f"{species} türü için {model} modeli bulunamadı. "
                f"Önce eğitim yapın: "
                f"python train_model.py --model {model} --species {species}"
            ),
        )

    temp_path = None
    try:
        # Görüntüyü geçici dosyaya kaydet
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        temp_path = f"temp_predict_{species}_{int(time.time())}.jpg"
        image.save(temp_path)

        # Tahmin
        start_time = time.time()
        result = predict_image(
            image_path=temp_path,
            model_name=model,
            species=species,
            top_k=top_k,
        )
        inference_time = time.time() - start_time

        result["inference_time_ms"] = round(inference_time * 1000, 2)
        result["filename"] = file.filename

        return JSONResponse(content=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")
    finally:
        # Temizle
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ─────────────────────────────────────────
# Startup
# ─────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 60)
    print("  🐦 Kümes Hayvanı Hastalık API Başlatılıyor...")
    print("=" * 60)

    for sp in SUPPORTED_SPECIES:
        report = check_dataset_exists(sp)
        status = "✅" if report["ready_for_training"] else "⏳"
        display = SPECIES_CONFIG[sp]["display_name"]
        print(f"  {status} {display}: {report['total_images']} görüntü")

    print("=" * 60 + "\n")
