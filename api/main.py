"""FastAPI REST API for poultry disease classification."""
import io
import sys
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_factory import get_model
from src.data.transforms import get_val_transforms
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionResponse(BaseModel):
    """Prediction response model."""
    success: bool
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    model_architecture: Optional[str] = None


# Global model state
model = None
device = None
class_names = None
class_to_idx = None
transform = None
model_config = None


def load_model():
    """Load the trained model."""
    global model, device, class_names, class_to_idx, transform, model_config
    
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
        if not config_path.exists():
            logger.warning(f"Config not found at {config_path}, using defaults")
            config = {}
        else:
            config = ConfigLoader.load(config_path)
        
        model_config = config.get('model', {})
        data_config = config.get('data', {})
        
        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load checkpoint
        checkpoint_path = Path(__file__).parent.parent / "models" / "best_model.pth"
        if not checkpoint_path.exists():
            checkpoint_path = Path(__file__).parent.parent / "best_poultry_disease_model.pth"
        
        if not checkpoint_path.exists():
            logger.error(f"No model checkpoint found")
            return False
        
        logger.info(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get class mapping
        if 'class_to_idx' in checkpoint:
            class_to_idx = checkpoint['class_to_idx']
        else:
            # Default classes
            class_to_idx = {
                'Avian_Influenza': 0,
                'Coccidiosis': 1,
                'Fowl_Pox': 2,
                'Healthy': 3,
                'Histomoniasis': 4,
                'Infectious_Bronchitis': 5,
                'Infectious_Bursal_Disease': 6,
                'Mareks_Disease': 7,
                'Newcastle_Disease': 8,
                'Salmonella': 9
            }
        
        class_names = list(class_to_idx.keys())
        num_classes = len(class_names)
        
        # Create model
        model_name = model_config.get('architecture', 'resnet50')
        model = get_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            dropout=model_config.get('dropout', 0.3)
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        # Create transform
        image_size = data_config.get('image_size', 224)
        transform = get_val_transforms(image_size)
        
        logger.info(f"Model loaded successfully: {model_name}")
        logger.info(f"Classes: {class_names}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting up API server...")
    success = load_model()
    if not success:
        logger.warning("Model could not be loaded, API will run without model")
    yield
    # Shutdown
    logger.info("Shutting down API server...")


# Create FastAPI app
app = FastAPI(
    title="Poultry Disease Classification API",
    description="REST API for classifying poultry diseases from histopathology images",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API info."""
    return HealthResponse(
        status="running",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        model_architecture=model_config.get('architecture') if model_config else None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        model_architecture=model_config.get('architecture') if model_config else None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict poultry disease from uploaded image.
    
    Args:
        file: Image file to classify
        
    Returns:
        Prediction results with class probabilities
    """
    import time
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'image/bmp']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            probs = probabilities[0].cpu().numpy()
        
        # Get prediction
        predicted_idx = int(np.argmax(probs))
        predicted_class = class_names[predicted_idx]
        confidence = float(probs[predicted_idx])
        
        # Create probability dict
        prob_dict = {
            class_names[i]: float(probs[i])
            for i in range(len(class_names))
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            success=True,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict poultry diseases for multiple images.
    
    Args:
        files: List of image files to classify
        
    Returns:
        List of prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            # Preprocess
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                probs = probabilities[0].cpu().numpy()
            
            # Get prediction
            predicted_idx = int(np.argmax(probs))
            predicted_class = class_names[predicted_idx]
            confidence = float(probs[predicted_idx])
            
            results.append({
                "filename": file.filename,
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"predictions": results}


@app.get("/classes")
async def get_classes():
    """Get list of supported disease classes."""
    if class_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": class_names,
        "num_classes": len(class_names)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
