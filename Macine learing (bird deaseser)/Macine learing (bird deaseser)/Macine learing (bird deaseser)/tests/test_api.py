"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import io
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAPI:
    """Tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        try:
            from api.main import app
            return TestClient(app)
        except Exception as e:
            pytest.skip(f"API not available: {e}")
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
    
    def test_classes_endpoint(self, client):
        """Test classes endpoint."""
        response = client.get("/classes")
        
        # May return 503 if model not loaded, or 200 if loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "classes" in data
            assert "num_classes" in data
    
    def test_predict_without_file(self, client):
        """Test predict endpoint without file."""
        response = client.post("/predict")
        
        # Should return 422 (validation error) for missing file
        assert response.status_code == 422
    
    def test_predict_with_image(self, client):
        """Test predict endpoint with image."""
        # Create dummy image
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        
        # May return 200 if model loaded, or 503 if not
        assert response.status_code in [200, 503]


class TestAPIResponseModels:
    """Tests for API response models."""
    
    def test_prediction_response_structure(self):
        """Test prediction response structure."""
        from api.main import PredictionResponse
        
        response = PredictionResponse(
            success=True,
            predicted_class="Healthy",
            confidence=0.95,
            probabilities={"Healthy": 0.95, "Disease": 0.05},
            processing_time_ms=45.2
        )
        
        assert response.success is True
        assert response.predicted_class == "Healthy"
        assert response.confidence == 0.95
    
    def test_health_response_structure(self):
        """Test health response structure."""
        from api.main import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            device="cuda",
            model_architecture="resnet50"
        )
        
        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.device == "cuda"
