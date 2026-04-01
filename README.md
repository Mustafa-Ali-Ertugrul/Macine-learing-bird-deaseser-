# 🐔 Poultry Disease Classification - Production Ready

Production-grade deep learning system for classifying poultry diseases using histopathology images.
Supports **multi-species** classification: Chicken 🐔, Goose 🦢, and Duck 🦆.

## 📊 Dataset

10 disease classes with 500+ images each per species:
- **Avian_Influenza** - Avian Influenza (Bird Flu)
- **Coccidiosis** - Coccidiosis infection
- **Fowl_Pox** - Fowl Pox disease
- **Healthy** - Normal healthy tissue
- **Histomoniasis** - Histomoniasis (Blackhead disease)
- **Infectious_Bronchitis** - Infectious Bronchitis (IB)
- **Infectious_Bursal_Disease** - Infectious Bursal Disease (IBD)
- **Mareks_Disease** - Marek's Disease
- **Newcastle_Disease** - Newcastle Disease (NDV)
- **Salmonella** - Salmonella infection

### Supported Species
| Species | Dataset | Status |
|---------|---------|--------|
| 🐔 Chicken | `final_dataset_10_classes` | ✅ Ready |
| 🦢 Goose | `goose_dataset_10_classes` | ✅ Ready |
| 🦆 Duck | `duck_dataset_10_classes` | ✅ Ready |

## 🚀 Quick Start

### Installation (Windows)

```bash
# Clone repository
git clone <repo-url>
cd <repo-name>

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train ViT-B/16 for chicken (default)
python train_model.py --model vit_b16

# Train for goose
python train_model.py --model vit_b16 --species goose

# Train for duck
python train_model.py --model vit_b16 --species duck

# Train ResNet50
python train_model.py --model resnet50 --species chicken

# With YAML config
python train_model.py --config config/training_config.yaml
```

### Evaluation

```bash
# Evaluate model
python evaluate_model.py --model vit_b16 --species chicken

# Compare all models
python compare_models.py
python compare_models.py --species goose
```

### Single Image Prediction

```bash
# Predict disease from image
python predict_single.py --image test_image.jpg --species chicken
python predict_single.py --image test_image.jpg --species goose --model resnet50
```

### API Server

```bash
# Start FastAPI server
uvicorn api.main:app --reload --port 8000

# Test prediction (PowerShell)
curl.exe -X POST "http://localhost:8000/predict" -F "file=@test_image.jpg"

# API docs: http://localhost:8000/docs
```

## 📁 Project Structure

```
├── config/                    # YAML configuration files
│   ├── training_config.yaml   # Chicken training config
│   ├── training_config_duck.yaml
│   └── training_config_goose.yaml
├── src/                       # Source code
│   ├── config.py              # Multi-species configuration
│   ├── models/                # Model factory (ResNet, EfficientNet, etc.)
│   ├── training/              # Training pipeline (Trainer, callbacks, metrics)
│   ├── data/                  # Dataset utilities
│   └── utils/                 # Logger, helpers
├── api/                       # FastAPI REST API
│   └── main.py                # Multi-species prediction API
├── scripts/                   # Training and utility scripts
├── tests/                     # Unit tests
├── final_dataset_10_classes/  # Chicken dataset
├── duck_dataset_10_classes/   # Duck dataset
├── goose_dataset_10_classes/  # Goose dataset
├── models/                    # Trained model weights
├── vit_poultry_results/       # ViT-B/16 training results
├── resnext_poultry_results/   # ResNeXt-50 results
├── resnest_poultry_results/   # ResNeSt-50d results
├── convnext_poultry_results/  # ConvNeXt-Tiny results
├── cvt_poultry_results/       # CvT-13 results
├── train_model.py             # Main training script
├── evaluate_model.py          # Model evaluation
├── predict_single.py          # Single image prediction
├── compare_models.py          # Model comparison
└── requirements.txt           # Python dependencies
```

## 🧠 Model Architectures

| Model | Parameters | Test Accuracy | Training Time |
|-------|-----------|---------------|---------------|
| **ViT-B/16** | ~86M | **98.14%** | 32.5 min |
| **ResNeXt-50** | ~25M | ✅ Completed | 30.4 min |
| **ResNeSt-50d** | ~27M | ✅ Completed | 41.4 min |
| **ConvNeXt-Tiny** | ~28M | ✅ Completed | - |
| **CvT-13** | ~11M | ✅ Completed | 22.5 min |

Additional supported models:
- **ResNet18/34/50/101** - Classic architectures
- **EfficientNet B0/B1/B2** - Efficient scaling
- **MobileNetV2** - Lightweight, mobile-friendly

## 🔬 Features

### Implemented ✅
- **Multi-Species Support**: Chicken, Goose, Duck classification
- **5+ Model Architectures**: ViT, ResNeXt, ResNeSt, ConvNeXt, CvT
- **Advanced Training Pipeline**: Cosine annealing, early stopping, mixed precision
- **Data Augmentation**: Flip, rotation, color jitter, affine transforms
- **FastAPI REST API**: Multi-species prediction endpoint
- **Model Evaluation**: Confusion matrix, classification report, F1 scores
- **Single Image Prediction**: Quick inference script
- **Model Comparison**: Compare all trained models at once
- **TensorBoard Logging**: Training visualization
- **Checkpoint Saving**: Best model + final model
- **Label Smoothing**: Better generalization
- **Class Weights**: Handle imbalanced data

### Planned 📋
- **Grad-CAM Visualization**: Model interpretability
- **K-Fold Cross-Validation**: Robust evaluation
- **Hyperparameter Tuning**: Optuna integration
- **ONNX Export**: Model optimization
- **Docker Deployment**: Containerization
- **MLflow Tracking**: Experiment management
- **CI/CD Pipeline**: Automated testing

## 🛠️ Development

### Code Quality

```bash
# Format code
black train_model.py evaluate_model.py predict_single.py src/ api/

# Lint code
ruff check train_model.py evaluate_model.py predict_single.py src/ api/

# Run tests
pytest tests/ --cov=src --cov-report=html
```

### Configuration

Edit `config/training_config.yaml` to customize:
- Model architecture
- Training hyperparameters
- Data augmentation
- Logging settings

Example:
```yaml
model:
  architecture: "vit_b16"  # Change to resnet50, efficientnet_b0, etc.
  pretrained: true
  dropout: 0.4

training:
  num_epochs: 30
  batch_size: 16
  optimizer:
    type: "adamw"
    lr: 0.0001
    weight_decay: 0.0001
  early_stopping:
    enabled: true
    patience: 10
  mixed_precision: true
```

## 📈 Performance

Current implementation features:
- **Mixed Precision Training**: Faster training with lower memory
- **Class Weights**: Handle imbalanced datasets
- **Early Stopping**: Prevent overfitting (patience=10)
- **Cosine Annealing LR**: Adaptive learning rate scheduling
- **Data Augmentation**: Comprehensive transforms (flip, rotation, color jitter, affine)
- **Label Smoothing**: Better generalization (0.1)
- **Multi-Species**: Separate models for chicken, goose, duck

## 📝 Usage Examples

### Load Configuration

```python
from src.config import get_config, SUPPORTED_SPECIES

config = get_config("vit_b16", species="chicken")
print(config["model_save_path"])
print(config["num_classes"])
```

### Train a Model

```bash
# Train ViT-B/16 for chicken
python train_model.py --model vit_b16 --species chicken

# Train ResNet50 for goose
python train_model.py --model resnet50 --species goose
```

### Evaluate a Model

```bash
# Evaluate and get confusion matrix
python evaluate_model.py --model vit_b16 --species chicken
```

### Predict Single Image

```python
from predict_single import predict

result = predict("test_image.jpg", model_name="vit_b16", species="chicken")
print(f"Disease: {result['top_prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
for pred in result['predictions']:
    print(f"  {pred['class']}: {pred['confidence']:.2%}")
```

## 🤝 Contributing

This is an educational and research project. Contributions are welcome!

1. Create a new branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## 📄 License

Educational and research use only.

## 🙏 Acknowledgments

- PyTorch team for the amazing framework
- Hugging Face for pre-trained transformers
- timm library for additional model architectures
- The open-source community

---

**Status**: Multi-species support implemented ✅ | 5 models trained ✅ | REST API ready ✅
