# 🐔 Poultry Disease Classification - Production Ready

Production-grade machine learning system for classifying poultry diseases using histopathology images.

## 📊 Dataset

10 disease classes with 500+ images each:
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

# Setup DVC (for data versioning)
dvc pull
```

### Training

```bash
# Basic training
python scripts/train.py --config config/training_config.yaml

# With TensorBoard monitoring
tensorboard --logdir runs
```

### API Server

```bash
# Start FastAPI server
uvicorn api.main:app --reload

# Test prediction (PowerShell)
curl.exe -X POST "http://localhost:8000/predict" -F "file=@test_image.jpg"

# API docs: http://localhost:8000/docs
```

## 📁 Project Structure

```
├── config/              # YAML configuration files
│   └── training_config.yaml
├── src/                 # Source code
│   ├── models/          # Model definitions
│   ├── training/        # Training pipeline
│   ├── data/            # Dataset and transforms
│   ├── utils/           # Logger, config loader, metrics
│   ├── preprocessing/   # Data preprocessing
│   ├── augmentation/    # Advanced augmentation (Phase 2)
│   ├── visualization/   # Grad-CAM, plots (Phase 2)
│   └── optimization/    # Hyperparameter tuning (Phase 2)
├── api/                 # FastAPI REST API (Phase 2)
├── tests/               # Unit tests (Phase 2)
├── scripts/             # Training and utility scripts
├── deployment/          # Docker, ONNX export (Phase 3)
├── mlops/               # MLflow tracking (Phase 3)
├── data/                # Dataset (managed by DVC)
│   └── final_dataset_10_classes/
├── models/              # Trained models (Git LFS/DVC)
├── logs/                # Log files
├── runs/                # TensorBoard logs
├── requirements.txt     # Python dependencies
└── README.md
```

## 🧠 Model Architectures

Supported models (configured in `config/training_config.yaml`):
- **ResNet18** - Fast, lightweight
- **ResNet50** - Better accuracy
- **EfficientNet B0, B2** - Efficient architecture
- **ConvNeXt Tiny** - Modern architecture

## 🔬 Features

### Phase 1 (Implemented) ✅
- **Modular Architecture**: Clean, maintainable code structure
- **Configuration Management**: YAML-based config files
- **Logging System**: Centralized logging to console and files
- **Type Hints**: Full type annotations for better code quality
- **Data Management**: Dataset class with transforms

### Phase 2 (Planned)
- **Advanced Augmentation**: CutMix, MixUp, AutoAugment
- **K-Fold Cross-Validation**: Robust model evaluation
- **Hyperparameter Tuning**: Optuna integration
- **Model Interpretability**: Grad-CAM visualization
- **REST API**: FastAPI for model serving
- **TensorBoard**: Training visualization
- **Unit Tests**: Comprehensive test coverage

### Phase 3 (Planned)
- **MLflow**: Experiment tracking
- **Ensemble Models**: Multiple model fusion
- **ONNX Export**: Model optimization
- **Docker**: Containerization
- **CI/CD**: Automated testing and deployment
- **DVC**: Data versioning pipeline

## 🛠️ Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/

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
  architecture: "resnet50"  # Change to resnet18, efficientnet_b0, etc.
  
training:
  num_epochs: 50
  batch_size: 32
  optimizer:
    lr: 0.0001
```

## 📈 Performance

Current implementation features:
- **Mixed Precision Training**: Faster training with lower memory
- **Class Weights**: Handle imbalanced datasets
- **Early Stopping**: Prevent overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **Data Augmentation**: Comprehensive transforms

## 🔧 Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0 (PyTorch Image Models)
- pyyaml >= 6.0
- tensorboard >= 2.13.0

## 📝 Usage Examples

### Load Configuration

```python
from src.utils.config_loader import ConfigLoader

config = ConfigLoader.load('config/training_config.yaml')
print(config['model']['architecture'])
```

### Create Logger

```python
from src.utils.logger import get_logger

logger = get_logger('training')
logger.info('Training started')
logger.warning('High memory usage detected')
```

### Create Model

```python
from src.models.model_factory import get_model

model = get_model(
    model_name='resnet50',
    num_classes=10,
    pretrained=True
)
```

### Load Dataset

```python
from src.data.dataset import PoultryDiseaseDataset
from src.data.transforms import get_train_transforms

dataset = PoultryDiseaseDataset(
    image_paths=image_paths,
    labels=labels,
    class_to_idx=class_to_idx,
    transform=get_train_transforms(224)
)
```

## 🤝 Contributing

This is an educational project. Contributions are welcome!

1. Create a new branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## 📄 License

Educational and research use only.

## 🙏 Acknowledgments

- PyTorch team for the amazing framework
- torchvision for pre-trained models
- The open-source community

---

**Status**: Phase 1 Implemented ✅ | Phase 2 In Progress 🚧 | Phase 3 Planned 📋
