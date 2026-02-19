# Technology Stack

## Core Technologies

### Machine Learning Framework
- **PyTorch** - Primary deep learning framework
- **Transformers (Hugging Face)** - Pre-trained model library
- **timm** - PyTorch image models library
- **scikit-learn** - Traditional ML utilities and metrics

### Model Architectures
- **Vision Transformer (ViT)** - `google/vit-base-patch16-224-in21k`
- **ConvNeXt** - `facebook/convnext-tiny-224`
- **ResNeXt** - `resnext50_32x4d`
- **ResNeSt** - Custom implementation
- **CVT** - `microsoft/cvt-21-384-224`

### Web Application
- **React 19.1.1** - Frontend framework
- **Vite** - Build tool and dev server
- **TailwindCSS 4.1.14** - Styling framework
- **Lucide React** - Icon library

### Data Processing
- **PIL/Pillow** - Image processing
- **OpenCV** - Computer vision utilities
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib** - Visualization

## Build System & Commands

### Python Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Set up Python path (Windows encoding fix included in scripts)
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Model Training
```bash
# Train specific model
python train_model.py --model vit_b16
python train_model.py --model convnext_tiny
python train_model.py --model resnext50

# Train all models sequentially
python train_all_models_sequential.py

# Individual model training scripts
python train_convnext.py
python train_vit_b16.py
python train_resnext.py
```

### Dataset Management
```bash
# Organize dataset splits
python organize_dataset_splits_physically.py

# Validate dataset integrity
python validate_dataset_integrity.py

# Clean dataset (remove duplicates)
python clean_dataset.py
```

### Web Application
```bash
# Navigate to React app directory
cd app/poultry-pathology-app
# or
cd poultry-pathology-app

# Install dependencies
npm install

# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

### Evaluation & Analysis
```bash
# Evaluate trained models
python evaluate_model.py

# Generate comprehensive reports
python generate_final_report.py

# Monitor training progress
python monitor_training.py
```

## Configuration

### Model Configuration
- Centralized config in `src/config.py`
- Default image size: 224x224
- Batch size: 16 (adjustable per model)
- Learning rate: 5e-5
- Weight decay: 0.01

### Data Paths
- Dataset: `final_dataset_split/` (train/val/test structure)
- Models: `*_poultry_results/` directories
- Output: Individual result directories per model

### Hardware Requirements
- CUDA-compatible GPU recommended
- Automatic fallback to CPU
- Windows encoding fixes included for console output

## Development Patterns

### File Naming Conventions
- Training scripts: `train_*.py`
- Evaluation scripts: `evaluate_*.py`
- Utility scripts: descriptive names in `src/`
- Model outputs: `{model_name}_poultry_results/`

### Error Handling
- Windows console encoding fixes in all main scripts
- Graceful GPU/CPU fallback
- Comprehensive error logging in training scripts