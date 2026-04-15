# Project Structure

## Root Directory Organization

### Core Directories

```
├── src/                          # Core source code modules
│   ├── config.py                 # Centralized configuration
│   ├── dataset_utils.py          # Dataset handling utilities
│   └── training_utils.py         # Training helper functions
├── data/                         # Dataset storage
│   ├── metadata/                 # CSV files and labels
│   ├── processed/                # Organized datasets
│   └── raw/                      # Original downloaded data
├── docs/                         # Project documentation
├── scripts/                      # Utility scripts organized by function
│   ├── collection/               # Data collection scripts
│   ├── labeling/                 # Annotation tools
│   ├── training/                 # Training utilities
│   └── utils/                    # General utilities
├── models/                       # Saved model files (.pth)
├── results/                      # Training results and outputs
└── tests/                        # Test files
```

### Web Applications

```
├── app/poultry-pathology-app/    # Main React application
├── poultry-pathology-app/        # Alternative React app location
└── templates/                    # HTML templates for tools
```

### Model Results Structure

Each trained model creates its own results directory:
```
├── {model_name}_poultry_results/
│   ├── checkpoint-*/             # Training checkpoints
│   ├── final_model/              # Best model artifacts
│   ├── confusion_matrix.png      # Evaluation visualizations
│   └── training_logs.txt         # Training progress logs
```

## Key File Patterns

### Training Scripts
- `train_model.py` - Universal training script (supports multiple models)
- `train_{model_name}.py` - Individual model training scripts
- `train_all_models_sequential.py` - Batch training orchestrator

### Dataset Management
- `organize_dataset_splits_physically.py` - Create train/val/test splits
- `validate_dataset_integrity.py` - Check for data leakage and quality issues
- `clean_dataset*.py` - Various dataset cleaning utilities

### Evaluation & Analysis
- `evaluate_{model_name}.py` - Model-specific evaluation
- `generate_final_report.py` - Comprehensive analysis reports
- `*_analysis.py` - Various analysis scripts

### Labeling Tools
- `poultry_labeling_tool.html` - Web-based image labeling interface
- `label_*.py` - Python-based labeling utilities

## Data Organization Patterns

### Dataset Structure (Target)
```
final_dataset_split/
├── train/                        # Training data (70%)
│   ├── Avian_Influenza/
│   ├── Coccidiosis/
│   ├── Fowl_Pox/
│   ├── Healthy/
│   ├── Histomoniasis/
│   ├── Infectious_Bronchitis/
│   ├── Infectious_Bursal_Disease/
│   ├── Mareks_Disease/
│   ├── Newcastle_Disease/
│   └── Salmonella/
├── val/                          # Validation data (15%)
│   └── [same class structure]
└── test/                         # Test data (15%)
    └── [same class structure]
```

### Metadata Files
```
data/metadata/
├── final_dataset_10_classes.csv  # Complete dataset manifest
├── poultry_labeled.csv           # Labeled subset
├── poultry_labeled_12k.csv       # Extended labeled set
└── labeling_batch.csv            # Current labeling batch
```

## Configuration Architecture

### Centralized Configuration (`src/config.py`)
- `COMMON_CONFIG` - Shared settings across all models
- `MODEL_CONFIGS` - Model-specific configurations
- `TRANSFORM_CONFIG` - Data augmentation settings
- `DISEASE_CLASSES` - Standardized class definitions

### Model-Specific Settings
Each model has its own configuration block with:
- Model name/path for Hugging Face or PyTorch
- Output directory
- Batch size optimizations
- Architecture-specific parameters

## Naming Conventions

### Files
- Python scripts: `snake_case.py`
- Model results: `{model_name}_poultry_results/`
- Checkpoints: `checkpoint-{step}/`
- Reports: `*_REPORT.md` or `*_report.txt`

### Classes and Functions
- Classes: `PascalCase` (e.g., `PoultryImageDataset`)
- Functions: `snake_case` (e.g., `prepare_datasets`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DISEASE_CLASSES`)

### Image Files
- Original: `{disease}_{id}.jpg`
- Augmented: `safe_aug_{id}_{original_name}.jpg` (⚠️ causes data leakage)
- Cleaned: Remove augmentation prefixes from dataset

## Critical Data Flow

### Training Pipeline
1. **Data Preparation**: `organize_dataset_splits_physically.py`
2. **Validation**: `validate_dataset_integrity.py`
3. **Training**: `train_model.py` or individual scripts
4. **Evaluation**: `evaluate_*.py`
5. **Reporting**: `generate_final_report.py`

### Dataset Quality Assurance
1. **Integrity Check**: Detect duplicates and data leakage
2. **Quality Assessment**: Check image sizes, corruption
3. **Class Balance**: Analyze distribution across classes
4. **Split Validation**: Ensure proper train/val/test isolation

## Important Notes

### Data Leakage Prevention
- ⚠️ **Never split after augmentation** - causes train/test leakage
- ✅ **Split first, then augment** training data only
- ✅ **Use hash-based duplicate detection** for quality assurance
- ✅ **Keep test set completely isolated** from any augmentation

### Windows Compatibility
- All main scripts include Windows console encoding fixes
- Use forward slashes in paths where possible
- Handle special characters in Turkish disease names properly

### Model Artifacts
- Each model saves to its own `{model_name}_poultry_results/` directory
- Final models saved in `final_model/` subdirectory
- Confusion matrices and training logs included automatically