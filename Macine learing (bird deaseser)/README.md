# 🐔 Poultry Disease Classification Project

Complete machine learning solution for classifying poultry diseases using histopathology images.

## 📊 Dataset

The project uses **final_dataset_10_classes** containing histopathology images organized into 10 categories:

### Disease Classes

- 🟢 **Healthy** - Normal poultry tissue
- 🔴 **Avian_Influenza** - Avian Influenza (Bird Flu)
- 🟡 **Coccidiosis** - Coccidiosis infection
- 🟠 **Fowl_Pox** - Fowl Pox disease
- 🔵 **Healthy** - Normal healthy tissue
- 🟤 **Histomoniasis** - Histomoniasis (Blackhead disease)
- 🟣 **Infectious_Bronchitis** - Infectious Bronchitis (IB)
- 🟠 **Infectious_Bursal_Disease** - Infectious Bursal Disease (IBD)
- 🟢 **Mareks_Disease** - Marek's Disease
- 🔶 **Newcastle_Disease** - Newcastle Disease (NDV)
- 🔵 **Salmonella** - Salmonella infection

## 🛠️ Available Tools

### Data Management
| Script | Description |
|---------|-------------|
| `verify_dataset.py` | Verify dataset integrity and check for corrupted images |
| `analyze_dataset_enhanced.py` | Comprehensive dataset analysis with statistics |
| `organize_dataset.py` | Organize images into class directories or create train/val/test splits |
| `prepare_training.py` | Prepare datasets for model training |

### Model Training
| Script | Description |
|---------|-------------|
| `train_model.py` | Train ResNet18 model with early stopping and learning rate scheduling |

### Labeling
| Tool | Description |
|------|-------------|
| `poultry_labeling_tool.html` | Interactive web-based tool for labeling images |

## 🚀 Quick Start

### 1. Verify Dataset
```bash
python verify_dataset.py
```

### 2. Analyze Dataset
```bash
python analyze_dataset_enhanced.py
```

### 3. Organize Dataset
```bash
# Create train/val/test splits (70/15/15)
python organize_dataset.py
# Select option 2
```

### 4. Train Model
```bash
python train_model.py
```

## 📦 Project Structure

```
.
├── final_dataset_10_classes/    # Main dataset (10 classes)
├── poultry_labeled_12k.csv      # Labeled image metadata
├── poultry_labeling_tool.html    # Interactive labeling tool
├── train_model.py               # Model training script
├── organize_dataset.py           # Dataset organization
├── prepare_training.py          # Training preparation
├── verify_dataset.py            # Dataset verification
├── analyze_dataset_enhanced.py  # Dataset analysis
├── reports/                     # Analysis reports and figures
└── CODE_IMPROVEMENTS.md       # Recent code improvements
```

## 🎯 Model Features

- **Architecture**: ResNet18 (pre-trained)
- **Training**: Transfer learning with frozen early layers
- **Optimization**: Adam optimizer with weight decay
- **Scheduling**: ReduceLROnPlateau learning rate scheduler
- **Regularization**: Early stopping (patience=5)
- **Augmentation**: Random flip, rotation, color jitter

## 📈 Performance

- **Batch Size**: 32 (optimized for GPU)
- **Epochs**: 20 with early stopping
- **Validation**: 80/20 train/validation split
- **Metrics**: Accuracy, classification report per class

## 🔧 Requirements

```
torch>=1.12.0
torchvision>=0.13.0
pandas>=1.4.0
scikit-learn>=1.1.0
Pillow>=9.0.0
tqdm>=4.64.0
```

Install dependencies:
```bash
pip install torch torchvision pandas scikit-learn Pillow tqdm
```

## 📝 Recent Improvements

### Code Quality
- ✅ Updated deprecated model loading syntax
- ✅ Added learning rate scheduling
- ✅ Implemented early stopping
- ✅ Enhanced error handling
- ✅ Progress bars for better UX
- ✅ Windows compatibility improvements

### Performance
- ✅ Increased batch size for better GPU utilization
- ✅ Added weight decay (L2 regularization)
- ✅ Improved data validation
- ✅ Better corrupted image handling

### New Features
- ✅ Enhanced dataset analyzer
- ✅ Flexible train/val/test splits
- ✅ Comprehensive error reporting

See `CODE_IMPROVEMENTS.md` for detailed changes.

## 📄 License

This project is for educational and research purposes.
