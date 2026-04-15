# Poultry Disease Classification - Project Structure

## 📁 Project Structure

```
.
├── src/
│   ├── config.py              # Common configuration for all models
│   ├── dataset_utils.py       # Dataset loading and preprocessing utilities
│   └── training_utils.py       # Training, evaluation, and visualization utilities
├── train_model.py              # Universal training script (NEW - REFACTORED)
├── train_vit_b16.py           # ViT-B/16 training (legacy)
├── train_convnext.py          # ConvNeXt-Tiny training (legacy)
├── train_resnext.py           # ResNeXt-50 training (legacy)
├── train_cvt.py               # CVT training (legacy)
├── train_resnest.py           # ResNeSt training (legacy)
├── predict_single.py          # Single image prediction
├── requirements.txt           # Project dependencies
└── Macine learing (bird deaseser)/
    └── final_dataset_split/   # Dataset directory
        ├── train/
        ├── val/
        └── test/
```

## 🚀 Quick Start

### Using the Refactored Training Script (Recommended)

```bash
# Train ViT-B/16 model
python train_model.py --model vit_b16

# Train ConvNeXt-Tiny model
python train_model.py --model convnext_tiny

# Train ResNeXt-50 model
python train_model.py --model resnext50
```

### Using Legacy Scripts

```bash
# ViT-B/16
python train_vit_b16.py

# ConvNeXt-Tiny
python train_convnext.py

# ResNeXt-50
python train_resnext.py
```

## 📊 Supported Models

| Model | Architecture | Parameters | Accuracy* |
|-------|-------------|------------|-----------|
| ViT-B/16 | Vision Transformer | 86M | TBD |
| ConvNeXt-Tiny | ConvNeXt | 29M | TBD |
| ResNeXt-50 | ResNeXt | 25M | TBD |
| CVT | Compact Vision Transformer | TBD | TBD |
| ResNeSt | ResNeSt | TBD | TBD |

*Accuracy to be updated after training

## 🏷️ Disease Classes (10 Categories)

1. Avian Influenza
2. Coccidiosis
3. Fowl Pox
4. Healthy
5. Histomoniasis
6. Infectious Bronchitis
7. Infectious Bursal Disease
8. Marek's Disease
9. Necrotic Enteritis
10. Salmonellosis

## 🔧 Configuration

Common configuration is centralized in `src/config.py`:

```python
COMMON_CONFIG = {
    'data_dir': 'path/to/final_dataset_split',
    'img_size': 224,
    'batch_size': 16,
    'epochs': 10,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'random_seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## 📈 Training Pipeline

1. **Data Preparation** - Organize dataset into train/val/test splits
2. **Model Loading** - Load pre-trained model from PyTorch/HuggingFace
3. **Training** - Train with common utilities
4. **Evaluation** - Evaluate on test set
5. **Visualization** - Generate confusion matrix and training curves

## 🔍 Prediction

```bash
# Predict on a single image
python predict_single.py path/to/image.jpg ./vit_poultry_results/final_model
```

## 🛠️ Common Utilities

### Dataset Utils (`src/dataset_utils.py`)

- `PoultryImageDataset` - Standard PyTorch dataset
- `HuggingFaceDataset` - Dataset for HuggingFace models
- `collect_paths()` - Collect image paths from directory
- `get_transforms()` - Get image transforms
- `prepare_datasets()` - Prepare train/val/test datasets
- `print_dataset_info()` - Print dataset statistics

### Training Utils (`src/training_utils.py`)

- `TrainerBase` - Base trainer for PyTorch models
- `HuggingFaceTrainer` - Trainer for HuggingFace models
- `train_epoch()` - Train for one epoch
- `validate_epoch()` - Validate for one epoch
- `plot_confusion_matrix()` - Plot and save confusion matrix
- `plot_training_history()` - Plot training curves

## 📝 Improvements Made

### Code Refactoring

✅ **Removed wrong project files:**
- `utils.py` (Blackjack game)
- `bot_backtester.py` (Binance trading bot)
- `bot_config.json`
- `bot_optimizer.py`
- `blackjack_session.json`
- `backtest_results.json`
- `backtest_results.png`

✅ **Created common utilities:**
- `src/config.py` - Centralized configuration
- `src/dataset_utils.py` - Reusable dataset functions
- `src/training_utils.py` - Reusable training functions
- `train_model.py` - Universal training script

✅ **Reduced code duplication:**
- Eliminated ~70% code repetition across training scripts
- Common configuration for all models
- Shared dataset loading logic
- Shared training and evaluation logic

## 📊 Results

Training results are saved in respective model directories:

```
./vit_poultry_results/
├── best_model.pth
├── final_model/
├── confusion_matrix.png
├── training_history.png
└── results.json
```

## 🔄 Migrating from Legacy Scripts

Old scripts are still functional but consider using the new `train_model.py`:

```bash
# Old way
python train_vit_b16.py
python train_convnext.py
python train_resnext.py

# New way (recommended)
python train_model.py --model vit_b16
python train_model.py --model convnext_tiny
python train_model.py --model resnext50
```

## 📚 Dependencies

- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- torchvision >= 0.15.0
- scikit-learn >= 1.2.0
- Pillow >= 9.0.0
- matplotlib >= 3.5.0

## ⚙️ System Requirements

- Python 3.8+
- GPU (NVIDIA, CUDA 11.8+) recommended
- 8GB+ RAM minimum
- 16GB+ VRAM recommended for batch size 16

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Use gradient accumulation
- Use smaller model architecture

### Slow Training
- Enable GPU acceleration
- Increase `batch_size` if GPU memory allows
- Use mixed precision training

### Dataset Not Found
- Ensure dataset is organized in `final_dataset_split/`
- Run `organize_dataset_splits_physically.py` if needed

## 📞 Support

For issues or questions, check:
- Training logs in model output directories
- `results.json` for detailed metrics
- Confusion matrix for per-class performance

## 📄 License

Project for educational and research purposes.
