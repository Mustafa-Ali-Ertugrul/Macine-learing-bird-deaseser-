# 🐔 Complete Poultry Disease Classification Workflow

## 📋 Project Overview

This project provides a complete workflow for classifying poultry diseases using histopathology images and machine learning. The workflow includes data collection, organization, labeling, and model training.

## 🗂️ Directory Structure

```
.
├── organized_poultry_dataset/          # Main dataset directory
│   ├── healthy/                        # Healthy poultry images
│   ├── ib/                             # Infectious Bronchitis images
│   ├── ibd/                            # Infectious Bursal Disease images
│   ├── coccidiosis/                    # Coccidiosis images
│   ├── salmonella/                     # Salmonella images
│   ├── fatty_liver/                    # Fatty Liver Syndrome images
│   ├── histomoniasis/                  # Histomoniasis images
│   ├── unclassified/                   # Unclassified images (need labeling)
│   ├── dataset.csv                     # Complete dataset listing
│   ├── train.csv                       # Training set (70%)
│   ├── val.csv                         # Validation set (15%)
│   └── test.csv                        # Test set (15%)
├── poultry_labeling_tool.html          # Interactive labeling interface
├── dataset_organizer_helper.py         # Tool to organize dataset
├── train_poultry_disease_vit.py        # Model training script
└── prepare_training.py                 # Dataset preparation script
```

## 🚀 Getting Started

### 1. Label Your Images

Open the interactive labeling tool:
```bash
# Open poultry_labeling_tool.html in your web browser
```

Classify images from the `unclassified` folder into the appropriate disease categories.

### 2. Organize Classified Images

Use the dataset organizer helper to move images to their class directories:
```bash
python dataset_organizer_helper.py
```

### 3. Prepare for Training

Run the preparation script to create train/validation/test splits:
```bash
python prepare_training.py
```

### 4. Train the Model

Train a Vision Transformer model on your classified dataset:
```bash
python train_poultry_disease_vit.py
```

## 🧠 Model Architecture

The training script uses a Vision Transformer (ViT) model:
- **Base Model**: `vit_b_16` (pre-trained on ImageNet)
- **Input Size**: 224×224 pixels
- **Classes**: 7 poultry diseases + healthy
- **Epochs**: 20
- **Learning Rate**: 3e-4
- **Batch Size**: 32

## 📊 Expected Results

With a properly classified dataset of 465+ images, you can expect:
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Test Accuracy**: 75-85%

## 🛠️ Tools & Scripts

### `poultry_labeling_tool.html`
An interactive web-based tool for classifying poultry disease images.

### `dataset_organizer_helper.py`
A command-line tool to help move images between class directories and update the dataset CSV.

### `prepare_training.py`
Creates train/validation/test splits and generates the model training script.

### `train_poultry_disease_vit.py`
Trains a Vision Transformer model on the classified dataset.

## 📈 Next Steps

1. **Improve Dataset Quality**
   - Manually review and correct misclassified images
   - Collect more images for underrepresented classes
   - Augment existing images to increase dataset size

2. **Enhance Model Performance**
   - Fine-tune hyperparameters
   - Try different model architectures
   - Implement data augmentation techniques
   - Use transfer learning from domain-specific models

3. **Deploy the Model**
   - Create a web application for real-time prediction
   - Integrate with microscopy equipment
   - Develop a mobile app for field use

## 🤝 Need Help?

If you encounter any issues or need assistance with:
- Dataset labeling
- Model training
- Performance optimization
- Deployment

Feel free to ask for help at any stage of the workflow!