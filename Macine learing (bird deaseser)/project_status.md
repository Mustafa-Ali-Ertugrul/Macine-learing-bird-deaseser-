# 🐔 Poultry Disease Classification Project - Current Status

## 📁 Current Dataset Structure

We have successfully organized a poultry disease dataset with:
- **465 histopathology images** from `poultry_microscopy/all_images`
- **8 class directories**:
  - `healthy` (0 images)
  - `ib` (Infectious Bronchitis) (0 images)
  - `ibd` (Infectious Bursal Disease) (0 images)
  - `coccidiosis` (0 images)
  - `salmonella` (0 images)
  - `fatty_liver` (0 images)
  - `histomoniasis` (0 images)
  - `unclassified` (465 images)

## 📊 Dataset CSV

A `dataset.csv` file has been created with:
- Image paths
- Class labels
- Filenames

## 🛠️ Tools Available

1. **HTML Labeling Tool**: `poultry_labeling_tool.html`
   - Interactive web interface for classifying images
   - Supports all 7 disease classes plus healthy
   - Easy to use with dropdown selectors

2. **Dataset Organizer Helper**: `dataset_organizer_helper.py`
   - Command-line tool to move images between classes
   - Automatically updates the dataset CSV file

3. **Dataset Structure**: `organized_poultry_dataset/`
   - Ready for model training
   - Properly organized class directories

## 📝 Next Steps

### Option 1: Manual Classification (Recommended)
1. Open `poultry_labeling_tool.html` in your browser
2. Classify images in the "unclassified" folder
3. Move classified images to appropriate class folders
4. Run `dataset_organizer_helper.py` to update the CSV

### Option 2: Automatic Classification
1. Use a pre-trained model to classify images
2. Move images based on model predictions
3. Manually review and correct classifications

### Option 3: Train Your Own Model
1. Manually classify a subset of images (50-100 per class)
2. Train a Vision Transformer model
3. Use the trained model to classify remaining images
4. Review and correct model predictions

## 🚀 Model Training Preparation

To train a Vision Transformer model, you'll need:
1. Classified dataset (images in appropriate class folders)
2. Updated dataset.csv file
3. Python environment with PyTorch/TensorFlow
4. GPU support (recommended)

## 📞 Need Help?

If you need assistance with any part of this project:
1. Dataset organization
2. Model training
3. Image classification
4. Deployment

Just let me know which step you'd like to work on next!