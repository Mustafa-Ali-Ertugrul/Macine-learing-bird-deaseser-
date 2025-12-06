# 🐔 Poultry Disease Image Labeling Guide

## 📊 Current Dataset Status

- **Total images**: 446
- **Already labeled**: 17 (all classified as "healthy")
- **Need labeling**: 429 (classified as "unknown")

## 🛠️ Labeling Tools Available

### 1. Interactive Command-Line Tool
```bash
python label_images_interactive.py
```
This tool will guide you through labeling images one by one with a simple text interface.

### 2. Web-Based Labeling Tool
Open `poultry_labeling_tool.html` in your web browser for a graphical interface.

### 3. CSV Template for Manual Labeling
We've created `poultry_labeling_template.csv` which you can:
- Open in Excel or any spreadsheet application
- Add disease labels in the "new_disease" column
- Save and use with our update script

## 🏷️ Disease Categories

1. **healthy** - Normal, healthy poultry tissue
2. **ib** - Infectious Bronchitis
3. **ibd** - Infectious Bursal Disease
4. **coccidiosis** - Coccidiosis infection
5. **salmonella** - Salmonella infection
6. **fatty_liver** - Fatty Liver Syndrome
7. **histomoniasis** - Histomoniasis (Blackhead disease)
8. **other** - For images that don't fit the above categories

## 🚀 Recommended Approach

### Option 1: Quick Start with Interactive Tool
```bash
python label_images_interactive.py
```
This will show you images one by one and let you classify them quickly.

### Option 2: Batch Labeling with Spreadsheet
1. Open `poultry_labeling_template.csv` in Excel
2. Add labels to the "new_disease" column
3. Save the file
4. Run `python update_labels.py` to update your main dataset

## 📋 Tips for Accurate Labeling

1. **Take your time** - Accurate labels are crucial for model performance
2. **Use reference materials** - Keep poultry disease guides handy
3. **When in doubt, use "other"** - It's better to mark uncertain images as "other" than to guess
4. **Save progress regularly** - The interactive tool saves automatically after each image
5. **Start with obvious cases** - Label clear healthy/diseased images first to build confidence

## 🔄 After Labeling

Once you've labeled some images:
1. Retrain your model with `python train_simple_model.py`
2. Evaluate performance
3. Continue labeling more images to improve the model

## 📞 Need Help?

If you're unsure about what disease an image shows:
1. Look at the filename for clues (e.g., "Liver" might indicate a liver-related condition)
2. Check if there are similar images in the dataset
3. Consult veterinary pathology resources