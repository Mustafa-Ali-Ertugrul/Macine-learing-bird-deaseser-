# 🐔 Poultry Disease Classification Project

This project provides a complete solution for classifying poultry diseases using histopathology images and machine learning.

## 📁 Dataset

We have collected and organized **465 histopathology images** of poultry tissues. These images are currently in the `organized_poultry_dataset/unclassified` folder and need to be classified into disease categories.

## 🛠️ Tools Included

1. **Interactive Labeling Tool** (`poultry_labeling_tool.html`)
   - Web-based interface for classifying images
   - Supports 10 disease categories plus healthy class

2. **Dataset Organization Scripts**
   - `dataset_organizer_helper.py` - Move images between classes
   - `prepare_training.py` - Create train/val/test splits

3. **Model Training**
   - `train_poultry_disease_vit.py` - Vision Transformer model training

## 🚀 Getting Started

1. **Label Images**
   - Open `poultry_labeling_tool.html` in your browser
   - Classify images from the unclassified folder

2. **Organize Dataset**
   ```bash
   python dataset_organizer_helper.py
   ```

3. **Prepare for Training**
   ```bash
   python prepare_training.py
   ```

4. **Train Model**
   ```bash
   python train_poultry_disease_vit.py
   ```

## 📊 Disease Categories

- 🟢 **Healthy** - Normal poultry tissue
- 🔴 **IB** - Infectious Bronchitis
- 🟠 **IBD** - Infectious Bursal Disease
- 🟡 **Coccidiosis** - Coccidiosis infection
- 🔵 **Salmonella** - Salmonella infection
- 🟣 **Fatty Liver** - Fatty Liver Syndrome
- 🟤 **Histomoniasis** - Histomoniasis (Blackhead disease)
- 🔶 **Newcastle** - Newcastle Disease (NDV)
- 🟢 **Marek** - Marek's Disease
- ⚫ **Avian Influenza** - Avian Influenza (Bird Flu)

## 📞 Support

For questions or issues with the project, please refer to the documentation files:
- `project_status.md` - Current project status
- `complete_workflow.md` - Detailed workflow instructions
=======
# Macine-learing-bird-deaseser-
>>>>>>> 14493a5a74487df461c40005ac51fac4e4106e69
