# Product Overview

## Poultry Disease Classification System

This is a machine learning project for automated classification of poultry diseases using histopathology images. The system aims to help veterinarians and researchers quickly identify diseases in poultry through computer vision.

### Core Features

- **Multi-class disease classification** supporting 10 disease categories:
  - Avian Influenza, Coccidiosis, Fowl Pox, Healthy, Histomoniasis
  - Infectious Bronchitis, Infectious Bursal Disease, Marek's Disease
  - Newcastle Disease, Salmonella

- **Multiple model architectures** including Vision Transformers (ViT), ConvNeXt, ResNeXt, ResNeSt, and CVT

- **Web-based labeling tools** for dataset annotation and management

- **React-based web application** for disease prediction and visualization

### Current Status

⚠️ **CRITICAL**: The project has identified severe data leakage issues with 567 duplicate images between training and test sets, invalidating current model performance metrics. Dataset cleanup is required before reliable model training.

The project contains 23,342+ histopathology images but requires comprehensive data cleaning and proper train/test splitting to achieve reliable results.

### Target Users

- Veterinary researchers and pathologists
- Poultry industry professionals
- Agricultural research institutions
- Machine learning researchers in medical imaging

### Key Challenges

- **Data Quality**: Severe data leakage detected requiring immediate cleanup
- **Dataset Organization**: Multiple duplicate categories and inconsistent naming
- **Model Validation**: Current high accuracy scores (99%+) are artificially inflated