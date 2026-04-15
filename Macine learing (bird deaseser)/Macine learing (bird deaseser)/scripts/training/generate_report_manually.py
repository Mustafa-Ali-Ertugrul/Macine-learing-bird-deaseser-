# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime

# Consolidated Results from manual observation of "evaluate_all.py" logs
results = {
    "ResNet18": {"accuracy": 0.8163, "precision": 0.81, "recall": 0.81, "f1": 0.81, "epoch": 10},
    "Simple CNN": {"accuracy": 0.00, "precision": 0.00, "recall": 0.00, "f1": 0.00, "epoch": 10},
    "ViT": {"accuracy": 0.8755, "precision": 0.88, "recall": 0.87, "f1": 0.87, "epoch": 10},
    "ResNeXt": {"accuracy": 0.8520, "precision": 0.85, "recall": 0.85, "f1": 0.85, "epoch": 10},
    "ConvNeXt": {"accuracy": 0.7011, "precision": 0.72, "recall": 0.70, "f1": 0.70, "epoch": 10},
    "CvT": {"accuracy": 0.8800, "precision": 0.84, "recall": 0.88, "f1": 0.85, "epoch": 10}
}

classes = ['Avian Influenza', 'Coccidiosis', 'Fowl Pox', 'Healthy', 'Histomoniasis', 
           'Infectious Bronchitis', 'Infectious Bursal Disease', "Marek's Disease", 
           'Newcastle Disease', 'Salmonella']

def generate_report():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    md_content = f"""# 🐔 Deep Learning Poultry Disease Classification Report
**Generated:** {timestamp}

## 1. Executive Summary
This report summarizes the performance of 6 deep learning models trained on the **10-Class Poultry Disease Dataset** (5000 images).

**Best Model:** **CvT (Convolutional Vision Transformer)**
**Accuracy:** **88.00%**
**Classes:** 10 (Avian Influenza, Coccidiosis, Fowl Pox, Healthy, Histomoniasis, IB, IBD, Marek's, NCD, Salmonella)

## 2. Model Performance Comparison

| Model | Accuracy | F1-Score | Status |
|-------|----------|----------|--------|
| **CvT** | **88.00%** | **0.85** | ✅ Best |
| ViT | 87.55% | 0.87 | ✅ Excellent |
| ResNeXt | 85.20% | 0.85 | ✅ Very Good |
| ResNet18 | 81.63% | 0.81 | ✅ Good |
| ConvNeXt | 70.11% | 0.70 | ⚠️ Underfitting |
| Simple CNN | 0.00% | 0.00 | ❌ Failed |

## 3. Detailed Class Performance (Best Model: CvT)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Avian Influenza | 0.44 | 0.11 | 0.17 | 102 |
| Coccidiosis | 0.52 | 0.94 | 0.66 | 110 |
| Fowl Pox | 1.00 | 0.99 | 0.99 | 94 |
| Healthy | 0.94 | 0.97 | 0.96 | 99 |
| Histomoniasis | 1.00 | 1.00 | 1.00 | 94 |
| Infectious Bronchitis | 1.00 | 1.00 | 1.00 | 104 |
| Infectious Bursal Disease (IBD) | 1.00 | 1.00 | 1.00 | 105 |
| Marek's Disease | 0.99 | 1.00 | 1.00 | 108 |
| Newcastle Disease | 0.94 | 0.95 | 0.95 | 85 |
| Salmonella | 0.98 | 0.95 | 0.96 | 100 |

### Insights
*   **Perfect Detection:** The model achieves near **100% accuracy** on Fowl Pox, Histomoniasis, IB, IBD, and Marek's Disease.
*   **Challenges:** The model struggles significantly with **Avian Influenza** (Recall 0.11), often confusing it with Coccidiosis (which has high Recall but low Precision).
*   **Recommendation:** Collect more distinct data for Avian Influenza to improve discrimination.

## 4. Training Statistics
*   **Total Images:** 4,976 (after cleaning)
*   **Training Time:** ~60 minutes total
*   **Hardware:** GPU Accelerated

"""
    
    with open("FINAL_REPORT.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print("Report generated: FINAL_REPORT.md")

if __name__ == "__main__":
    generate_report()
