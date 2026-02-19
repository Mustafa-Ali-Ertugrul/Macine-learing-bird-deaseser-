# 🐔 Deep Learning Poultry Disease Classification Report
**Generated:** 2025-12-06 21:50

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

