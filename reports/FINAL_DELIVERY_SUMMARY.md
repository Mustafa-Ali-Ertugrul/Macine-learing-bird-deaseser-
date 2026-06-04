# Final Delivery Summary

This project contains a multi-species poultry and cattle disease classification workflow. The final usable assets were reviewed and consolidated on 2026-06-03.

## Cattle Model

The strongest and safest model package is:

- `cattle_strict_efficientnet_kasa/best_model.pth`
- Architecture: `EfficientNet-B0`
- Classes: 10 cattle disease/health classes
- Internal strict split accuracy: 90.95%
- Macro F1: 90.17%
- Independent external candidate test accuracy: 75.85%

Supported cattle classes:

1. `Bovine_Tuberculosis`
2. `Dermatophilosis`
3. `Digital_Dermatitis`
4. `Foot_and_Mouth_Disease`
5. `Healthy`
6. `Hoof_Overgrowth`
7. `Lumpy_Skin_Disease`
8. `Mastitis`
9. `Pediculosis`
10. `Ringworm`

The package also includes:

- `final_summary.json`
- `classification_report.txt`
- `independent_test_summary.json`
- `KULLANIM.txt`

## Internet Video Test

A public cow video was downloaded from Pexels and tested with the cattle model:

- Source: https://www.pexels.com/video/herd-of-cows-during-daylight-855974/
- Local report folder: `reports/internet_cow_video_test`
- Tested frames: 12
- Top prediction on all frames: `Healthy`
- Average top confidence: 0.9968

Important files:

- `reports/internet_cow_video_test/internet_cow_video_summary.json`
- `reports/internet_cow_video_test/internet_cow_video_predictions.csv`
- `test_internet_cow_video.py`

## Duck Dataset Audit

The duck configuration targets 10 common poultry classes, but the real available duck image data is not 10-class complete.

Actual populated duck classes found across the workspace:

- `Bumblefoot`
- `Duck_Plague`
- `Fowl_Pox`

Key dataset counts:

- `dataset/duck`: `Bumblefoot 58`, `Duck_Plague 41`, `Fowl_Pox 58`
- `cleaned_dataset/duck`: `Bumblefoot 58`, `Duck_Plague 39`, `Fowl_Pox 58`
- `prepared_duck_3class_cleaned`: `Bumblefoot 58`, `Duck_Plague 39`, `Fowl_Pox 58`
- `duck_dataset_10_classes`: `Bumblefoot 34`, `Duck_Plague 40`, `Fowl_Pox 58`

`duck_split_dataset` exists, but no image files were found in its split folders.

## Goose Dataset Audit

The goose configuration also targets a 10-class poultry setup, but the real available goose image data is not 10-class complete.

Actual populated goose classes found across the workspace:

- `Fowl_Pox`
- `Goose_Parvovirus`

Key dataset counts:

- `dataset/goose`: `Fowl_Pox 46`, `Goose_Parvovirus 31`
- `cleaned_dataset/goose`: `Fowl_Pox 46`, `Goose_Parvovirus 29`
- `prepared_goose_2class_cleaned`: `Fowl_Pox 46`, `Goose_Parvovirus 29`
- `goose_dataset_10_classes`: `Fowl_Pox 46`, `Goose_Parvovirus 28`

`goose_split_dataset` exists, but no image files were found in its split folders.

## Audit Report

Detailed machine-readable audit:

- `reports/duck_goose_asset_audit.json`
- Script: `audit_duck_goose_assets.py`

## Recommended Thesis Wording

For cattle:

> A 10-class cattle disease classification model was trained with EfficientNet-B0 and tested on both the strict internal split and independent external candidate images.

For duck and goose:

> Duck and goose datasets were initially structured for a 10-class poultry disease classification target. However, after data collection and cleaning, reliable species-specific images were available only for a smaller subset of classes. Therefore, duck and goose experiments are reported as species-specific prototype datasets/models rather than complete 10-class models.

This avoids overstating the duck/goose results while keeping the project defensible.
