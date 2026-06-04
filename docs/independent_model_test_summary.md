# Independent Model Test Summary

Date: 2026-06-04

Imported archive:

- `imports/kanatli_hayvan_modelleri_final_v2/kanatli_hayvan_modelleri_final_v2.zip`

Evaluation command:

```powershell
.\ml_venv\Scripts\python.exe .\scripts\evaluation\evaluate_independent_models.py
```

The detailed prediction CSV files were intentionally left out of git. The script can regenerate them under `reports/independent_model_tests/`.

## Dataset Coverage

| Species | Independent samples | Unsupported samples skipped |
| --- | ---: | ---: |
| chicken | 517 | 44 |
| duck | 242 | 19 |
| goose | 141 | 40 |
| cattle | 307 | 0 |

Exact training-overlap hashing is available with `--check-overlap`, but it is disabled by default because hashing all local training images is slow.

## Results

| Model | Species | Images | Accuracy | Macro F1 | Status |
| --- | --- | ---: | ---: | ---: | --- |
| chicken_hf_vit_b16 | chicken | 517 | 0.9516 | 0.1950 | ok |
| chicken_hf_convnext_tiny | chicken | 517 | 0.9652 | 0.2456 | ok |
| chicken_hf_cvt_13 | chicken | 517 | 0.9323 | 0.1930 | ok |
| chicken_pth_resnext50 | chicken | 517 | 0.9981 | 0.4995 | ok |
| chicken_pth_resnest50d | chicken | 517 | 0.9632 | 0.1635 | ok |
| legacy_resnet18 | chicken | 517 | 0.8801 | 0.1170 | ok |
| legacy_convnext_tiny | chicken | 517 | 0.8259 | 0.1131 | ok |
| legacy_cvt_13 | chicken | 517 | 0.9710 | 0.1642 | ok |
| cattle_efficientnet_b0 | cattle | 307 | 0.3876 | 0.2800 | ok |
| cattle_mobilenet_v2 | cattle | 307 | 0.3648 | 0.2304 | ok |
| cattle_resnet50 | cattle | 307 | 0.3518 | 0.2502 | ok |
| cattle_resnext50 | cattle | 307 | 0.3550 | 0.2517 | ok |
| cattle_vit_b16 | cattle | 307 | 0.2769 | 0.1610 | ok |
| cattle_strict_efficientnet_b0 | cattle | 307 | 0.5831 | 0.3505 | ok |
| duck_efficientnet_b0 | duck | 242 | 0.6446 | 0.2767 | ok |
| goose_efficientnet_b0 | goose | 141 | 0.4610 | 0.1780 | ok |

## Incompatible Checkpoints

- `chicken_pth_vit_b16`: saved with a 1-class classifier head, so it cannot be loaded into the current 10-class ViT model.
- `legacy_simple_cnn`: saved with an 8-class classifier head, so it cannot be loaded into the current 10-class simple CNN.

## Notes

- Chicken independent samples are heavily concentrated in `Fowl_Pox`, so high chicken accuracy should not be interpreted as broad 10-class generalization.
- Cattle candidate labels `Bovine_Respiratory_Disease` and `Dermatophytosis` are mapped to configured classes `Bovine_Tuberculosis` and `Ringworm` for evaluation compatibility.
