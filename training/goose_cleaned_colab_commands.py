
# ════════════════════════════════════════════
# GOOSE CLEANED - COLAB EGITIM KOMUTLARI
# ════════════════════════════════════════════

# 1. Repo guncelle
!cd Macine-learing-bird-deaseser- && git pull

# ────────────────────────────
# DENEY 1: Frozen baseline
# ────────────────────────────
!cd Macine-learing-bird-deaseser- && python training/train.py \
    --data-dir prepared_goose_2class_cleaned \
    --output-dir outputs/goose_cleaned_v2_frozen \
    --model efficientnet_b0 \
    --freeze-backbone \
    --epochs 40 --lr 1e-3 --batch-size 8 \
    --label-smoothing 0.05 --patience 12

# ────────────────────────────
# DENEY 2: Partial unfreeze
# ────────────────────────────
!cd Macine-learing-bird-deaseser- && python training/train.py \
    --data-dir prepared_goose_2class_cleaned \
    --output-dir outputs/goose_cleaned_v3_unfreeze \
    --model efficientnet_b0 \
    --freeze-backbone --unfreeze-last-n 2 \
    --epochs 30 --lr 1e-4 --batch-size 8 \
    --label-smoothing 0.05 --patience 10 \
    --optimizer adamw --weight-decay 1e-4

# ────────────────────────────
# DENEY 3: ResNet18
# ────────────────────────────
!cd Macine-learing-bird-deaseser- && python training/train.py \
    --data-dir prepared_goose_2class_cleaned \
    --output-dir outputs/goose_cleaned_resnet18 \
    --model resnet18 \
    --freeze-backbone --unfreeze-last-n 1 \
    --epochs 30 --lr 5e-4 --batch-size 8 \
    --patience 10

# ────────────────────────────
# SONUC KARSILASTIRMA
# ────────────────────────────
import json, os, csv

experiments = [
    "goose_v2_frozen", "goose_v3_unfreeze", "goose_resnet18",
    "goose_cleaned_v2_frozen", "goose_cleaned_v3_unfreeze", "goose_cleaned_resnet18",
]

results = []
print(f"{'Deney':<35} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
print("-" * 70)

for exp in experiments:
    path = f"Macine-learing-bird-deaseser-/outputs/{exp}/metrics.json"
    dtype = "cleaned" if "cleaned" in exp else "original"
    if os.path.exists(path):
        m = json.load(open(path))
        print(f"{exp:<35} {m['accuracy']:>7.2f}% {m['macro_f1']:>7.2f}% "
              f"{m['macro_precision']:>7.2f}% {m['macro_recall']:>7.2f}%")
        results.append({"experiment": exp, "dataset": dtype, **m})
    else:
        print(f"{exp:<35} {'--':>8} {'--':>8}")

# En iyi model
if results:
    best = max(results, key=lambda x: (x["macro_f1"], x["accuracy"]))
    print(f"\nEN IYI MODEL: {best['experiment']}")
    print(f"  Macro F1: {best['macro_f1']}% | Accuracy: {best['accuracy']}%")

    # Karsilastirma kaydet
    with open("Macine-learing-bird-deaseser-/outputs/goose_model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # En iyi modeli kopyala
    best_src = f"Macine-learing-bird-deaseser-/outputs/{best['experiment']}"
    best_dst = "Macine-learing-bird-deaseser-/outputs/goose_best_final"
    os.makedirs(best_dst, exist_ok=True)
    for fn in ["best_model.pth", "final_summary.json", "confusion_matrix.png",
               "classification_report.txt", "metrics.json"]:
        src = f"{best_src}/{fn}"
        if os.path.exists(src):
            import shutil; shutil.copy2(src, f"{best_dst}/{fn}")
    print(f"  En iyi model kopyalandi: {best_dst}/")
