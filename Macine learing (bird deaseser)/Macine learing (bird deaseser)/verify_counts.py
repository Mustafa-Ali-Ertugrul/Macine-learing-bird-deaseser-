
import os

data_dir = 'Macine learing (bird deaseser)/final_dataset_10_classes'
print(f"--- Final Dataset Validation: {data_dir} ---")

if os.path.exists(data_dir):
    all_ok = True
    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            images = [f for f in os.listdir(cls_path) if f.casefold().endswith(('.jpg', '.png', '.jpeg'))]
            count = len(images)
            status = "✅ OK" if count >= 500 else "❌ LOW"
            print(f"{cls}: {count} {status}")
            if count < 500:
                all_ok = False
    
    if all_ok:
        print("\nSUCCESS: All classes have at least 500 images.")
    else:
        print("\nFAILURE: Some classes are still under 500.")
else:
    print("Directory not found!")
