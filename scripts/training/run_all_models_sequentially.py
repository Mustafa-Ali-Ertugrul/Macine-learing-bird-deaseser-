import subprocess
import os
import time

scripts = [
    "train_model.py",                  # ResNet
    "train_simple_model.py",           # Simple CNN
    "train_poultry_disease_vit.py",    # ViT
    "train_poultry_disease_resnext.py",# ResNeXt
    "train_poultry_disease_convnext.py",# ConvNeXt
    "train_poultry_disease_cvt.py"     # CvT
]

print("🚀 Starting SEQUENTIAL training of ALL models...")
print(f"📋 Plan: {', '.join(scripts)}\n")

for i, script in enumerate(scripts):
    print(f"[{i+1}/{len(scripts)}] 🏃‍♂️ Running {script}...")
    start_time = time.time()
    
    try:
        # Check if file exists first
        if not os.path.exists(script):
             print(f"⚠️ Script not found, skipping: {script}")
             continue
             
        subprocess.run(["python", script], check=True)
        duration = time.time() - start_time
        print(f"✅ {script} finished successfully in {duration:.1f}s.\n")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {script} failed with error {e}\n")
    except Exception as e:
        print(f"❌ {script} failed with unexpected error {e}\n")

print("🎉 ALL DONE!")
