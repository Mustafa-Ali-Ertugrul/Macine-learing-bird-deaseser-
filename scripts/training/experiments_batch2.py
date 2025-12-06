import subprocess
import os

scripts = [
    # "train_model.py", # Already running/ran in previous batch?
    # "train_simple_model.py",
    # "train_poultry_disease_vit.py",
    # I will add the new ones to run now.
    "train_poultry_disease_resnext.py",
    "train_poultry_disease_convnext.py",
    "train_poultry_disease_cvt.py"
]

for script in scripts:
    print(f"🚀 Running {script}...")
    try:
        subprocess.run(["python", script], check=True)
        print(f"✅ {script} finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {script} failed with error {e}")
